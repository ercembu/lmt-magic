"""Self-play training loop for one MageZero agent that plays any sealed deck of a set.

MageZero (github.com/WillWroble/MageZero) trains an agent per fixed deck against
named opponents. Card values for a set need one agent that plays thousands of
random sealed pools, so this loop drives their pieces -- the XMage runner (on
the lmt AI-only port of their AI), their inference server and their trainer --
over random deck pairs:

    gen g:  play N games, both seats the agent at version g (gen 0: search only)
            -> pool/gen{g}/*.hdf5
            train version g+1 on the last WINDOW generations, starting from g

Every heavy step runs under a systemd memory limit (the laptop has 30GB and the
main simulation is running alongside). A finished generation is skipped on
restart.

    mz_loop.py --gens 6 --games 120 --budget 150
"""
import argparse, glob, json, os, random, re, shutil, subprocess, sys, time, urllib.request
import yaml

import mz_split

SIM = os.environ.get("LMT_SIM", "/home/ahmet/pers/lmt-magic-sim")   # workspace made by sim/mzport/setup.sh
MZ = f"{SIM}/mzport/magezero-py"
PY = f"{SIM}/.venv-ml/bin/python"
AGENT = "FRA_SEALED"
PORT = 50061
PLAY_MEM = "14G"   # the runner's cap; its heap is MZ_XMX (11g) in mz-xmage.sh, plus ~2G off-heap


def curriculum(g):
    """MageZero's own schedule (configs/curriculum.yml)."""
    td = {0: 0.95, 1: 0.92, 2: 0.85}.get(g, 0.70)
    return {"td_discount": td, "prior_temperature": 1.5,
            "binary": g >= 2, "priority": g >= 3, "target": g >= 10, "opponent": False}


def capped(unit, mem, cmd, cwd, log, runtime=None, env=None):
    args = ["systemd-run", "--user", f"--unit={unit}", "-p", f"MemoryMax={mem}",
            "-p", "MemorySwapMax=512M", "--wait", "--collect"]
    if runtime:
        args += ["-p", f"RuntimeMaxSec={runtime}"]
    for k, v in (env or {}).items():
        args += ["-E", f"{k}={v}"]
    args += ["sh", "-c", f"cd {cwd} && exec {cmd} > {log} 2>&1"]
    return subprocess.run(args, capture_output=True, text=True).returncode


def write_game_yml(g, out_dir, pairs_file, games, threads, budget, online=(False, False)):
    """online: whether seat A / seat B searches with the network (else XMage's heuristic only)."""
    cfg = yaml.safe_load(open(f"{MZ}/configs/game.yml"))
    c = curriculum(g)
    for side, net in zip(("player_a", "player_b"), online):
        p = cfg[side]
        p["deckPath"] = open(pairs_file).readline().split("\t")[0]   # overridden per game
        p["type"] = "mcts"
        p["output_file"] = f"{out_dir}/{side}.hdf5"
        p["mcts"].update(offline_mode=not net, td_discount=c["td_discount"],
                         search_budget=budget, timeout_ms=4000)
        p["priors"].update(binary=c["binary"] and net, priority=c["priority"] and net,
                           target=c["target"] and net, opponent=c["opponent"] and net,
                           prior_temperature=c["prior_temperature"])
    cfg["training"].update(games=games, threads=threads, pairs_file=pairs_file,
                           minimum_deck_size=40, max_turns=30)
    # self-play: the same model on two server processes, one per seat -- a single Python server pegged one
    # core and left the GPU ~90% idle (each request waited ~53ms, mostly in preprocessing)
    cfg["server"].update(port=PORT, opponent_port=PORT + 1)
    cfg["logging"]["record_dir"] = f"{out_dir}/replays"   # replays for sim/viewer.html
    path = f"{out_dir}/game.yml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path


# server processes per port (they share it via SO_REUSEPORT). Measured on the gate: 4 processes on one port
# played 16 games in 6.7 min against 4.5 min for 1 -- separate CUDA contexts time-slice the GPU on tiny batches
WORKERS_PER_PORT = 1


def start_server(version, log, ports=(PORT, PORT + 1)):
    """One port per seat that searches with the network. A single Python process per port is GIL-bound on
    request decoding: the MCTS threads waited on it ~60% of the time while the GPU sat at ~10%."""
    per = WORKERS_PER_PORT
    for port in ports:
        for k in range(per):
            subprocess.run(["systemd-run", "--user", f"--unit=lmt-mzserver{port}-{k}", "-p", "MemoryMax=3G",
                            "--collect", "sh", "-c",
                            f"cd {MZ} && exec {PY} src/magezero/server.py --deck {AGENT} "
                            f"--version {version} --port {port} > {log}.{port}.{k} 2>&1"], capture_output=True)
    for port in ports:
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
                break
            except Exception:
                time.sleep(1)
        else:
            return False
    time.sleep(10)   # the port answers once the first worker is up; give the rest time to load the model
    return True


def stop_server():
    subprocess.run(["systemctl", "--user", "stop", "lmt-mzserver*"], capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--games", type=int, default=120, help="games per generation")
    ap.add_argument("--threads", type=int, default=14, help="game threads for search-only play (CPU-bound)")
    ap.add_argument("--threads-online", type=int, default=24,
                    help="game threads when a seat uses the network: they park while waiting on it, and more of "
                         "them in flight fill the server's batches (a batch costs ~25ms from 1 state to 128)")
    ap.add_argument("--budget", type=int, default=300, help="MCTS simulations per decision")
    ap.add_argument("--window", type=int, default=8, help="generations of data to train on")
    ap.add_argument("--gate-games", type=int, default=60, help="network vs search-only games before it may lead")
    ap.add_argument("--gate-rate", type=float, default=0.55, help="win rate the network needs to lead self-play")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--decks", default=f"{SIM}/simdecks")
    ap.add_argument("--profile-minutes", type=int, default=0, help="flight-record the first N minutes of play")
    a = ap.parse_args()

    decks = sorted(glob.glob(f"{a.decks}/*.dck"))
    pool = f"{MZ}/pool/{AGENT}"
    os.makedirs(pool, exist_ok=True)
    state_p = f"{pool}/state.json"
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"gens": {}}

    for g in range(a.gens):
        gdir = f"{pool}/gen{g}"
        done = state["gens"].get(str(g), {})
        os.makedirs(gdir, exist_ok=True)
        t0 = time.time()

        # Gate: a network leads self-play only after it beats the search-only agent head-to-head. v1, trained
        # on 120 games, made the search worse than XMage's heuristic (players kept cards, skipped land drops,
        # half the games hit the turn limit) -- the classic AlphaZero bootstrap failure.
        online = False
        if g > 0 and not done.get("played"):
            gate = done.get("gate")
            if gate is None:
                gd = f"{gdir}/gate"
                os.makedirs(gd, exist_ok=True)
                rng = random.Random(500 + g)
                gp = f"{gd}/pairs.tsv"
                with open(gp, "w") as f:
                    for _ in range(a.gate_games):
                        x, y = rng.sample(decks, 2)
                        f.write(f"{x}\t{y}\n")
                gy = write_game_yml(g, gd, gp, a.gate_games, a.threads_online, a.budget, online=(True, False))
                if not start_server(g, f"{gd}/server.log", ports=(PORT,)):
                    stop_server()
                    sys.exit(f"gen {g}: model server for v{g} did not come up (see {gd}/server.log)")
                capped("lmt-mzplay", PLAY_MEM, f"{MZ}/xmage/mz-xmage.sh {gy}", gd, f"{gd}/play.log", runtime=6 * 3600)
                stop_server()
                glog = open(f"{gd}/play.log", errors="replace").read()
                m = re.findall(r"Player A win rate: [\d.NaN]+% \((\d+)/(\d+)\)", glog)
                w, n = (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)
                gate = {"wins": w, "games": n, "rate": round(w / n, 3) if n else None,
                        "passed": bool(n) and w / n >= a.gate_rate}
                done["gate"] = gate
                state["gens"][str(g)] = done
                json.dump(state, open(state_p, "w"), indent=1)
                print(f"gen {g}: gate v{g} vs search-only: {w}/{n} = {gate['rate']} -> "
                      f"{'network leads' if gate['passed'] else 'search-only this generation'}", flush=True)
            online = gate["passed"]

        if not done.get("played"):
            rng = random.Random(1000 + g)
            pairs = f"{gdir}/pairs.tsv"
            with open(pairs, "w") as f:
                for _ in range(a.games):
                    x, y = rng.sample(decks, 2)
                    f.write(f"{x}\t{y}\n")
            yml = write_game_yml(g, gdir, pairs, a.games, a.threads_online if online else a.threads, a.budget,
                                 online=(online, online))
            if online and not start_server(g, f"{gdir}/server.log"):
                stop_server()
                sys.exit(f"gen {g}: model server for v{g} did not come up (see {gdir}/server.log)")
            env = {}
            if a.profile_minutes:
                # Java Flight Recorder over the first minutes of play: where the CPU time goes
                env["MZ_JAVA_OPTS"] = (f"-XX:StartFlightRecording=duration={a.profile_minutes}m,"
                                       f"settings=profile,filename={gdir}/profile.jfr")
            rc = capped("lmt-mzplay", PLAY_MEM, f"{MZ}/xmage/mz-xmage.sh {yml}", gdir, f"{gdir}/play.log",
                        runtime=6 * 3600, env=env)
            stop_server()
            log = open(f"{gdir}/play.log", errors="replace").read()
            ok = log.count("Successful:") and int(log.rsplit("Successful:", 1)[1].split()[0])
            print(f"gen {g}: played {ok}/{a.games} games (exit {rc}) in {(time.time() - t0) / 60:.0f} min",
                  flush=True)
            done.update(played=True, games=ok, play_min=round((time.time() - t0) / 60, 1), online=online)
            state["gens"][str(g)] = done
            json.dump(state, open(state_p, "w"), indent=1)

        if not done.get("trained"):
            v = g + 1
            tdir, vdir = f"{MZ}/data/{AGENT}/ver{v}/training", f"{MZ}/data/{AGENT}/ver{v}/testing"
            shutil.rmtree(f"{MZ}/data/{AGENT}/ver{v}", ignore_errors=True)
            os.makedirs(tdir), os.makedirs(vdir)
            for k in range(max(0, g - a.window + 1), g + 1):
                for f in glob.glob(f"{pool}/gen{k}/*.hdf5"):
                    name = f"gen{k}_{os.path.basename(f)}"
                    if k == g:
                        # both seats train; the newest generation's last 10% of games (both seats) validate.
                        # Holding out seat B validated on the same games from the other side, with flipped
                        # labels: v1 memorized which cards won each game and scored worse than the mean
                        # (value 0.344 vs 0.300), then won 5/60 at the gate
                        mz_split.split(f, f"{tdir}/{name}", f"{vdir}/{name}")
                    else:
                        os.symlink(f, f"{tdir}/{name}")
            ckpt = []
            if g > 0:
                os.makedirs(f"{MZ}/models/{AGENT}/ver{v}", exist_ok=True)
                shutil.copy(f"{MZ}/models/{AGENT}/ver{g}/model.pt.gz", f"{MZ}/models/{AGENT}/ver{v}/model.pt.gz")
                ckpt = ["--checkpoint"]
            t1 = time.time()
            rc = capped("lmt-mztrain", "8G",
                        f"{PY} src/magezero/train.py --deck {AGENT} --version {v} --epochs {a.epochs} "
                        + " ".join(ckpt), MZ, f"{gdir}/train.log", env={"PYTHONUNBUFFERED": "1"})
            tail = [l for l in open(f"{gdir}/train.log", errors="replace") if "Validation loss" in l]
            print(f"gen {g}: trained v{v} (exit {rc}) in {(time.time() - t1) / 60:.0f} min; "
                  f"{tail[-1].strip() if tail else 'no validation line'}", flush=True)
            if rc != 0 or not os.path.exists(f"{MZ}/models/{AGENT}/ver{v}/model.pt.gz"):
                sys.exit(f"gen {g}: training failed (see {gdir}/train.log)")
            # deploy the best-validation epoch, not the last: on small early generations the last epoch
            # overfits (gen 0 -> v1: value loss 0.31 at epoch 1, 0.81 at epoch 3)
            mdir = f"{MZ}/models/{AGENT}/ver{v}"
            if os.path.exists(f"{mdir}/best.pt.gz"):
                shutil.copy(f"{mdir}/model.pt.gz", f"{mdir}/last_epoch.pt.gz")
                shutil.copy(f"{mdir}/best.pt.gz", f"{mdir}/model.pt.gz")
            done.update(trained=True, val=tail[-1].strip() if tail else None)
            state["gens"][str(g)] = done
            json.dump(state, open(state_p, "w"), indent=1)


if __name__ == "__main__":
    main()
