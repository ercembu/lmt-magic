"""Progress of the MageZero self-play loop (scripts/mz_loop.py), for the viewer's Progress tab.

    selfplay_stats.py <pool dir> <out.json> [games per generation]

Per generation: games played and failed, minutes, results of its games, the gate
match, and every epoch the trainer reported (training and validation losses,
priority accuracy) -- the numbers that say whether the network is learning. Plus
what is running right now: the training step in progress, and the machine's load.
"""
import glob, gzip, json, os, re, subprocess, sys, time
from datetime import datetime

LOSS = re.compile(r"(priority|choose_target|choose_use|value)_loss=([\d.]+)")
TS = re.compile(r"^\w+\s+(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
NUM = re.compile(r"(\w+)=([\d.]+)")


def losses(line):
    return {k: float(v) for k, v in LOSS.findall(line or "")}


def play_progress(log, cache=None):
    if not os.path.exists(log):
        return None
    st = os.stat(log)
    key = f"log:{log}:{st.st_size}:{int(st.st_mtime)}"
    if cache is not None and key in cache:
        return cache[key]
    done = failed = 0
    first = last = None
    with open(log, errors="replace") as f:
        for line in f:
            if "completed successfully" in line:
                done += 1
            elif "A game simulation failed" in line:
                failed += 1
            m = TS.match(line)
            if m:
                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                first = first or t
                last = t
    out = {"done": done, "failed": failed, "start": first, "last": last}
    if cache is not None:
        for k in [k for k in cache if k.startswith(f"log:{log}:")]:
            del cache[k]   # older sizes of the same log
        cache[key] = out
    return out


def gate_games(gdir):
    try:
        return sum(1 for _ in open(os.path.join(gdir, "gate", "pairs.tsv")))
    except OSError:
        return None


def train_curve(path):
    """Every epoch of a train.log ({epoch, train, val, acc, val_acc, val_acc_nonpass, seconds}) and the
    last in-epoch progress line ({epoch, epochs, step, steps, elapsed, losses, acc}), if any."""
    epochs, live = [], None
    if not os.path.exists(path):
        return epochs, live
    cur = None
    for line in open(path, errors="replace"):
        if line.startswith("Progress "):
            d = dict(NUM.findall(line))
            e, ep = line.split("epoch=")[1].split()[0].split("/")
            s, st = line.split("step=")[1].split()[0].split("/")
            live = {"epoch": int(e), "epochs": int(ep), "step": int(s), "steps": int(st),
                    "elapsed": float(d.get("elapsed", "0").rstrip("s") or 0), "losses": losses(line),
                    "acc": float(d["priority_acc"]) if "priority_acc" in d else None}
        elif line.startswith("Epoch "):
            n = int(line.split()[1])
            if cur is None or cur["epoch"] != n:
                cur = {"epoch": n, "train": {}, "val": {}}
                epochs.append(cur)
            if "_loss=" in line:
                cur["train"] = losses(line)
            d = dict(NUM.findall(line))
            if "priority_acc" in d:
                cur["acc"] = float(d["priority_acc"])
            if "time" in d:
                cur["seconds"] = float(d["time"])
        elif line.startswith("Validation loss") and cur is not None:
            cur["val"] = losses(line)
            m = re.search(r"avg_total_loss=([\d.]+)", line)
            if m:
                cur["val_total"] = float(m.group(1))
        elif line.startswith("Test priority_accuracy") and cur is not None:
            d = dict(NUM.findall(line))
            cur["val_acc"] = float(d.get("priority_accuracy", 0))
            cur["val_acc_nonpass"] = float(d.get("non", 0)) if "non" in d else None
            m = re.search(r"non-pass=([\d.]+)", line)
            if m:
                cur["val_acc_nonpass"] = float(m.group(1))
    # a progress line from an epoch that has since finished is not live
    if live and epochs and epochs[-1]["epoch"] >= live["epoch"] and epochs[-1].get("train"):
        live = None
    return epochs, live


_CACHE = {}


def results(replay_dir, cache):
    """Winner and length of each recorded game (cached by file name: replays don't change once written)."""
    out = []
    for f in sorted(glob.glob(os.path.join(replay_dir, "*.json.gz"))):
        key = os.path.relpath(f, "/")
        if key not in cache:
            try:
                g = json.load(gzip.open(f, "rt", encoding="utf8"))
            except (OSError, EOFError, ValueError):
                continue   # still being written
            cache[key] = [g.get("winner"), g.get("turns")]
        out.append(cache[key])
    if not out:
        return None
    turns = [t for _, t in out if t]
    return {"games": len(out), "A": sum(w == "A" for w, _ in out), "B": sum(w == "B" for w, _ in out),
            "draws": sum(w not in ("A", "B") for w, _ in out),
            "avg_turns": round(sum(turns) / len(turns), 1) if turns else None}


def system():
    """What is running and how loaded the machine is."""
    s = {}
    try:
        u, m, t = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=5).stdout.strip().split(", ")
        s["gpu"] = {"util": int(u), "mem_gb": round(int(m) / 1024, 1), "mem_total_gb": round(int(t) / 1024, 1)}
    except Exception:
        # AMD (amdgpu) has no nvidia-smi; the driver exposes the same numbers in sysfs
        for d in sorted(glob.glob("/sys/class/drm/card*/device")):
            try:
                rd = lambda f: int(open(f"{d}/{f}").read())
                s["gpu"] = {"util": rd("gpu_busy_percent"), "mem_gb": round(rd("mem_info_vram_used") / 2**30, 1),
                            "mem_total_gb": round(rd("mem_info_vram_total") / 2**30, 1)}
                break
            except Exception:
                pass
    try:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        s["ram"] = {"used_gb": round((mi["MemTotal"] - mi["MemAvailable"]) / 1e6, 1),
                    "total_gb": round(mi["MemTotal"] / 1e6, 1)}
        s["load"] = round(float(open("/proc/loadavg").read().split()[0]), 1)
        s["cores"] = os.cpu_count()
    except Exception:
        pass
    try:
        units = subprocess.run(["systemctl", "--user", "list-units", "lmt-mz*", "--no-legend", "--plain",
                                "--state=active"], capture_output=True, text=True, timeout=5).stdout.split()
        units = [u.split(".")[0] for u in units if u.endswith(".service")]
        names = {"lmt-mzplay": "games", "lmt-mztrain": "training"}
        s["running"] = sorted({names.get(u, "model server" if u.startswith("lmt-mzserver") else None)
                               for u in units} - {None})
    except Exception:
        pass
    return s


def main(pool, out, per_gen=120):
    state_p = os.path.join(pool, "state.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"gens": {}}
    cache_p = out + ".cache"
    try:
        cache = json.load(open(cache_p))
    except (OSError, ValueError):
        cache = {}
    gens, live_train = [], None
    for gdir in sorted(glob.glob(os.path.join(pool, "gen*")), key=lambda d: int(re.sub(r"\D", "", os.path.basename(d)) or 0)):
        if not re.fullmatch(r"gen\d+", os.path.basename(gdir)):
            continue
        g = int(re.sub(r"\D", "", os.path.basename(gdir)))
        st = state["gens"].get(str(g), {})
        pp = play_progress(os.path.join(gdir, "play.log"), cache) or {}
        epochs, live = train_curve(os.path.join(gdir, "train.log"))
        done_ep = [e for e in epochs if e.get("val_total") is not None]
        best = min(done_ep, key=lambda e: e["val_total"]) if done_ep else (epochs[-1] if epochs else None)
        status = ("trained" if st.get("trained") else "training" if st.get("played")
                  else "playing" if pp else "waiting")
        if status == "training" and live:
            live_train = dict(live, gen=g, version=g + 1)
        target = per_gen
        gate = st.get("gate")
        gp = play_progress(os.path.join(gdir, "gate", "play.log"), cache)
        gate_live = results(os.path.join(gdir, "gate", "replays"), cache)
        if not pp and gp and not gate:
            # before a generation plays, its network has to beat the search-only agent (mz_loop.py's gate)
            pp, status, target = gp, f"gate: v{g} vs search-only", gate_games(gdir) or per_gen
        gate_txt = None
        if gate:
            gate_txt = (f"{gate['wins']}/{gate['games']} = {gate['rate']}: "
                        + ("network leads" if gate.get("passed") else "search-only")) if gate.get("games") else "skipped"
        gens.append({"gen": g, "status": status, "target": target, "gate": gate_txt, "gate_live": gate_live,
                     "online": st.get("online"),
                     "done": pp.get("done", 0), "failed": pp.get("failed", 0),
                     "minutes": st.get("play_min") or (round((pp["last"] - pp["start"]) / 60, 1)
                                                         if pp.get("start") and pp.get("last") else None),
                     "results": results(os.path.join(gdir, "replays"), cache),
                     "val": (best or {}).get("val", {}), "train": (best or {}).get("train", {}),
                     "val_acc": (best or {}).get("val_acc"), "val_acc_nonpass": (best or {}).get("val_acc_nonpass"),
                     "epochs": epochs, "note": st.get("note")})
    cur = next((g for g in gens if g["status"] in ("playing", "training") or g["status"].startswith("gate")),
               gens[-1] if gens else None)
    rate = None
    if cur and (cur["status"] == "playing" or cur["status"].startswith("gate")) and cur["minutes"]:
        rate = round(60 * cur["done"] / max(cur["minutes"], 1e-6), 1)
    stats = {"mode": "selfplay", "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "per_gen": per_gen,
             "current": cur, "games_per_hour": rate, "training": live_train, "system": system(),
             "eta_min": round((cur["target"] - cur["done"] - cur["failed"]) / rate * 60) if rate else None,
             "total_games": sum(g["done"] for g in gens if not g["status"].startswith("gate")),
             "total_failed": sum(g["failed"] for g in gens),
             "gens": gens}
    # who played each seat, by the replay-link prefix the viewer's refresher uses (genN_, genNgate_)
    labels = {}
    for g in gens:
        n = g["gen"]
        if g["status"] == "waiting":
            continue
        net = bool(g["online"]) or (g["online"] is None and g["status"] == "playing"
                                    and (state["gens"].get(str(n), {}).get("gate") or {}).get("passed"))
        who = f"v{n}" if net else "search"
        labels[f"gen{n}"] = {"A": who, "B": who, "kind": "network" if net else "search"}
        labels[f"gen{n}gate"] = {"A": f"v{n}", "B": "search", "kind": "gate"}
    lp = os.path.join(os.path.dirname(out), "labels.json")
    json.dump(labels, open(lp + ".tmp", "w"))
    os.replace(lp + ".tmp", lp)
    tmp = out + ".tmp"
    json.dump(stats, open(tmp, "w"))
    os.replace(tmp, out)
    json.dump(cache, open(cache_p + ".tmp", "w"))
    os.replace(cache_p + ".tmp", cache_p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120)
