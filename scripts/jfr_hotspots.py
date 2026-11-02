"""Where a self-play JVM spends its CPU, from a Java Flight Recorder file.

    jfr_hotspots.py <recording.jfr> [jfr-binary]

Two views of the sampled stacks: self time (the method on top of the stack) and
inclusive time (any method anywhere on it), plus a breakdown into the operations
an MCTS simulation repeats -- copying the game, applying effects, finding
playable abilities, encoding the state -- so the copy path can be costed.
"""
import collections, re, subprocess, sys

FRAME = re.compile(r"^\s+([\w$.<>]+)\(.*\)\s+line:")

# substrings of a frame that mark an operation, checked outermost-first per stack
OPERATIONS = [
    ("game copy (GameImpl/GameState copy)", ("GameImpl.<init>(", "GameState.<init>(", "GameImpl.copy(", "GameState.copy(")),
    ("apply effects", ("ContinuousEffects.apply", "GameState.applyEffects", "GameImpl.applyEffects")),
    ("playable abilities", ("getPlayable", "PlayerImpl.getPlayableFromObject", "getManaAvailable")),
    ("state encoder", ("StateEncoder.",)),
    ("model client (HTTP to GPU server)", ("RemoteModelEvaluator",)),
    ("trigger / state-based checks", ("checkStateAndTriggered", "checkStateBasedActions", "checkTriggered")),
    ("card / ability copy", ("CardImpl.<init>(", "AbilityImpl.<init>(", "Abilities", ".copy(")),
]


def stacks(path, jfr):
    # streamed: the text form of a big recording runs to gigabytes
    proc = subprocess.Popen([jfr, "print", "--events", "jdk.ExecutionSample", "--stack-depth", "96", path],
                            stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    cur = None
    for line in proc.stdout:
        if line.startswith("jdk.ExecutionSample"):
            if cur:
                yield cur
            cur = []
        elif cur is not None:
            m = FRAME.match(line)
            if m:
                cur.append(m.group(1))
    if cur:
        yield cur


def main(path, jfr):
    self_t, incl, ops = collections.Counter(), collections.Counter(), collections.Counter()
    n = 0
    for st in stacks(path, jfr):
        if not st:
            continue
        n += 1
        self_t[st[0]] += 1
        for f in set(st):
            incl[f] += 1
        joined = "\n".join(st)
        for name, keys in OPERATIONS:
            if any(k in joined for k in keys):
                ops[name] += 1
    print(f"{n} samples\n")
    print("== operations (share of samples whose stack contains it; they overlap) ==")
    for name, c in ops.most_common():
        print(f"  {100 * c / n:5.1f}%  {name}")
    print("\n== top self time ==")
    for f, c in self_t.most_common(25):
        print(f"  {100 * c / n:5.1f}%  {f}")
    print("\n== top inclusive (mage.* only) ==")
    for f, c in [(f, c) for f, c in incl.most_common(400) if f.startswith(("mage.", "org.mage."))][:35]:
        print(f"  {100 * c / n:5.1f}%  {f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/home/ahmet/pers/lmt-magic-sim/xm/jdk/bin/jfr")
