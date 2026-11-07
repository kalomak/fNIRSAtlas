import subprocess
from collections import Counter
cmd = ["squeue", "--me", "-h", "-o", "%i|%T|%P|%j"]
out = subprocess.check_output(cmd, universal_newlines=True)
rows = [line.strip().split("|", 3) for line in out.splitlines() if line.strip()]
state_counter = Counter()
partition_counter = Counter()
combo_counter = Counter()
setting_counter = Counter()
model_counter = Counter()
for jobid, state, partition, name in rows:
    state_counter[state] += 1
    partition_counter[partition] += 1
    parts = name.split("_")
    setting = parts[0] if parts else "unknown"
    model = parts[1] if len(parts) > 1 else "unknown"
    setting_counter[setting] += 1
    model_counter[model] += 1
    combo_counter[(state, setting, model)] += 1
print("TOTAL", len(rows))
for (state, setting, model), v in sorted(combo_counter.items()):
    print(f"{state}|{setting}|{model}|{v}")
