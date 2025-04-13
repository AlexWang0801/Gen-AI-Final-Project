import json


raw = '''
admiration     : 0.35
amusement      : 0.25
anger          : 0.20
annoyance      : 0.50
approval       : 0.30
caring         : 0.15
confusion      : 0.30
curiosity      : 0.30
desire         : 0.35
disappointment : 0.35
disapproval    : 0.55
disgust        : 0.50
embarrassment  : 0.45
excitement     : 0.10
fear           : 0.35
gratitude      : 0.50
grief          : 0.30
joy            : 0.55
love           : 0.15
nervousness    : 0.40
optimism       : 0.35
pride          : 0.20
realization    : 0.10
relief         : 0.15
remorse        : 0.40
sadness        : 0.55
surprise       : 0.15
neutral        : 0.50
'''


thresholds = {}
for line in raw.strip().splitlines():
    if ":" in line:
        key, val = line.split(":")
        thresholds[key.strip()] = float(val.strip())


with open("src/optimal_thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

print("✅ Saved to src/optimal_thresholds.json")
