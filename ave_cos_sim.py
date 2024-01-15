import csv
import numpy as np

with open('cos_sim_tra1.csv') as f:
    reader = csv.reader(f)
    l = [row for row in reader]

conv_better = []
prop_better = []
cnt = 0
for i, row in enumerate(l):
    if np.isnan(float(row[1])) or np.isnan(float(row[2])):
        continue
    conv = float(row[1])
    prop = float(row[2])
    if conv > prop:
        conv_better.append((i, round(conv - prop,4)))
    else:
        cnt += 1
        prop_better.append((i, round(prop - conv,4)))

conv_better.sort(key=lambda x: -x[1])
prop_better.sort(key=lambda x: -x[1])

for i in range(min(15, len(conv_better))):
    print("{}:{}".format(*conv_better[i]))
print()
for i in range(min(15, len(prop_better))):
    print("{}:{}".format(*prop_better[i]))
print()
print(cnt)