import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

single = pd.read_csv("comparable/Episode Length.csv")
dist = pd.read_csv("comparable/494.csv")

smoothingWeight = 0.8

smoothed_single = []
smoothed_dist = []


last = single["length"][0]
smoothed_single.append(last)
for dat in list(single["length"])[1:]:
    last = last * smoothingWeight + (1 - smoothingWeight) * dat
    smoothed_single.append(last)

single["smoothed"] = smoothed_single

dist = dist.groupby(["train_step"], as_index=False).mean()[["train_step", "EpisodeLength"]]
last = dist["EpisodeLength"][0]
smoothed_dist.append(last)
for dat in list(dist["EpisodeLength"])[1:]:
    last = last * smoothingWeight + (1 - smoothingWeight) * dat
    smoothed_dist.append(last)

dist["smoothed"] = smoothed_dist

info = {"train_step": [], "Episode Length": [], "type":[]}

info["train_step"].extend(single["step"])
info["Episode Length"].extend(single["smoothed"])
info["type"].extend(["single" for _ in range(len(single["step"]))])
info["train_step"].extend(dist["train_step"])
info["Episode Length"].extend(dist["smoothed"])
info["type"].extend(["distributed" for _ in range(len(dist["train_step"]))])

sns.lineplot(x="train_step", y="Episode Length", hue="type", data=info)
plt.show()


