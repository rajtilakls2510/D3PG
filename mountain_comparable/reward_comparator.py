import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

single = pd.concat([pd.read_csv("Total Reward1.csv"), pd.read_csv("Total Reward2.csv"), pd.read_csv("Total Reward3.csv"), pd.read_csv("Total Reward4.csv"), pd.read_csv("Total Reward5.csv")])
single["step"] = pd.cut(single["step"], bins=np.linspace(0, 5_00_000, 500), labels=np.linspace(0, 5_00_000, 499)+1000).astype(int)
single = single.groupby("step", as_index=False)["total_reward"].agg(["mean", "std"]).reset_index().fillna(0)

dist = pd.concat([pd.read_csv("301.csv"), pd.read_csv("917.csv"), pd.read_csv("2503.csv"), pd.read_csv("2911.csv"), pd.read_csv("3163.csv")])
dist["train_step"] = pd.cut(dist["train_step"], bins=np.linspace(0, 6_00_000, 600), labels=np.linspace(0, 6_00_000, 599)+1000).astype(int)
dist = dist.groupby("train_step", as_index=False)["TotalReward"].agg(["mean", "std"]).reset_index().fillna(0)

smoothingWeight = 0.8

smoothed_single = []
smoothed_dist = []


last = single["mean"][0]
smoothed_single.append(last)
for dat in list(single["mean"])[1:]:
    last = last * smoothingWeight + (1 - smoothingWeight) * dat
    smoothed_single.append(last)

single["smoothed"] = smoothed_single

# dist = dist.groupby(["train_step"], as_index=False).mean()[["train_step", "TotalReward"]]
last = dist["mean"][0]
smoothed_dist.append(last)
for dat in list(dist["mean"])[1:]:
    last = last * smoothingWeight + (1 - smoothingWeight) * dat
    smoothed_dist.append(last)

dist["smoothed"] = smoothed_dist

info = {"train_step": [], "TotalReward": [], "type":[]}

info["train_step"].extend(single["step"])
info["TotalReward"].extend(single["smoothed"])
info["type"].extend(["single" for _ in range(len(single["step"]))])
info["train_step"].extend(dist["train_step"])
info["TotalReward"].extend(dist["smoothed"])
info["type"].extend(["distributed" for _ in range(len(dist["train_step"]))])

fig, ax = plt.subplots()
fig.suptitle("Total Reward vs Training Steps in Mountain Car Continuous")
sns.lineplot(x="train_step", y="TotalReward", hue="type", data=info, ax=ax)
ax.fill_between(single["step"], single["mean"]-single["std"]/2, single["mean"]+single["std"]/2, color="blue", alpha=0.1)
ax.fill_between(dist["train_step"], dist["mean"]-dist["std"]/2, dist["mean"]+dist["std"]/2, color="orange", alpha=0.1)
plt.savefig("mountain_reward.png", format="png")
plt.show()


