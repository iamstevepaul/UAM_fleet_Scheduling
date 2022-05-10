"""
Author: Steve Paul 
Date: 4/29/22 """
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import pandas as pd

dir = "Results/"
mlp_file_name = "r_mlp_results.pkl"
gcaps_file_name = "r_gcaps_results.pkl"
feas_rnd_file_name = "Feas_Rnd_results.pkl"

xtick_fontsize = 40
ytick_fontsize = 40
ylabel_fontsize = 40

with open(dir+feas_rnd_file_name, "rb") as f:
    feas_rnd_data = pickle.load(f)
    f.close()
with open(dir+mlp_file_name, "rb") as f:
    mlp_data = pickle.load(f)
    f.close()

with open(dir+gcaps_file_name, "rb") as f:
    gcaps_data = pickle.load(f)
    f.close()

df = sb.load_dataset('iris')
rewards_ls = []
profit_ls = []
method_ls = []
for i in range(100):
    rewards_ls.append(feas_rnd_data["total_reward"][i])
    profit_ls.append(feas_rnd_data["profit"][i])
    method_ls.append("Feas-RND")

for i in range(100):
    rewards_ls.append(mlp_data["total_reward"][i])
    profit_ls.append(mlp_data["profit"][i])
    method_ls.append("MLP-RL")

for i in range(100):
    rewards_ls.append(gcaps_data["total_reward"][i])
    profit_ls.append(gcaps_data["profit"][i])
    method_ls.append("GCAPS-RL")

data = {
    "Method": method_ls,
    "Profit": profit_ls,
    "Total Reward": rewards_ls
}
my_pal = {"Feas-RND": "c", "MLP-RL": "r", "GCAPS-RL":"b"}
df = pd.DataFrame(data)

# sb.boxplot(x=df["Method"], y=df["Total Reward"], palette=my_pal)
# plt.xticks(fontsize = xtick_fontsize)
# plt.xlabel("")
# plt.yticks(fontsize = ytick_fontsize)
# plt.ylabel(ylabel="Total reward per episode",fontsize = ylabel_fontsize)
# plt.show()
sb.boxplot(x=df["Method"], y=df["Profit"], palette=my_pal)
plt.xticks(fontsize = xtick_fontsize)
plt.xlabel("")
label=["10K", "15K", "20K", "25K", "30K"]
ticks = [10000, 15000, 20000, 25000, 30000]
plt.yticks(ticks,label,fontsize = ytick_fontsize)
plt.ylabel(ylabel="Total profit per day ($)",fontsize = ylabel_fontsize)
plt.show()