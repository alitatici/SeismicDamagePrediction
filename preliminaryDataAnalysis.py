from Preprocess4ML import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dirname = os.path.dirname(__file__)
data_file = os.path.join(dirname, 'all_data_4_ML.h5')
data_input = pd.read_hdf(data_file, 'df')
df = preprocessDataAnalysis(data_input)

# # IM VS MIDR PLOTS
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
# axes[0, 0].scatter(df["PGA"], df["MIDR"], s=15, color='darkred', linewidths=0.15, alpha=0.25, edgecolors='black')
# axes[0, 0].set_xlabel("PGA", fontsize=14)
# axes[0, 0].set_ylabel("MIDR", fontsize=14)
# axes[0, 0].grid(True, linestyle='-', alpha=0.25)
# axes[0, 1].scatter(df["PGV"], df["MIDR"], s=15, color='darkred', linewidths=0.15, alpha=0.25, edgecolors='black')
# axes[0, 1].set_xlabel("PGV", fontsize=14)
# axes[0, 1].grid(True, linestyle='-', alpha=0.25)
# axes[1, 0].scatter(df["PGD"], df["MIDR"], s=15, color='darkred', linewidths=0.15, alpha=0.25, edgecolors='black')
# axes[1, 0].set_xlabel("PGD", fontsize=14)
# axes[1, 0].set_ylabel("MIDR", fontsize=14)
# axes[1, 0].grid(True, linestyle='-', alpha=0.25)
# axes[1, 1].scatter(df["Sa(T1)"], df["MIDR"], s=15, color='darkred', linewidths=0.15, alpha=0.25, edgecolors='black')
# axes[1, 1].set_xlabel("Sa(T1)", fontsize=14)
# axes[1, 1].grid(True, linestyle='-', alpha=0.25)
# axes[2, 0].scatter(df["Sd(T1)"], df["MIDR"], s=15, color='darkred', linewidths=0.15, alpha=0.25, edgecolors='black')
# axes[2, 0].set_xlabel("Sd(T1)", fontsize=14)
# axes[2, 0].set_ylabel("MIDR", fontsize=14)
# axes[2, 0].grid(True, linestyle='-', alpha=0.25)
# axes[2, 1].scatter(df["CAV"], df["MIDR"], s=15, color='darkred', linewidths=0.15, alpha=0.25, edgecolors='black')
# axes[2, 1].set_xlabel("CAV", fontsize=14)
# axes[2, 1].grid(True, linestyle='-', alpha=0.25)
#
# plt.show()

# # Create a joint plot with scatter and histograms
# sns.jointplot(x="PGA", y="MIDR", data=df, kind="scatter", s=25, color='darkred', alpha=0.25, marginal_kws=dict(bins=30, fill=False, color='darkred'))
# plt.xlabel('PGA (g)', fontsize=16)
# plt.ylabel('MIDR', fontsize=16)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.show()

# sns.jointplot(x="PGV", y="MIDR", data=df, kind="scatter", s=25, color='darkred', alpha=0.25, marginal_kws=dict(bins=30, fill=False, color='darkred'))
# plt.xlabel('PGV (cm/s)', fontsize=16)
# plt.ylabel('MIDR', fontsize=16)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.show()

# sns.jointplot(x="PGD", y="MIDR", data=df, kind="scatter", s=25, color='darkred', alpha=0.25, marginal_kws=dict(bins=30, fill=False, color='darkred'))
# plt.xlabel('PGD (cm)', fontsize=16)
# plt.ylabel('MIDR', fontsize=16)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.show()

# sns.jointplot(x="Sa(T1)", y="MIDR", data=df, kind="scatter", s=25, color='darkred', alpha=0.25, marginal_kws=dict(bins=30, fill=False, color='darkred'))
# plt.xlabel('Sa(T1) (g)', fontsize=16)
# plt.ylabel('MIDR', fontsize=16)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.show()

# sns.jointplot(x="Sd(T1)", y="MIDR", data=df, kind="scatter", s=25, color='darkred', alpha=0.25, marginal_kws=dict(bins=30, fill=False, color='darkred'))
# plt.xlabel('Sd(T1) (cm)', fontsize=16)
# plt.ylabel('MIDR', fontsize=16)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.show()

# sns.jointplot(x="CAV", y="MIDR", data=df, kind="scatter", s=25, color='darkred', alpha=0.25, marginal_kws=dict(bins=30, fill=False, color='darkred'))
# plt.xlabel('CAV (m/s)', fontsize=16)
# plt.ylabel('MIDR', fontsize=16)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.show()


# # Soil Condition vs MIDR plots
# unique_values = np.sort(np.unique(df["Soil Condition"]))
# condition1 = df.loc[df["Soil Condition"] == unique_values[2], 'MIDR'].values
# condition2 = df.loc[df["Soil Condition"] == unique_values[1], 'MIDR'].values
# condition3 = df.loc[df["Soil Condition"] == unique_values[0], 'MIDR'].values
#
# plt.hist(condition1, bins=20, alpha=0.5, color='blue', label='Good', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Good Soil Condition')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.show()
#
# plt.hist(condition2, bins=20, alpha=0.5, color='darkgreen', label='Moderate', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Moderate Soil Condition')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.show()
#
# plt.hist(condition3, bins=20, alpha=0.5, color='red', label='Poor', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Poor Soil Condition')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.show()


# # First Storey - Commercial Use vs MIDR plots
# unique_values = np.sort(np.unique(df["First Storey - Commercial Use"]))
# condition2 = df.loc[df["First Storey - Commercial Use"] == unique_values[1], 'MIDR'].values
# condition3 = df.loc[df["First Storey - Commercial Use"] == unique_values[0], 'MIDR'].values

# plt.hist(condition2, bins=20, alpha=0.5, color='darkgreen', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Commercial Use')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.ylim([0,7000])
# plt.tight_layout()
# plt.show()
# print("abc")

# plt.hist(condition3, bins=20, alpha=0.5, color='red', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Normal Floor Height')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.ylim([0,7000])
# plt.tight_layout()
# plt.show()
# print("abc")


# # Number of Story vs MIDR plots
# unique_values = np.sort(np.unique(df["Number of Storey"]))
# condition2 = df.loc[df["Number of Storey"] == unique_values[1], 'MIDR'].values
# condition3 = df.loc[df["Number of Storey"] == unique_values[0], 'MIDR'].values
#
# plt.hist(condition3, bins=20, alpha=0.5, color='darkgreen', label='3 Story Buildings', edgecolor='black')
# plt.hist(condition2, bins=20, alpha=0.5, color='navy', label='5 Story Buildings', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.title('Distribution of MIDR Depending on Number of Stories')
# plt.legend()
# plt.show()


# Span Length vs MIDR plots
# unique_values = np.sort(np.unique(df["Span Length"]))
# condition1 = df.loc[df["Span Length"] == unique_values[0], 'MIDR'].values
# condition2 = df.loc[df["Span Length"] == unique_values[1], 'MIDR'].values
# condition3 = df.loc[df["Span Length"] == unique_values[2], 'MIDR'].values
#
# plt.hist(condition1, bins=20, alpha=0.5, color='blue', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Span Length = 2.5 m')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.ylim([0,6000])
# plt.tight_layout()
# plt.show()
# print("abc")

# plt.hist(condition2, bins=20, alpha=0.5, color='darkgreen', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Span Length = 3.0 m')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.ylim([0,6000])
# plt.tight_layout()
# plt.show()
# print("abc")

# plt.hist(condition3, bins=20, alpha=0.5, color='red', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Span Length = 4.0 m')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.ylim([0,6000])
# plt.tight_layout()
# plt.show()
# print("abc")


# # Story Height vs MIDR plots
# unique_values = np.sort(np.unique(df["Storey Height"]))
# condition1 = df.loc[df["Storey Height"] == unique_values[0], 'MIDR'].values
# condition2 = df.loc[df["Storey Height"] == unique_values[1], 'MIDR'].values
# condition3 = df.loc[df["Storey Height"] == unique_values[2], 'MIDR'].values
#
# plt.hist(condition1, bins=20, alpha=0.5, color='blue', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Story Height = 2.6 m')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.ylim([0,4500])
# plt.tight_layout()
# plt.show()
# print("abc")

# plt.hist(condition2, bins=20, alpha=0.5, color='darkgreen', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Story Height = 2.8 m')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.ylim([0,4500])
# plt.tight_layout()
# plt.show()
# print("abc")
#
# plt.hist(condition3, bins=20, alpha=0.5, color='red', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.title('Story Height = 3.0 m')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.ylim([0,4500])
# plt.tight_layout()
# plt.show()
# print("abc")


# # Number of Span vs MIDR plots
# unique_values = np.sort(np.unique(df["Number of Span"]))
# condition2 = df.loc[df["Number of Span"] == unique_values[0], 'MIDR'].values
# condition3 = df.loc[df["Number of Span"] == unique_values[1], 'MIDR'].values
#
# plt.hist(condition3, bins=20, alpha=0.5, color='darkgreen', label='3 Span Buildings', edgecolor='black')
# plt.hist(condition2, bins=20, alpha=0.5, color='navy', label='4 Span Buildings', edgecolor='black')
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Number of instances', fontsize=12)
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.title('Distribution of MIDR Depending on Number of Spans')
# plt.legend()
# plt.show()

#
# # Concrete Strength vs MIDR plots
# # 5, 12, 20, 28, and 35 MPa
# unique_values = np.sort(np.unique(df["Concrete Strength"]))
# condition1 = df.loc[df["Concrete Strength"] == unique_values[0], 'MIDR'].values
# # hist, bins = np.histogram(condition1, bins=25, density=True)
# # bin_centers = (bins[:-1] + bins[1:]) / 2
# # plt.plot(bin_centers, hist, color='rebeccapurple', linestyle='-', linewidth=1, label='5 MPa')
# sns.kdeplot(condition1, color='rebeccapurple', linestyle='-', linewidth=1, label='5 MPa')
# condition2 = df.loc[df["Concrete Strength"] == unique_values[1], 'MIDR'].values
# # hist, bins = np.histogram(condition2, bins=25, density=True)
# # bin_centers = (bins[:-1] + bins[1:]) / 2
# # plt.plot(bin_centers, hist, color='navy', linestyle='-', linewidth=1, label='12 MPa')
# sns.kdeplot(condition2, color='mediumblue', linestyle='-', linewidth=1, label='12 MPa')
# condition3 = df.loc[df["Concrete Strength"] == unique_values[2], 'MIDR'].values
# # hist, bins = np.histogram(condition3, bins=25, density=True)
# # bin_centers = (bins[:-1] + bins[1:]) / 2
# # plt.plot(bin_centers, hist, color='darkgreen', linestyle='-', linewidth=1, label='20 MPa')
# sns.kdeplot(condition3, color='darkorange', linestyle='-', linewidth=1, label='20 MPa')
# condition4 = df.loc[df["Concrete Strength"] == unique_values[3], 'MIDR'].values
# # hist, bins = np.histogram(condition4, bins=25, density=True)
# # bin_centers = (bins[:-1] + bins[1:]) / 2
# # plt.plot(bin_centers, hist, color='black', linestyle='-', linewidth=1, label='28 MPa')
# sns.kdeplot(condition4, color='green', linestyle='-', linewidth=1, label='28 MPa')
# condition5 = df.loc[df["Concrete Strength"] == unique_values[4], 'MIDR'].values
# # hist, bins = np.histogram(condition5, bins=25, density=True)
# # bin_centers = (bins[:-1] + bins[1:]) / 2
# # plt.plot(bin_centers, hist, color='red', linestyle='-', linewidth=1, label='35 MPa')
# sns.kdeplot(condition5, color='red', linestyle='-', linewidth=1, label='35 MPa')
#
# plt.axvline(x=0.001, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(x=0.002, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(x=0.005, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(x=0.01, color='k', linestyle='--', linewidth=0.8)
#
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Density', fontsize=12)
# plt.title('PDF of MIDR Depending on Concrete Strength')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.xlim([0, 0.02])
# plt.legend()
# plt.show()


# # Number of Story = 3
# df_nos_3 = df[df['Number of Storey'] == 3]
#
# # Story Height vs MIDR plots
# unique_values = np.sort(np.unique(df_nos_3["Column Area"]))
# condition1 = df_nos_3.loc[df_nos_3["Column Area"] == unique_values[0], 'MIDR'].values
# sns.kdeplot(condition1, color='red', linestyle='-', linewidth=1, label='0.0625 m²')
# condition2 = df_nos_3.loc[df_nos_3["Column Area"] == unique_values[1], 'MIDR'].values
# sns.kdeplot(condition2, color='mediumblue', linestyle='-', linewidth=1, label='0.9 m²')
# condition3 = df_nos_3.loc[df_nos_3["Column Area"] == unique_values[2], 'MIDR'].values
# sns.kdeplot(condition3, color='darkgreen', linestyle='-', linewidth=1, label='0.1225 m²')
#
# plt.axvline(x=0.001, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(x=0.002, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(x=0.005, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(x=0.01, color='k', linestyle='--', linewidth=0.8)
#
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Density', fontsize=12)
# plt.title('PDF of MIDR Depending on Column Area for 3-Story Buildings')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.xlim([0, 0.02])
# plt.legend()
# plt.show()
#
# # Number of Story = 5
# df_nos_5 = df[df['Number of Storey'] == 5]
#
# # Story Height vs MIDR plots
# unique_values = np.sort(np.unique(df_nos_5["Column Area"]))
# condition1 = df_nos_5.loc[df_nos_5["Column Area"] == unique_values[0], 'MIDR'].values
# sns.kdeplot(condition1, color='red', linestyle='-', linewidth=1, label='0.9 m²')
# condition2 = df_nos_5.loc[df_nos_5["Column Area"] == unique_values[1], 'MIDR'].values
# sns.kdeplot(condition2, color='mediumblue', linestyle='-', linewidth=1, label='0.1225 m²')
# condition3 = df_nos_5.loc[df_nos_5["Column Area"] == unique_values[2], 'MIDR'].values
# sns.kdeplot(condition3, color='darkgreen', linestyle='-', linewidth=1, label='0.16 m²')
#
# plt.axvline(x=0.001, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(x=0.002, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(x=0.005, color='k', linestyle='--', linewidth=0.8)
# plt.axvline(x=0.01, color='k', linestyle='--', linewidth=0.8)
#
# plt.xlabel('MIDR', fontsize=12)
# plt.ylabel('Density', fontsize=12)
# plt.title('PDF of MIDR Depending on Column Area for 5-Story Buildings')
# plt.grid(True, linestyle='-', alpha=0.25)
# plt.xlim([0, 0.02])
# plt.legend()
# plt.show()


print("abc")

