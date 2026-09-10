"""
Task 3 — Goals vs Shot Volume (team level)
FIFA World Cup 2026

Analytic question:
Is there a significant difference in shot conversion efficiency
(goals per shot on target) between teams that reached the knockout
stage and teams eliminated in the group stage?

Inputs required (place in the same folder as this script):
  - squad_shooting.csv   : exported directly from fbref's "Squad Shooting"
                           table (Share & Export -> Get table as CSV) at
                           https://fbref.com/en/comps/1/shooting/World-Cup-Stats
  - knockout_status.csv  : Team, Knockout (1 = reached Round of 32, 0 = group
                           stage only) -- already provided, verified against
                           the official FIFA bracket.

Outputs:
  - Descriptive statistics table (printed + saved as CSV)
  - Boxplot comparing the two groups (PNG)
  - 95% confidence interval for mean conversion rate
  - Two-sample t-test result
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

RANDOM_SEED = 42
SAMPLE_SIZE = 24  # simple random sample drawn from the 48-team population

# ---------------------------------------------------------------------------
# 1. DATA WRANGLING
# ---------------------------------------------------------------------------

# fbref sometimes prefixes squad names with confederation codes / flags when
# exported; if your CSV has an extra unnamed first column, this drops it.
shooting = pd.read_csv("squad_shooting.csv")
shooting.columns = [c.strip() for c in shooting.columns]

# Keep just what we need. fbref's shooting table typically has these columns:
# Squad, # Pl, 90s, Gls, Sh, SoT, SoT%, Sh/90, SoT/90, G/Sh, G/SoT, PK, PKatt
shooting = shooting.rename(columns={"Squad": "Team"})
shooting["Team"] = shooting["Team"].str.replace(r"^[a-z]{2,3}\s", "", regex=True).str.strip()

# Name reconciliation: fbref vs common bracket naming
NAME_FIX = {
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Bosnia–Herz": "Bosnia and Herzegovina",  # en dash, matches fbref export
    "Cabo Verde": "Cabo Verde",
    "Türkiye": "Turkiye",
    "Curaçao": "Curacao",
    "USA": "United States",
}
shooting["Team"] = shooting["Team"].replace(NAME_FIX)

knockout = pd.read_csv("knockout_status.csv")

df = shooting.merge(knockout, on="Team", how="inner")
missing = set(knockout["Team"]) - set(df["Team"])
if missing:
    print("WARNING: these teams did not match between the two files:", missing)
    print("Fix the NAME_FIX dictionary above and rerun.\n")

# Conversion rate: goals per shot on target
# (fbref already computes this as 'G/SoT' -- recompute manually to be safe)
df["Conversion"] = df["Gls"] / df["SoT"].replace(0, np.nan)
df = df.dropna(subset=["Conversion"])

print(f"Population size after merge: {len(df)} teams\n")

# 2. DATA PREPARATION AND SAMPLING
# Population = all 48 World Cup 2026 teams.
# We draw a simple random sample (without replacement) to demonstrate
# sampling technique, rather than using the full census.

np.random.seed(RANDOM_SEED)
sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)

knockout_sample = sample[sample["Knockout"] == 1]["Conversion"]
group_only_sample = sample[sample["Knockout"] == 0]["Conversion"]

print(f"Sample size: {len(sample)}  (Knockout={len(knockout_sample)}, "
      f"Group-only={len(group_only_sample)})\n")

# 3. DESCRIPTIVE STATISTICS
desc = sample.groupby("Knockout")["Conversion"].agg(["count", "mean", "median", "std", "min", "max"])
desc.index = desc.index.map({1: "Knockout", 0: "Group-stage-only"})
print("Descriptive statistics (goals per shot on target):")
print(desc, "\n")
desc.to_csv("task3_descriptive_stats.csv")

plt.figure(figsize=(6, 5))
sample.boxplot(column="Conversion", by="Knockout")
plt.xticks([1, 2], ["Group-stage-only", "Knockout"])
plt.title("Shot Conversion Rate: Knockout vs Group-stage-only")
plt.suptitle("")
plt.ylabel("Goals per Shot on Target")
plt.xlabel("")
plt.tight_layout()
plt.savefig("task3_boxplot.png", dpi=150)
print("Saved chart: task3_boxplot.png\n")

# 4. CONFIDENCE INTERVAL (95%) -- mean conversion rate, Knockout teams
ci_data = knockout_sample
mean = ci_data.mean()
sem = stats.sem(ci_data)
ci = stats.t.interval(0.95, df=len(ci_data) - 1, loc=mean, scale=sem)
print(f"95% CI for mean conversion rate (Knockout teams): "
      f"mean={mean:.3f}, CI=({ci[0]:.3f}, {ci[1]:.3f})\n")

# 5. TWO-SAMPLE T-TEST
# H0: mean conversion rate is equal for Knockout and Group-stage-only teams
# H1: mean conversion rate differs between the two groups
t_stat, p_value = stats.ttest_ind(knockout_sample, group_only_sample, equal_var=False)  # Welch's t-test
print("Two-sample t-test (Welch's, unequal variances assumed):")
print(f"  t-statistic = {t_stat:.3f}")
print(f"  p-value     = {p_value:.4f}")
alpha = 0.05
if p_value < alpha:
    print(f"  Result: p < {alpha} -> reject H0. There IS a significant "
          f"difference in conversion efficiency between the two groups.")
else:
    print(f"  Result: p >= {alpha} -> fail to reject H0. No significant "
          f"difference detected in conversion efficiency between the groups.")
