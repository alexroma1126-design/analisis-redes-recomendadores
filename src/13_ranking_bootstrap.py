from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "outputs/research"
FIG = ROOT / "figures/research"
rng = np.random.default_rng(2026)

df = pd.read_csv(OUT / "ranking_per_user.csv")

metrics = [
    "precision_at_5",
    "recall_at_5",
    "ndcg_at_5"
]

print("=" * 65)
print("MOLNIYA H5 - BOOTSTRAP PAREADO DEL RANKING")
print("=" * 65)

records = []
samples = []

for metric in metrics:
    paired = df.pivot(
        index="user",
        columns="model",
        values=metric
    )[["svd", "hybrid"]].dropna()

    difference = (
        paired["hybrid"] - paired["svd"]
    ).to_numpy()

    observed = float(difference.mean())
    n = len(difference)
    B = 5000

    boot = np.empty(B)

    for b in range(B):
        indices = rng.integers(0, n, size=n)
        boot[b] = difference[indices].mean()

    lo, hi = np.quantile(boot, [0.025, 0.975])

    records.append({
        "metric": metric,
        "users": n,
        "observed_gain": observed,
        "ci_95_low": lo,
        "ci_95_high": hi,
        "fraction_positive_bootstrap": float(
            (boot > 0).mean()
        ),
        "users_improved": int((difference > 0).sum()),
        "users_worsened": int((difference < 0).sum()),
        "users_unchanged": int((difference == 0).sum())
    })

    samples.append(pd.DataFrame({
        "metric": metric,
        "bootstrap_gain": boot
    }))

results = pd.DataFrame(records)
results.to_csv(
    OUT / "ranking_bootstrap_summary.csv",
    index=False
)

pd.concat(samples, ignore_index=True).to_csv(
    OUT / "ranking_bootstrap_samples.csv",
    index=False
)

print(results.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(results))
gains = results.observed_gain.to_numpy()
lower = gains - results.ci_95_low.to_numpy()
upper = results.ci_95_high.to_numpy() - gains

ax.errorbar(
    x, gains,
    yerr=[lower, upper],
    fmt="o", capsize=5
)
ax.axhline(0, linestyle="--", color="black")
ax.set_xticks(x)
ax.set_xticklabels(results.metric)
ax.set_ylabel("Hibrido - SVD")
ax.set_title("IC bootstrap 95% - metricas Top-5")
fig.tight_layout()
fig.savefig(
    FIG / "09_ranking_bootstrap.png",
    dpi=200
)
plt.close(fig)

assert len(results) == 3
assert results.users.eq(839).all()
assert np.isfinite(
    results.select_dtypes(include="number").to_numpy()
).all()

print("\nVALIDACION H5: OK")
