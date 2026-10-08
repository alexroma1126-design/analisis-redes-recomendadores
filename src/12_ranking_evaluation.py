from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUT = ROOT / "outputs/research"
FIG = ROOT / "figures/research"

df = pd.read_csv(OUT / "hybrid_predictions.csv")
assert len(df) == 20000
assert not df.duplicated(["user", "movie"]).any()

K = 5
rows = []

for user, group in df.groupby("user"):
    if len(group) < K:
        continue

    relevant = (group.actual >= 4).astype(int)
    total_relevant = int(relevant.sum())

    if total_relevant == 0:
        continue

    for model in ["svd", "graph", "hybrid"]:
        ranked = group.sort_values(
            [model, "movie"],
            ascending=[False, True]
        ).head(K)

        hits = (ranked.actual >= 4).astype(int).to_numpy()
        precision = hits.sum() / K
        recall = hits.sum() / total_relevant

        discounts = 1 / np.log2(np.arange(2, K + 2))
        dcg = float(np.sum(hits * discounts))

        ideal = np.zeros(K)
        ideal[:min(K, total_relevant)] = 1
        idcg = float(np.sum(ideal * discounts))

        rows.append({
            "user": user,
            "model": model,
            "precision_at_5": precision,
            "recall_at_5": recall,
            "ndcg_at_5": dcg / idcg
        })

result = pd.DataFrame(rows)
summary = result.groupby("model")[
    ["precision_at_5", "recall_at_5", "ndcg_at_5"]
].agg(["mean", "std"])

result.to_csv(OUT / "ranking_per_user.csv", index=False)
summary.to_csv(OUT / "ranking_summary.csv")

means = result.groupby("model")[
    ["precision_at_5", "recall_at_5", "ndcg_at_5"]
].mean()

print("=== MOLNIYA H4: RANKING TOP-5 ===")
print("Usuarios evaluados:", result.user.nunique())
print("Relevancia: rating >= 4")
print("\n=== RESULTADOS ===")
print(means.to_string())

fig, axes = plt.subplots(figsize=(9, 5))
means.plot.bar(ax=axes, rot=0)
axes.set_title("Ranking Top-5 sobre calificaciones observadas")
axes.set_ylabel("Promedio por usuario")
axes.legend(loc="best")
fig.tight_layout()
fig.savefig(FIG / "08_ranking_comparison.png", dpi=200)
plt.close(fig)

assert result.user.nunique() > 0
assert result.groupby("user").model.nunique().eq(3).all()
assert result[[
    "precision_at_5", "recall_at_5", "ndcg_at_5"
]].apply(lambda c: c.between(0,1).all()).all()

print("\nVALIDACION H4: OK")
