from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

ROOT = Path.cwd()
OUT = ROOT / "outputs/research"
source = ROOT / "src/09_hybrid_recommender.py"

from hybrid_core import GraphRecommender, raw_rows, evaluate, metrics

df = pd.read_csv(
    ROOT / "data/ml-100k/u.data",
    sep="\t", header=None,
    names=["user", "movie", "rating", "timestamp"]
)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    df[["user", "movie", "rating"]], reader
)

params = json.loads(
    (ROOT / "outputs/best_hyperparameters.json").read_text(
        encoding="utf-8"
    )
)["SVD"]

alphas = [0, .05, .10, .15, .20, .30, .50]
records = []

print("=== MOLNIYA H3: ROBUSTEZ MULTISPLIT ===", flush=True)

for seed in [11, 23, 42, 67, 101]:
    outer_train, outer_test = train_test_split(
        data, test_size=.2, random_state=seed
    )

    outer_df = raw_rows(outer_train)
    inner_data = Dataset.load_from_df(
        outer_df[["user", "movie", "rating"]], reader
    )
    inner_train, validation = train_test_split(
        inner_data, test_size=.2, random_state=seed + 1000
    )

    s_inner = SVD(**params).fit(inner_train)
    g_inner = GraphRecommender(k=30).fit(
        raw_rows(inner_train)
    )
    y, s, g = evaluate(s_inner, g_inner, validation)

    scores = [
        (metrics(y, (1-a)*s+a*g)[0], a)
        for a in alphas
    ]
    _, alpha = min(scores)

    s_final = SVD(**params).fit(outer_train)
    g_final = GraphRecommender(k=30).fit(outer_df)

    y, s, g = evaluate(s_final, g_final, outer_test)
    hybrid = (1-alpha)*s+alpha*g

    svd_rmse, svd_mae = metrics(y, s)
    hybrid_rmse, hybrid_mae = metrics(y, hybrid)

    records.append({
        "seed": seed,
        "alpha": alpha,
        "svd_rmse": svd_rmse,
        "hybrid_rmse": hybrid_rmse,
        "rmse_gain": svd_rmse-hybrid_rmse,
        "svd_mae": svd_mae,
        "hybrid_mae": hybrid_mae,
        "mae_gain": svd_mae-hybrid_mae
    })

    print(
        f"Seed {seed}: alpha={alpha:.2f} | "
        f"SVD={svd_rmse:.6f} | "
        f"Hibrido={hybrid_rmse:.6f} | "
        f"Ganancia={svd_rmse-hybrid_rmse:+.6f}",
        flush=True
    )

results = pd.DataFrame(records)
results.to_csv(
    OUT / "hybrid_multisplit_5seeds.csv", index=False
)

print("\n=== RESUMEN ===")
print(results.to_string(index=False))
print("\nMejora RMSE media:",
      round(results.rmse_gain.mean(), 6))
print("Desviacion entre splits:",
      round(results.rmse_gain.std(ddof=1), 6))
print("Splits favorables:",
      int((results.rmse_gain > 0).sum()), "/5")

assert len(results) == 5
assert np.isfinite(
    results.select_dtypes(include="number").to_numpy()
).all()
print("\nVALIDACION H3: OK")
