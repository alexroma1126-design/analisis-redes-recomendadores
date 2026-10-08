from pathlib import Path
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

ratings = pd.read_csv(
    ROOT / "data/ml-100k/u.data",
    sep="\t",
    header=None,
    names=["user_id", "movie_id", "rating", "timestamp"]
)
assert len(ratings) == 100000
assert ratings.rating.between(1, 5).all()

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    ratings[["user_id", "movie_id", "rating"]],
    reader
)
trainset, testset = train_test_split(
    data, test_size=0.20, random_state=42
)

models = {
    "SVD": SVD(
        n_factors=100, n_epochs=20,
        lr_all=0.005, reg_all=0.02,
        random_state=42
    ),
    "KNN_item": KNNBasic(
        k=40, min_k=1,
        sim_options={
            "name": "cosine",
            "user_based": False
        },
        verbose=False
    )
}

print("=== MOLNIYA: SISTEMAS DE RECOMENDACION ===", flush=True)
print("Entrenamiento:", trainset.n_ratings, flush=True)
print("Prueba:", len(testset), flush=True)

results = []
predictions_out = []

for name, model in models.items():
    print(f"\nEntrenando {name}...", flush=True)
    start = time.perf_counter()
    model.fit(trainset)
    elapsed = time.perf_counter() - start

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    results.append({
        "model": name,
        "RMSE": rmse,
        "MAE": mae,
        "train_seconds": elapsed
    })

    predictions_out.extend([
        {
            "model": name,
            "user_id": p.uid,
            "movie_id": p.iid,
            "actual_rating": p.r_ui,
            "predicted_rating": p.est
        }
        for p in predictions
    ])

    print(
        f"{name}: RMSE={rmse:.5f} "
        f"MAE={mae:.5f} Tiempo={elapsed:.2f}s",
        flush=True
    )

metrics = pd.DataFrame(results).sort_values("RMSE")
metrics.to_csv(OUT / "recommender_metrics.csv", index=False)
pd.DataFrame(predictions_out).to_csv(
    OUT / "recommender_predictions.csv", index=False
)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics.model, metrics.RMSE)
ax.bar_label(bars, fmt="%.4f", padding=4)
ax.set_ylabel("RMSE (menor es mejor)")
ax.set_title("MovieLens 100K | SVD vs KNN")
ax.set_ylim(0, max(metrics.RMSE) * 1.18)
fig.tight_layout()
fig.savefig(FIG / "03_recommender_comparison.png", dpi=180)
plt.close(fig)

assert len(metrics) == 2
assert len(predictions_out) == 40000
assert metrics.RMSE.notna().all()
assert metrics.MAE.notna().all()

print("\n=== CLASIFICACION FINAL ===")
print(metrics.to_string(index=False))
print("\nGanador por RMSE:", metrics.iloc[0]["model"])
print("VALIDACION OK", flush=True)
