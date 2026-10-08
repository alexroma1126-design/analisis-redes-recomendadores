from pathlib import Path
import json
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from hybrid_core import GraphRecommender, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "data/ml-100k/u.data",
    sep="\t", header=None,
    names=["user", "movie", "rating", "timestamp"]
).sort_values(
    ["timestamp", "user", "movie"]
).reset_index(drop=True)

train = df.iloc[:70000].copy()
validation = df.iloc[70000:85000].copy()
test = df.iloc[85000:].copy()

assert train.timestamp.max() <= validation.timestamp.min()
assert validation.timestamp.max() <= test.timestamp.min()

params = json.loads(
    (ROOT / "outputs/best_hyperparameters.json").read_text(
        encoding="utf-8"
    )
)["SVD"]

reader = Reader(rating_scale=(1, 5))

def train_models(frame):
    data = Dataset.load_from_df(
        frame[["user", "movie", "rating"]], reader
    )
    svd = SVD(**params).fit(data.build_full_trainset())
    graph = GraphRecommender(k=30).fit(frame)
    return svd, graph

def evaluate(frame, svd, graph):
    actual = frame.rating.to_numpy(dtype=float)
    ps = np.array([
        svd.predict(int(u), int(m)).est
        for u, m in zip(frame.user, frame.movie)
    ])
    pg = np.array([
        graph.predict(int(u), int(m))
        for u, m in zip(frame.user, frame.movie)
    ])
    return actual, ps, pg

print("=== MOLNIYA H7: CRONOLOGIA GLOBAL ===")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))
print("Ultima fecha train:",
      pd.to_datetime(train.timestamp.max(), unit="s", utc=True))
print("Primera fecha test:",
      pd.to_datetime(test.timestamp.min(), unit="s", utc=True))

svd, graph = train_models(train)
y, s, g = evaluate(validation, svd, graph)

alphas = [0, .025, .05, .075, .10, .15, .20, .30]
rows = []
for a in alphas:
    rmse, mae = metrics(y, (1-a)*s+a*g)
    rows.append({"alpha": a, "rmse": rmse, "mae": mae})

val = pd.DataFrame(rows)
val.to_csv(OUT / "global_temporal_validation.csv", index=False)
alpha = float(val.sort_values(["rmse", "alpha"]).iloc[0].alpha)

print("\nVALIDACION:")
print(val.to_string(index=False))
print("Alpha seleccionado:", alpha)

development = df.iloc[:85000].copy()
svd, graph = train_models(development)
y, s, g = evaluate(test, svd, graph)
h = (1-alpha)*s + alpha*g

results = []
for name, pred in [
    ("SVD", s), ("Grafo", g), ("Hibrido", h)
]:
    rmse, mae = metrics(y, pred)
    results.append({"model": name, "rmse": rmse, "mae": mae})

report = pd.DataFrame(results)
report.to_csv(OUT / "global_temporal_metrics.csv", index=False)

known_users = set(development.user)
known_movies = set(development.movie)
cold = (
    ~test.user.isin(known_users)
    | ~test.movie.isin(known_movies)
)

predictions = pd.DataFrame({
    "user": test.user.to_numpy(),
    "movie": test.movie.to_numpy(),
    "actual": y,
    "svd": s,
    "graph": g,
    "hybrid": h,
    "cold_start": cold.to_numpy()
})
predictions.to_csv(
    OUT / "global_temporal_predictions.csv", index=False
)

print("\n=== TEST GLOBAL ===")
print(report.to_string(index=False))
print("Casos cold-start:", int(cold.sum()))
print("Mejora RMSE:", metrics(y,s)[0]-metrics(y,h)[0])

assert len(predictions) == 15000
assert np.isfinite(
    report[["rmse", "mae"]].to_numpy()
).all()

print("VALIDACION H7: OK")
print("Nota: hiperparametros SVD heredados;")
print("esto sigue siendo una evaluacion exploratoria.")
