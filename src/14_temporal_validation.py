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
)

df = df.sort_values(
    ["user", "timestamp", "movie"]
).reset_index(drop=True)

position = df.groupby("user").cumcount()
size = df.groupby("user")["movie"].transform("size")
remaining = size - position

train = df[remaining > 4].copy()
validation = df[(remaining > 2) & (remaining <= 4)].copy()
test = df[remaining <= 2].copy()

assert len(train) + len(validation) + len(test) == 100000
assert set(train.user) == set(test.user)

params = json.loads(
    (ROOT / "outputs/best_hyperparameters.json").read_text(
        encoding="utf-8"
    )
)["SVD"]

reader = Reader(rating_scale=(1, 5))

def fit_svd(frame):
    dataset = Dataset.load_from_df(
        frame[["user", "movie", "rating"]],
        reader
    )
    return SVD(**params).fit(
        dataset.build_full_trainset()
    )

def predict_models(frame, svd, graph):
    y = frame.rating.to_numpy(dtype=float)

    s = np.array([
        svd.predict(int(u), int(m)).est
        for u, m in zip(frame.user, frame.movie)
    ])

    g = np.array([
        graph.predict(int(u), int(m))
        for u, m in zip(frame.user, frame.movie)
    ])

    return y, s, g

print("=== MOLNIYA H6: VALIDACION TEMPORAL ===")
print("Train:", len(train))
print("Validacion:", len(validation))
print("Test:", len(test))

svd = fit_svd(train)
graph = GraphRecommender(k=30).fit(train)

y, s, g = predict_models(validation, svd, graph)

alphas = [0, .05, .10, .15, .20, .30, .50]
scores = []

for a in alphas:
    rmse, mae = metrics(y, (1-a)*s+a*g)
    scores.append({
        "alpha": a,
        "rmse": rmse,
        "mae": mae
    })

selection = pd.DataFrame(scores)
selection.to_csv(
    OUT / "temporal_validation.csv",
    index=False
)

alpha = float(
    selection.sort_values(
        ["rmse", "alpha"]
    ).iloc[0].alpha
)

print("\nAlpha elegido:", alpha)
print(selection.to_string(index=False))

development = pd.concat(
    [train, validation],
    ignore_index=True
)

svd_final = fit_svd(development)
graph_final = GraphRecommender(k=30).fit(development)

y, s, g = predict_models(
    test, svd_final, graph_final
)

h = (1-alpha)*s + alpha*g

rows = []

for name, pred in [
    ("SVD", s),
    ("Grafo", g),
    ("Hibrido", h)
]:
    rmse, mae = metrics(y, pred)
    rows.append({
        "model": name,
        "rmse": rmse,
        "mae": mae
    })

result = pd.DataFrame(rows)
result.to_csv(
    OUT / "temporal_test_metrics.csv",
    index=False
)

pd.DataFrame({
    "user": test.user.to_numpy(),
    "movie": test.movie.to_numpy(),
    "actual": y,
    "svd": s,
    "graph": g,
    "hybrid": h
}).to_csv(
    OUT / "temporal_predictions.csv",
    index=False
)

print("\n=== RESULTADOS TEMPORALES ===")
print(result.to_string(index=False))
print(
    "\nMejora RMSE:",
    metrics(y,s)[0] - metrics(y,h)[0]
)

assert len(y) == len(test)
assert np.isfinite(
    result[["rmse","mae"]].to_numpy()
).all()

print("\nVALIDACION H6: OK")
print("Nota: separacion cronologica por usuario,")
print("no un corte temporal global.")
