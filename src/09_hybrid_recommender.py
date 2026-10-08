from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research"
FIG = ROOT / "figures/research"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ratings = pd.read_csv(
    ROOT / "data/ml-100k/u.data", sep="\t", header=None,
    names=["user", "movie", "rating", "timestamp"]
)
with open(ROOT / "outputs/best_hyperparameters.json", encoding="utf-8") as f:
    params = json.load(f)["SVD"]

reader = Reader(rating_scale=(1, 5))
full = Dataset.load_from_df(ratings[["user", "movie", "rating"]], reader)
outer_train, outer_test = train_test_split(full, test_size=.2, random_state=42)

def raw_rows(train):
    return pd.DataFrame(
        [(int(train.to_raw_uid(u)), int(train.to_raw_iid(i)), float(r))
         for u, i, r in train.all_ratings()],
        columns=["user", "movie", "rating"]
    )

outer_df = raw_rows(outer_train)
assert len(outer_df) == 80000
inner_data = Dataset.load_from_df(outer_df[["user", "movie", "rating"]], reader)
inner_train, inner_val = train_test_split(
    inner_data, test_size=.2, random_state=17
)

class GraphRecommender:
    def __init__(self, k=30, shrinkage=25):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, frame):
        self.global_mean = float(frame.rating.mean())
        self.user_means = frame.groupby("user").rating.mean().to_dict()
        users = sorted(frame.user.unique())
        movies = sorted(frame.movie.unique())
        self.uindex = {v: j for j, v in enumerate(users)}
        self.mindex = {v: j for j, v in enumerate(movies)}
        u = frame.user.map(self.uindex).to_numpy()
        m = frame.movie.map(self.mindex).to_numpy()
        shape = (len(users), len(movies))
        binary = coo_matrix(
            (np.ones(len(frame)), (u, m)), shape=shape
        ).tocsr()
        counts = (binary.T @ binary).toarray()
        popularity = np.diag(counts).copy()
        denominator = np.sqrt(np.outer(popularity, popularity))
        sim = np.divide(
            counts, denominator + self.shrinkage,
            out=np.zeros_like(counts), where=denominator > 0
        )
        np.fill_diagonal(sim, 0)
        self.neighbors = {}
        for j, movie in enumerate(movies):
            order = np.argsort(-sim[j], kind="stable")[:self.k]
            self.neighbors[movie] = [
                (movies[t], float(sim[j, t]))
                for t in order if sim[j, t] > 0
            ]
        self.histories = {}
        for user, movie, rating in frame[
            ["user","movie","rating"]
        ].itertuples(index=False, name=None):
            self.histories.setdefault(user, {})[movie] = float(rating)
        return self

    def predict(self, user, movie):
        mean = self.user_means.get(user, self.global_mean)
        history = self.histories.get(user, {})
        numerator = 0.0
        denominator = 0.0
        for neighbor, weight in self.neighbors.get(movie, []):
            if neighbor in history:
                numerator += weight * (history[neighbor] - mean)
                denominator += abs(weight)
        if denominator == 0:
            return float(np.clip(mean, 1, 5))
        return float(np.clip(mean + numerator / denominator, 1, 5))

def evaluate(svd, graph, test):
    actual = np.array([float(r) for _, _, r in test])
    pred_svd = np.array([
        svd.predict(u, i).est for u, i, _ in test
    ])
    pred_graph = np.array([
        graph.predict(int(u), int(i)) for u, i, _ in test
    ])
    assert np.isfinite(pred_graph).all()
    return actual, pred_svd, pred_graph

def metrics(actual, predicted):
    error = actual - predicted
    return (
        float(np.sqrt(np.mean(error ** 2))),
        float(np.mean(np.abs(error)))
    )

print("=== MOLNIYA H1: SVD + GRAFO ===", flush=True)
print("Inner train:", inner_train.n_ratings,
      "Validacion:", len(inner_val), flush=True)
print("Holdout reservado:", len(outer_test), flush=True)

svd_inner = SVD(**params).fit(inner_train)
graph_inner = GraphRecommender(k=30).fit(raw_rows(inner_train))
y, s, g = evaluate(svd_inner, graph_inner, inner_val)

alphas = [0.0, .05, .10, .20, .35, .50, .75, 1.0]
rows = []
for alpha in alphas:
    pred = (1 - alpha) * s + alpha * g
    rmse, mae = metrics(y, pred)
    rows.append({
        "alpha": alpha,
        "validation_rmse": rmse,
        "validation_mae": mae
    })

validation = pd.DataFrame(rows)
validation.to_csv(OUT / "hybrid_validation.csv", index=False)
chosen = validation.sort_values(
    ["validation_rmse","alpha"]
).iloc[0]
alpha = float(chosen.alpha)

print("\n=== VALIDACION ===")
print(validation.to_string(index=False))
print("Alpha elegido:", alpha, flush=True)

print("\nEntrenamiento externo y evaluacion final...", flush=True)
svd_final = SVD(**params).fit(outer_train)
graph_final = GraphRecommender(k=30).fit(outer_df)
y, s, g = evaluate(svd_final, graph_final, outer_test)
h = (1 - alpha) * s + alpha * g

results = []
for name, pred in [
    ("SVD", s),
    ("Grafo", g),
    ("Hibrido", h)
]:
    rmse, mae = metrics(y, pred)
    results.append({
        "model": name,
        "rmse": rmse,
        "mae": mae
    })

report = pd.DataFrame(results)
report.to_csv(OUT / "hybrid_test_metrics.csv", index=False)

pd.DataFrame({
    "user": [int(u) for u, _, _ in outer_test],
    "movie": [int(i) for _, i, _ in outer_test],
    "actual": y,
    "svd": s,
    "graph": g,
    "hybrid": h
}).to_csv(OUT / "hybrid_predictions.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    validation.alpha,
    validation.validation_rmse, "o-"
)
ax.axvline(alpha, linestyle="--", color="black")
ax.set_xlabel("Peso alpha del modelo de grafos")
ax.set_ylabel("RMSE en validacion")
ax.set_title("MOLNIYA H1 - Seleccion del peso hibrido")
fig.tight_layout()
fig.savefig(
    FIG / "06_hybrid_alpha_validation.png", dpi=200
)
plt.close(fig)

assert len(report) == 3
assert len(y) == 20000
assert np.isfinite(
    report[["rmse","mae"]].to_numpy()
).all()
assert abs(metrics(y, s)[0] - 0.921526) < 0.0001

print("\n=== RESULTADOS EN PRUEBA ===")
print(report.to_string(index=False))
print(
    "\nGanancia Hibrido vs SVD:",
    round(metrics(y, s)[0] - metrics(y, h)[0], 6)
)
print("VALIDACION H1: OK")
