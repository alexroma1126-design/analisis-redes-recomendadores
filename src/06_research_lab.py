from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "research"
FIG = ROOT / "figures" / "research"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

ratings = pd.read_csv(
    ROOT / "data/ml-100k/u.data", sep="\t", header=None,
    names=["user", "movie", "rating", "timestamp"]
)
movies = pd.read_csv(
    ROOT / "data/ml-100k/u.item", sep="|",
    header=None, encoding="latin-1",
    usecols=[0, 1], names=["movie", "title"]
)

print("=" * 65, flush=True)
print("MOLNIYA RESEARCH LAB - ESTOCASTICA Y CONTROLABILIDAD", flush=True)
print("=" * 65, flush=True)

# A: Bootstrap agrupado por usuario
print("\n[A] ENTRENAMIENTO Y BOOTSTRAP", flush=True)

data = Dataset.load_from_df(
    ratings[["user", "movie", "rating"]],
    Reader(rating_scale=(1, 5))
)
train, test = train_test_split(
    data, test_size=0.2, random_state=42
)

with open(ROOT / "outputs/best_hyperparameters.json",
          encoding="utf-8") as f:
    params = json.load(f)

svd = SVD(**params["SVD"])
knn = KNNBasic(**params["KNN_item"])
svd.fit(train)
knn.fit(train)
p_svd = svd.test(test)
p_knn = knn.test(test)

assert len(p_svd) == len(p_knn) == 20000
assert all(
    (a.uid, a.iid, a.r_ui) == (b.uid, b.iid, b.r_ui)
    for a, b in zip(p_svd, p_knn)
)

df = pd.DataFrame({
    "user": [p.uid for p in p_svd],
    "sq_svd": [(p.r_ui - p.est) ** 2 for p in p_svd],
    "sq_knn": [(p.r_ui - p.est) ** 2 for p in p_knn]
})
summary = df.groupby("user").agg(
    n=("sq_svd", "size"),
    sse_svd=("sq_svd", "sum"),
    sse_knn=("sq_knn", "sum")
).reset_index()

n_users = len(summary)
B = 1000
delta = np.empty(B)
svd_scores = np.empty(B)
knn_scores = np.empty(B)
counts = summary["n"].to_numpy()
sse_svd = summary["sse_svd"].to_numpy()
sse_knn = summary["sse_knn"].to_numpy()

for b in range(B):
    idx = rng.integers(0, n_users, size=n_users)
    denominator = counts[idx].sum()
    a = np.sqrt(sse_svd[idx].sum() / denominator)
    c = np.sqrt(sse_knn[idx].sum() / denominator)
    svd_scores[b] = a
    knn_scores[b] = c
    delta[b] = c - a

observed = np.sqrt(df.sq_knn.mean()) - np.sqrt(df.sq_svd.mean())
ci = np.quantile(delta, [0.025, 0.975])
prob = np.mean(delta > 0)

pd.DataFrame({
    "rmse_svd": svd_scores,
    "rmse_knn": knn_scores,
    "delta_knn_minus_svd": delta
}).to_csv(OUT / "bootstrap_rmse.csv", index=False)

print(f"Usuarios de prueba: {n_users}")
print(f"Replicas bootstrap: {B}")
print(f"Diferencia RMSE observada KNN-SVD: {observed:.6f}")
print(f"IC bootstrap percentil 95%: [{ci[0]:.6f}, {ci[1]:.6f}]")
print(f"Proporcion de replicas favorables a SVD: {prob:.3f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(delta, bins=45, alpha=0.8)
ax.axvline(0, linestyle="--", color="black",
           label="Igual rendimiento")
ax.axvline(observed, linestyle="-", color="red",
           label="Diferencia observada")
ax.set_xlabel("RMSE KNN - RMSE SVD")
ax.set_ylabel("Frecuencia")
ax.set_title("Bootstrap agrupado por usuario")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "01_bootstrap_rmse.png", dpi=200)
plt.close(fig)

assert np.isfinite(delta).all()
assert ci[0] <= ci[1]

# B: Controlabilidad lineal sobre red de peliculas
print("\n[B] CONTROLABILIDAD SOBRE RED DE PELICULAS", flush=True)

top_movies = ratings["movie"].value_counts().head(30).index.tolist()
subset = ratings[ratings.movie.isin(top_movies)]
binary = pd.crosstab(subset.user, subset.movie)
binary = (binary > 0).astype(float)
binary = binary.reindex(columns=top_movies)

X = binary.to_numpy()
C = X.T @ X
np.fill_diagonal(C, 0)

degrees = np.sum(X, axis=0)
denom = np.sqrt(np.outer(degrees, degrees))
S = np.divide(C, denom, out=np.zeros_like(C), where=denom > 0)
np.fill_diagonal(S, 0)

spectral_norm = np.linalg.norm(S, ord=2)
assert spectral_norm > 0
A = 0.8 * S / spectral_norm

rho = np.max(np.abs(np.linalg.eigvalsh(A)))
assert rho < 1.0
print(f"Peliculas: {len(top_movies)}")
print(f"Radio espectral de A: {rho:.6f}")

records = []
for j in range(10):
    b = np.zeros((30, 1))
    b[j, 0] = 1.0
    Q = b @ b.T
    W = solve_discrete_lyapunov(A, Q)
    W = (W + W.T) / 2

    eig = np.linalg.eigvalsh(W)
    residual = np.linalg.norm(A @ W @ A.T - W + Q)
    assert residual < 1e-7
    assert eig.min() > -1e-8

    probabilities = np.maximum(eig, 0)
    probabilities /= max(probabilities.sum(), 1e-15)
    positive = probabilities[probabilities > 1e-15]
    effective_rank = np.exp(
        -np.sum(positive * np.log(positive))
    )

    records.append({
        "movie": top_movies[j],
        "trace_gramian": np.trace(W),
        "min_eigenvalue": eig.min(),
        "effective_rank": effective_rank,
        "lyapunov_residual": residual
    })

control = pd.DataFrame(records)
control = control.merge(movies, on="movie", how="left")
control = control.sort_values("trace_gramian", ascending=False)
control.to_csv(
    OUT / "controllability_gramian.csv", index=False
)

pd.DataFrame(
    A, index=top_movies, columns=top_movies
).to_csv(OUT / "diffusion_matrix.csv")

fig, ax = plt.subplots(figsize=(11, 6))
plot = control.sort_values("trace_gramian")
ax.barh(plot.title, plot.trace_gramian)
ax.set_xlabel("Traza del Gramiano")
ax.set_title("Controlabilidad promedio - entradas individuales")
fig.tight_layout()
fig.savefig(FIG / "02_control_gramian.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(A, cmap="viridis")
fig.colorbar(im, ax=ax, label="Peso de difusion")
ax.set_title("Operador A - red de 30 peliculas")
ax.set_xlabel("Indice de pelicula")
ax.set_ylabel("Indice de pelicula")
fig.tight_layout()
fig.savefig(FIG / "03_diffusion_matrix.png", dpi=200)
plt.close(fig)

print("\nTOP 5 EN CONTROLABILIDAD PROMEDIO:")
print(
    control[[
        "title", "trace_gramian",
        "min_eigenvalue", "effective_rank"
    ]].head(5).to_string(index=False)
)
print("\nNOTA: traza alta no implica controlabilidad completa.")
print("VALIDACION OK - outputs/research y figures/research")
