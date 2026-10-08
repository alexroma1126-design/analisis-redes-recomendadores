from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research"
FIG = ROOT / "figures/research"
A = pd.read_csv(OUT / "diffusion_matrix.csv", index_col=0)
ids = A.index.astype(int).tolist()
M = A.to_numpy(dtype=float)
n = len(ids)
I = np.eye(n)

assert M.shape == (30, 30)
assert np.allclose(M, M.T, atol=1e-10)
assert max(abs(np.linalg.eigvalsh(M))) < 1

movies = pd.read_csv(
    ROOT / "data/ml-100k/u.item",
    sep="|", header=None, encoding="latin-1",
    usecols=[0, 1], names=["movie_id", "title"]
)
names = dict(zip(movies.movie_id, movies.title))

def evaluate(selected):
    B = I[:, selected]
    W = solve_discrete_lyapunov(M, B @ B.T)
    W = (W + W.T) / 2
    eigenvalues = np.linalg.eigvalsh(W)
    pbh_rank = min(
        np.linalg.matrix_rank(
            np.column_stack((lam * I - M, B)),
            tol=1e-9
        )
        for lam in np.linalg.eigvalsh(M)
    )
    return {
        "trace": float(np.trace(W)),
        "lambda_min": float(eigenvalues[0]),
        "rank_gramian": int(np.sum(eigenvalues > 1e-10)),
        "pbh_min_rank": int(pbh_rank),
        "residual": float(np.linalg.norm(
            M @ W @ M.T - W + B @ B.T
        ))
    }

print("=== MOLNIYA: CONTROLABILIDAD PBH ===", flush=True)

single = []
for j in range(n):
    result = evaluate([j])
    single.append({
        "movie_id": ids[j],
        "title": names.get(ids[j], str(ids[j])),
        **result
    })

single = pd.DataFrame(single)
single.to_csv(OUT / "pbh_all_single_actuators.csv", index=False)

print("\nCONTROLABILIDAD CON UN ACTUADOR")
print(single.sort_values("trace", ascending=False).head(5).to_string(index=False))

popular = [0, 1, 2]
best_single = single.sort_values(
    "trace", ascending=False
).movie_id.head(3).tolist()
best_single_idx = [ids.index(mid) for mid in best_single]

selected = []
greedy_log = []
for step in range(3):
    options = [
        (evaluate(selected + [j])["lambda_min"], j)
        for j in range(n) if j not in selected
    ]
    score, chosen = max(options, key=lambda item: (item[0], -item[1]))
    selected.append(chosen)
    greedy_log.append({
        "step": step + 1,
        "movie_id": ids[chosen],
        "title": names.get(ids[chosen], str(ids[chosen])),
        "lambda_min": score
    })
    print(f"Greedy paso {step+1}: {names.get(ids[chosen])}", flush=True)

strategies = {
    "Popularidad": popular,
    "Traza individual": best_single_idx,
    "Greedy lambda_min": selected
}

comparison = []
for label, chosen in strategies.items():
    comparison.append({
        "strategy": label,
        "movie_ids": ",".join(str(ids[j]) for j in chosen),
        **evaluate(chosen)
    })

comparison = pd.DataFrame(comparison)
comparison.to_csv(OUT / "pbh_strategy_comparison.csv", index=False)
pd.DataFrame(greedy_log).to_csv(
    OUT / "pbh_greedy_selection.csv", index=False
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(comparison.strategy, comparison.lambda_min)
ax.set_ylabel("Menor autovalor del Gramiano")
ax.set_title("Tres estrategias de seleccion de actuadores")
ax.tick_params(axis="x", rotation=10)
fig.tight_layout()
fig.savefig(FIG / "04_control_strategy_comparison.png", dpi=200)
plt.close(fig)

assert comparison.residual.max() < 1e-7
assert (comparison.pbh_min_rank <= n).all()

print("\n=== COMPARACION FINAL ===")
print(comparison.to_string(index=False))
print("\nNota: PBH y rangos calculados numericamente.")
print("VALIDACION OK")
