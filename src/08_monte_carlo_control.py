from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path.cwd()
out = root / "outputs/research"
fig = root / "figures/research"
out.mkdir(parents=True, exist_ok=True)
fig.mkdir(parents=True, exist_ok=True)

matrix = pd.read_csv(out / "diffusion_matrix.csv", index_col=0)
A = matrix.to_numpy(dtype=float)
ids = [int(x) for x in matrix.index]
n = len(ids)
I = np.eye(n)
rng = np.random.default_rng(2026)
epsilons = [1e-4, 1e-6, 1e-8]

previous = pd.read_csv(out / "pbh_strategy_comparison.csv")
strategies = {}

for row in previous.itertuples():
    movie_ids = [int(x) for x in row.movie_ids.split(",")]
    strategies[row.strategy] = [ids.index(x) for x in movie_ids]

for k in range(100):
    strategies[f"Random_{k:03d}"] = sorted(
        rng.choice(n, 3, replace=False).tolist()
    )

records = []
for name, selected in strategies.items():
    B = I[:, selected]
    W = solve_discrete_lyapunov(A, B @ B.T)
    W = (W + W.T) / 2
    eig = np.linalg.eigvalsh(W)
    row = {
        "strategy": name,
        "type": "random" if name.startswith("Random") else "designed",
        "trace": float(np.trace(W)),
        "min_eigenvalue": float(eig[0]),
    }
    for eps in epsilons:
        row[f"logdet_{eps:g}"] = float(
            np.log(np.maximum(eig + eps, 1e-300)).sum()
        )
    records.append(row)

df = pd.DataFrame(records)
df.to_csv(out / "monte_carlo_strategy_comparison.csv", index=False)

print("=== MOLNIYA: MONTE CARLO DE CONTROLABILIDAD ===")
print("100 selecciones aleatorias | 3 actuadores")
print("Resultados descriptivos; no son pruebas de significancia.")

random = df[df.type == "random"]
designed = df[df.type == "designed"]

for eps in epsilons:
    key = f"logdet_{eps:g}"
    print(f"\nREGULARIZACION epsilon={eps:g}")
    print("Aleatorio media:", round(random[key].mean(), 5))
    print("Aleatorio desviacion:", round(random[key].std(), 5))
    for row in designed.itertuples():
        score = df.loc[df.strategy == row.strategy, key].iloc[0]
        percentile = 100 * (random[key] < score).mean()
        print(f"{row.strategy}: {score:.5f} | "
              f"supera al {percentile:.1f}% de muestras aleatorias")

figura, ax = plt.subplots(figsize=(10, 6))
key = "logdet_1e-06"
ax.hist(random[key], bins=20, alpha=.65, label="Aleatorio")
for row in designed.itertuples():
    score = df.loc[df.strategy == row.strategy, key].iloc[0]
    ax.axvline(score, label=row.strategy)
ax.set_xlabel("Log-determinante regularizado")
ax.set_ylabel("Frecuencia")
ax.set_title("Controlabilidad: estrategias vs Monte Carlo")
ax.legend()
figura.tight_layout()
figura.savefig(fig / "05_monte_carlo_control.png", dpi=200)
plt.close(figura)

assert len(random) == 100
assert np.isfinite(df.select_dtypes(include="number").to_numpy()).all()
print("\nVALIDACION OK")
