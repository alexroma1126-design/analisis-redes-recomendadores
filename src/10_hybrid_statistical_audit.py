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

df = pd.read_csv(OUT / "hybrid_predictions.csv")
assert len(df) == 20000
assert not df[["user","movie","actual","svd","hybrid"]].isna().any().any()

df["se_svd"] = (df.actual - df.svd)**2
df["se_hybrid"] = (df.actual - df.hybrid)**2
df["ae_svd"] = abs(df.actual - df.svd)
df["ae_hybrid"] = abs(df.actual - df.hybrid)

groups = df.groupby("user").agg(
    n=("actual","size"),
    sse_svd=("se_svd","sum"),
    sse_hybrid=("se_hybrid","sum"),
    sae_svd=("ae_svd","sum"),
    sae_hybrid=("ae_hybrid","sum")
)

values = groups.to_numpy()
n_users = len(groups)
B = 5000
deltas = np.empty(B)
mae_deltas = np.empty(B)

for b in range(B):
    ix = rng.integers(0,n_users,n_users)
    sample = values[ix].sum(axis=0)
    n, ss, sh, aa, ah = sample
    deltas[b] = np.sqrt(ss/n)-np.sqrt(sh/n)
    mae_deltas[b] = aa/n-ah/n

observed = (
    np.sqrt(df.se_svd.mean()) -
    np.sqrt(df.se_hybrid.mean())
)
ci = np.quantile(deltas,[.025,.975])
ci_mae = np.quantile(mae_deltas,[.025,.975])

print("="*65)
print("MOLNIYA H2 - ROBUSTEZ ESTADISTICA")
print("="*65)
print("Usuarios:",n_users)
print("Evaluaciones:",len(df))
print("Bootstrap:",B)
print(f"Mejora RMSE observada: {observed:.8f}")
print(f"IC percentil 95%: [{ci[0]:.8f}, {ci[1]:.8f}]")
print(f"Fraccion favorable al hibrido: {(deltas>0).mean():.4f}")
print(f"IC mejora MAE 95%: [{ci_mae[0]:.8f}, {ci_mae[1]:.8f}]")

wins = groups.sse_hybrid < groups.sse_svd
print(f"Usuarios con menor SSE hibrido: {wins.sum()}/{n_users}")

pd.DataFrame({
    "rmse_improvement":deltas,
    "mae_improvement":mae_deltas
}).to_csv(OUT/"hybrid_bootstrap_5000.csv",index=False)

fig,ax = plt.subplots(figsize=(9,5))
ax.hist(deltas,bins=50,alpha=.8)
ax.axvline(0,color="black",linestyle="--")
ax.axvline(observed,color="red",linestyle="-")
ax.set_title("MOLNIYA - Bootstrap pareado por usuario")
ax.set_xlabel("RMSE SVD - RMSE hibrido")
ax.set_ylabel("Frecuencia")
fig.tight_layout()
fig.savefig(FIG/"07_hybrid_bootstrap.png",dpi=200)
plt.close(fig)

assert np.isfinite(deltas).all()
assert ci[0] <= ci[1]
print("VALIDACION H2: OK")
