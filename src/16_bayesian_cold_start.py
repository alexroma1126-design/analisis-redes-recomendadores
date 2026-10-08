from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "outputs/research"

r = pd.read_csv(
    ROOT / "data/ml-100k/u.data", sep="\t",
    header=None,
    names=["user","movie","rating","timestamp"]
).sort_values(
    ["timestamp","user","movie"]
).reset_index(drop=True)

train = r.iloc[:70000]
val = r.iloc[70000:85000]
dev = r.iloc[:85000]
test = r.iloc[85000:]

def predict(train, target, lam):
    mu = train.rating.mean()
    stats = train.groupby("movie").rating.agg(
        ["count","mean"]
    )
    smooth = (
        stats["count"] * stats["mean"] + lam * mu
    ) / (stats["count"] + lam)
    return target.movie.map(smooth).fillna(mu).to_numpy()

grid = [0,1,5,10,20,50,100,200]
rows = []

for lam in grid:
    p = predict(train,val,lam)
    rmse = np.sqrt(np.mean((val.rating-p)**2))
    rows.append({"lambda":lam,"validation_rmse":rmse})

cv = pd.DataFrame(rows)
lam = int(cv.sort_values(
    ["validation_rmse","lambda"]
).iloc[0]["lambda"])

cv.to_csv(OUT/"bayesian_temporal_validation.csv",index=False)

p = predict(dev,test,lam)
original = pd.read_csv(
    OUT/"global_temporal_predictions.csv"
)

assert len(original)==len(test)==15000
assert np.array_equal(original.user.to_numpy(),test.user.to_numpy())
assert np.array_equal(original.movie.to_numpy(),test.movie.to_numpy())

original["bayesian"] = p

results = []
for scenario,group in original.groupby("cold_start"):
    for model in ["svd","graph","bayesian"]:
        error = group.actual-group[model]
        results.append({
            "cold_start":bool(scenario),
            "model":model,
            "n":len(group),
            "rmse":float(np.sqrt(np.mean(error**2))),
            "mae":float(np.mean(abs(error)))
        })

for model in ["svd","graph","bayesian"]:
    error = original.actual-original[model]
    results.append({
        "cold_start":"GLOBAL",
        "model":model,
        "n":len(original),
        "rmse":float(np.sqrt(np.mean(error**2))),
        "mae":float(np.mean(abs(error)))
    })

report = pd.DataFrame(results)
report.to_csv(
    OUT/"bayesian_temporal_comparison.csv",
    index=False
)
original.to_csv(
    OUT/"bayesian_temporal_predictions.csv",
    index=False
)

print("=== MOLNIYA H9: BASELINE BAYESIANO ===")
print("\nVALIDACION:")
print(cv.to_string(index=False))
print("\nLambda seleccionado:",lam)
print("\nRESULTADOS DE PRUEBA:")
print(report.to_string(index=False))

assert np.isfinite(report[["rmse","mae"]].to_numpy()).all()
print("\nVALIDACION H9: OK")
