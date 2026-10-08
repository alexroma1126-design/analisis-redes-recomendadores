from pathlib import Path
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, BaselineOnly

ROOT = Path.cwd()
OUT = ROOT / "outputs/research"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "data/ml-100k/u.data",
    sep="\t", header=None,
    names=["user", "movie", "rating", "timestamp"]
).sort_values(
    ["timestamp", "user", "movie"]
).reset_index(drop=True)

train = df.iloc[:70000]
validation = df.iloc[70000:85000]
development = df.iloc[:85000]
test = df.iloc[85000:]

reader = Reader(rating_scale=(1, 5))

def fit(model, frame):
    data = Dataset.load_from_df(
        frame[["user", "movie", "rating"]], reader
    )
    return model.fit(data.build_full_trainset())

def evaluate(model, frame):
    y = frame.rating.to_numpy(dtype=float)
    p = np.array([
        model.predict(int(u), int(i)).est
        for u, i in zip(frame.user, frame.movie)
    ])
    return (
        float(np.sqrt(np.mean((y-p)**2))),
        float(np.mean(abs(y-p)))
    )

configs = []

for factors in [30, 60, 100]:
    for reg in [0.02, 0.08, 0.15]:
        configs.append((
            "SVD",
            {"n_factors": factors,
             "n_epochs": 30,
             "reg_all": reg,
             "random_state": 42}
        ))

for reg in [5, 10, 20]:
    configs.append((
        "BaselineOnly",
        {"bsl_options": {
            "method": "als",
            "reg_u": reg,
            "reg_i": reg,
            "n_epochs": 15
        },
         "verbose": False}
    ))

rows = []

print("=== MOLNIYA H12: BASELINES TEMPORALES ===", flush=True)

for name, params in configs:
    cls = SVD if name == "SVD" else BaselineOnly
    model = fit(cls(**params), train)
    rmse, mae = evaluate(model, validation)
    rows.append({
        "model": name,
        "params": repr(params),
        "validation_rmse": rmse,
        "validation_mae": mae
    })
    print(name, params, "RMSE:", round(rmse, 6), flush=True)

search = pd.DataFrame(rows)
search.to_csv(
    OUT / "temporal_baseline_search.csv", index=False
)

best = search.sort_values(
    ["model", "validation_rmse"]
).groupby("model").head(1)

results = []
for row in best.itertuples():
    import ast
    params = ast.literal_eval(row.params)
    cls = SVD if row.model == "SVD" else BaselineOnly
    model = fit(cls(**params), development)
    rmse, mae = evaluate(model, test)
    results.append({
        "model": row.model,
        "validation_rmse": row.validation_rmse,
        "test_rmse": rmse,
        "test_mae": mae,
        "params": row.params
    })

report = pd.DataFrame(results)
report.to_csv(
    OUT / "temporal_baseline_comparison.csv",
    index=False
)

print("\n=== COMPARACION FINAL ===")
print(report.to_string(index=False))
assert len(report) == 2
assert np.isfinite(report.test_rmse).all()
print("\nVALIDACION H12: OK")
print("Evaluacion exploratoria: test ya utilizado en H7-H11.")
