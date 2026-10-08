from pathlib import Path
import json
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split, GridSearchCV, KFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ratings = pd.read_csv(
    ROOT / "data/ml-100k/u.data",
    sep="\t", header=None,
    names=["user_id", "movie_id", "rating", "timestamp"]
)
movies = pd.read_csv(
    ROOT / "data/ml-100k/u.item",
    sep="|", header=None, encoding="latin-1",
    usecols=[0, 1], names=["movie_id", "title"]
)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    ratings[["user_id", "movie_id", "rating"]], reader
)
trainset, testset = train_test_split(
    data, test_size=0.2, random_state=42
)

training_rows = [
    (
        int(trainset.to_raw_uid(uid)),
        int(trainset.to_raw_iid(iid)),
        float(rating)
    )
    for uid, iid, rating in trainset.all_ratings()
]
train_df = pd.DataFrame(
    training_rows, columns=["user_id", "movie_id", "rating"]
)
assert len(train_df) == 80000
train_data = Dataset.load_from_df(train_df, reader)
cv = KFold(n_splits=3, random_state=42, shuffle=True)

searches = [
    (
        "SVD", SVD,
        {
            "n_factors": [50, 100],
            "n_epochs": [20, 30],
            "reg_all": [0.02, 0.08],
            "random_state": [42]
        }
    ),
    (
        "KNN_item", KNNBasic,
        {
            "k": [20, 40],
            "min_k": [1],
            "sim_options": {
                "name": ["cosine", "pearson"],
                "user_based": [False]
            },
            "verbose": [False]
        }
    )
]

final_results = []
best_models = {}

print("=== MOLNIYA: GRID SEARCH ===", flush=True)
print("CV: 3 folds | Training: 80000 | Holdout: 20000", flush=True)

for name, model_class, params in searches:
    print(f"\nOptimizando {name}...", flush=True)
    grid = GridSearchCV(
        model_class, params, measures=["rmse"],
        cv=cv, n_jobs=1
    )
    grid.fit(train_data)

    best_params = grid.best_params["rmse"]
    print("Mejor RMSE CV:", round(grid.best_score["rmse"], 5))
    print("Parametros:", best_params, flush=True)

    pd.DataFrame(grid.cv_results).to_csv(
        OUT / f"gridsearch_{name}.csv", index=False
    )

    model = model_class(**best_params)
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    final_results.append({
        "model": name,
        "cv_rmse": grid.best_score["rmse"],
        "test_rmse": rmse,
        "test_mae": mae
    })
    best_models[name] = (model_class, best_params)

    print(f"Holdout: RMSE={rmse:.5f} MAE={mae:.5f}", flush=True)

results = pd.DataFrame(final_results).sort_values("test_rmse")
results.to_csv(OUT / "optimized_model_metrics.csv", index=False)

with open(OUT / "best_hyperparameters.json", "w", encoding="utf-8") as f:
    json.dump(
        {name: params for name, (_, params) in best_models.items()},
        f, indent=2
    )

print("\n=== RESULTADOS FINALES ===")
print(results.to_string(index=False))

champion_name = min(
    final_results, key=lambda x: x["cv_rmse"]
)["model"]
model_class, champion_params = best_models[champion_name]

champion = model_class(**champion_params)
champion.fit(data.build_full_trainset())

user_id = 405
seen = set(ratings.loc[ratings.user_id == user_id, "movie_id"])
candidates = movies.loc[~movies.movie_id.isin(seen)].copy()
candidates["predicted_rating"] = [
    champion.predict(user_id, int(mid)).est
    for mid in candidates.movie_id
]
recommendations = candidates.sort_values(
    ["predicted_rating", "movie_id"],
    ascending=[False, True]
).head(5)

assert len(recommendations) == 5
assert not set(recommendations.movie_id).intersection(seen)

recommendations.to_csv(
    OUT / "top5_recommendations_user405.csv", index=False
)

print(f"\nMODELO ELEGIDO POR CV: {champion_name}")
print("\n=== TOP 5 PARA USUARIO 405 ===")
print(recommendations.to_string(index=False))
print("\nVALIDACION OK: GridSearch y recomendaciones.")
