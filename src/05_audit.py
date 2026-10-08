from pathlib import Path
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
errors = []

def check(label, condition):
    ok = bool(condition)
    print(f"{'OK' if ok else 'ERROR'} | {label}")
    if not ok:
        errors.append(label)

print("=" * 65)
print("OPERACION MOLNIYA - AUDITORIA INTEGRAL")
print("=" * 65)

required = [
    OUT / "users_centrality.csv",
    OUT / "movies_centrality.csv",
    OUT / "communities.csv",
    OUT / "bipartite_clustering_sample.csv",
    OUT / "recommender_metrics.csv",
    OUT / "recommender_predictions.csv",
    OUT / "gridsearch_SVD.csv",
    OUT / "gridsearch_KNN_item.csv",
    OUT / "optimized_model_metrics.csv",
    OUT / "best_hyperparameters.json",
    OUT / "top5_recommendations_user405.csv",
    FIG / "01_top_movies.png",
    FIG / "02_network_communities.png",
    FIG / "03_recommender_comparison.png",
]
for path in required:
    check(str(path.relative_to(ROOT)), path.is_file() and path.stat().st_size > 0)

if not errors:
    ratings = pd.read_csv(
        ROOT / "data/ml-100k/u.data",
        sep="\t", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    metrics = pd.read_csv(OUT / "optimized_model_metrics.csv")
    recs = pd.read_csv(OUT / "top5_recommendations_user405.csv")
    preds = pd.read_csv(OUT / "recommender_predictions.csv")
    community = pd.read_csv(OUT / "communities.csv")
    cluster = pd.read_csv(OUT / "bipartite_clustering_sample.csv")

    check("100000 calificaciones", len(ratings) == 100000)
    check("Dos modelos optimizados", set(metrics.model) == {"SVD", "KNN_item"})
    check("RMSE y MAE finitos", metrics[["cv_rmse","test_rmse","test_mae"]].notna().all().all())
    check("40000 predicciones iniciales", len(preds) == 40000)
    check("220 nodos de comunidades", len(community) == 220)
    check("Tres comunidades", community.community.nunique() == 3)
    check("Clustering bipartito valido", cluster.bipartite_clustering.between(0,1).all())
    check("Cinco recomendaciones distintas", len(recs) == 5 and recs.movie_id.nunique() == 5)
    seen = set(ratings.loc[ratings.user_id == 405, "movie_id"])
    check("Recomendaciones no vistas", not set(recs.movie_id) & seen)
    check("Predicciones entre 1 y 5", recs.predicted_rating.between(1,5).all())

    data = Dataset.load_from_df(
        ratings[["user_id","movie_id","rating"]],
        Reader(rating_scale=(1,5))
    )
    train, test = train_test_split(data, test_size=.2, random_state=42)
    check("Split 80000 / 20000", train.n_ratings == 80000 and len(test) == 20000)
    train_pairs = {
        (train.to_raw_uid(u), train.to_raw_iid(i))
        for u,i,_ in train.all_ratings()
    }
    test_pairs = {(u,i) for u,i,_ in test}
    check("Sin pares duplicados entre train/test", not train_pairs & test_pairs)

print("=" * 65)
print("AUDITORIA APROBADA" if not errors else f"AUDITORIA CON {len(errors)} ERROR(ES)")
if errors:
    raise SystemExit(1)
