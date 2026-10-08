from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from surprise import Dataset, Reader, SVD

ROOT = Path.cwd()
OUT = ROOT / "outputs/research"
OUT.mkdir(parents=True, exist_ok=True)

ratings = pd.read_csv(
    ROOT / "data/ml-100k/u.data",
    sep="\t", header=None,
    names=["user", "movie", "rating", "timestamp"]
).sort_values(
    ["timestamp", "user", "movie"]
).reset_index(drop=True)

genre_cols = [f"genre_{i}" for i in range(19)]
items = pd.read_csv(
    ROOT / "data/ml-100k/u.item",
    sep="|", header=None, encoding="latin-1",
    usecols=[0] + list(range(5,24))
)
items.columns = ["movie"] + genre_cols
assert items.movie.is_unique

features = items.set_index("movie")[genre_cols]

train = ratings.iloc[:70000].copy()
val = ratings.iloc[70000:85000].copy()
dev = ratings.iloc[:85000].copy()
test = ratings.iloc[85000:].copy()

params = json.loads(
    (ROOT / "outputs/best_hyperparameters.json")
    .read_text(encoding="utf-8")
)["SVD"]

reader = Reader(rating_scale=(1,5))

def fit_svd(frame):
    data = Dataset.load_from_df(
        frame[["user","movie","rating"]], reader
    )
    return SVD(**params).fit(
        data.build_full_trainset()
    )

def genre_matrix(frame):
    X = features.reindex(frame.movie.to_numpy())
    assert X.notna().all().all()
    return X.to_numpy(dtype=float)

def evaluate(svd, content, frame):
    y = frame.rating.to_numpy(dtype=float)
    s = np.array([
        svd.predict(int(u),int(m)).est
        for u,m in zip(frame.user,frame.movie)
    ])
    c = np.clip(content.predict(genre_matrix(frame)),1,5)
    return y,s,c

def rmse(y,p):
    return float(np.sqrt(np.mean((y-p)**2)))

print("=== MOLNIYA H11: MODELO BASADO EN GENEROS ===")
print("Entrenamiento:",len(train))
print("Validacion:",len(val))
print("Prueba:",len(test))

svd = fit_svd(train)
new_val = ~val.user.isin(train.user)

candidates = []
for reg in [1,10,100,1000]:
    content = Ridge(alpha=reg)
    content.fit(genre_matrix(train),train.rating)

    y,s,c = evaluate(svd,content,val)

    for weight in [0,.1,.25,.5,.75,1]:
        hybrid = (1-weight)*s+weight*c
        candidates.append({
            "ridge_alpha":reg,
            "content_weight":weight,
            "rmse_new_users":rmse(
                y[new_val],hybrid[new_val]
            ),
            "rmse_all":rmse(y,hybrid)
        })

search = pd.DataFrame(candidates)
search.to_csv(
    OUT / "genre_coldstart_validation.csv",
    index=False
)

best = search.sort_values(
    ["rmse_new_users","rmse_all"]
).iloc[0]

regularization = float(best.ridge_alpha)
weight = float(best.content_weight)

print("\nPARAMETROS SELECCIONADOS:")
print(best.to_string())

svd = fit_svd(dev)
content = Ridge(alpha=regularization)
content.fit(genre_matrix(dev),dev.rating)

y,s,c = evaluate(svd,content,test)
hybrid = (1-weight)*s+weight*c

new_users = ~test.user.isin(dev.user)

records = []
for group,mask in [
    ("GLOBAL",np.ones(len(test),dtype=bool)),
    ("USUARIOS_NUEVOS",new_users.to_numpy()),
    ("USUARIOS_CONOCIDOS",(~new_users).to_numpy())
]:
    for name,pred in [
        ("SVD",s),
        ("Generos",c),
        ("Hibrido_contenido",hybrid)
    ]:
        records.append({
            "group":group,
            "model":name,
            "n":int(mask.sum()),
            "rmse":rmse(y[mask],pred[mask]),
            "mae":float(np.mean(
                np.abs(y[mask]-pred[mask])
            ))
        })

report = pd.DataFrame(records)
report.to_csv(
    OUT / "genre_coldstart_test.csv",
    index=False
)

print("\n=== RESULTADOS H11 ===")
print(report.to_string(index=False))

assert len(y)==15000
assert np.isfinite(
    report[["rmse","mae"]].to_numpy()
).all()

print("\nVALIDACION H11: OK")
