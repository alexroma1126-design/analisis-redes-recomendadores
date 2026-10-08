import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

def raw_rows(train):
    return pd.DataFrame(
        [(int(train.to_raw_uid(u)), int(train.to_raw_iid(i)), float(r))
         for u, i, r in train.all_ratings()],
        columns=["user", "movie", "rating"]
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
