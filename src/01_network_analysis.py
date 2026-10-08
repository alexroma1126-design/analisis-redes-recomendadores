from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ml-100k'
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

ratings = pd.read_csv(DATA / 'u.data', sep='\t', header=None, names=['user_id','movie_id','rating','timestamp'])
movies = pd.read_csv(DATA / 'u.item', sep='|', header=None, encoding='latin-1', usecols=[0,1], names=['movie_id','title'])

G = nx.Graph()
G.add_nodes_from((('u', int(u)) for u in ratings.user_id.unique()), bipartite=0)
G.add_nodes_from((('m', int(m)) for m in ratings.movie_id.unique()), bipartite=1)
G.add_weighted_edges_from((('u', int(r.user_id)), ('m', int(r.movie_id)), float(r.rating)) for r in ratings.itertuples(index=False))

users = ratings.groupby('user_id').agg(degree=('movie_id','nunique'), mean_rating=('rating','mean')).reset_index().sort_values(['degree','user_id'],ascending=[False,True])
films = ratings.groupby('movie_id').agg(degree=('user_id','nunique'), mean_rating=('rating','mean')).reset_index().merge(movies,on='movie_id',how='left').sort_values(['degree','movie_id'],ascending=[False,True])
users.to_csv(OUT / 'users_centrality.csv',index=False)
films.to_csv(OUT / 'movies_centrality.csv',index=False)

assert G.number_of_nodes() == 2625
assert G.number_of_edges() == 100000
assert nx.is_bipartite(G)

print('=== OPERACION MOLNIYA: ANALISIS DE RED ===')
print('Nodos:',G.number_of_nodes())
print('Aristas:',G.number_of_edges())
print('Bipartito:',nx.is_bipartite(G))
print('\n=== TOP 5 USUARIOS ===')
print(users.head(5).to_string(index=False))
print('\n=== TOP 10 PELICULAS ===')
print(films[['movie_id','title','degree','mean_rating']].head(10).to_string(index=False))

fig,ax=plt.subplots(figsize=(11,6))
top=films.head(10).sort_values('degree')
ax.barh(top.title,top.degree)
ax.set_xlabel('Numero de usuarios que calificaron la pelicula')
ax.set_title('MovieLens 100K | Peliculas con mayor grado')
fig.tight_layout()
fig.savefig(FIG / '01_top_movies.png',dpi=180)
plt.close(fig)
print('\nVALIDACION OK. CSV y figura generados.')
