from pathlib import Path
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'
r = pd.read_csv(ROOT / 'data/ml-100k/u.data',sep='\t',header=None,names=['user','movie','rating','timestamp'])

print('=== MOLNIYA: COMUNIDADES ===',flush=True)
print('Preparando subred reproducible...',flush=True)

users = r.user.value_counts().head(120).index
movies = r.movie.value_counts().head(100).index
sample = r[r.user.isin(users) & r.movie.isin(movies)]

G = nx.Graph()
G.add_nodes_from([f'u{x}' for x in users],bipartite=0)
G.add_nodes_from([f'm{x}' for x in movies],bipartite=1)
G.add_edges_from(
    (f'u{u}',f'm{m}')
    for u,m in sample[['user','movie']].itertuples(index=False,name=None)
)
assert nx.is_bipartite(G)

print('Nodos:',G.number_of_nodes())
print('Aristas:',G.number_of_edges())

print('Calculando comunidades...',flush=True)
communities = nx.community.louvain_communities(
    G,weight=None,seed=42
)
Q = nx.community.modularity(G,communities,weight=None)

labels = {
    node:i
    for i,group in enumerate(communities)
    for node in group
}

pd.DataFrame([
    {
        'node':n,
        'type':'user' if n.startswith('u') else 'movie',
        'community':labels[n],
        'degree':G.degree(n)
    }
    for n in G
]).to_csv(OUT/'communities.csv',index=False)

print('Comunidades:',len(communities))
print('Modularidad:',round(Q,4))

print('Calculando clustering bipartito...',flush=True)
selected = sorted(
    G.nodes,key=lambda n:G.degree(n),reverse=True
)[:30]
coeff = bipartite.clustering(
    G,nodes=selected,mode='dot'
)

pd.DataFrame([
    {
        'node':n,
        'degree':G.degree(n),
        'bipartite_clustering':coeff[n]
    }
    for n in selected
]).to_csv(
    OUT/'bipartite_clustering_sample.csv',index=False
)

print(
    'Clustering promedio muestra:',
    round(sum(coeff.values())/len(coeff),4)
)

print('Generando figura...',flush=True)
pos = nx.spring_layout(G,seed=42,iterations=70)

fig,ax = plt.subplots(figsize=(13,10))
nx.draw_networkx_edges(
    G,pos,ax=ax,alpha=.07,width=.35
)
nx.draw_networkx_nodes(
    G,pos,
    nodelist=list(G.nodes),
    node_color=[labels[n] for n in G],
    cmap=plt.cm.tab20,
    node_size=[
        max(12,min(150,G.degree(n)*2))
        for n in G
    ],
    ax=ax
)
ax.set_title(
    'MovieLens 100K | Comunidades Louvain (subred 220 nodos)'
)
ax.axis('off')
fig.tight_layout()
fig.savefig(
    FIG/'02_network_communities.png',dpi=180
)
plt.close(fig)

assert len(labels)==G.number_of_nodes()
assert all(0<=x<=1 for x in coeff.values())
print('VALIDACION OK: comunidades, clustering y figura generados.')
