from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research"
FIG = ROOT / "figures/research"
failures = []
passed = 0

def check(name, condition):
    global passed
    ok = bool(condition)
    print(f"{'OK' if ok else 'ERROR'} | {name}")
    if ok:
        passed += 1
    else:
        failures.append(name)

print("=" * 68)
print("MOLNIYA - AUDITORIA CIENTIFICA AMPLIADA")
print("=" * 68)

required = [
    "hybrid_validation.csv",
    "hybrid_test_metrics.csv",
    "hybrid_predictions.csv",
    "hybrid_bootstrap_5000.csv",
    "hybrid_multisplit_5seeds.csv",
    "ranking_per_user.csv",
    "ranking_bootstrap_summary.csv",
    "temporal_test_metrics.csv",
    "global_temporal_metrics.csv",
    "global_temporal_predictions.csv",
    "cold_start_diagnostic.csv",
    "bayesian_temporal_comparison.csv",
    "genre_coldstart_test.csv",
    "temporal_baseline_comparison.csv",
    "temporal_cohort_audit.csv",
    "bootstrap_rmse.csv",
    "controllability_gramian.csv",
    "pbh_strategy_comparison.csv",
    "monte_carlo_strategy_comparison.csv"
]

for name in required:
    check(f"Archivo {name}", (OUT/name).is_file())

if not failures:
    pred = pd.read_csv(OUT/"hybrid_predictions.csv")
    multi = pd.read_csv(OUT/"hybrid_multisplit_5seeds.csv")
    ranking = pd.read_csv(OUT/"ranking_bootstrap_summary.csv")
    temporal = pd.read_csv(OUT/"global_temporal_predictions.csv")
    cold = pd.read_csv(OUT/"cold_start_diagnostic.csv")
    cohorts = pd.read_csv(OUT/"temporal_cohort_audit.csv")
    control = pd.read_csv(OUT/"pbh_strategy_comparison.csv")
    mc = pd.read_csv(OUT/"monte_carlo_strategy_comparison.csv")
    h1 = pd.read_csv(OUT/"hybrid_test_metrics.csv")

    check("H1: 20000 predicciones",len(pred)==20000)
    check("H1: sin duplicados usuario-pelicula",
          not pred.duplicated(["user","movie"]).any())

    def rmse(actual, predicted):
        return np.sqrt(np.mean((actual-predicted)**2))

    recalculated = {
        "SVD":rmse(pred.actual,pred.svd),
        "Grafo":rmse(pred.actual,pred.graph),
        "Hibrido":rmse(pred.actual,pred.hybrid)
    }
    check("H1: RMSE reproducido",
          all(np.isclose(
              h1.loc[h1.model==name,"rmse"].iloc[0],
              value,atol=1e-10
          ) for name,value in recalculated.items()))

    check("H3: cinco particiones",len(multi)==5)
    check("H3: mejoras positivas",
          (multi.rmse_gain>0).all())
    check("H3: diferencias coherentes",
          np.allclose(
              multi.svd_rmse-multi.hybrid_rmse,
              multi.rmse_gain
          ))

    check("H5: tres metricas",len(ranking)==3)
    check("H5: 839 usuarios",
          ranking.users.eq(839).all())
    check("H5: intervalos ordenados",
          (ranking.ci_95_low<=ranking.ci_95_high).all())

    check("H7: 15000 predicciones",len(temporal)==15000)
    check("H7: 12568 casos cold-start",
          int(temporal.cold_start.sum())==12568)
    check("H8: escenarios suman 15000",
          int(cold.n.sum())==15000)

    check("H10: cohortes de 100000 registros",
          int(cohorts.ratings.sum())==100000)

    check("Control: tres estrategias",len(control)==3)
    check("Control: residuos Lyapunov bajos",
          (control.residual<1e-7).all())
    check("Monte Carlo: 100 selecciones aleatorias",
          int((mc.type=="random").sum())==100)

    check("Todos los resultados numericos son finitos",
          all(np.isfinite(
              frame.select_dtypes(include="number").to_numpy()
          ).all() for frame in [
              pred,multi,ranking,temporal,cold,control,mc
          ]))

for filename in [
    "06_hybrid_alpha_validation.png",
    "07_hybrid_bootstrap.png",
    "08_ranking_comparison.png",
    "09_ranking_bootstrap.png"
]:
    check(f"Figura {filename}",(FIG/filename).is_file())

print("=" * 68)
print(f"APROBADAS: {passed}")
print(f"FALLIDAS: {len(failures)}")
if failures:
    raise SystemExit(1)
print("AUDITORIA CIENTIFICA AMPLIADA: APROBADA")
