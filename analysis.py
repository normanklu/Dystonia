"""
Robustness and sensitivity suite for the FS / DS crossover claim.

Addresses four reviewer concerns:
  A  crossover robustness across random seeds and structural assumptions
  B  crossover robustness across network size
  C  OAT sensitivity at multiple operating points, multiple seeds,
     with an FS/DS-comparable metric (no percent change on a 0 Hz baseline)
  D  global (all-parameters-at-once) sensitivity via Latin hypercube +
     standardised regression coefficients, and the fraction of parameter
     sets in which the crossover survives at all
"""
import numpy as np, pandas as pd, json, time, sys
from ei_sim import simulate, BASELINE, PARAM_KEYS

BASE = dict(BASELINE)
BASE['g_ext'] = 15.0      # calibrated: ~0.6 spikes per external event (see note)

# Four structural variants. The manuscript does not specify DS connectivity
# or whether the drive amplitude A scales with the synaptic weight, so both
# are treated as structural uncertainty rather than settled facts.
VARIANTS = {
    'V1_recurrentDS_fixeddrive':  dict(ds_recurrent=True,  shared_inh=False, drive_mode='fixed'),
    'V2_recurrentDS_sharedinh':   dict(ds_recurrent=True,  shared_inh=True,  drive_mode='fixed'),
    'V3_feedforwardDS':           dict(ds_recurrent=False, shared_inh=False, drive_mode='fixed'),
    'V4_drivecoupled_asoriginal': dict(ds_recurrent=True,  shared_inh=False, drive_mode='weight_coupled'),
}

S_GRID = np.round(np.linspace(0.5, 3.0, 11), 3)
N_SEEDS = 40


def crossover(s_grid, d):
    """
    First input strength at which DS overtakes FS (d = fr_DS - fr_FS crosses 0
    upward), by linear interpolation. Returns (value, status).
    """
    if np.all(d > 0):
        return np.nan, 'DS_always_higher'
    if np.all(d < 0):
        return np.nan, 'FS_always_higher'
    for i in range(len(d) - 1):
        if d[i] <= 0 < d[i + 1]:
            f = -d[i] / (d[i + 1] - d[i])
            return float(s_grid[i] + f * (s_grid[i + 1] - s_grid[i])), 'crossover'
    return np.nan, 'no_upward_crossing'


def bootstrap_ci(x, n=5000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


# ===========================================================================
# A. Seeds x structural variants
# ===========================================================================
def analysis_A():
    rows, cross_rows = [], []
    for vname, vkw in VARIANTS.items():
        curves_FS = np.zeros((N_SEEDS, len(S_GRID)))
        curves_DS = np.zeros((N_SEEDS, len(S_GRID)))
        for j, s in enumerate(S_GRID):
            par = {k: np.full(N_SEEDS, BASE[k]) for k in PARAM_KEYS}
            out = simulate(par, s, seed=1000 + j, **vkw)
            curves_FS[:, j] = out['fr_FS']
            curves_DS[:, j] = out['fr_DS']
            print(f"  A {vname} s={s:.2f}", flush=True)
        for k in range(N_SEEDS):
            xs, st = crossover(S_GRID, curves_DS[k] - curves_FS[k])
            cross_rows.append(dict(variant=vname, seed=k, crossover_s=xs, status=st))
        for j, s in enumerate(S_GRID):
            rows.append(dict(variant=vname, s=s,
                             FS_mean=curves_FS[:, j].mean(), FS_sd=curves_FS[:, j].std(ddof=1),
                             DS_mean=curves_DS[:, j].mean(), DS_sd=curves_DS[:, j].std(ddof=1),
                             diff_mean=(curves_DS[:, j] - curves_FS[:, j]).mean(),
                             diff_sd=(curves_DS[:, j] - curves_FS[:, j]).std(ddof=1),
                             frac_DS_gt_FS=float((curves_DS[:, j] > curves_FS[:, j]).mean())))
        np.savez(f'curves_{vname}.npz', FS=curves_FS, DS=curves_DS, s=S_GRID)
    pd.DataFrame(rows).to_csv('A_sweep_by_variant.csv', index=False)
    pd.DataFrame(cross_rows).to_csv('A_crossover_by_seed.csv', index=False)


# ===========================================================================
# B. Network size scaling
# ===========================================================================
def analysis_B(variant='V1_recurrentDS_fixeddrive', tag=''):
    S_B = np.round(np.linspace(0.5, 3.0, 7), 3)
    SIZES = [(10, 3), (20, 5), (40, 10), (80, 20)]
    NS = 15
    rows, cross_rows = [], []
    for scaling in ['none', 'inv_n']:
        for (Nex, Ninh) in SIZES:
            cFS = np.zeros((NS, len(S_B))); cDS = np.zeros((NS, len(S_B)))
            for j, s in enumerate(S_B):
                par = {k: np.full(NS, BASE[k]) for k in PARAM_KEYS}
                out = simulate(par, s, Nex=Nex, Ninh=Ninh, seed=2000 + j,
                               weight_scaling=scaling, **VARIANTS[variant])
                cFS[:, j] = out['fr_FS']; cDS[:, j] = out['fr_DS']
                print(f"  B scaling={scaling} Nex={Nex} s={s:.2f}", flush=True)
            for k in range(NS):
                xs, st = crossover(S_B, cDS[k] - cFS[k])
                cross_rows.append(dict(scaling=scaling, Nex=Nex, Ninh=Ninh,
                                       seed=k, crossover_s=xs, status=st))
            for j, s in enumerate(S_B):
                rows.append(dict(scaling=scaling, Nex=Nex, s=s,
                                 FS_mean=cFS[:, j].mean(), DS_mean=cDS[:, j].mean(),
                                 FS_sd=cFS[:, j].std(ddof=1), DS_sd=cDS[:, j].std(ddof=1)))
    pd.DataFrame(rows).to_csv(f'B_size_sweep{tag}.csv', index=False)
    pd.DataFrame(cross_rows).to_csv(f'B_size_crossover{tag}.csv', index=False)


# ===========================================================================
# C. OAT sensitivity, multi-seed, multi-operating-point, comparable metric
# ===========================================================================
OPS = [1.0, 2.0, 3.0]


def analysis_C(base=None, variant='V1_recurrentDS_fixeddrive', tag=''):
    BASEC = dict(BASE) if base is None else {**BASE, **base}
    NS = 30
    PERT = 0.20
    rows = []
    for s_op in OPS:
        par0 = {k: np.full(NS, BASEC[k]) for k in PARAM_KEYS}
        b = simulate(par0, s_op, seed=3000, **VARIANTS[variant])
        bFS, bDS = b['fr_FS'], b['fr_DS']
        bDIFF = bDS - bFS
        for pk in PARAM_KEYS:
            for direction, fac in [('low', 1 - PERT), ('high', 1 + PERT)]:
                par = {k: np.full(NS, BASEC[k]) for k in PARAM_KEYS}
                par[pk] = np.full(NS, BASEC[pk] * fac)
                o = simulate(par, s_op, seed=3000, **VARIANTS[variant])
                dFS = o['fr_FS'] - bFS
                dDS = o['fr_DS'] - bDS
                dDIFF = (o['fr_DS'] - o['fr_FS']) - bDIFF
                rows.append(dict(
                    s_op=s_op, parameter=pk, direction=direction, factor=fac,
                    base_FS=bFS.mean(), base_DS=bDS.mean(),
                    dFS_mean=dFS.mean(), dFS_ci_lo=np.percentile(dFS, 2.5), dFS_ci_hi=np.percentile(dFS, 97.5),
                    dDS_mean=dDS.mean(), dDS_ci_lo=np.percentile(dDS, 2.5), dDS_ci_hi=np.percentile(dDS, 97.5),
                    dDIFF_mean=dDIFF.mean(), dDIFF_ci_lo=np.percentile(dDIFF, 2.5), dDIFF_ci_hi=np.percentile(dDIFF, 97.5)))
                print(f"  C s={s_op} {pk} {direction}", flush=True)
    pd.DataFrame(rows).to_csv(f'C_oat_sensitivity{tag}.csv', index=False)


# ===========================================================================
# D. Global sensitivity (Latin hypercube) + crossover survival
# ===========================================================================
def lhs(n, d, seed=0):
    rng = np.random.default_rng(seed)
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    return u


def analysis_D(variant='V1_recurrentDS_fixeddrive', tag=''):
    NSAMP, PERT = 400, 0.20
    u = lhs(NSAMP, len(PARAM_KEYS), seed=99)
    P = {k: BASE[k] * ((1 - PERT) + 2 * PERT * u[:, i]) for i, k in enumerate(PARAM_KEYS)}
    out = {}
    for s_op in [1.0, 2.0, 3.0]:
        o = simulate(P, s_op, seed=4000, **VARIANTS[variant])
        out[s_op] = o
        print(f"  D LHS s={s_op}", flush=True)
    df = pd.DataFrame({k: P[k] for k in PARAM_KEYS})
    for s_op in [1.0, 2.0, 3.0]:
        df[f'FS_s{s_op}'] = out[s_op]['fr_FS']
        df[f'DS_s{s_op}'] = out[s_op]['fr_DS']
        df[f'DIFF_s{s_op}'] = out[s_op]['fr_DS'] - out[s_op]['fr_FS']
    df.to_csv(f'D_lhs_samples{tag}.csv', index=False)

    # Standardised regression coefficients
    src_rows = []
    X = np.column_stack([P[k] for k in PARAM_KEYS])
    Xz = (X - X.mean(0)) / X.std(0)
    Xz = np.column_stack([np.ones(NSAMP), Xz])
    for s_op in [1.0, 2.0, 3.0]:
        for tgt in ['FS', 'DS', 'DIFF']:
            y = df[f'{tgt}_s{s_op}'].values
            if y.std() == 0:
                continue
            yz = (y - y.mean()) / y.std()
            beta, *_ = np.linalg.lstsq(Xz, yz, rcond=None)
            r2 = 1 - ((yz - Xz @ beta) ** 2).sum() / (yz ** 2).sum()
            for i, k in enumerate(PARAM_KEYS):
                src_rows.append(dict(s_op=s_op, target=tgt, parameter=k,
                                     src=beta[i + 1], model_R2=r2))
    pd.DataFrame(src_rows).to_csv(f'D_src{tag}.csv', index=False)

    surv = [dict(s_op=s_op,
                 frac_DS_gt_FS=float((df[f'DIFF_s{s_op}'] > 0).mean()),
                 median_DIFF=float(df[f'DIFF_s{s_op}'].median()),
                 q05_DIFF=float(df[f'DIFF_s{s_op}'].quantile(0.05)),
                 q95_DIFF=float(df[f'DIFF_s{s_op}'].quantile(0.95)))
            for s_op in [1.0, 2.0, 3.0]]
    pd.DataFrame(surv).to_csv(f'D_crossover_survival{tag}.csv', index=False)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    t0 = time.time()
    if which in ('all', 'A'): analysis_A(); print(f"A done {time.time()-t0:.0f}s", flush=True)
    if which in ('all', 'B'): analysis_B();
    if which == 'B4': analysis_B('V4_drivecoupled_asoriginal', '_V4'); print(f"B done {time.time()-t0:.0f}s", flush=True)
    if which in ('all', 'C'): analysis_C();
    if which == 'Cr': analysis_C(dict(gE_DS_mult=0.15, gI_FS_mult=0.10, g_ext=25.0), tag='_retuned'); print(f"C done {time.time()-t0:.0f}s", flush=True)
    if which in ('all', 'D'): analysis_D();
    if which == 'D4': analysis_D('V4_drivecoupled_asoriginal', '_V4'); print(f"D done {time.time()-t0:.0f}s", flush=True)
    print("ALL DONE")
