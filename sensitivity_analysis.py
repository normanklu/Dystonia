#!/usr/bin/env python3
"""
Sensitivity Analysis for the E/I Network Model of Functional vs. Dystonic Synergy
-----------------------------------------------------------------------------------
This script performs a one-at-a-time (OAT) local sensitivity analysis on the
spiking network model used to simulate the "Functional Synergy" (Population 1,
balanced E/I drive) and the "Dystonic Synergy" (Population 2, unbalanced E drive)
described in the manuscript.

Each model parameter is perturbed by +/-20% from its baseline value (all other
parameters held fixed), the network is re-simulated at a fixed high-drive input
strength (s = 3.0, the peak of the sweep used in the main analysis), and the
resulting percent change in the population-averaged firing rate is recorded for
both populations. Results are summarized as a tornado plot, which is a standard
way to present local sensitivity analyses in a manuscript figure.

A fixed random seed is used for every run (baseline and perturbed) so that
differences in output are attributable to the parameter change rather than to
stochastic variation in the Poisson input (variance-reduction / "common random
numbers" approach).

Output:
    sensitivity_tornado.png  - two-panel tornado plot (300 dpi, ready for the paper)
    sensitivity_results.csv  - underlying numbers, for the supplementary material
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# ============================================================================
# 1. Neuron model (identical dynamics to the main simulation code)
# ============================================================================
class PYR:
    """Single-compartment conductance-based LIF neuron."""

    def __init__(self, nid, Cm=200.0, gL=10.0, EL=-65.0, Vth=-50.0, Vreset=-60.0,
                 tref=2.0, Esyn=0.0, tausyn=6.0, gsyn=2.0):
        self.nid = nid
        self.Cm, self.gL, self.EL = Cm, gL, EL
        self.Vth, self.Vreset, self.tref = Vth, Vreset, tref
        self.Esyn = Esyn              # >= -40 mV -> excitatory, else inhibitory
        self.tausyn = tausyn
        self.gmax = gsyn              # nS added per incoming/external spike

        self.v = Vreset
        self.reftime = 0.0
        self.gE = 0.0
        self.gI = 0.0
        self.IsynE = 0.0
        self.IsynI = 0.0
        self.sendspk = 0
        self.spktimes = []

    def add_external(self):
        self.gE += self.gmax

    def add_pre_spike(self):
        if self.Esyn >= -40:
            self.gE += self.gmax
        else:
            self.gI += self.gmax

    def step(self, dt, t_ms):
        # exponential decay of conductances
        self.gE -= dt * self.gE / self.tausyn
        self.gI -= dt * self.gI / self.tausyn

        # synaptic currents (Erev_E = 0 mV, Erev_I = -75 mV)
        self.IsynE = self.gE * (self.v - 0.0)
        self.IsynI = self.gI * (self.v - (-75.0))
        Isyn_tot = self.IsynE + self.IsynI

        if self.reftime > 0:
            self.reftime -= dt
            self.v = self.Vreset
            self.sendspk = 0
            return

        dv = dt / self.Cm * (-self.gL * (self.v - self.EL) - Isyn_tot)
        self.v += dv

        self.sendspk = 0
        if self.v >= self.Vth:
            self.v = self.Vreset
            self.reftime = self.tref
            self.sendspk = 1
            self.spktimes.append(t_ms)


# ============================================================================
# 2. Network builder / simulator (parametrized version of the main script)
# ============================================================================
def run_network(params, input_strength=3.0, tf=2000.0, dt=0.1, seed=42):
    """
    Build and simulate the two-population network for one parameter set.

    Returns
    -------
    fr1, fr2 : float
        Population-averaged firing rates (Hz) of Population 1 (Functional
        Synergy, E cells only) and Population 2 (Dystonic Synergy).
    """
    np.random.seed(seed)  # common random numbers across all sensitivity runs

    Nex, Ninh = 20, 5
    Nneuron = Nex + Ninh  # 25

    nt = int(tf / dt)
    sig = params['r'] * np.ones(nt)
    sig2 = params['r'] * np.ones(nt)

    neuron_list, syn_list = [], []

    # ---- Population 1: Functional Synergy (20 E + 5 I) ----
    for i in range(Nneuron):
        if i < Nex:  # excitatory cell, fixed weight
            neuron_list.append(PYR(i, Cm=params['Cm'], gL=params['gL'], Vth=params['Vth'],
                                    Esyn=0.0, tausyn=params['tausyn'], gsyn=params['gsyn_FS_E']))
            syn_list.append(np.arange(Nex, Nneuron))
        else:        # inhibitory cell, weight scales with drive
            neuron_list.append(PYR(i, Cm=params['Cm'], gL=params['gL'], Vth=params['Vth'],
                                    Esyn=-75.0, tausyn=params['tausyn'],
                                    gsyn=params['gsyn_FS_I_mult'] * input_strength))
            syn_list.append(np.r_[np.arange(0, Nex), np.arange(Nneuron, Nneuron + Nex)])

    # ---- Population 2: Dystonic Synergy (20 E cells, no dedicated I pop) ----
    for i in range(Nex):
        idx = Nneuron + i
        neuron_list.append(PYR(idx, Cm=params['Cm'], gL=params['gL'], Vth=params['Vth'],
                                Esyn=0.0, tausyn=params['tausyn'],
                                gsyn=params['gsyn_DS_E_mult'] * input_strength))
        syn_list.append(np.arange(Nex, Nneuron))

    # ---- simulation loop ----
    for i in range(nt - 1):
        for n, cell in enumerate(neuron_list):
            if n < Nex and np.random.rand() < sig[i] * 1e-3 * dt:
                cell.add_external()
            if n >= Nneuron and np.random.rand() < sig2[i] * 1e-3 * dt:
                cell.add_external()

            if cell.sendspk == 1:
                for tgt in syn_list[n]:
                    if tgt < len(neuron_list):
                        neuron_list[tgt].add_pre_spike()

            cell.step(dt, i * dt)

    fr1 = sum(len(c.spktimes) for c in neuron_list[:Nex]) / (tf * Nex * 1e-3)
    fr2 = sum(len(c.spktimes) for c in neuron_list[Nneuron:]) / (tf * Nex * 1e-3)
    return fr1, fr2


# ============================================================================
# 3. Baseline parameters and sensitivity sweep
# ============================================================================
baseline = dict(
    r=8.0,               # Hz, external Poisson drive rate
    tausyn=6.0,           # ms, synaptic conductance decay
    Cm=200.0,             # pF, membrane capacitance
    gL=10.0,              # nS, leak conductance
    Vth=-50.0,            # mV, spike threshold
    gsyn_FS_E=2.0,        # nS, Population-1 excitatory synaptic weight
    gsyn_FS_I_mult=6.0,   # nS per unit input strength, Population-1 inhibitory weight
    gsyn_DS_E_mult=4.0,   # nS per unit input strength, Population-2 excitatory weight
)

# Human-readable labels for the figure
labels = {
    'r': 'External drive rate  r',
    'tausyn': 'Synaptic time constant  \u03C4syn',
    'Cm': 'Membrane capacitance  Cm',
    'gL': 'Leak conductance  gL',
    'Vth': 'Spike threshold  Vth',
    'gsyn_FS_E': 'FS excitatory weight  gsyn(FS,E)',
    'gsyn_FS_I_mult': 'FS inhibitory weight  gsyn(FS,I)',
    'gsyn_DS_E_mult': 'DS excitatory weight  gsyn(DS,E)',
}

PERTURB = 0.20          # +/- 20%
INPUT_STRENGTH = 3.0    # peak drive, where FS/DS divergence is largest
TF = 2000.0
DT = 0.1

print("Running baseline simulation...")
t0 = time.time()
fr1_base, fr2_base = run_network(baseline, input_strength=INPUT_STRENGTH, tf=TF, dt=DT)
print(f"  Baseline: FS = {fr1_base:.2f} Hz, DS = {fr2_base:.2f} Hz "
      f"({time.time() - t0:.1f} s)")

results = []
for pname in baseline:
    for direction, factor in [('low', 1 - PERTURB), ('high', 1 + PERTURB)]:
        params = dict(baseline)
        params[pname] = baseline[pname] * factor
        t0 = time.time()
        fr1, fr2 = run_network(params, input_strength=INPUT_STRENGTH, tf=TF, dt=DT)
        dt_run = time.time() - t0
        results.append(dict(
            parameter=pname, direction=direction, factor=factor,
            fr1=fr1, fr2=fr2,
            # FS (Population 1) is essentially silent at baseline under this drive
            # level, so a percent change is undefined -> report the absolute
            # change in Hz instead. DS (Population 2) fires robustly at baseline,
            # so percent change is used for that population.
            abs_change_fr1=fr1 - fr1_base,
            pct_change_fr2=100 * (fr2 - fr2_base) / fr2_base if fr2_base > 0 else np.nan,
        ))
        print(f"  {pname:16s} {direction:4s} ({factor:.2f}x): "
              f"FS={fr1:6.2f} Hz ({results[-1]['abs_change_fr1']:+6.2f} Hz)  "
              f"DS={fr2:6.2f} Hz ({results[-1]['pct_change_fr2']:+6.1f}%)  [{dt_run:.1f}s]")

df = pd.DataFrame(results)
df.to_csv('sensitivity_results.csv', index=False)

# ============================================================================
# 4. Tornado plot
# ============================================================================
def tornado_data(df, metric):
    rows = []
    for pname in baseline:
        sub = df[df.parameter == pname]
        low = sub[sub.direction == 'low'][metric].values[0]
        high = sub[sub.direction == 'high'][metric].values[0]
        rows.append((pname, low, high, max(abs(low), abs(high))))
    rows.sort(key=lambda r: r[3])  # ascending -> largest at top after barh
    return rows

def plot_tornado(ax, rows, title, xlabel, symlog=False):
    ynames = [labels[r[0]] for r in rows]
    ypos = np.arange(len(rows))
    for y, (pname, low, high, _) in zip(ypos, rows):
        ax.barh(y, high - 0, left=0, height=0.6,
                color='#c0392b' if high > 0 else '#2980b9', alpha=0.85, zorder=2)
        ax.barh(y, low - 0, left=0, height=0.6,
                color='#c0392b' if low > 0 else '#2980b9', alpha=0.55, zorder=2)
        # annotate the two most extreme bars with their value so the
        # symlog-compressed axis stays readable
        for val in (low, high):
            if abs(val) == max(abs(low), abs(high)) and abs(val) > 100:
                ax.text(val, y, f'  {val:+.0f}%', va='center',
                        ha='left' if val > 0 else 'right', fontsize=7.5, color='#444')
    ax.axvline(0, color='k', linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ynames, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    if symlog:
        ax.set_xscale('symlog', linthresh=10)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

rows_fs = tornado_data(df, 'abs_change_fr1')
rows_ds = tornado_data(df, 'pct_change_fr2')

plot_tornado(axes[0], rows_fs, f'Functional Synergy (Pop. 1)\nbaseline = {fr1_base:.2f} Hz (quiescent)',
             'Absolute change in firing rate (Hz)')
plot_tornado(axes[1], rows_ds, f'Dystonic Synergy (Pop. 2)\nbaseline = {fr2_base:.2f} Hz',
             '% change in firing rate\nrelative to baseline (symlog scale)', symlog=True)

# shared legend
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor='#c0392b', alpha=0.85, label=f'Parameter +{int(PERTURB*100)}%'),
    Patch(facecolor='#2980b9', alpha=0.55, label=f'Parameter \u2212{int(PERTURB*100)}%'),
]
fig.legend(handles=legend_elems, loc='lower center', ncol=2, frameon=False, fontsize=9)

fig.suptitle('Local Sensitivity Analysis (OAT, \u00B120%) at Input Strength s = 3.0',
             fontsize=12, y=1.02)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig('sensitivity_tornado.png', dpi=300, bbox_inches='tight')
print("\nFinished - figure written to sensitivity_tornado.png")
print("Underlying values written to sensitivity_results.csv")
