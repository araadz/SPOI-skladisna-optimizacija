"""
================================================================================
WAREHOUSE OPTIMIZATION MODEL
================================================================================

Formule:
- Cost: C = α·H + β·max(0, V-2) + γ·(4-E)
- Quality: Q = exp(-C/τ)
- Utility: U = Q · (1 + λ·d/d_max)
================================================================================
"""

import numpy as np
import pandas as pd
import random

# ============================================================
# DEFAULT PARAMETRI
# ============================================================

DEFAULT_PARAMS = {
    'COST_A': 2.0,
    'COST_B': 5.0,
    'COST_C': 2.0,
    'TAU': 8.0,
    'DEMAND_MULTIPLIER': 15,
    'N_PICKS': 1000
}

# ============================================================
# COST / QUALITY / UTILITY
# ============================================================

def calculate_cost(H, V, E, params=None):
    if params is None:
        params = DEFAULT_PARAMS
    return (
        params['COST_A'] * H +
        params['COST_B'] * max(0, V - 2) +
        params['COST_C'] * (4 - E)
    )

def calculate_quality(H, V, E, params=None):
    if params is None:
        params = DEFAULT_PARAMS
    return np.exp(-calculate_cost(H, V, E, params) / params['TAU'])

def calculate_utility(izlaz_norm, H, V, E, params=None):
    if params is None:
        params = DEFAULT_PARAMS
    return calculate_quality(H, V, E, params) * (1 + izlaz_norm * params['DEMAND_MULTIPLIER'])

# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):
    df = df.dropna(subset=['H','V','E','izlaz']).reset_index(drop=True)
    if 'TEZINA_KAT' not in df.columns:
        df['TEZINA_KAT'] = 4
    df['TEZINA_KAT'] = df['TEZINA_KAT'].fillna(4)
    max_izlaz = df['izlaz'].max() if df['izlaz'].max() > 0 else 1
    df['izlaz_norm'] = df['izlaz'] / max_izlaz
    return df

# ============================================================
# UTILITY MATRIX
# ============================================================

def generate_utility_matrix(df, df_positions, params=None):
    if params is None:
        params = DEFAULT_PARAMS
    n = len(df)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            U[i, j] = calculate_utility(
                df.iloc[i]['izlaz_norm'],
                df_positions.iloc[j]['H'],
                df_positions.iloc[j]['V'],
                df_positions.iloc[j]['E'],
                params
            )
    return U

# ============================================================
# SIMULATION
# ============================================================

def simulate_picks(assignment, df, df_positions, params=None):
    if params is None:
        params = DEFAULT_PARAMS
    n = len(df)

    utils = np.array([
        calculate_utility(
            df.iloc[i]['izlaz_norm'],
            df_positions.iloc[assignment[i]]['H'],
            df_positions.iloc[assignment[i]]['V'],
            df_positions.iloc[assignment[i]]['E'],
            params
        ) for i in range(n)
    ])

    costs = np.array([
        calculate_cost(
            df_positions.iloc[assignment[i]]['H'],
            df_positions.iloc[assignment[i]]['V'],
            df_positions.iloc[assignment[i]]['E'],
            params
        ) for i in range(n)
    ])

    izlaz = df['izlaz'].values
    probs = izlaz / izlaz.sum() if izlaz.sum() > 0 else np.ones(n)/n
    np.random.seed(42)
    picked = np.random.choice(n, size=params['N_PICKS'], p=probs)
    sim_cost = costs[picked].sum()

    wH = np.sum(df_positions.iloc[list(assignment.values())]['H'].values * izlaz) / izlaz.sum()
    wV = np.sum(df_positions.iloc[list(assignment.values())]['V'].values * izlaz) / izlaz.sum()

    return utils, costs, sim_cost, wH, wV

# ============================================================
# ILP SOLVER
# ============================================================

def solve_ilp(U, df, df_positions):
    import pulp
    n = len(df)
    prob = pulp.LpProblem("Warehouse", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", ((i,j) for i in range(n) for j in range(n)), cat='Binary')
    prob += pulp.lpSum(U[i,j]*x[i,j] for i in range(n) for j in range(n))

    for i in range(n):
        prob += pulp.lpSum(x[i,j] for j in range(n)) == 1
    for j in range(n):
        prob += pulp.lpSum(x[i,j] for i in range(n)) <= 1

    heavy = df[df['TEZINA_KAT']>=4].index.tolist()
    high_v = df_positions[df_positions['V']>3].index.tolist()
    for i in heavy:
        for j in high_v:
            prob += x[i,j] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    assignment = {i:j for i in range(n) for j in range(n) if pulp.value(x[i,j])==1}
    return assignment, pulp.LpStatus[prob.status]

# ============================================================
# CORE OPTIMIZATION
# ============================================================

def optimize(df, params=None):
    if params is None:
        params = DEFAULT_PARAMS

    df = prepare_data(df)
    df_pos = df[['H','V','E']].copy().reset_index(drop=True)
    n = len(df)

    init_assign = {i:i for i in range(n)}
    init_utils, init_costs, init_sim, init_wH, init_wV = simulate_picks(init_assign, df, df_pos, params)

    U = generate_utility_matrix(df, df_pos, params)
    opt_assign, status = solve_ilp(U, df, df_pos)
    opt_utils, opt_costs, opt_sim, opt_wH, opt_wV = simulate_picks(opt_assign, df, df_pos, params)

    return {
        'df': df,
        'df_positions': df_pos,
        'n_items': n,
        'params': params,
        'status': status,
        'init_assign': init_assign,
        'init_utils': init_utils,
        'init_costs': init_costs,
        'init_sim': init_sim,
        'init_wH': init_wH,
        'init_wV': init_wV,
        'opt_assign': opt_assign,
        'opt_utils': opt_utils,
        'opt_costs': opt_costs,
        'opt_sim': opt_sim,
        'opt_wH': opt_wH,
        'opt_wV': opt_wV,
        'improvement': (opt_utils.sum()-init_utils.sum())/init_utils.sum()*100,
        'cost_reduction': (init_sim-opt_sim)/init_sim*100,
        'h_reduction': (init_wH-opt_wH)/init_wH*100,
        'v_reduction': (init_wV-opt_wV)/init_wV*100,
        'moved': sum(1 for i in range(n) if opt_assign[i]!=i)
    }

# ============================================================
# META-OPTIMIZATION OF PARAMETERS
# ============================================================

PARAM_SPACE = {
    'COST_A': (0.5,5.0),
    'COST_B': (1.0,10.0),
    'COST_C': (0.5,5.0),
    'TAU': (2.0,20.0),
    'DEMAND_MULTIPLIER': (5,30)
}

def optimize_parameters(df, n_trials=20, base_params=None, progress_callback=None):
    if base_params is None:
        base_params = DEFAULT_PARAMS.copy()

    best_score = -np.inf
    best_params = None

    for i in range(1, n_trials + 1):
        params = base_params.copy()
        for k, (lo, hi) in PARAM_SPACE.items():
            if k == 'DEMAND_MULTIPLIER':
                params[k] = int(np.random.uniform(lo, hi))
            else:
                params[k] = np.random.uniform(lo, hi)

        df_p = prepare_data(df)
        df_pos = df_p[['H','V','E']].reset_index(drop=True)
        U = generate_utility_matrix(df_p, df_pos, params)
        score = U.sum()

        if score > best_score:
            best_score = score
            best_params = params

        if progress_callback is not None:
            progress_callback(i, n_trials, best_score)

    return best_params, best_score


# ============================================================
# EXPORT
# ============================================================

def create_output_dataframe(results):
    df = results['df'].copy()
    pos = results['df_positions']
    opt = results['opt_assign']
    n = results['n_items']
    df['NOVI_H'] = [int(pos.iloc[opt[i]]['H']) for i in range(n)]
    df['NOVI_V'] = [int(pos.iloc[opt[i]]['V']) for i in range(n)]
    df['NOVI_E'] = [int(pos.iloc[opt[i]]['E']) for i in range(n)]
    df['utility'] = results['opt_utils']
    df['position_cost'] = results['opt_costs']
    if 'izlaz_norm' in df.columns:
        df = df.drop(columns=['izlaz_norm'])
    return df
