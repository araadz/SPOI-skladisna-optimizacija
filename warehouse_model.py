"""
================================================================================
WAREHOUSE OPTIMIZATION MODEL
================================================================================
Modul sa svim funkcijama za optimizaciju skladišta.
Koristi ga Streamlit aplikacija.

Formule:
- Cost: C = α·H + β·max(0, V-2) + γ·(4-E)
- Quality: Q = exp(-C/τ)
- Utility: U = Q · (1 + λ·d/d_max)
================================================================================
"""

import numpy as np
import pandas as pd

# ============================================================
# DEFAULT PARAMETRI
# ============================================================
DEFAULT_PARAMS = {
    'COST_A': 2.0,       # α - horizontalna udaljenost
    'COST_B': 5.0,       # β - vertikalna penalizacija
    'COST_C': 2.0,       # γ - dubina
    'TAU': 8.0,          # τ - temperatura
    'DEMAND_MULTIPLIER': 15,  # λ - multiplikator potražnje
    'N_PICKS': 500       # broj pickova za simulaciju
}


# ============================================================
# FUNKCIJE TROŠKA I KVALITETE
# ============================================================
def calculate_cost(H, V, E, params=None):
    """
    Funkcija troška pozicije
    C = α·H + β·max(0, V-2) + γ·(4-E)
    """
    if params is None:
        params = DEFAULT_PARAMS
    return params['COST_A'] * H + params['COST_B'] * max(0, V - 2) + params['COST_C'] * (4 - E)


def calculate_quality(H, V, E, params=None):
    """
    Kvaliteta pozicije - eksponencijalno opadanje
    Q = exp(-C/τ)
    """
    if params is None:
        params = DEFAULT_PARAMS
    cost = calculate_cost(H, V, E, params)
    return np.exp(-cost / params['TAU'])


def calculate_utility(izlaz_norm, H, V, E, params=None):
    """
    Utility funkcija
    U = Q · (1 + λ·d/d_max)
    """
    if params is None:
        params = DEFAULT_PARAMS
    quality = calculate_quality(H, V, E, params)
    demand_weight = 1.0 + izlaz_norm * params['DEMAND_MULTIPLIER']
    return quality * demand_weight


# ============================================================
# PRIPREMA PODATAKA
# ============================================================
def prepare_data(df):
    """
    Priprema DataFrame za optimizaciju
    """
    # Čišćenje
    df = df.dropna(subset=['H', 'V', 'E', 'izlaz']).reset_index(drop=True)
    
    # Dodaj TEZINA_KAT ako ne postoji
    if 'TEZINA_KAT' not in df.columns:
        df['TEZINA_KAT'] = 4
    df['TEZINA_KAT'] = df['TEZINA_KAT'].fillna(4)
    
    # Normalizacija potražnje
    max_izlaz = df['izlaz'].max() if df['izlaz'].max() > 0 else 1
    df['izlaz_norm'] = df['izlaz'] / max_izlaz
    
    return df


# ============================================================
# UTILITY MATRICA
# ============================================================
def generate_utility_matrix(df, df_positions, params=None):
    """
    Generira n×n utility matricu
    """
    if params is None:
        params = DEFAULT_PARAMS
        
    n = len(df)
    U = np.zeros((n, n))
    
    for i in range(n):
        izlaz_norm = df.iloc[i]['izlaz_norm']
        for j in range(n):
            H = df_positions.iloc[j]['H']
            V = df_positions.iloc[j]['V']
            E = df_positions.iloc[j]['E']
            U[i, j] = calculate_utility(izlaz_norm, H, V, E, params)
    
    return U


# ============================================================
# SIMULACIJA
# ============================================================
def simulate_picks(assignment, df, df_positions, params=None):
    """
    Simulira pickove i vraća metrike
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    n = len(df)
    
    # Izračunaj utility i cost za svaki artikal
    utils = np.array([
        calculate_utility(df.iloc[i]['izlaz_norm'],
                         df_positions.iloc[assignment[i]]['H'],
                         df_positions.iloc[assignment[i]]['V'],
                         df_positions.iloc[assignment[i]]['E'],
                         params)
        for i in range(n)
    ])
    
    costs = np.array([
        calculate_cost(df_positions.iloc[assignment[i]]['H'],
                      df_positions.iloc[assignment[i]]['V'],
                      df_positions.iloc[assignment[i]]['E'],
                      params)
        for i in range(n)
    ])
    
    # Simulacija pickova
    izlaz = df['izlaz'].values
    total_izlaz = izlaz.sum()
    probs = izlaz / total_izlaz if total_izlaz > 0 else np.ones(n) / n
    
    np.random.seed(42)
    picked = np.random.choice(n, size=params['N_PICKS'], p=probs)
    sim_cost = costs[picked].sum()
    
    # Weighted metrics
    wH = sum(df_positions.iloc[assignment[i]]['H'] * izlaz[i] for i in range(n)) / total_izlaz
    wV = sum(df_positions.iloc[assignment[i]]['V'] * izlaz[i] for i in range(n)) / total_izlaz
    
    return utils, costs, sim_cost, wH, wV


# ============================================================
# ILP SOLVER
# ============================================================
def solve_ilp(U, df, df_positions):
    """
    Rješava ILP problem i vraća optimalni assignment
    
    max Σ U[i,j] · x[i,j]
    s.t. Σj x[i,j] = 1  ∀i
         Σi x[i,j] ≤ 1  ∀j
         x[i,j] = 0     za teške artikle na V > 3
    """
    import pulp
    
    n = len(df)
    
    # Kreiraj problem
    prob = pulp.LpProblem("Warehouse", pulp.LpMaximize)
    
    # Binarne varijable
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n)), cat='Binary')
    
    # Funkcija cilja
    prob += pulp.lpSum(U[i, j] * x[i, j] for i in range(n) for j in range(n))
    
    # Ograničenje 1: svaki artikal na jednu poziciju
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n)) == 1
    
    # Ograničenje 2: svaka pozicija max jedan artikal
    for j in range(n):
        prob += pulp.lpSum(x[i, j] for i in range(n)) <= 1
    
    # Ograničenje 3: teški artikli ne na V > 3
    heavy = df[(df['TEZINA_KAT'] >= 4) & (df['izlaz'] > 0)].index.tolist()
    high_v = df_positions[df_positions['V'] > 3].index.tolist()
    for i in heavy:
        for j in high_v:
            prob += x[i, j] == 0
    
    # Riješi
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=120))
    
    # Ekstrahiraj assignment
    assignment = {}
    for i in range(n):
        for j in range(n):
            if pulp.value(x[i, j]) == 1:
                assignment[i] = j
                break
    
    status = pulp.LpStatus[prob.status]
    
    return assignment, status


# ============================================================
# GLAVNA FUNKCIJA OPTIMIZACIJE
# ============================================================
def optimize(df, params=None):
    """
    Glavna funkcija koja pokreće cijelu optimizaciju
    
    Args:
        df: DataFrame sa kolonama H, V, E, izlaz
        params: dict sa parametrima (opcionalno)
    
    Returns:
        dict sa svim rezultatima
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    # Pripremi podatke
    df = prepare_data(df)
    df_positions = df[['H', 'V', 'E']].copy().reset_index(drop=True)
    n = len(df)
    
    # Početno stanje
    init_assign = {i: i for i in range(n)}
    init_utils, init_costs, init_sim, init_wH, init_wV = simulate_picks(init_assign, df, df_positions, params)
    
    # Generiraj utility matricu
    U = generate_utility_matrix(df, df_positions, params)
    
    # Riješi ILP
    opt_assign, status = solve_ilp(U, df, df_positions)
    
    # Metrike za optimalno rješenje
    opt_utils, opt_costs, opt_sim, opt_wH, opt_wV = simulate_picks(opt_assign, df, df_positions, params)
    
    # Izračunaj poboljšanja
    improvement = (opt_utils.sum() - init_utils.sum()) / init_utils.sum() * 100
    cost_reduction = (init_sim - opt_sim) / init_sim * 100
    h_reduction = (init_wH - opt_wH) / init_wH * 100
    v_reduction = (init_wV - opt_wV) / init_wV * 100
    moved = sum(1 for i in range(n) if opt_assign[i] != i)
    
    return {
        'df': df,
        'df_positions': df_positions,
        'n_items': n,
        'params': params,
        'status': status,
        # Početno
        'init_assign': init_assign,
        'init_utils': init_utils,
        'init_costs': init_costs,
        'init_sim': init_sim,
        'init_wH': init_wH,
        'init_wV': init_wV,
        # Optimizirano
        'opt_assign': opt_assign,
        'opt_utils': opt_utils,
        'opt_costs': opt_costs,
        'opt_sim': opt_sim,
        'opt_wH': opt_wH,
        'opt_wV': opt_wV,
        # Poboljšanja
        'improvement': improvement,
        'cost_reduction': cost_reduction,
        'h_reduction': h_reduction,
        'v_reduction': v_reduction,
        'moved': moved
    }


# ============================================================
# EXPORT REZULTATA
# ============================================================
def create_output_dataframe(results):
    """
    Kreira output DataFrame sa novim pozicijama
    """
    df = results['df'].copy()
    df_positions = results['df_positions']
    opt_assign = results['opt_assign']
    n = results['n_items']
    
    df['NOVI_H'] = [int(df_positions.iloc[opt_assign[i]]['H']) for i in range(n)]
    df['NOVI_V'] = [int(df_positions.iloc[opt_assign[i]]['V']) for i in range(n)]
    df['NOVI_E'] = [int(df_positions.iloc[opt_assign[i]]['E']) for i in range(n)]
    df['utility'] = results['opt_utils']
    df['position_cost'] = results['opt_costs']
    
    # Ukloni pomoćne kolone
    if 'izlaz_norm' in df.columns:
        df = df.drop(columns=['izlaz_norm'])
    
    return df
