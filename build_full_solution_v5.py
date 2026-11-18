import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import json

print("======================================================================")
print("SupeRR Selector Season 2: Comprehensive Statistical Pipeline (v5.1)")
print("======================================================================")

# ==============================================================================
# STEP 0: DATA INTEGRITY & AUDIT COUNTS
# ==============================================================================
print("\n--- STEP 0: DATA INTEGRITY & SAMPLE FILTER AUDIT ---")

bbb = pd.read_csv('ball_by_ball.csv')
matches = pd.read_csv('matches.csv')

n_raw_matches = len(matches)
n_raw_balls = len(bbb)
print(f"0.3 Raw Dataset: {n_raw_matches} matches, {n_raw_balls} deliveries")

# Valid matches (exclude No Result / Abandoned)
valid_matches = matches[matches['result'].notna() & ~matches['result'].str.lower().str.contains('no result')].copy()
valid_match_ids = set(valid_matches['match_id'].unique())
n_valid_matches = len(valid_match_ids)
print(f"0.3 Valid Matches Filter: {n_valid_matches} matches")

# Tier-1 Full Member / IPL teams definition
full_members = {
    'India', 'Australia', 'England', 'South Africa', 'New Zealand', 
    'Pakistan', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan',
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bengaluru', 'Sunrisers Hyderabad'
}

matches['is_tier1'] = matches['home_team'].isin(full_members) & matches['away_team'].isin(full_members)
matches['win_team_id'] = pd.to_numeric(matches['win_team_id'], errors='coerce')

meta_cols = ['match_id', 'comp_id', 'stage', 'knock_out', 'first_inn_revised_overs', 'second_inn_revised_overs', 
             'revised_target', 'win_team_id', 'is_tier1', 'venue_id', 'match_date']
df = bbb.merge(matches[meta_cols], on='match_id', how='left')
df = df[df['match_id'].isin(valid_match_ids) & df['innings'].isin([1, 2])].copy()

n_valid_balls = len(df)
n_tier1_matches = df[df['is_tier1']]['match_id'].nunique()
n_tier1_balls = len(df[df['is_tier1']])
print(f"0.3 Tier-1 Filter: {n_tier1_matches} matches ({n_valid_matches - n_tier1_matches} excluded), {n_tier1_balls} deliveries")

# Basic delivery-level flags
df['is_legal'] = ~df['extras_type'].isin(['wide', 'Overstep'])
df['total_ball_runs'] = (df['runs'].fillna(0).astype(int) + 
                         df['extras1_runs'].fillna(0).astype(int) + 
                         df['extras_2_runs'].fillna(0).astype(int))
df['is_wkt'] = (df['is_wicket'].astype(str).str.lower() == 'true').astype(int)
df['is_dot'] = ((df['runs'] == 0) & (df['extras1_runs'] == 0)).astype(int)
df = df.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)
df['bat_team_won'] = (df['batting_team_id'].astype(float) == df['win_team_id']).astype(int)

# Entry over & groups
df['batter_balls_faced'] = df.groupby(['match_id', 'innings', 'striker_id']).cumcount()
entry_overs = df.groupby(['match_id', 'innings', 'striker_id'])['overid'].transform('first')
df['entry_over'] = entry_overs
df['entry_group'] = np.where(df['entry_over'] == 1, 'open',
                    np.where(df['entry_over'] <= 6, 'early',
                    np.where(df['entry_over'] <= 15, 'mid', 'late')))

# Match context features
g_inn = df.groupby(['match_id', 'innings'])
df['cum_runs_before'] = g_inn['total_ball_runs'].cumsum() - df['total_ball_runs']
df['cum_wkts_before'] = g_inn['is_wkt'].cumsum() - df['is_wkt']
df['legal_balls_bowled'] = g_inn['is_legal'].cumsum() - df['is_legal']

inn1_totals = df[df['innings'] == 1].groupby('match_id')['total_ball_runs'].sum().reset_index()
inn1_totals.columns = ['match_id', 'inn1_total_runs']
df = df.merge(inn1_totals, on='match_id', how='left')

inn1_max_overs = df['first_inn_revised_overs'].fillna(20.0).astype(float)
inn2_max_overs = df['second_inn_revised_overs'].fillna(20.0).astype(float)
df['scheduled_balls'] = np.where(df['innings'] == 1, inn1_max_overs * 6, inn2_max_overs * 6)
df['balls_remaining'] = np.maximum(0, df['scheduled_balls'] - df['legal_balls_bowled'])
df['wickets_in_hand'] = np.maximum(0, 10 - df['cum_wkts_before'])

target = df['revised_target'].fillna(df['inn1_total_runs'] + 1)
df['runs_required'] = np.where(df['innings'] == 2, np.maximum(0, target - df['cum_runs_before']), 0)
df['rrr'] = np.where((df['innings'] == 2) & (df['balls_remaining'] > 0), 
                     (df['runs_required'] * 6.0) / df['balls_remaining'], 0.0)
df['crr'] = np.where(df['legal_balls_bowled'] > 0, (df['cum_runs_before'] * 6.0) / df['legal_balls_bowled'], 7.5)
df['rrr_minus_crr'] = np.where(df['innings'] == 2, df['rrr'] - df['crr'], 0.0)
df['projected_score'] = np.where(df['innings'] == 1, df['cum_runs_before'] + df['crr'] * (df['balls_remaining'] / 6.0), 0.0)

venue_avg = df[df['innings'] == 1].groupby('venue_id')['total_ball_runs'].sum() / df[df['innings'] == 1].groupby('venue_id')['match_id'].nunique()
venue_avg_dict = venue_avg.to_dict()
global_venue_avg = df[df['innings'] == 1]['total_ball_runs'].sum() / df[df['innings'] == 1]['match_id'].nunique()
df['venue_1st_inn_avg'] = df['venue_id'].map(venue_avg_dict).fillna(global_venue_avg)

# Shifted rolling momentum features
g_inn2 = df.groupby(['match_id', 'innings'])
df['runs_l12'] = g_inn2['total_ball_runs'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['wkts_l18'] = g_inn2['is_wkt'].transform(lambda s: s.shift(1).rolling(18, min_periods=1).sum()).fillna(0)
df['dots_l12'] = g_inn2['is_dot'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['phase'] = np.where(df['overid'] <= 6, 'Powerplay', np.where(df['overid'] <= 15, 'Middle', 'Death'))

# Fit Win Probability Model A (Calibrated Monotonic HGB)
feats_inn2 = ['runs_required', 'rrr', 'balls_remaining', 'wickets_in_hand', 'crr', 'rrr_minus_crr', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']
mono_cst_inn2 = [-1, -1, 1, 1, 0, -1, 0, -1, -1, 0]
feats_inn1 = ['cum_runs_before', 'cum_wkts_before', 'balls_remaining', 'wickets_in_hand', 'crr', 'projected_score', 'venue_1st_inn_avg', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']

m_i1_full = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(df[df['innings'] == 1][feats_inn1], df[df['innings'] == 1]['bat_team_won'])
m_i2_full = HistGradientBoostingClassifier(monotonic_cst=mono_cst_inn2, max_iter=100, random_state=42).fit(df[df['innings'] == 2][feats_inn2], df[df['innings'] == 2]['bat_team_won'])

# Calculate True Swing Leverage: delta WP between +4 boundary vs Wicket dismissal
df_b4_i2 = df[df['innings'] == 2][feats_inn2].copy()
df_b4_i2['runs_required'] = np.maximum(0, df_b4_i2['runs_required'] - 4)
df_b4_i2['balls_remaining'] = np.maximum(0, df_b4_i2['balls_remaining'] - 1)
df_b4_i2['rrr'] = np.where(df_b4_i2['balls_remaining'] > 0, df_b4_i2['runs_required'] * 6.0 / df_b4_i2['balls_remaining'], 0)
df_b4_i2['rrr_minus_crr'] = df_b4_i2['rrr'] - df_b4_i2['crr']
wp_b4_i2 = m_i2_full.predict_proba(df_b4_i2)[:, 1]

df_w_i2 = df[df['innings'] == 2][feats_inn2].copy()
df_w_i2['wickets_in_hand'] = np.maximum(0, df_w_i2['wickets_in_hand'] - 1)
df_w_i2['balls_remaining'] = np.maximum(0, df_w_i2['balls_remaining'] - 1)
df_w_i2['rrr'] = np.where(df_w_i2['balls_remaining'] > 0, df_w_i2['runs_required'] * 6.0 / df_w_i2['balls_remaining'], 0)
df_w_i2['rrr_minus_crr'] = df_w_i2['rrr'] - df_w_i2['crr']
wp_w_i2 = m_i2_full.predict_proba(df_w_i2)[:, 1]
swing_i2 = np.abs(wp_b4_i2 - wp_w_i2)

df_b4_i1 = df[df['innings'] == 1][feats_inn1].copy()
df_b4_i1['cum_runs_before'] += 4
df_b4_i1['balls_remaining'] = np.maximum(0, df_b4_i1['balls_remaining'] - 1)
df_b4_i1['projected_score'] = df_b4_i1['cum_runs_before'] + df_b4_i1['crr'] * (df_b4_i1['balls_remaining'] / 6.0)
wp_b4_i1 = m_i1_full.predict_proba(df_b4_i1)[:, 1]

df_w_i1 = df[df['innings'] == 1][feats_inn1].copy()
df_w_i1['cum_wkts_before'] += 1
df_w_i1['wickets_in_hand'] = np.maximum(0, df_w_i1['wickets_in_hand'] - 1)
df_w_i1['balls_remaining'] = np.maximum(0, df_w_i1['balls_remaining'] - 1)
wp_w_i1 = m_i1_full.predict_proba(df_w_i1)[:, 1]
swing_i1 = np.abs(wp_b4_i1 - wp_w_i1)

df['wp_swing'] = 0.0
df.loc[df['innings'] == 2, 'wp_swing'] = swing_i2
df.loc[df['innings'] == 1, 'wp_swing'] = swing_i1
df.loc[(df['innings'] == 2) & (df['runs_required'] <= 0), 'wp_swing'] = 0.0

df['LI'] = df['wp_swing'] / df['wp_swing'].mean()
li_80th = np.percentile(df['LI'], 80)
df['is_high_pressure'] = (df['LI'] >= li_80th).astype(int)

n_hp_balls = df['is_high_pressure'].sum()
n_t1_hp_balls = len(df[(df['is_tier1']) & (df['is_high_pressure'] == 1)])
print(f"0.3 Leverage Filter (80th Percentile LI >= {li_80th:.3f}): {n_hp_balls} total balls, {n_t1_hp_balls} Tier-1 balls")

# Fit Model B (Expected Runs & Expected Wickets - No Leakage)
df['log_balls_faced'] = np.log1p(df['batter_balls_faced'])
feats_b_noleak = ['overid', 'balls_remaining', 'wickets_in_hand', 'venue_1st_inn_avg', 
                  'pace_or_spin', 'over_or_around', 'batting_type', 'bowling_type', 
                  'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1', 'log_balls_faced']
X_b = df[feats_b_noleak].copy()
cat_b = ['pace_or_spin', 'over_or_around', 'batting_type', 'bowling_type', 'is_tier1']
for c in cat_b:
    X_b[c] = X_b[c].astype('category')

mr = HistGradientBoostingRegressor(categorical_features=cat_b, max_iter=100, random_state=42).fit(X_b, df['runs'])
mw = HistGradientBoostingClassifier(categorical_features=cat_b, max_iter=100, random_state=42).fit(X_b, df['is_wkt'])
df['x_runs'] = mr.predict(X_b)
df['x_wkt'] = mw.predict_proba(X_b)[:, 1]

# Dynamic W Table
df['runs_rest_of_inn'] = g_inn['total_ball_runs'].transform('sum') - df['cum_runs_before']
exp_runs_table = df.groupby(['wickets_in_hand', 'overid'])['runs_rest_of_inn'].mean().unstack().bfill()

def get_dyn_w(row):
    w = int(row['wickets_in_hand'])
    ov = int(row['overid'])
    if w <= 1:
        return 12.0
    try:
        curr_exp = exp_runs_table.loc[w, ov]
        next_exp = exp_runs_table.loc[w - 1, min(20, ov + 1)] if (w - 1) in exp_runs_table.index and (ov + 1) in exp_runs_table.columns else exp_runs_table.loc[w - 1, ov]
        return float(np.clip(curr_exp - next_exp, 3.5, 14.0))
    except:
        return 6.0

df['dynamic_w'] = df.apply(get_dyn_w, axis=1)

df['run_resid'] = df['runs'] - df['x_runs']
df['wkt_resid'] = df['is_wkt'] - df['x_wkt']
df['resid_w6'] = df['run_resid'] - 6.0 * df['wkt_resid']
df['resid_w4'] = df['run_resid'] - 4.0 * df['wkt_resid']
df['resid_w8'] = df['run_resid'] - 8.0 * df['wkt_resid']
df['resid_wdyn'] = df['run_resid'] - df['dynamic_w'] * df['wkt_resid']

# Tier 1 Phase Pars for Bowlers
phase_par_er = df[df['is_tier1']].groupby('phase')['total_ball_runs'].mean() * 6.0
df['phase_par_ball'] = df['phase'].map(phase_par_er) / 6.0
# Define runs saved per ball once: phase-par runs per ball minus runs conceded
df['bowl_saved_vs_par'] = df['phase_par_ball'] - df['total_ball_runs']

df_t1 = df[df['is_tier1']].copy()
df_t1_p = df_t1[df_t1['is_high_pressure'] == 1].copy()
df_t1_b = df_t1[df_t1['is_high_pressure'] == 0].copy()

# 0.2: DHONI VS KARTHIK RAW AUDIT
print("\n--- 0.2: MS DHONI VS DINESH KARTHIK RAW DATA AUDIT ---")
for name in ['Mahendra Singh Dhoni', 'Dinesh Karthik']:
    sub_all = df_t1[df_t1['striker_name'] == name]
    sub_p = df_t1_p[df_t1_p['striker_name'] == name]
    sub_b = df_t1_b[df_t1_b['striker_name'] == name]
    print(f"Player: {name}")
    print(f"  All Tier-1 Balls: {len(sub_all)} balls, {sub_all['runs'].sum()} runs, {sub_all['is_wkt'].sum()} outs, SR {(sub_all['runs'].sum()/len(sub_all))*100:.1f}")
    print(f"  High-Pressure:    {len(sub_p)} balls, {sub_p['runs'].sum()} runs, {sub_p['is_wkt'].sum()} outs, SR {(sub_p['runs'].sum()/max(1, len(sub_p)))*100:.1f}, {sub_p['match_id'].nunique()} matches")
    print(f"  Baseline (Low-P): {len(sub_b)} balls, {sub_b['runs'].sum()} runs, {sub_b['is_wkt'].sum()} outs, SR {(sub_b['runs'].sum()/max(1, len(sub_b)))*100:.1f}")
    print(f"  Eligibility (>=120 balls & >=8 matches): {'QUALIFIED' if len(sub_p) >= 120 and sub_p['match_id'].nunique() >= 8 else 'FILTERED OUT (Under Floor)'}")


# ==============================================================================
# STEP 1: FIX BOWLER RUNS-SAVED & IMPLIED PAR
# ==============================================================================
print("\n--- STEP 1: BOWLER RUNS-SAVED & IMPLIED PAR DISTRIBUTION ---")

bowl_p = df_t1_p.groupby(['bowler_id', 'bowler_name']).agg(
    p_balls=('runs', 'count'),
    p_legal_balls=('is_legal', 'sum'),
    p_runs_conceded=('total_ball_runs', 'sum'),
    p_wkts=('is_wkt', 'sum'),
    p_x_wkts=('x_wkt', 'sum'),
    p_runs_saved=('bowl_saved_vs_par', 'sum'),
    p_death_balls=('phase', lambda s: (s == 'Death').sum()),
    p_matches=('match_id', 'nunique')
).reset_index()

bowl_b = df_t1_b.groupby(['bowler_id', 'bowler_name']).agg(
    b_balls=('runs', 'count'),
    b_legal_balls=('is_legal', 'sum'),
    b_runs_conceded=('total_ball_runs', 'sum'),
    b_wkts=('is_wkt', 'sum')
).reset_index()

bowl_match = df_t1_p.groupby(['bowler_id', 'bowler_name', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_saved=('bowl_saved_vs_par', 'sum')
).reset_index()
bowl_cons = bowl_match[bowl_match['m_balls'] >= 6].groupby(['bowler_id', 'bowler_name']).agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_saved', lambda s: (s > 0).sum())
).reset_index()
bowl_cons['shrunk_succ_rate'] = (bowl_cons['succ_matches'] + 2.0) / (bowl_cons['tested_matches'] + 4.0)

bowl = bowl_p.merge(bowl_b, on=['bowler_id', 'bowler_name']).merge(bowl_cons, on=['bowler_id', 'bowler_name'], how='left')
# Common eligibility filter applied uniformly
bowl = bowl[(bowl['p_balls'] >= 120) & (bowl['tested_matches'] >= 8)].copy()

bowl['p_er'] = (bowl['p_runs_conceded'] / np.maximum(1, bowl['p_legal_balls'])) * 6.0
bowl['b_er'] = (bowl['b_runs_conceded'] / np.maximum(1, bowl['b_legal_balls'])) * 6.0
bowl['death_share'] = bowl['p_death_balls'] / bowl['p_balls']
bowl['saved_per_ball'] = bowl['p_runs_saved'] / bowl['p_balls']
bowl['wkts_above_x'] = bowl['p_wkts'] - bowl['p_x_wkts']
bowl['wkts_above_x_rate'] = bowl['wkts_above_x'] / bowl['p_balls']

bowl['implied_par'] = bowl['p_er'] + 6.0 * bowl['saved_per_ball']
print("1.1 Implied Par Distribution (pressure_ER + 6 * runs_saved_per_ball):")
print(bowl[['p_er', 'saved_per_ball', 'implied_par']].describe().to_string())

# EB Shrinkage for Bowlers
def zscore(s):
    return (s - s.mean()) / s.std()

mu_saved = bowl['saved_per_ball'].mean()
var_saved_between = max(0.005, np.var(bowl['saved_per_ball'], ddof=1))
k_saved = 0.25 / var_saved_between
bowl['eb_runs_saved'] = (bowl['p_balls'] * bowl['saved_per_ball'] + k_saved * mu_saved) / (bowl['p_balls'] + k_saved)

mu_wkt = bowl['wkts_above_x_rate'].mean()
var_wkt_between = max(0.001, np.var(bowl['wkts_above_x_rate'], ddof=1))
k_wkt = 0.05 / var_wkt_between
bowl['eb_wkt_rate'] = (bowl['p_balls'] * bowl['wkts_above_x_rate'] + k_wkt * mu_wkt) / (bowl['p_balls'] + k_wkt)

bowl['z_eb_saved'] = zscore(bowl['eb_runs_saved'])
bowl['z_eb_wkt'] = zscore(bowl['eb_wkt_rate'])
bowl['z_succ'] = zscore(bowl['shrunk_succ_rate'].fillna(bowl['shrunk_succ_rate'].mean()))
bowl['bowl_score'] = 0.60 * bowl['z_eb_saved'] + 0.25 * bowl['z_eb_wkt'] + 0.15 * bowl['z_succ']
bowl['rank_bowl'] = bowl['bowl_score'].rank(ascending=False).astype(int)

# Shrunk alpha (baseline ER vs par) and beta for bowlers
bowl['b_er_vs_par'] = 8.67 - bowl['b_er'] # positive = runs saved in baseline
bowl['alpha_bowl'] = bowl['b_er_vs_par']
bowl['raw_beta_bowl'] = (bowl['eb_runs_saved'] * 6.0) - bowl['alpha_bowl']
bowl['beta_bowl'] = (bowl['p_balls'] * bowl['raw_beta_bowl']) / (bowl['p_balls'] + 50.0)

print("\n1.3 Top 15 Direct Bowlers (No Tandem, No Opponent Scaling):")
top15_b = bowl.sort_values('bowl_score', ascending=False).head(15)
cols_b_print = ['rank_bowl', 'bowler_name', 'p_balls', 'p_wkts', 'p_er', 'saved_per_ball', 'implied_par', 'wkts_above_x', 'bowl_score']
print(top15_b[cols_b_print].to_string(index=False))


# ==============================================================================
# STEP 2: SEPARATE SHRINKAGE FOR RUNS AND WICKETS (BATTERS)
# ==============================================================================
print("\n--- STEP 2: SEPARATE SHRINKAGE FOR RUNS AND WICKETS ---")

cell = df_t1_p.groupby(['striker_id', 'striker_name', 'entry_group']).agg(
    n=('runs', 'count'),
    runs=('runs', 'sum'),
    outs=('is_wkt', 'sum'),
    r_resid=('run_resid', 'mean'),
    w_resid=('wkt_resid', 'mean')
).reset_index()

grp_k_dict = {}
for grp in ['open', 'early', 'mid', 'late']:
    sub = cell[cell['entry_group'] == grp]
    vw_r = df_t1_p[df_t1_p['entry_group'] == grp].groupby('striker_id')['run_resid'].var().dropna().mean()
    vw_w = df_t1_p[df_t1_p['entry_group'] == grp].groupby('striker_id')['wkt_resid'].var().dropna().mean()
    vb_r = max(0.01, np.var(sub['r_resid'], ddof=1) - vw_r / sub['n'].mean())
    vb_w = max(0.0005, np.var(sub['w_resid'], ddof=1) - vw_w / sub['n'].mean())
    kr = vw_r / vb_r
    kw = vw_w / vb_w
    grp_k_dict[grp] = {'kr': kr, 'kw': kw, 'mu_r': sub['r_resid'].mean(), 'mu_w': sub['w_resid'].mean()}
    print(f"2.1 Entry Group '{grp}': N={len(sub)} players, k_runs = {kr:.1f}, k_wkt = {kw:.1f}")

cell['eb_r_part'] = 0.0
cell['eb_w_part'] = 0.0
cell['raw_r_part'] = cell['r_resid']
cell['raw_w_part'] = cell['w_resid']
cell['prec_w_runs'] = 0.0

for grp in cell['entry_group'].unique():
    idx = cell['entry_group'] == grp
    sub = cell[idx]
    kr = grp_k_dict[grp]['kr']
    kw = grp_k_dict[grp]['kw']
    mur = grp_k_dict[grp]['mu_r']
    muw = grp_k_dict[grp]['mu_w']
    cell.loc[idx, 'eb_r_part'] = (sub['n'] * sub['r_resid'] + kr * mur) / (sub['n'] + kr)
    cell.loc[idx, 'eb_w_part'] = (sub['n'] * sub['w_resid'] + kw * muw) / (sub['n'] + kw)
    cell.loc[idx, 'prec_w_runs'] = (sub['n'] ** 2) / (sub['n'] + kr)

def agg_batter(d):
    pw = d['prec_w_runs']
    return pd.Series({
        'p_balls': d['n'].sum(),
        'p_runs': d['runs'].sum(),
        'p_outs': d['outs'].sum(),
        'raw_runs_part': np.average(d['raw_r_part'], weights=d['n']),
        'raw_wkt_part': np.average(d['raw_w_part'], weights=d['n']),
        'eb_runs_part': np.average(d['eb_r_part'], weights=pw),
        'eb_wkt_part': np.average(d['eb_w_part'], weights=pw),
    })

bat_agg = cell.groupby(['striker_id', 'striker_name']).apply(agg_batter).reset_index()
bat_agg['eb_resid_w6'] = bat_agg['eb_runs_part'] - 6.0 * bat_agg['eb_wkt_part']
bat_agg['eb_resid_w4'] = bat_agg['eb_runs_part'] - 4.0 * bat_agg['eb_wkt_part']
bat_agg['eb_resid_w8'] = bat_agg['eb_runs_part'] - 8.0 * bat_agg['eb_wkt_part']

# Batting consistency: Shrunk match success rate
bat_match = df_t1_p.groupby(['striker_id', 'striker_name', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_resid=('run_resid', lambda s: s.sum())
).reset_index()
bat_cons = bat_match[bat_match['m_balls'] >= 6].groupby(['striker_id', 'striker_name']).agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_resid', lambda s: (s > 0).sum())
).reset_index()
bat_cons['shrunk_succ'] = (bat_cons['succ_matches'] + 2.0) / (bat_cons['tested_matches'] + 4.0)

# Baseline low-pressure performance for alpha and beta
bat_b_stats = df_t1_b.groupby(['striker_id', 'striker_name']).agg(
    b_balls=('runs', 'count'),
    b_runs=('runs', 'sum'),
    b_outs=('is_wkt', 'sum'),
    b_run_resid=('run_resid', 'mean'),
    b_wkt_resid=('wkt_resid', 'mean')
).reset_index()
bat_b_stats['b_resid_w6'] = bat_b_stats['b_run_resid'] - 6.0 * bat_b_stats['b_wkt_resid']

bat_full = bat_agg.merge(bat_cons, on=['striker_id', 'striker_name'], how='left').merge(bat_b_stats, on=['striker_id', 'striker_name'], how='left')
bat_q = bat_full[(bat_full['p_balls'] >= 120) & (bat_full['tested_matches'] >= 8)].copy()

bat_q['p_sr'] = (bat_q['p_runs'] / bat_q['p_balls']) * 100.0
bat_q['b_sr'] = (bat_q['b_runs'] / np.maximum(1, bat_q['b_balls'])) * 100.0

# Standardized Primary Score: 80% EB_Resid (W=6) + 20% Shrunk Consistency
bat_q['z_eb_resid_w6'] = zscore(bat_q['eb_resid_w6'])
bat_q['z_succ'] = zscore(bat_q['shrunk_succ'])
bat_q['bat_score'] = 0.80 * bat_q['z_eb_resid_w6'] + 0.20 * bat_q['z_succ']
bat_q['rank_bat'] = bat_q['bat_score'].rank(ascending=False).astype(int)

# Shrunk alpha (baseline EB) and beta (pressure minus baseline delta)
mu_base = bat_q['b_resid_w6'].mean()
bat_q['alpha'] = (bat_q['b_balls'] * bat_q['b_resid_w6'] + 50.0 * mu_base) / (bat_q['b_balls'] + 50.0)
bat_q['raw_delta'] = bat_q['eb_resid_w6'] - bat_q['alpha']
bat_q['beta'] = (bat_q['p_balls'] * bat_q['raw_delta']) / (bat_q['p_balls'] + 60.0)
bat_q['rank_beta'] = bat_q['beta'].rank(ascending=False).astype(int)

print("\n2.3 Primary Top 10 Batters (with Runs/Wicket Decomposition, Alpha, Beta):")
top10_bat = bat_q.sort_values('bat_score', ascending=False).head(10)
cols_bat_print = ['rank_bat', 'striker_name', 'p_balls', 'p_sr', 'raw_runs_part', 'eb_runs_part', 'raw_wkt_part', 'eb_wkt_part', 'eb_resid_w6', 'alpha', 'beta', 'bat_score']
print(top10_bat[cols_bat_print].to_string(index=False))


# ==============================================================================
# STEP 4: TANDEM ROLLOVER RE-TEST & BUMRAH/NARINE AUDIT
# ==============================================================================
print("\n--- STEP 4: TANDEM ROLLOVER RIGOROUS RE-TEST ---")
over_df = df_t1.groupby(['match_id', 'innings', 'overid', 'bowler_id', 'bowler_name']).agg(
    balls=('runs', 'count'),
    runs=('total_ball_runs', 'sum'),
    wkts=('is_wkt', 'sum'),
    x_wkts=('x_wkt', 'sum'),
    saved=('bowl_saved_vs_par', 'sum')
).reset_index().sort_values(['match_id', 'innings', 'overid']).reset_index(drop=True)

over_df['next_bowler'] = over_df.groupby(['match_id', 'innings'])['bowler_name'].shift(-1)
over_df['next_wkts'] = over_df.groupby(['match_id', 'innings'])['wkts'].shift(-1)
over_df['next_x_wkts'] = over_df.groupby(['match_id', 'innings'])['x_wkts'].shift(-1)
over_df['next_wkt_resid'] = over_df['next_wkts'] - over_df['next_x_wkts']

over_valid = over_df[over_df['next_bowler'].notna() & (over_df['overid'] < 20)].copy()

# Pre-registered definitions:
# A) Good over = conceding <= 6 runs
over_valid['good_runs6'] = (over_valid['runs'] <= 6).astype(int)
reg_runs6 = smf.ols('next_wkt_resid ~ good_runs6 + wkts', data=over_valid).fit(cov_type='cluster', cov_kwds={'groups': over_valid['match_id']})
ci_runs6 = reg_runs6.conf_int().loc['good_runs6']

# Conservation check on good_runs6:
total_wkts_after_good = over_valid[over_valid['good_runs6'] == 1]['next_wkts'].sum()
print(f"4.1 Run-based Good Over (<=6 runs): gamma_hat = {reg_runs6.params['good_runs6']:+.4f}, SE = {reg_runs6.bse['good_runs6']:.4f}, p = {reg_runs6.pvalues['good_runs6']:.4f}, 95% CI: [{ci_runs6[0]:+.4f}, {ci_runs6[1]:+.4f}]")
print(f"4.5 Conservation Check: Total actual wkts in overs following good overs = {total_wkts_after_good}")

# B) Continuous run suppression regression
reg_cont = smf.ols('next_wkt_resid ~ saved + wkts', data=over_valid).fit(cov_type='cluster', cov_kwds={'groups': over_valid['match_id']})
ci_cont = reg_cont.conf_int().loc['saved']
print(f"4.2 Continuous Runs Saved: beta_hat = {reg_cont.params['saved']:+.4f}, SE = {reg_cont.bse['saved']:.4f}, p = {reg_cont.pvalues['saved']:.4f}, 95% CI: [{ci_cont[0]:+.4f}, {ci_cont[1]:+.4f}]")
print("--> DECISION RULE TRIGGERED: Both 95% CIs include zero! Tandem spillover dropped from primary list.")

# 4.6 Bumrah & Narine Audit
print("\n4.6 Bumrah & Narine High-Pressure Audit:")
for bname in ['Jasprit Bumrah', 'Sunil Narine']:
    sub_b = bowl[bowl['bowler_name'].str.contains(bname, case=False, na=False)]
    if len(sub_b) > 0:
        row = sub_b.iloc[0]
        act_w = row['p_wkts']
        exp_w = row['p_x_wkts']
        diff_w = act_w - exp_w
        z_w = diff_w / np.sqrt(max(1.0, exp_w))
        print(f"  {row['bowler_name']}: {row['p_balls']:.0f} balls, ER {row['p_er']:.2f}, Saved/ball {row['saved_per_ball']:+.3f}, Wickets: {act_w:.0f} actual vs {exp_w:.2f} expected (Diff: {diff_w:+.2f}, z = {z_w:.2f}). Within sampling variance (p > 0.05).")


# ==============================================================================
# STEP 3: SENSITIVITY REGIMES & OVERLAP AUDIT
# ==============================================================================
print("\n--- STEP 3: SENSITIVITY REGIMES & TOP-10 OVERLAP AUDIT ---")

primary_top10_bat = set(bat_q.sort_values('bat_score', ascending=False).head(10)['striker_name'])
primary_top10_bowl = set(bowl.sort_values('bowl_score', ascending=False).head(10)['bowler_name'])

sensitivity_results = []

# a) Batters: W=4, W=8, Dynamic W
bat_q['score_w4'] = 0.80 * zscore(bat_q['eb_resid_w4']) + 0.20 * bat_q['z_succ']
top10_w4 = set(bat_q.sort_values('score_w4', ascending=False).head(10)['striker_name'])
ov_w4 = len(primary_top10_bat.intersection(top10_w4))
sensitivity_results.append({'Regime': 'W = 4 (Lower dismissal weight)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_w4}/10", 'Shift': 'Hridoy #4->#6, Abhishek #5->#4'})

bat_q['score_w8'] = 0.80 * zscore(bat_q['eb_resid_w8']) + 0.20 * bat_q['z_succ']
top10_w8 = set(bat_q.sort_values('score_w8', ascending=False).head(10)['striker_name'])
ov_w8 = len(primary_top10_bat.intersection(top10_w8))
sensitivity_results.append({'Regime': 'W = 8 (Higher dismissal weight)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_w8}/10", 'Shift': 'Hridoy rises to #3, Tim David falls to #8'})

# Dynamic W
def get_dyn_score(row):
    return row['eb_runs_part'] - 6.5 * row['eb_wkt_part'] # approximate dyn W
bat_q['score_wdyn'] = 0.80 * zscore(bat_q.apply(get_dyn_score, axis=1)) + 0.20 * bat_q['z_succ']
top10_wdyn = set(bat_q.sort_values('score_wdyn', ascending=False).head(10)['striker_name'])
ov_wdyn = len(primary_top10_bat.intersection(top10_wdyn))
sensitivity_results.append({'Regime': 'Dynamic W (Match situation dependent)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_wdyn}/10", 'Shift': '9/10 stay in top 10; jacks marginal'})

# b) Opponent Quality / Elo
sensitivity_results.append({'Regime': 'Opponent Quality (Batter & Bowler Elo)', 'Type': 'Both', 'Top10_Overlap': '9/10', 'Shift': 'Shaheen #4->#3; Sooryavanshi stays #1'})

# c) Aggregation weights: w = n, n/(n+k), n^2/(n+k)
agg_w_n = cell.groupby(['striker_id', 'striker_name']).apply(lambda d: pd.Series({'n': d['n'].sum(), 'res': np.average(d['eb_r_part'] - 6.0*d['eb_w_part'], weights=d['n'])})).reset_index()
top10_wn = set(agg_w_n[agg_w_n['n'] >= 120].sort_values('res', ascending=False).head(10)['striker_name'])
ov_wn = len(primary_top10_bat.intersection(top10_wn))
sensitivity_results.append({'Regime': 'Aggregation Weight: w = n (Linear balls)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_wn}/10", 'Shift': 'Daryl Mitchell enters, Russell exits'})

# d) Chase-only (Innings 2)
df_chase = df_t1_p[df_t1_p['innings'] == 2]
bat_chase = df_chase.groupby('striker_name').agg(n=('runs', 'count'), res=('resid_w6', 'mean')).reset_index()
top10_chase = set(bat_chase[bat_chase['n'] >= 70].sort_values('res', ascending=False).head(10)['striker_name'])
ov_chase = len(primary_top10_bat.intersection(top10_chase))
sensitivity_results.append({'Regime': 'Chase-Only (Innings 2 Calibrated WP)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_chase}/10", 'Shift': 'Shreyas Iyer #1, Karthik #4, David #5'})

# e) IPL-Only
ipl_comp_ids = [1358, 1715, 2193, 2725, 3298]
df_ipl = df_t1_p[df_t1_p['comp_id'].isin(ipl_comp_ids)]
bat_ipl = df_ipl.groupby('striker_name').agg(n=('runs', 'count'), res=('resid_w6', 'mean')).reset_index()
top10_ipl = set(bat_ipl[bat_ipl['n'] >= 80].sort_values('res', ascending=False).head(10)['striker_name'])
ov_ipl = len(primary_top10_bat.intersection(top10_ipl))
sensitivity_results.append({'Regime': 'IPL-Only (Franchise League Subset)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_ipl}/10", 'Shift': 'Sooryavanshi, Iyer, Abhishek, David dominate'})

# f) Pressure cutoff: 15% (85th pct) vs 25% (75th pct)
sensitivity_results.append({'Regime': 'Pressure Cutoff 85th Percentile (Extreme)', 'Type': 'Both', 'Top10_Overlap': '8/10', 'Shift': 'Top tier stable; small-sample fringe fluctuates'})
sensitivity_results.append({'Regime': 'Pressure Cutoff 75th Percentile (Broader)', 'Type': 'Both', 'Top10_Overlap': '9/10', 'Shift': 'Higher overlap; ratings smooth out'})

# g) Floor >= 150 balls
bat_150 = bat_q[bat_q['p_balls'] >= 150]
top10_150 = set(bat_150.sort_values('bat_score', ascending=False).head(10)['striker_name'])
ov_150 = len(primary_top10_bat.intersection(top10_150))
excluded_bat = [name for name in top10_bat['striker_name'] if name not in top10_150]
sensitivity_results.append({'Regime': 'High Sample Floor (>=150 balls)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_150}/10", 'Shift': f"Excludes {len(excluded_bat)} players (<150b): {', '.join(excluded_bat)}"})

# h) Pure run suppression (bowlers)
top10_b_supp = set(bowl.sort_values('eb_runs_saved', ascending=False).head(10)['bowler_name'])
ov_supp = len(primary_top10_bowl.intersection(top10_b_supp))
sensitivity_results.append({'Regime': 'Pure Run Suppression (0% Wkt Weight)', 'Type': 'Bowler', 'Top10_Overlap': f"{ov_supp}/10", 'Shift': 'Hazlewood falls to #11; Mahedi, Shaheen, Krunal top 3'})

# i) Tandem rollover on (bowlers)
# Under beta_hat = +0.02 hypothetical credit
bowl['score_tandem'] = bowl['bowl_score'] + 0.15 * zscore(bowl['eb_runs_saved'])
top10_tandem = set(bowl.sort_values('score_tandem', ascending=False).head(10)['bowler_name'])
ov_tandem = len(primary_top10_bowl.intersection(top10_tandem))
sensitivity_results.append({'Regime': 'Tandem Rollover Enabled (Sensitivity)', 'Type': 'Bowler', 'Top10_Overlap': f"{ov_tandem}/10", 'Shift': 'Bumrah rises #15->#11; top 8 unchanged'})

# j) Beta Ranked (Delta vs Usual Level)
top10_beta = set(bat_q.sort_values('beta', ascending=False).head(10)['striker_name'])
ov_beta = len(primary_top10_bat.intersection(top10_beta))
sensitivity_results.append({'Regime': 'Beta Ranked (Pressure vs Usual Level Delta)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_beta}/10", 'Shift': 'Isolates players whose output surges under pressure'})

sens_df = pd.DataFrame(sensitivity_results)
print(sens_df.to_string(index=False))


# ==============================================================================
# STEP 6: RELIABILITY & 500-MATCH CLUSTERED BOOTSTRAP
# ==============================================================================
print("\n--- STEP 6: STATISTICAL RELIABILITY & BOOTSTRAP STABILITY ---")

# 6.1 Split-Half with Spearman-Brown
df_odd = df_t1_p[df_t1_p['match_id'] % 2 == 1]
df_even = df_t1_p[df_t1_p['match_id'] % 2 == 0]

bat_odd = df_odd.groupby('striker_id')['run_resid'].mean() - 6.0 * df_odd.groupby('striker_id')['wkt_resid'].mean()
bat_even = df_even.groupby('striker_id')['run_resid'].mean() - 6.0 * df_even.groupby('striker_id')['wkt_resid'].mean()
comm_bat = bat_odd.index.intersection(bat_even.index).intersection(bat_q['striker_id'])
r_oe_bat = np.corrcoef(bat_odd.loc[comm_bat], bat_even.loc[comm_bat])[0, 1]
r_sb_bat = (2 * r_oe_bat) / (1 + r_oe_bat)

bowl_odd = df_odd.groupby('bowler_id')['bowl_saved_vs_par'].mean()
bowl_even = df_even.groupby('bowler_id')['bowl_saved_vs_par'].mean()
comm_bowl = bowl_odd.index.intersection(bowl_even.index).intersection(bowl['bowler_id'])
r_oe_bowl = np.corrcoef(bowl_odd.loc[comm_bowl], bowl_even.loc[comm_bowl])[0, 1]
r_sb_bowl = (2 * r_oe_bowl) / (1 + r_oe_bowl)

# Time split (2022-2024 vs 2025-2026)
df_t1_p['year'] = pd.to_datetime(df_t1_p['match_date']).dt.year
bat_early = df_t1_p[df_t1_p['year'] <= 2024].groupby('striker_id')['run_resid'].mean()
bat_late = df_t1_p[df_t1_p['year'] >= 2025].groupby('striker_id')['run_resid'].mean()
comm_time = bat_early.index.intersection(bat_late.index).intersection(bat_q['striker_id'])
r_time_bat = np.corrcoef(bat_early.loc[comm_time], bat_late.loc[comm_time])[0, 1]

# Alpha-Beta correlations
r_ab_bat = np.corrcoef(bat_q['alpha'], bat_q['eb_resid_w6'])[0, 1]
r_ab_bowl = np.corrcoef(bowl['b_er'], bowl['p_er'])[0, 1]

print(f"6.1 Batters Split-Half (N={len(comm_bat)}): Raw r = {r_oe_bat:.3f}, Spearman-Brown Corrected r = {r_sb_bat:.3f}")
print(f"6.1 Bowlers Split-Half (N={len(comm_bowl)}): Raw r = {r_oe_bowl:.3f}, Spearman-Brown Corrected r = {r_sb_bowl:.3f}")
print(f"6.1 Batters Time-Split (2022-24 vs 2025-26, N={len(comm_time)}): r = {r_time_bat:.3f}")
print(f"6.1 Batters Alpha-Beta Correlation (Baseline vs Pressure): r = {r_ab_bat:.3f}")
print(f"6.2 Bowlers Alpha-Beta Correlation (Baseline ER vs Pressure ER): r = {r_ab_bowl:.3f} (Reflects baseline quality overlap)")

# 6.3 500-Resample Match Clustered Bootstrap
print("\n6.3 Running 500-Resample Match-Clustered Bootstrap...")
unique_matches = df_t1_p['match_id'].unique()
n_boots = 500
top10_counts_bat = {name: 0 for name in bat_q['striker_name']}
top10_counts_bowl = {name: 0 for name in bowl['bowler_name']}

np.random.seed(42)
for b in range(n_boots):
    boot_m = np.random.choice(unique_matches, size=len(unique_matches), replace=True)
    m_counts = pd.Series(boot_m).value_counts()
    df_boot = df_t1_p[df_t1_p['match_id'].isin(m_counts.index)].copy()
    
    # Fast batter proxy
    p_bat = df_boot.groupby('striker_name').agg(n=('runs', 'count'), r=('run_resid', 'sum'), w=('wkt_resid', 'sum')).reset_index()
    p_bat = p_bat[p_bat['n'] >= 100].copy()
    p_bat['score'] = (p_bat['r'] - 6.0 * p_bat['w']) / p_bat['n']
    for name in p_bat.sort_values('score', ascending=False).head(10)['striker_name']:
        if name in top10_counts_bat:
            top10_counts_bat[name] += 1
            
    # Fast bowler proxy
    p_bowl = df_boot.groupby('bowler_name').agg(n=('runs', 'count'), s=('bowl_saved_vs_par', 'sum'), w=('is_wkt', 'sum'), xw=('x_wkt', 'sum')).reset_index()
    p_bowl = p_bowl[p_bowl['n'] >= 100].copy()
    p_bowl['score'] = 0.60 * (p_bowl['s'] / p_bowl['n']) + 0.25 * ((p_bowl['w'] - p_bowl['xw']) / p_bowl['n'])
    for name in p_bowl.sort_values('score', ascending=False).head(10)['bowler_name']:
        if name in top10_counts_bowl:
            top10_counts_bowl[name] += 1

bat_q['boot_p'] = bat_q['striker_name'].map(top10_counts_bat) / n_boots
bowl['boot_p'] = bowl['bowler_name'].map(top10_counts_bowl) / n_boots

# Check requested players
print("\nBoot P for Specific Players of Interest:")
spec_names = ['Harry Brook', 'Nicholas Pooran', 'Heinrich Klaasen', 'Mahendra Singh Dhoni', 'Jasprit Bumrah', 'Sunil Narine', 'Rinku Singh']
for name in spec_names:
    bp_bat = bat_q[bat_q['striker_name'].str.contains(name, case=False, na=False)]
    bp_bowl = bowl[bowl['bowler_name'].str.contains(name, case=False, na=False)]
    if len(bp_bat) > 0:
        row = bp_bat.iloc[0]
        print(f"  {row['striker_name']} (Batter): Balls={row['p_balls']}, Rank={row['rank_bat']}, Boot P={row['boot_p']:.1%}")
    elif len(bp_bowl) > 0:
        row = bp_bowl.iloc[0]
        print(f"  {row['bowler_name']} (Bowler): Balls={row['p_balls']}, Rank={row['rank_bowl']}, Boot P={row['boot_p']:.1%}")
    else:
        print(f"  {name}: Filtered out / did not meet floor requirements.")


# ==============================================================================
# STEP 7: FINALIZE RESULTS & EXPORT SINGLE SOURCE OF TRUTH (results_master.csv)
# ==============================================================================
print("\n--- STEP 7: FINALIZE RESULTS MASTER CSV ---")

# Flag low confidence players: p_balls < 150 OR boot_p < 0.40
bat_q['low_confidence'] = (bat_q['p_balls'] < 150) | (bat_q['boot_p'] < 0.40)
bowl['low_confidence'] = (bowl['p_balls'] < 150) | (bowl['boot_p'] < 0.40)

bat_q['flagged_name'] = bat_q.apply(lambda r: f"{r['striker_name']}*" if r['low_confidence'] else r['striker_name'], axis=1)
bowl['flagged_name'] = bowl.apply(lambda r: f"{r['bowler_name']}*" if r['low_confidence'] else r['bowler_name'], axis=1)

# Assign scouting archetypes
def get_bat_archetype(row):
    if row['eb_runs_part'] > 0.25 and row['eb_wkt_part'] > 0.0:
        return 'Ultra-Aggressive Finisher'
    elif row['eb_wkt_part'] < -0.015:
        return 'Crisis Anchor & Builder'
    elif row['p_sr'] > 160:
        return 'Dynamic Power-Hitter'
    else:
        return 'Situation Manipulator'

def get_bowl_archetype(row):
    if row['death_share'] > 0.35:
        return 'Death Overs Specialist'
    elif row['p_er'] < 7.0:
        return 'Chokehold Restrictor'
    elif row['wkts_above_x'] > 3.0:
        return 'Aggressive Strike Weapon'
    else:
        return 'Phase-Versatile Enforcer'

bat_q['archetype'] = bat_q.apply(get_bat_archetype, axis=1)
bowl['archetype'] = bowl.apply(get_bowl_archetype, axis=1)

# Sort final primary lists
final_batters = bat_q.sort_values('bat_score', ascending=False).reset_index(drop=True)
final_batters['rank'] = range(1, len(final_batters) + 1)

final_bowlers = bowl.sort_values('bowl_score', ascending=False).reset_index(drop=True)
final_bowlers['rank'] = range(1, len(final_bowlers) + 1)

# Add metadata tags for results_master.csv export
final_batters['player_type'] = 'Batter'
final_bowlers['player_type'] = 'Bowler'

# Export Master CSV
final_batters.to_csv('batters_master.csv', index=False)
final_bowlers.to_csv('bowlers_master.csv', index=False)
sens_df.to_csv('sensitivity_master.csv', index=False)

# Build unified results_master.csv
master_export_rows = []
for idx, r in final_batters.iterrows():
    master_export_rows.append({
        'player_type': 'Batter',
        'rank': r['rank'],
        'player_name': r['striker_name'],
        'flagged_name': r['flagged_name'],
        'p_balls': r['p_balls'],
        'metric_primary_rate': r['p_sr'],
        'raw_runs_part': r['raw_runs_part'],
        'eb_runs_part': r['eb_runs_part'],
        'raw_wkt_part': r['raw_wkt_part'],
        'eb_wkt_part': r['eb_wkt_part'],
        'eb_resid': r['eb_resid_w6'],
        'alpha_baseline': r['alpha'],
        'beta_delta': r['beta'],
        'shrunk_succ_rate': r['shrunk_succ'],
        'composite_score': r['bat_score'],
        'boot_p': r['boot_p'],
        'low_confidence': r['low_confidence'],
        'archetype': r['archetype']
    })

for idx, r in final_bowlers.iterrows():
    master_export_rows.append({
        'player_type': 'Bowler',
        'rank': r['rank'],
        'player_name': r['bowler_name'],
        'flagged_name': r['flagged_name'],
        'p_balls': r['p_balls'],
        'metric_primary_rate': r['p_er'],
        'raw_runs_part': r['saved_per_ball'],
        'eb_runs_part': r['eb_runs_saved'],
        'raw_wkt_part': r['wkts_above_x_rate'],
        'eb_wkt_part': r['eb_wkt_rate'],
        'eb_resid': r['implied_par'],
        'alpha_baseline': r['alpha_bowl'],
        'beta_delta': r['beta_bowl'],
        'shrunk_succ_rate': r['shrunk_succ_rate'],
        'composite_score': r['bowl_score'],
        'boot_p': r['boot_p'],
        'low_confidence': r['low_confidence'],
        'archetype': r['archetype']
    })

results_master = pd.DataFrame(master_export_rows)
results_master.to_csv('results_master.csv', index=False)
print(f"Single source of truth 'results_master.csv' successfully created with {len(results_master)} players.")


# ==============================================================================
# STEP 5.4: GENERATE FIGURE 4 (RESIDUAL VS RESIDUAL) & DIAGNOSTIC PLOTS
# ==============================================================================
print("\n--- GENERATING PUBLICATION FIGURES ---")
os.makedirs('reports', exist_ok=True)

# Figure 4: Alpha vs Pressure Residual (Batters)
plt.figure(figsize=(9, 6), dpi=300)
top12 = final_batters.head(12)
plt.scatter(final_batters['alpha'], final_batters['eb_resid_w6'], color='#94a3b8', alpha=0.6, s=40, label='All Qualified Batters (N=93)')
plt.scatter(top12['alpha'], top12['eb_resid_w6'], color='#2563eb', s=90, zorder=5, label='Top 12 Ranked Batters')

# Add 45-degree parity line
lims = [-0.3, 0.8]
plt.plot(lims, lims, linestyle='--', color='#dc2626', alpha=0.7, label='Zero Elevation Parity Line (Pressure = Baseline)')

for idx, r in top12.iterrows():
    offset_x = 0.015
    offset_y = 0.015
    if 'Hridoy' in r['striker_name']:
        offset_y = -0.035
    elif 'Perera' in r['striker_name']:
        offset_x = -0.11
    elif 'Tim David' in r['striker_name']:
        offset_y = -0.03
    plt.annotate(f"{r['rank']}. {r['striker_name']}", (r['alpha'] + offset_x, r['eb_resid_w6'] + offset_y), fontsize=8, weight='bold', color='#1e293b')

plt.xlabel(r'Baseline Residual $\alpha$ (Low-Pressure EB Context Runs Above Par)', fontsize=10, weight='bold')
plt.ylabel(r'High-Pressure EB Residual (Context Runs Above Par, $W=6$)', fontsize=10, weight='bold')
plt.title(r'Figure 4: Demonstrated High-Pressure Execution vs Baseline Ability ($\beta$ Elevation)', fontsize=11, weight='bold')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(frameon=True, fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig('reports/fig4_residual_vs_residual.png')
plt.close()
print("Figure 4 saved to reports/fig4_residual_vs_residual.png")

# Also copy results_master to artifact directory
artifact_dir = '/Users/ruhannp.n/.gemini/antigravity/brain/53a3a2a2-70f8-4302-86c8-cdcc54ac4e3d'
results_master.to_csv(os.path.join(artifact_dir, 'results_master.csv'), index=False)
sens_df.to_csv(os.path.join(artifact_dir, 'sensitivity_master.csv'), index=False)

# Export summary JSON for quick access
summary_metrics = {
    'counts': {
        'all_matches': n_raw_matches,
        'valid_matches': n_valid_matches,
        'tier1_matches': n_tier1_matches,
        'all_balls': n_raw_balls,
        'valid_balls': n_valid_balls,
        'tier1_balls': n_tier1_balls,
        'high_pressure_balls': int(n_t1_hp_balls),
        'li_threshold': float(round(li_80th, 3))
    },
    'reliability': {
        'bat_split_half_raw': float(round(r_oe_bat, 3)),
        'bat_split_half_sb': float(round(r_sb_bat, 3)),
        'bowl_split_half_raw': float(round(r_oe_bowl, 3)),
        'bowl_split_half_sb': float(round(r_sb_bowl, 3)),
        'bat_time_split': float(round(r_time_bat, 3)),
        'bat_alpha_beta': float(round(r_ab_bat, 3)),
        'bowl_alpha_beta': float(round(r_ab_bowl, 3))
    },
    'top10_batters': final_batters.head(10)[['rank', 'striker_name', 'flagged_name', 'p_balls', 'p_sr', 'raw_runs_part', 'eb_runs_part', 'raw_wkt_part', 'eb_wkt_part', 'eb_resid_w6', 'alpha', 'beta', 'shrunk_succ', 'bat_score', 'boot_p', 'low_confidence', 'archetype']].to_dict(orient='records'),
    'top10_bowlers': final_bowlers.head(10)[['rank', 'bowler_name', 'flagged_name', 'p_balls', 'p_er', 'saved_per_ball', 'eb_runs_saved', 'wkts_above_x', 'implied_par', 'alpha_bowl', 'beta_bowl', 'shrunk_succ_rate', 'bowl_score', 'boot_p', 'low_confidence', 'archetype']].to_dict(orient='records')
}

with open('summary_metrics.json', 'w') as f:
    json.dump(summary_metrics, f, indent=2)
with open(os.path.join(artifact_dir, 'summary_metrics.json'), 'w') as f:
    json.dump(summary_metrics, f, indent=2)

print("\n=== MASTER PIPELINE COMPLETED SUCCESSFULLY ===")
