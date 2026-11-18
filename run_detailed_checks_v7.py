import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

print("=== STARTING DETAILED CHECKS SCRIPT (V7 AUDIT) ===")

# --- 1. LOAD DATA ---
bbb = pd.read_csv('ball_by_ball.csv')
matches = pd.read_csv('matches.csv')

valid_matches = matches[matches['result'].notna() & ~matches['result'].str.lower().str.contains('no result')].copy()
valid_match_ids = set(valid_matches['match_id'].unique())

full_members = {
    'India', 'Australia', 'England', 'South Africa', 'New Zealand', 
    'Pakistan', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan',
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bengaluru', 'Sunrisers Hyderabad'
}

matches['is_tier1'] = matches['home_team'].isin(full_members) & matches['away_team'].isin(full_members)
matches['win_team_id'] = pd.to_numeric(matches['win_team_id'], errors='coerce')

meta_cols = ['match_id', 'stage', 'knock_out', 'first_inn_revised_overs', 'second_inn_revised_overs', 
             'revised_target', 'win_team_id', 'is_tier1', 'venue_id', 'match_date']
df = bbb.merge(matches[meta_cols], on='match_id', how='left')
df = df[df['match_id'].isin(valid_match_ids) & df['innings'].isin([1, 2])].copy()

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

# Shifted rolling
g_inn2 = df.groupby(['match_id', 'innings'])
df['runs_l12'] = g_inn2['total_ball_runs'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['wkts_l18'] = g_inn2['is_wkt'].transform(lambda s: s.shift(1).rolling(18, min_periods=1).sum()).fillna(0)
df['dots_l12'] = g_inn2['is_dot'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['phase'] = np.where(df['overid'] <= 6, 'Powerplay', np.where(df['overid'] <= 15, 'Middle', 'Death'))

# Fit Model A
feats_inn2 = ['runs_required', 'rrr', 'balls_remaining', 'wickets_in_hand', 'crr', 'rrr_minus_crr', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']
mono_cst_inn2 = [-1, -1, 1, 1, 0, -1, 0, -1, -1, 0]
feats_inn1 = ['cum_runs_before', 'cum_wkts_before', 'balls_remaining', 'wickets_in_hand', 'crr', 'projected_score', 'venue_1st_inn_avg', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']

m_i1_full = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(df[df['innings'] == 1][feats_inn1], df[df['innings'] == 1]['bat_team_won'])
m_i2_full = HistGradientBoostingClassifier(monotonic_cst=mono_cst_inn2, max_iter=100, random_state=42).fit(df[df['innings'] == 2][feats_inn2], df[df['innings'] == 2]['bat_team_won'])

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

# MODEL B
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

# Dynamic W
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

# Phase Par for Bowlers
phase_par_er = df[df['is_tier1']].groupby('phase')['total_ball_runs'].mean() * 6.0
df['phase_par_ball'] = df['phase'].map(phase_par_er) / 6.0
df['bowl_saved_vs_par'] = df['phase_par_ball'] - df['total_ball_runs']

df_t1 = df[df['is_tier1']].copy()
df_t1_p = df_t1[df_t1['is_high_pressure'] == 1].copy()
df_t1_b = df_t1[df_t1['is_high_pressure'] == 0].copy()

# ==============================================================================
# ITEM 1 & 2: DIRECT BOWLER RUNS SAVED & IMPLIED PAR
# ==============================================================================
print("\n" + "="*50)
print("ITEM 1 & 2: DIRECT BOWLER RUNS SAVED & IMPLIED PAR")
print("="*50)

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
bowl = bowl[(bowl['p_balls'] >= 120) & (bowl['tested_matches'] >= 8)].copy()

bowl['p_er'] = (bowl['p_runs_conceded'] / np.maximum(1, bowl['p_legal_balls'])) * 6.0
bowl['b_er'] = (bowl['b_runs_conceded'] / np.maximum(1, bowl['b_legal_balls'])) * 6.0
bowl['death_share'] = bowl['p_death_balls'] / bowl['p_balls']
bowl['saved_per_ball'] = bowl['p_runs_saved'] / bowl['p_balls']
bowl['wkts_above_x'] = bowl['p_wkts'] - bowl['p_x_wkts']
bowl['wkts_above_x_rate'] = bowl['wkts_above_x'] / bowl['p_balls']

bowl['implied_par'] = bowl['p_er'] + 6.0 * bowl['saved_per_ball']

print("Direct Bowler Implied Par describe():")
print(bowl[['p_er', 'saved_per_ball', 'implied_par']].describe())

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

top15_bowl = bowl.sort_values('bowl_score', ascending=False).head(15)
cols_b_print = ['bowler_name', 'p_balls', 'p_wkts', 'p_er', 'saved_per_ball', 'implied_par', 'wkts_above_x', 'bowl_score']
print("\nTop 15 Direct Bowlers:")
print(top15_bowl[cols_b_print].to_string(index=False))

# Check specific bowlers requested: Krunal Pandya, Josh Hazlewood, Shadab Khan
print("\nSpecific Bowler Inspection (Krunal, Hazlewood, Shadab, Bumrah):")
spec_bowl = bowl[bowl['bowler_name'].str.contains('Krunal|Hazlewood|Shadab|Bumrah', case=False, na=False)]
print(spec_bowl[cols_b_print].to_string(index=False))

# ==============================================================================
# ITEM 3: ESTIMATE SEPARATE k FOR RUNS AND WICKETS
# ==============================================================================
print("\n" + "="*50)
print("ITEM 3: SEPARATE k ESTIMATION FOR RUNS PART AND WICKET PART")
print("="*50)

# Cell level dataset for batters in Tier-1 high pressure
cell = df_t1_p.groupby(['striker_id', 'striker_name', 'entry_group']).agg(
    n=('runs', 'count'),
    runs=('runs', 'sum'),
    outs=('is_wkt', 'sum'),
    r_resid=('run_resid', 'mean'),
    w_resid=('wkt_resid', 'mean')
).reset_index()

# Overall empirical variances
var_within_runs = df_t1_p.groupby(['striker_id', 'entry_group'])['run_resid'].var().dropna().mean()
var_within_wkt = df_t1_p.groupby(['striker_id', 'entry_group'])['wkt_resid'].var().dropna().mean()

# Between variance of batter cell means (weighted by sample size or unweighted)
var_between_runs = max(0.01, np.var(cell['r_resid'], ddof=1) - var_within_runs / cell['n'].mean())
var_between_wkt = max(0.0005, np.var(cell['w_resid'], ddof=1) - var_within_wkt / cell['n'].mean())

k_runs = var_within_runs / var_between_runs
k_wkt = var_within_wkt / var_between_wkt

print(f"Overall Empirical Variance - Runs: within={var_within_runs:.4f}, between={var_between_runs:.4f} -> k_runs = {k_runs:.1f}")
print(f"Overall Empirical Variance - Wkts: within={var_within_wkt:.4f}, between={var_between_wkt:.4f} -> k_wkt = {k_wkt:.1f}")

# Group-by-group k estimation
print("\nGroup-by-Group k Estimation:")
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
    print(f"  Entry Group '{grp}': k_runs = {kr:.1f}, k_wkt = {kw:.1f}")

# Compute cell-level EB with separate k
cell['eb_r_part'] = 0.0
cell['eb_w_part'] = 0.0
cell['raw_r_part'] = cell['r_resid']
cell['raw_w_part'] = cell['w_resid']
cell['prec_w_runs'] = 0.0
cell['prec_w_wkt'] = 0.0

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

# Aggregate across entry groups
def agg_batter_separate_k(d):
    pw = d['prec_w_runs']
    n_tot = d['n'].sum()
    return pd.Series({
        'p_balls': n_tot,
        'p_runs': d['runs'].sum(),
        'p_outs': d['outs'].sum(),
        'raw_runs_part': np.average(d['raw_r_part'], weights=d['n']),
        'raw_wkt_part': np.average(d['raw_w_part'], weights=d['n']),
        'eb_runs_part': np.average(d['eb_r_part'], weights=pw),
        'eb_wkt_part': np.average(d['eb_w_part'], weights=pw),
    })

bat_agg = cell.groupby(['striker_id', 'striker_name']).apply(agg_batter_separate_k).reset_index()

# Reconciled residuals
bat_agg['eb_resid_w6'] = bat_agg['eb_runs_part'] - 6.0 * bat_agg['eb_wkt_part']
bat_agg['eb_resid_w4'] = bat_agg['eb_runs_part'] - 4.0 * bat_agg['eb_wkt_part']

# Consistency
bat_match = df_t1_p.groupby(['striker_id', 'striker_name', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_resid=('run_resid', lambda s: s.sum()) # match run resid
).reset_index()
bat_cons = bat_match[bat_match['m_balls'] >= 6].groupby(['striker_id', 'striker_name']).agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_resid', lambda s: (s > 0).sum())
).reset_index()
bat_cons['shrunk_succ'] = (bat_cons['succ_matches'] + 2.0) / (bat_cons['tested_matches'] + 4.0)

# Baseline stats for beta calculation
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

# Score with separate k
bat_q['z_eb_resid_w6'] = zscore(bat_q['eb_resid_w6'])
bat_q['z_succ'] = zscore(bat_q['shrunk_succ'])
bat_q['score_w6'] = 0.80 * bat_q['z_eb_resid_w6'] + 0.20 * bat_q['z_succ']

# Score with W=4
bat_q['z_eb_resid_w4'] = zscore(bat_q['eb_resid_w4'])
bat_q['score_w4'] = 0.80 * bat_q['z_eb_resid_w4'] + 0.20 * bat_q['z_succ']

bat_q['rank_w6'] = bat_q['score_w6'].rank(ascending=False).astype(int)
bat_q['rank_w4'] = bat_q['score_w4'].rank(ascending=False).astype(int)

# Add alpha (baseline EB) and beta (pressure minus baseline delta)
# Shrunk baseline residual
mu_base = bat_q['b_resid_w6'].mean()
k_base = 50.0
bat_q['alpha'] = (bat_q['b_balls'] * bat_q['b_resid_w6'] + k_base * mu_base) / (bat_q['b_balls'] + k_base)
bat_q['raw_delta'] = bat_q['eb_resid_w6'] - bat_q['alpha']
# Shrunk beta
k_beta = 60.0
bat_q['beta'] = (bat_q['p_balls'] * bat_q['raw_delta']) / (bat_q['p_balls'] + k_beta)
bat_q['rank_beta'] = bat_q['beta'].rank(ascending=False).astype(int)

print("\n" + "="*50)
print("BATTER RANKS WITH SEPARATE k (W=6)")
print("="*50)
top12_bat = bat_q.sort_values('score_w6', ascending=False).head(12)
cols_show = ['rank_w6', 'striker_name', 'p_balls', 'raw_runs_part', 'eb_runs_part', 'raw_wkt_part', 'eb_wkt_part', 'eb_resid_w6', 'alpha', 'beta', 'rank_beta', 'rank_w4']
print(top12_bat[cols_show].to_string(index=False))

# ==============================================================================
# ITEM 4: DHONI VS KARTHIK AUDIT
# ==============================================================================
print("\n" + "="*50)
print("ITEM 4: DHONI VS KARTHIK AUDIT ROWS")
print("="*50)
dk_msd_audit = bat_full[bat_full['striker_name'].str.contains('Dhoni|Karthik', case=False, na=False)].copy()
dk_msd_audit['p_sr'] = (dk_msd_audit['p_runs'] / dk_msd_audit['p_balls']) * 100.0
dk_msd_audit['is_qualified_120'] = dk_msd_audit['p_balls'] >= 120
cols_dk = ['striker_name', 'p_balls', 'p_runs', 'p_outs', 'p_sr', 'raw_runs_part', 'raw_wkt_part', 'eb_resid_w6', 'tested_matches', 'shrunk_succ', 'is_qualified_120']
print(dk_msd_audit[cols_dk].to_string(index=False))

# ==============================================================================
# ITEM 5: REMAINING CHECKS
# ==============================================================================
print("\n" + "="*50)
print("ITEM 5A: AGGREGATION WEIGHTS OVERLAP (w = n vs n/(n+k) vs n^2/(n+k))")
print("="*50)

# w1 = n
agg_w1 = cell.groupby(['striker_id', 'striker_name']).apply(
    lambda d: pd.Series({
        'p_balls': d['n'].sum(),
        'eb_resid': np.average(d['eb_r_part'] - 6.0 * d['eb_w_part'], weights=d['n'])
    })
).reset_index()
# w2 = n / (n + k)
agg_w2 = cell.groupby(['striker_id', 'striker_name']).apply(
    lambda d: pd.Series({
        'p_balls': d['n'].sum(),
        'eb_resid': np.average(d['eb_r_part'] - 6.0 * d['eb_w_part'], weights=d['n'] / (d['n'] + 30.0))
    })
).reset_index()

q1 = agg_w1[agg_w1['p_balls'] >= 120].sort_values('eb_resid', ascending=False).head(10)['striker_name'].tolist()
q2 = agg_w2[agg_w2['p_balls'] >= 120].sort_values('eb_resid', ascending=False).head(10)['striker_name'].tolist()
q3 = top12_bat.head(10)['striker_name'].tolist()

print("Top 10 with w = n:", q1)
print("Top 10 with w = n/(n+k):", q2)
print("Top 10 with w = n^2/(n+k) (Primary):", q3)
print(f"Overlap between w=n and w=n^2/(n+k): {len(set(q1).intersection(q3))}/10")
print(f"Overlap between w=n/(n+k) and w=n^2/(n+k): {len(set(q2).intersection(q3))}/10")

print("\n" + "="*50)
print("ITEM 5B: RELIABILITY (ODD/EVEN WITH SPEARMAN-BROWN, TIME-SPLIT)")
print("="*50)
df_odd = df_t1_p[df_t1_p['match_id'] % 2 == 1]
df_even = df_t1_p[df_t1_p['match_id'] % 2 == 0]

bat_odd = df_odd.groupby('striker_id')['run_resid'].mean() - 6.0 * df_odd.groupby('striker_id')['wkt_resid'].mean()
bat_even = df_even.groupby('striker_id')['run_resid'].mean() - 6.0 * df_even.groupby('striker_id')['wkt_resid'].mean()
comm_bat = bat_odd.index.intersection(bat_even.index).intersection(bat_q['striker_id'])
r_oe_bat = np.corrcoef(bat_odd.loc[comm_bat], bat_even.loc[comm_bat])[0, 1]
r_sb_bat = (2 * r_oe_bat) / (1 + r_oe_bat)
print(f"Batters Split-Half (N={len(comm_bat)}): Raw r = {r_oe_bat:.3f}, Spearman-Brown Corrected r = {r_sb_bat:.3f}")

bowl_odd = df_odd.groupby('bowler_id')['bowl_saved_vs_par'].mean()
bowl_even = df_even.groupby('bowler_id')['bowl_saved_vs_par'].mean()
comm_bowl = bowl_odd.index.intersection(bowl_even.index).intersection(bowl['bowler_id'])
r_oe_bowl = np.corrcoef(bowl_odd.loc[comm_bowl], bowl_even.loc[comm_bowl])[0, 1]
r_sb_bowl = (2 * r_oe_bowl) / (1 + r_oe_bowl)
print(f"Bowlers Split-Half (N={len(comm_bowl)}): Raw r = {r_oe_bowl:.3f}, Spearman-Brown Corrected r = {r_sb_bowl:.3f}")

# Time Split (2022-2024 vs 2025-2026)
df_t1_p['year'] = pd.to_datetime(df_t1_p['match_date']).dt.year
bat_early = df_t1_p[df_t1_p['year'] <= 2024].groupby('striker_id')['run_resid'].mean()
bat_late = df_t1_p[df_t1_p['year'] >= 2025].groupby('striker_id')['run_resid'].mean()
comm_time = bat_early.index.intersection(bat_late.index).intersection(bat_q['striker_id'])
if len(comm_time) > 5:
    r_time = np.corrcoef(bat_early.loc[comm_time], bat_late.loc[comm_time])[0, 1]
    print(f"Batters Time-Split (2022-24 vs 2025-26, N={len(comm_time)}): r = {r_time:.3f}")
else:
    print(f"Batters Time-Split: Only {len(comm_time)} players overlap across 2022-24 and 2025-26 in high-pressure sample.")

# Alpha-Beta correlation
r_ab_bat = np.corrcoef(bat_q['alpha'], bat_q['eb_resid_w6'])[0, 1]
print(f"Batters Alpha-Beta (Baseline vs Pressure Residual) correlation: r = {r_ab_bat:.3f}")

# Bowlers Alpha-Beta correlation
bowl['alpha_b_er'] = bowl['b_er']
r_ab_bowl = np.corrcoef(bowl['b_er'], bowl['p_er'])[0, 1]
print(f"Bowlers Alpha-Beta (Baseline ER vs Pressure ER) correlation: r = {r_ab_bowl:.3f}")

print("\n" + "="*50)
print("ITEM 5C: BOOTSTRAP STABILITY (P(Top 10) OVER 200 MATCH-CLUSTERED RESAMPLES)")
print("="*50)

# Resample matches with replacement
unique_matches = df_t1_p['match_id'].unique()
top10_counts_bat = {name: 0 for name in bat_q['striker_name']}
n_boots = 200

np.random.seed(42)
for b in range(n_boots):
    boot_m = np.random.choice(unique_matches, size=len(unique_matches), replace=True)
    # Weight matches by frequency
    m_counts = pd.Series(boot_m).value_counts()
    df_boot = df_t1_p[df_t1_p['match_id'].isin(m_counts.index)].copy()
    # Fast proxy: player total resid
    player_res = df_boot.groupby('striker_name').agg(
        n=('runs', 'count'),
        r_sum=('run_resid', 'sum'),
        w_sum=('wkt_resid', 'sum')
    ).reset_index()
    player_res = player_res[player_res['n'] >= 100].copy()
    player_res['score'] = (player_res['r_sum'] - 6.0 * player_res['w_sum']) / player_res['n']
    b_top10 = player_res.sort_values('score', ascending=False).head(10)['striker_name'].tolist()
    for name in b_top10:
        if name in top10_counts_bat:
            top10_counts_bat[name] += 1

bat_q['boot_p'] = bat_q['striker_name'].map(top10_counts_bat) / n_boots
print("Bootstrap Probabilities for Top Batters:")
print(bat_q.sort_values('score_w6', ascending=False).head(12)[['striker_name', 'p_balls', 'score_w6', 'boot_p']].to_string(index=False))

print("\n=== ALL DETAILED CHECKS COMPLETE ===")

