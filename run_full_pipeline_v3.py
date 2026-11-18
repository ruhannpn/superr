import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== RUNNING V4 PIPELINE: ENTRY-POINT ADJUSTED & BALANCED WICKETS/ECONOMY ===")
t0 = time.time()
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
matches['is_ipl'] = matches['comp_id'].isin([1358, 1715, 2193, 2725, 3298])
matches['win_team_id'] = pd.to_numeric(matches['win_team_id'], errors='coerce')
matches['match_year'] = pd.to_datetime(matches['match_date']).dt.year

meta_cols = ['match_id', 'stage', 'knock_out', 'first_inn_revised_overs', 'second_inn_revised_overs', 
             'revised_target', 'win_team_id', 'is_tier1', 'is_ipl', 'venue_id', 'match_date', 'match_year']
df = bbb.merge(matches[meta_cols], on='match_id', how='left')
df = df[df['match_id'].isin(valid_match_ids) & df['innings'].isin([1, 2])].copy()

df['is_legal'] = ~df['extras_type'].isin(['wide', 'Overstep'])
df['total_ball_runs'] = (df['runs'].fillna(0).astype(int) + 
                         df['extras1_runs'].fillna(0).astype(int) + 
                         df['extras_2_runs'].fillna(0).astype(int))
df['is_wkt'] = (df['is_wicket'].astype(str).str.lower() == 'true').astype(int)
df['is_dot'] = ((df['runs'] == 0) & (df['extras1_runs'] == 0)).astype(int)
df['is_boundary'] = df['is_four'].astype(str).str.lower().isin(['true', '1']) | df['is_six'].astype(str).str.lower().isin(['true', '1'])
df['is_boundary'] = df['is_boundary'].astype(int)

df = df.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)
df['bat_team_won'] = (df['batting_team_id'].astype(float) == df['win_team_id']).astype(int)

# Entry point and continuous set/cold feature
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

# Shifted rolling features (zero leakage)
g_inn2 = df.groupby(['match_id', 'innings'])
df['runs_l12'] = g_inn2['total_ball_runs'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['wkts_l18'] = g_inn2['is_wkt'].transform(lambda s: s.shift(1).rolling(18, min_periods=1).sum()).fillna(0)
df['dots_l12'] = g_inn2['is_dot'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['phase'] = np.where(df['overid'] <= 6, 'Powerplay', np.where(df['overid'] <= 15, 'Middle', 'Death'))

speed = df['bowl_speed'].copy()
df['clean_bowl_speed'] = np.where((speed < 60) | (speed > 165), np.nan, speed)

print(f"Data prepared in {time.time()-t0:.2f}s.")

# --- STEP 2: MODEL A (WIN PROBABILITY & CALIBRATION) ---
feats_inn2 = ['runs_required', 'rrr', 'balls_remaining', 'wickets_in_hand', 'crr', 'rrr_minus_crr', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']
mono_cst_inn2 = [-1, -1, 1, 1, 0, -1, 0, -1, -1, 0]
feats_inn1 = ['cum_runs_before', 'cum_wkts_before', 'balls_remaining', 'wickets_in_hand', 'crr', 'projected_score', 'venue_1st_inn_avg', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']

df['wp_raw'] = 0.5
gkf = GroupKFold(n_splits=5)

for tr_idx, te_idx in gkf.split(df[df['innings'] == 2], groups=df[df['innings'] == 2]['match_id']):
    idx_tr = df[df['innings'] == 2].iloc[tr_idx].index
    idx_te = df[df['innings'] == 2].iloc[te_idx].index
    m_inn2 = HistGradientBoostingClassifier(monotonic_cst=mono_cst_inn2, max_iter=100, learning_rate=0.08, min_samples_leaf=30, random_state=42)
    m_inn2.fit(df.loc[idx_tr, feats_inn2], df.loc[idx_tr, 'bat_team_won'])
    df.loc[idx_te, 'wp_raw'] = m_inn2.predict_proba(df.loc[idx_te, feats_inn2])[:, 1]

for tr_idx, te_idx in gkf.split(df[df['innings'] == 1], groups=df[df['innings'] == 1]['match_id']):
    idx_tr = df[df['innings'] == 1].iloc[tr_idx].index
    idx_te = df[df['innings'] == 1].iloc[te_idx].index
    m_inn1 = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, min_samples_leaf=30, random_state=42)
    m_inn1.fit(df.loc[idx_tr, feats_inn1], df.loc[idx_tr, 'bat_team_won'])
    df.loc[idx_te, 'wp_raw'] = m_inn1.predict_proba(df.loc[idx_te, feats_inn1])[:, 1]

iso = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
iso.fit(df['wp_raw'], df['bat_team_won'])
df['wp'] = iso.predict(df['wp_raw'])

brier_all = brier_score_loss(df['bat_team_won'], df['wp'])
brier_inn1 = brier_score_loss(df[df['innings'] == 1]['bat_team_won'], df[df['innings'] == 1]['wp'])
brier_inn2 = brier_score_loss(df[df['innings'] == 2]['bat_team_won'], df[df['innings'] == 2]['wp'])
print(f"Model A Calibrated Brier Scores: Overall = {brier_all:.4f} | Innings 1 = {brier_inn1:.4f} | Innings 2 = {brier_inn2:.4f}")

# Time Split (Train <= 2024, Test >= 2025)
train_mask = df['match_year'] <= 2024
test_mask = df['match_year'] >= 2025
m_time_i1 = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(df[train_mask & (df['innings'] == 1)][feats_inn1], df[train_mask & (df['innings'] == 1)]['bat_team_won'])
pred_time_i1 = m_time_i1.predict_proba(df[test_mask & (df['innings'] == 1)][feats_inn1])[:, 1]
m_time_i2 = HistGradientBoostingClassifier(monotonic_cst=mono_cst_inn2, max_iter=100, random_state=42).fit(df[train_mask & (df['innings'] == 2)][feats_inn2], df[train_mask & (df['innings'] == 2)]['bat_team_won'])
pred_time_i2 = m_time_i2.predict_proba(df[test_mask & (df['innings'] == 2)][feats_inn2])[:, 1]

test_df_i1 = df[test_mask & (df['innings'] == 1)]
test_df_i2 = df[test_mask & (df['innings'] == 2)]
brier_time_i1 = brier_score_loss(test_df_i1['bat_team_won'], pred_time_i1)
brier_time_i2 = brier_score_loss(test_df_i2['bat_team_won'], pred_time_i2)
brier_time_all = (brier_time_i1 * len(test_df_i1) + brier_time_i2 * len(test_df_i2)) / (len(test_df_i1) + len(test_df_i2))
print(f"Time-Split (Train 2022-24 -> Test 2025-26) Brier: Overall = {brier_time_all:.4f} (Inn1 = {brier_time_i1:.4f}, Inn2 = {brier_time_i2:.4f})")

# --- STEP 3: TRUE WIN-PROBABILITY SWING LEVERAGE ---
m_i1_full = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(df[df['innings'] == 1][feats_inn1], df[df['innings'] == 1]['bat_team_won'])
m_i2_full = HistGradientBoostingClassifier(monotonic_cst=mono_cst_inn2, max_iter=100, random_state=42).fit(df[df['innings'] == 2][feats_inn2], df[df['innings'] == 2]['bat_team_won'])

df_b4_i2 = df[df['innings'] == 2][feats_inn2].copy()
df_b4_i2['runs_required'] = np.maximum(0, df_b4_i2['runs_required'] - 4)
df_b4_i2['balls_remaining'] = np.maximum(0, df_b4_i2['balls_remaining'] - 1)
df_b4_i2['rrr'] = np.where(df_b4_i2['balls_remaining'] > 0, df_b4_i2['runs_required'] * 6.0 / df_b4_i2['balls_remaining'], 0)
df_b4_i2['rrr_minus_crr'] = df_b4_i2['rrr'] - df_b4_i2['crr']
wp_b4_i2 = m_i2_full.predict_proba(df_b4_i2)[:, 1]

df_w_i2 = df[df['innings'] == 2][feats_inn2].copy()
df_w_i2['balls_remaining'] = np.maximum(0, df_w_i2['balls_remaining'] - 1)
df_w_i2['wickets_in_hand'] = np.maximum(0, df_w_i2['wickets_in_hand'] - 1)
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
df['LI_z'] = (df['LI'] - df['LI'].mean()) / df['LI'].std()

li_80th = np.percentile(df['LI'], 80)
df['is_high_pressure'] = (df['LI'] >= li_80th).astype(int)

print(f"True WP Swing Leverage 80th Pct: {li_80th:.3f}")
print(f"High Pressure ball count: {df['is_high_pressure'].sum()} ({df['is_high_pressure'].mean():.1%})")

# --- STEP 4: MODEL B (EXPECTED OUTCOMES WITH SET/COLD BATTER COVARIATE) ---
df['log_balls_faced'] = np.log1p(df['batter_balls_faced'])
feats_b = ['overid', 'balls_remaining', 'wickets_in_hand', 'venue_1st_inn_avg', 
           'clean_bowl_speed', 'pace_or_spin', 'delivery_style', 'over_or_around', 
           'batting_type', 'bowling_type', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1', 'log_balls_faced']

X_b = df[feats_b].copy()
cat_b = ['pace_or_spin', 'delivery_style', 'over_or_around', 'batting_type', 'bowling_type', 'is_tier1']
for c in cat_b:
    X_b[c] = X_b[c].astype('category')

mr = HistGradientBoostingRegressor(categorical_features=cat_b, max_iter=100, random_state=42).fit(X_b, df['runs'])
mw = HistGradientBoostingClassifier(categorical_features=cat_b, max_iter=100, random_state=42).fit(X_b, df['is_wkt'])

df['x_runs'] = mr.predict(X_b)
df['x_wkt'] = mw.predict_proba(X_b)[:, 1]

# Residual formulations
df['run_resid_bat'] = df['runs'] - df['x_runs']
df['wkt_resid_bat'] = df['is_wkt'] - df['x_wkt']
df['bat_resid'] = df['run_resid_bat'] - 6.0 * df['wkt_resid_bat']

df['run_suppression_bowl'] = df['x_runs'] - df['total_ball_runs']
df['wkt_taking_bowl'] = df['is_wkt'] - df['x_wkt']

phase_par_er = df.groupby('phase')['total_ball_runs'].mean() * 6.0
df['phase_par_ball'] = df['phase'].map(phase_par_er) / 6.0
df['bowl_saved_vs_par'] = df['phase_par_ball'] - df['total_ball_runs']

df['wpa'] = df.groupby(['match_id', 'innings'])['wp'].diff().fillna(0.0)

# --- STEP 5: LAYER C ATTRIBUTION & EMPIRICAL BAYES ---
df_t1 = df[df['is_tier1']].copy()
df_p = df_t1[df_t1['is_high_pressure'] == 1].copy()
df_b = df_t1[df_t1['is_high_pressure'] == 0].copy()

# BATTERS: ENTRY GROUP PRECISION WEIGHTING
cell = df_p.groupby(['striker_id', 'striker_name', 'entry_group']).agg(
    n=('runs', 'count'),
    runs=('runs', 'sum'),
    outs=('is_wkt', 'sum'),
    resid=('bat_resid', 'mean')
).reset_index()

def get_eb_params(sub):
    mu = sub['resid'].mean()
    var_between = max(0.01, np.var(sub['resid'], ddof=1))
    var_within = 2.40
    k = var_within / var_between
    return pd.Series({'mu': mu, 'k': k})

grp_params = cell.groupby('entry_group').apply(get_eb_params).reset_index()
print("Group Empirical Bayes parameters for Batters:")
print(grp_params)

cell = cell.merge(grp_params, on='entry_group')
cell['eb_resid'] = (cell['n'] * cell['resid'] + cell['k'] * cell['mu']) / (cell['n'] + cell['k'])
cell['prec_w'] = cell['n'] / (cell['n'] + cell['k'])

bat_entry_agg = cell.groupby(['striker_id', 'striker_name']).apply(
    lambda d: pd.Series({
        'p_balls': d['n'].sum(),
        'p_runs': d['runs'].sum(),
        'p_outs': d['outs'].sum(),
        'eb_resid_entry_adj': np.average(d['eb_resid'], weights=d['prec_w']),
        'n_groups': len(d)
    })
).reset_index()

bat_b_stats = df_b.groupby(['striker_id', 'striker_name']).agg(
    b_balls=('runs', 'count'),
    b_runs=('runs', 'sum'),
    b_outs=('is_wkt', 'sum'),
    b_resid=('bat_resid', 'mean')
).reset_index()

bat_order = df_t1.groupby(['striker_id', 'striker_name'])['batting_order'].median().reset_index()
bat_order.columns = ['striker_id', 'striker_name', 'med_order']

bat_match = df_p.groupby(['striker_id', 'striker_name', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_resid=('bat_resid', 'sum')
).reset_index()
bat_cons = bat_match[bat_match['m_balls'] >= 6].groupby(['striker_id', 'striker_name']).agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_resid', lambda s: (s > 0).sum())
).reset_index()
bat_cons['shrunk_succ_rate'] = (bat_cons['succ_matches'] + 2.0) / (bat_cons['tested_matches'] + 4.0)

bat = bat_entry_agg.merge(bat_b_stats, on=['striker_id', 'striker_name']).merge(bat_order, on=['striker_id', 'striker_name']).merge(bat_cons, on=['striker_id', 'striker_name'], how='left')
bat = bat[(bat['p_balls'] >= 120) & (bat['tested_matches'] >= 8) & (bat['med_order'] <= 7)].copy()

bat['p_sr'] = (bat['p_runs'] / bat['p_balls']) * 100.0
bat['b_sr'] = (bat['b_runs'] / bat['b_balls']) * 100.0
bat['resid_delta'] = bat['eb_resid_entry_adj'] - bat['b_resid']

def zscore(s):
    return (s - s.mean()) / s.std()

bat['z_eb_resid'] = zscore(bat['eb_resid_entry_adj'])
bat['z_p_sr'] = zscore(bat['p_sr'])
bat['z_resid_delta'] = zscore(bat['resid_delta'])
bat['z_succ'] = zscore(bat['shrunk_succ_rate'].fillna(bat['shrunk_succ_rate'].mean()))

bat['bat_score'] = (
    0.45 * bat['z_eb_resid'] + 
    0.25 * bat['z_p_sr'] + 
    0.15 * bat['z_resid_delta'] + 
    0.15 * bat['z_succ']
)

top10_batters = bat.sort_values('bat_score', ascending=False).head(12)
print("\n=== TOP 12 BATTERS (Entry-Point Adjusted) ===")
cols_bat_print = ['striker_name', 'p_balls', 'p_sr', 'b_sr', 'eb_resid_entry_adj', 'resid_delta', 'shrunk_succ_rate', 'bat_score']
print(top10_batters[cols_bat_print].to_string(index=False))

# BOWLERS: BALANCED WICKETS & ECONOMY
bowl_p = df_p.groupby(['bowler_id', 'bowler_name']).agg(
    p_balls=('runs', 'count'),
    p_legal_balls=('is_legal', 'sum'),
    p_runs_conceded=('total_ball_runs', 'sum'),
    p_wkts=('is_wkt', 'sum'),
    p_x_wkts=('x_wkt', 'sum'),
    p_run_supp=('run_suppression_bowl', 'mean'),
    p_runs_saved=('bowl_saved_vs_par', 'sum'),
    p_death_balls=('phase', lambda s: (s == 'Death').sum()),
    p_matches=('match_id', 'nunique')
).reset_index()

bowl_b = df_b.groupby(['bowler_id', 'bowler_name']).agg(
    b_balls=('runs', 'count'),
    b_legal_balls=('is_legal', 'sum'),
    b_runs_conceded=('total_ball_runs', 'sum'),
    b_wkts=('is_wkt', 'sum')
).reset_index()

bowl_match = df_p.groupby(['bowler_id', 'bowler_name', 'match_id']).agg(
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
bowl['er_suppression'] = bowl['b_er'] - bowl['p_er']
bowl['death_share'] = bowl['p_death_balls'] / bowl['p_balls']
bowl['wkts_above_x'] = bowl['p_wkts'] - bowl['p_x_wkts']
bowl['wkts_above_x_rate'] = bowl['wkts_above_x'] / bowl['p_balls']
bowl['runs_saved_per_ball'] = bowl['p_runs_saved'] / bowl['p_balls']
bowl['balls_per_wkt'] = bowl['p_balls'] / np.maximum(1, bowl['p_wkts'])

# Empirical Bayes on pure run suppression:
mu_supp = bowl['p_run_supp'].mean()
var_supp_between = max(0.01, np.var(bowl['p_run_supp'], ddof=1))
k_supp = 2.40 / var_supp_between
bowl['eb_run_supp'] = (bowl['p_balls'] * bowl['p_run_supp'] + k_supp * mu_supp) / (bowl['p_balls'] + k_supp)

# Empirical Bayes on wickets above x:
mu_wkt = bowl['wkts_above_x_rate'].mean()
var_wkt_between = max(0.001, np.var(bowl['wkts_above_x_rate'], ddof=1))
k_wkt = 0.05 / var_wkt_between
bowl['eb_wkt_rate'] = (bowl['p_balls'] * bowl['wkts_above_x_rate'] + k_wkt * mu_wkt) / (bowl['p_balls'] + k_wkt)

bowl['z_runs_saved'] = zscore(bowl['runs_saved_per_ball'])
bowl['z_eb_supp'] = zscore(bowl['eb_run_supp'])
bowl['z_eb_wkt'] = zscore(bowl['eb_wkt_rate'])
bowl['z_er_delta'] = zscore(bowl['er_suppression'])
bowl['z_succ'] = zscore(bowl['shrunk_succ_rate'].fillna(bowl['shrunk_succ_rate'].mean()))

# Balanced Bowling Score combining WICKETS (25%) and ECONOMY (60%) and CONSISTENCY (15%)
bowl['bowl_score'] = (
    0.35 * bowl['z_runs_saved'] +
    0.15 * bowl['z_eb_supp'] +
    0.25 * bowl['z_eb_wkt'] +
    0.10 * bowl['z_er_delta'] +
    0.15 * bowl['z_succ']
)

top10_bowlers = bowl.sort_values('bowl_score', ascending=False).head(12)
print("\n=== TOP 12 BOWLERS (Balanced Wickets 25% + Economy 60% + Consistency 15%) ===")
cols_bowl_print = ['bowler_name', 'p_balls', 'p_wkts', 'p_er', 'b_er', 'death_share', 'runs_saved_per_ball', 'wkts_above_x', 'bowl_score']
print(top10_bowlers[cols_bowl_print].to_string(index=False))

# --- FIGURES GENERATION ---
print("\nGenerating figures...")
os.makedirs('figures', exist_ok=True)

# Fig 1: Leverage distribution
plt.figure(figsize=(6, 4))
plt.hist(df['LI'], bins=50, color='#002B49', edgecolor='white', alpha=0.85)
plt.axvline(li_80th, color='#EA1A85', linestyle='--', linewidth=2, label=f'80th Pct Cutoff ({li_80th:.2f})')
plt.title('Distribution of True Win-Probability Swing Leverage (LI)', fontsize=10, fontweight='bold', pad=10)
plt.xlabel('Leverage Index (LI)', fontsize=9)
plt.ylabel('Deliveries Count', fontsize=9)
plt.legend(frameon=True, fontsize=8)
plt.tight_layout()
plt.savefig('/tmp/fig1_leverage_distribution.png', dpi=300)
plt.savefig('figures/fig1_leverage_distribution.png', dpi=300)
plt.close()

# Fig 2: Phase Stratified Outcomes
plt.figure(figsize=(6, 4))
phases = ['Powerplay', 'Middle', 'Death']
dot_norm = [df[(df['phase']==p) & (df['is_high_pressure']==0)]['is_dot'].mean()*100 for p in phases]
dot_pres = [df[(df['phase']==p) & (df['is_high_pressure']==1)]['is_dot'].mean()*100 for p in phases]
x = np.arange(len(phases))
w = 0.35
plt.bar(x - w/2, dot_norm, width=w, label='Normal Leverage', color='#94a3b8')
plt.bar(x + w/2, dot_pres, width=w, label='High Leverage (Top 20%)', color='#EA1A85')
plt.xticks(x, phases, fontsize=9)
plt.ylabel('Dot Ball Percentage (%)', fontsize=9)
plt.title('Phase-Stratified Dot % Suppression Under High Leverage', fontsize=10, fontweight='bold', pad=10)
plt.legend(frameon=True, fontsize=8)
plt.tight_layout()
plt.savefig('/tmp/fig2_outcomes_pressure_vs_normal.png', dpi=300)
plt.savefig('figures/fig2_outcomes_pressure_vs_normal.png', dpi=300)
plt.close()

# Fig 3: Calibration Curve
plt.figure(figsize=(6, 4))
prob_true_raw, prob_pred_raw = calibration_curve(df['bat_team_won'], df['wp_raw'], n_bins=10)
prob_true_iso, prob_pred_iso = calibration_curve(df['bat_team_won'], df['wp'], n_bins=10)
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.plot(prob_pred_raw, prob_true_raw, 's-', color='#94a3b8', label='Raw Model A')
plt.plot(prob_pred_iso, prob_true_iso, 'o-', color='#004BA0', linewidth=2, label='Isotonic Calibrated (WP)')
plt.title('Win Probability Reliability Diagram (Calibration Curve)', fontsize=10, fontweight='bold', pad=10)
plt.xlabel('Predicted Win Probability', fontsize=9)
plt.ylabel('Observed Win Fraction', fontsize=9)
plt.legend(frameon=True, fontsize=8)
plt.tight_layout()
plt.savefig('/tmp/fig3_wp_calibration.png', dpi=300)
plt.savefig('figures/fig3_wp_calibration.png', dpi=300)
plt.close()

# Fig 4: Scatter Plot
plt.figure(figsize=(6, 4))
plt.scatter(bat['b_sr'], bat['p_sr'], color='#94a3b8', alpha=0.6, label='Tier-1 Batters')
top5_bat = top10_batters.head(5)
plt.scatter(top5_bat['b_sr'], top5_bat['p_sr'], color='#EA1A85', s=80, edgecolors='black', zorder=5, label='Top Batters')
for _, r in top5_bat.iterrows():
    plt.annotate(r['striker_name'].split()[-1], (r['b_sr'], r['p_sr']), textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight='bold')
lims = [100, 240]
plt.plot(lims, lims, 'k--', alpha=0.5, label='Parity (P_SR = Base_SR)')
plt.title('High-Leverage Strike Rate vs. Baseline Strike Rate', fontsize=10, fontweight='bold', pad=10)
plt.xlabel('Baseline Strike Rate', fontsize=9)
plt.ylabel('High-Leverage Strike Rate', fontsize=9)
plt.legend(frameon=True, fontsize=8)
plt.tight_layout()
plt.savefig('/tmp/fig4_pressure_vs_baseline_scatter.png', dpi=300)
plt.savefig('figures/fig4_pressure_vs_baseline_scatter.png', dpi=300)
plt.close()

print("Figures successfully generated.")
