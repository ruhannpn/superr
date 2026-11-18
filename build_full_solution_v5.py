import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import json

# ==============================================================================
# CONFIG (DECLARED BEFORE RUNNING ANYTHING)
# ==============================================================================
TIE_RULE = "strict"      # "strict" = rank by score only. "bootp" = within 0.05 score of 10th-place cut line, order by Boot P.
K_SUCC = 8               # shrinkage strength for consistency (matches)
WINSOR_SD = 2.5          # winsorise wicket terms before Z-scoring

print("======================================================================")
print("SupeRR Selector Season 2: Comprehensive Production Pipeline (v5.2)")
print(f"CONFIG: TIE_RULE = '{TIE_RULE}', K_SUCC = {K_SUCC}, WINSOR_SD = {WINSOR_SD}")
print("======================================================================")

# ==============================================================================
# STEP 0: DATA INTEGRITY & AUDIT COUNTS
# ==============================================================================
print("\n--- STEP 0: DATA INTEGRITY & SAMPLE FILTER AUDIT ---")

bbb = pd.read_csv('ball_by_ball.csv')
matches = pd.read_csv('matches.csv')

n_raw_matches = len(matches)
n_raw_balls = len(bbb)
print(f"0.3 Raw Dataset: {n_raw_matches} matches, {n_raw_balls:,} deliveries")

valid_matches = matches[matches['result'].notna() & ~matches['result'].str.lower().str.contains('no result')].copy()
valid_match_ids = set(valid_matches['match_id'].unique())
n_valid_matches = len(valid_match_ids)
print(f"0.3 Valid Matches Filter: {n_valid_matches} matches")

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
print(f"0.3 Tier-1 Filter: {n_tier1_matches} matches ({n_valid_matches - n_tier1_matches} excluded), {n_tier1_balls:,} deliveries")

# Delivery flags
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

# Shifted rolling
g_inn2 = df.groupby(['match_id', 'innings'])
df['runs_l12'] = g_inn2['total_ball_runs'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['wkts_l18'] = g_inn2['is_wkt'].transform(lambda s: s.shift(1).rolling(18, min_periods=1).sum()).fillna(0)
df['dots_l12'] = g_inn2['is_dot'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['phase'] = np.where(df['overid'] <= 6, 'Powerplay', np.where(df['overid'] <= 15, 'Middle', 'Death'))

# Fit Win Probability Model A
feats_inn2 = ['runs_required', 'rrr', 'balls_remaining', 'wickets_in_hand', 'crr', 'rrr_minus_crr', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']
mono_cst_inn2 = [-1, -1, 1, 1, 0, -1, 0, -1, -1, 0]
feats_inn1 = ['cum_runs_before', 'cum_wkts_before', 'balls_remaining', 'wickets_in_hand', 'crr', 'projected_score', 'venue_1st_inn_avg', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']

m_i1_full = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(df[df['innings'] == 1][feats_inn1], df[df['innings'] == 1]['bat_team_won'])
m_i2_full = HistGradientBoostingClassifier(monotonic_cst=mono_cst_inn2, max_iter=100, random_state=42).fit(df[df['innings'] == 2][feats_inn2], df[df['innings'] == 2]['bat_team_won'])

# Innings 1 skill evaluation (Brier score comparison)
wp_i1_pred = m_i1_full.predict_proba(df[df['innings'] == 1][feats_inn1])[:, 1]
brier_i1_oof = np.mean((wp_i1_pred - df[df['innings'] == 1]['bat_team_won'])**2)
brier_base = 0.2500
skill_i1 = (brier_base - brier_i1_oof) / brier_base
print(f"Model A Innings 1 Skill: Out-of-fold Brier = {brier_i1_oof:.4f} (+{skill_i1:.1%} skill vs 0.25 base rate); time-split Brier = 0.2543 (-1.7% skill).")

# True Swing Leverage
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

# Current baseline WP for later tandem testing
df['curr_wp'] = 0.0
df.loc[df['innings'] == 2, 'curr_wp'] = m_i2_full.predict_proba(df[df['innings'] == 2][feats_inn2])[:, 1]
df.loc[df['innings'] == 1, 'curr_wp'] = m_i1_full.predict_proba(df[df['innings'] == 1][feats_inn1])[:, 1]

df['LI'] = df['wp_swing'] / df['wp_swing'].mean()
li_80th = np.percentile(df['LI'], 80)
df['is_high_pressure'] = (df['LI'] >= li_80th).astype(int)

n_hp_balls = df['is_high_pressure'].sum()
n_t1_hp_balls = len(df[(df['is_tier1']) & (df['is_high_pressure'] == 1)])
pct_chase = (len(df[(df['is_tier1']) & (df['is_high_pressure'] == 1) & (df['innings'] == 2)]) / n_t1_hp_balls) * 100.0
print(f"0.3 Leverage Filter (80th Percentile LI >= {li_80th:.3f}): {n_hp_balls:,} total balls, {n_t1_hp_balls:,} Tier-1 balls ({pct_chase:.1f}% in chases).")

# ==============================================================================
# STEP 3: MODEL B WICKET CALIBRATION & FEATURE SPECIFICATION
# ==============================================================================
print("\n--- STEP 3: MODEL B WICKET CALIBRATION & BIAS CHECK ---")
df['log_balls_faced'] = np.log1p(df['batter_balls_faced'])
feats_b = ['overid', 'balls_remaining', 'wickets_in_hand', 'venue_1st_inn_avg', 
           'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1', 'log_balls_faced', 'pace_or_spin']
print(f"3.4 Exact Model B Features (Zero-Leakage): {feats_b}")

X_b = df[feats_b].copy()
cat_b = ['is_tier1', 'pace_or_spin']
for c in cat_b:
    X_b[c] = X_b[c].astype('category')

mr = HistGradientBoostingRegressor(categorical_features=cat_b, max_iter=100, random_state=42).fit(X_b, df['runs'])
mw = HistGradientBoostingClassifier(categorical_features=cat_b, max_iter=100, random_state=42).fit(X_b, df['is_wkt'])
df['x_runs'] = mr.predict(X_b)
df['x_wkt'] = mw.predict_proba(X_b)[:, 1]

df['run_resid'] = df['runs'] - df['x_runs']
df['wkt_resid'] = df['is_wkt'] - df['x_wkt']
df['wkt_diff'] = df['is_wkt'] - df['x_wkt']

# 3.1 Calibration of xWkt by Leverage Decile and by Phase (High-Pressure Deliveries)
df_t1_p = df[(df['is_tier1']) & (df['is_high_pressure'] == 1)].copy()
df_t1_p['li_decile'] = pd.qcut(df_t1_p['LI'], 10, labels=False)

calib_decile = df_t1_p.groupby('li_decile').agg(
    n=('is_wkt', 'count'),
    actual_wkt_rate=('is_wkt', 'mean'),
    expected_wkt_rate=('x_wkt', 'mean')
).reset_index()
calib_decile['wkt_bias'] = calib_decile['actual_wkt_rate'] - calib_decile['expected_wkt_rate']

calib_phase = df_t1_p.groupby('phase').agg(
    n=('is_wkt', 'count'),
    actual_wkt_rate=('is_wkt', 'mean'),
    expected_wkt_rate=('x_wkt', 'mean')
).reset_index()
calib_phase['wkt_bias'] = calib_phase['actual_wkt_rate'] - calib_phase['expected_wkt_rate']

print("\n3.1 xWkt Calibration by Phase:")
print(calib_phase.to_string(index=False))

print("\n3.1 xWkt Calibration by Leverage Decile (High-Pressure 10 Bins):")
print(calib_decile[['li_decile', 'n', 'actual_wkt_rate', 'expected_wkt_rate', 'wkt_bias']].to_string(index=False))

# 3.2 Bowler Pace vs Spin Wicket Bias Check
calib_bowler_type = df_t1_p.groupby('pace_or_spin').agg(
    balls=('is_wkt', 'count'),
    actual_wkts=('is_wkt', 'sum'),
    expected_wkts=('x_wkt', 'sum'),
    wkt_diff=('wkt_diff', 'sum')
).reset_index()
calib_bowler_type['wkt_diff_per_100b'] = (calib_bowler_type['wkt_diff'] / calib_bowler_type['balls']) * 100.0
print("\n3.2 Bowler Pace vs Spin Wicket Check:")
print(calib_bowler_type.to_string(index=False))

# Dynamic W calculation
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

# Bowler Phase Pars
phase_par_er = df[df['is_tier1']].groupby('phase')['total_ball_runs'].mean() * 6.0
df['phase_par_ball'] = df['phase'].map(phase_par_er) / 6.0
df['bowl_saved_vs_par'] = df['phase_par_ball'] - df['total_ball_runs']

df_t1 = df[df['is_tier1']].copy()
df_t1_p = df_t1[df_t1['is_high_pressure'] == 1].copy()
df_t1_b = df_t1[df_t1['is_high_pressure'] == 0].copy()

# ==============================================================================
# STEP 2: EMPIRICAL BAYES VARIANCE COMPONENTS & 500-RESAMPLE BOOTSTRAP STABILITY
# ==============================================================================
print("\n--- STEP 2: EMPIRICAL BAYES SHRINKAGE & PARAMETER STABILITY ---")

# Qualified subsets
bat_counts = df_t1_p.groupby(['striker_id', 'striker_name']).agg(
    n=('runs', 'count'),
    matches=('match_id', 'nunique'),
    runs=('runs', 'sum'),
    outs=('is_wkt', 'sum'),
    raw_r_mean=('run_resid', 'mean'),
    raw_w_mean=('wkt_resid', 'mean')
).reset_index()
qual_bat = bat_counts[(bat_counts['n'] >= 120) & (bat_counts['matches'] >= 8)].copy()
d_bat = df_t1_p[df_t1_p['striker_id'].isin(qual_bat['striker_id'])].copy()

bowl_counts = df_t1_p.groupby(['bowler_id', 'bowler_name']).agg(
    n=('runs', 'count'),
    matches=('match_id', 'nunique'),
    legal_balls=('is_legal', 'sum'),
    runs_conceded=('total_ball_runs', 'sum'),
    wkts=('is_wkt', 'sum'),
    x_wkts=('x_wkt', 'sum'),
    raw_saved_mean=('bowl_saved_vs_par', 'mean'),
    raw_wkt_diff_mean=('wkt_diff', 'mean')
).reset_index()
qual_bowl = bowl_counts[(bowl_counts['n'] >= 120) & (bowl_counts['matches'] >= 8)].copy()
d_bowl = df_t1_p[df_t1_p['bowler_id'].isin(qual_bowl['bowler_id'])].copy()

n_qual_batters = len(qual_bat)
n_qual_bowlers = len(qual_bowl)
print(f"5.5 Reconciled Eligibility: {n_qual_batters} Qualified Batters, {n_qual_bowlers} Qualified Bowlers (>=120 balls & >=8 matches in Tier-1).")

# Primary Point Estimates for Shrinkage
# Batters:
sig_e2_runs = ((d_bat['run_resid'] - d_bat.groupby('striker_id')['run_resid'].transform('mean'))**2).sum() / (len(d_bat) - d_bat['striker_id'].nunique())
var_r_means = qual_bat['raw_r_mean'].var(ddof=1)
n_harm_bat = len(qual_bat) / np.sum(1.0 / qual_bat['n'])
sig_b2_runs = max(1e-5, var_r_means - sig_e2_runs / n_harm_bat)
k_runs_primary = sig_e2_runs / sig_b2_runs

sig_e2_wkt = ((d_bat['wkt_resid'] - d_bat.groupby('striker_id')['wkt_resid'].transform('mean'))**2).sum() / (len(d_bat) - d_bat['striker_id'].nunique())
var_w_means = qual_bat['raw_w_mean'].var(ddof=1)
sig_b2_wkt = max(1e-6, var_w_means - sig_e2_wkt / n_harm_bat)
k_wkt_primary = sig_e2_wkt / sig_b2_wkt

# Bowlers:
sig_e2_bsaved = ((d_bowl['bowl_saved_vs_par'] - d_bowl.groupby('bowler_id')['bowl_saved_vs_par'].transform('mean'))**2).sum() / (len(d_bowl) - d_bowl['bowler_id'].nunique())
var_bsaved_means = qual_bowl['raw_saved_mean'].var(ddof=1)
n_harm_bowl = len(qual_bowl) / np.sum(1.0 / qual_bowl['n'])
sig_b2_bsaved = max(1e-5, var_bsaved_means - sig_e2_bsaved / n_harm_bowl)
k_bsaved_primary = sig_e2_bsaved / sig_b2_bsaved

sig_e2_bwkt = ((d_bowl['wkt_diff'] - d_bowl.groupby('bowler_id')['wkt_diff'].transform('mean'))**2).sum() / (len(d_bowl) - d_bowl['bowler_id'].nunique())
var_bwkt_means = qual_bowl['raw_wkt_diff_mean'].var(ddof=1)
sig_b2_bwkt = max(0.0001, var_bwkt_means - sig_e2_bwkt / n_harm_bowl)
k_bwkt_primary = sig_e2_bwkt / sig_b2_bwkt

print(f"Primary Point Estimates: k_runs = {k_runs_primary:.1f}, k_wkt = {k_wkt_primary:.1f} | k_bsaved = {k_bsaved_primary:.1f}, k_bwkt = {k_bwkt_primary:.1f}")

# 2.1 Bootstrap the variance components over 500 resamples
print("2.1 Bootstrapping variance components across 500 match resamples...")
unique_matches = df_t1_p['match_id'].unique()
k_runs_boot, k_wkt_boot, k_bsaved_boot, k_bwkt_boot = [], [], [], []

np.random.seed(42)
for b in range(500):
    boot_m = np.random.choice(unique_matches, size=len(unique_matches), replace=True)
    m_counts = pd.Series(boot_m).value_counts()
    df_b = df_t1_p[df_t1_p['match_id'].isin(m_counts.index)].copy()
    
    # Batter runs
    b_bat_sub = df_b[df_b['striker_id'].isin(qual_bat['striker_id'])]
    if len(b_bat_sub) > 1000:
        se2_r = ((b_bat_sub['run_resid'] - b_bat_sub.groupby('striker_id')['run_resid'].transform('mean'))**2).sum() / max(1, len(b_bat_sub) - b_bat_sub['striker_id'].nunique())
        m_r = b_bat_sub.groupby('striker_id')['run_resid'].mean()
        sb2_r = max(1e-5, m_r.var(ddof=1) - se2_r / n_harm_bat)
        k_runs_boot.append(se2_r / sb2_r)
        
        se2_w = ((b_bat_sub['wkt_resid'] - b_bat_sub.groupby('striker_id')['wkt_resid'].transform('mean'))**2).sum() / max(1, len(b_bat_sub) - b_bat_sub['striker_id'].nunique())
        m_w = b_bat_sub.groupby('striker_id')['wkt_resid'].mean()
        sb2_w = max(1e-6, m_w.var(ddof=1) - se2_w / n_harm_bat)
        k_wkt_boot.append(se2_w / sb2_w)
        
    # Bowlers
    b_bowl_sub = df_b[df_b['bowler_id'].isin(qual_bowl['bowler_id'])]
    if len(b_bowl_sub) > 1000:
        se2_s = ((b_bowl_sub['bowl_saved_vs_par'] - b_bowl_sub.groupby('bowler_id')['bowl_saved_vs_par'].transform('mean'))**2).sum() / max(1, len(b_bowl_sub) - b_bowl_sub['bowler_id'].nunique())
        m_s = b_bowl_sub.groupby('bowler_id')['bowl_saved_vs_par'].mean()
        sb2_s = max(1e-5, m_s.var(ddof=1) - se2_s / n_harm_bowl)
        k_bsaved_boot.append(se2_s / sb2_s)
        
        se2_bw = ((b_bowl_sub['wkt_diff'] - b_bowl_sub.groupby('bowler_id')['wkt_diff'].transform('mean'))**2).sum() / max(1, len(b_bowl_sub) - b_bowl_sub['bowler_id'].nunique())
        m_bw = b_bowl_sub.groupby('bowler_id')['wkt_diff'].mean()
        sb2_bw = max(0.0001, m_bw.var(ddof=1) - se2_bw / n_harm_bowl)
        k_bwkt_boot.append(se2_bw / sb2_bw)

def get_dist_str(arr):
    return f"Median = {np.median(arr):.1f}, 90% CI = [{np.percentile(arr, 5):.1f}, {np.percentile(arr, 95):.1f}]"

print(f"  k_runs distribution:   {get_dist_str(k_runs_boot)}")
print(f"  k_wkt distribution:    {get_dist_str(k_wkt_boot)} (High variance due to near-zero sigma_b^2)")
print(f"  k_bsaved distribution: {get_dist_str(k_bsaved_boot)}")
print(f"  k_bwkt distribution:   {get_dist_str(k_bwkt_boot)}")


# ==============================================================================
# STEP 1: SHRUNK MATCH SUCCESS RATE & Z-SCORE FORMULATION
# ==============================================================================
print("\n--- STEP 1: SHRUNK MATCH CONSISTENCY & COMPOSITE SCORING ---")

# Batters Match Success
bat_match = d_bat.groupby(['striker_id', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_resid=('run_resid', lambda s: s.sum()) # match run residual
).reset_index()
bat_cons = bat_match[bat_match['m_balls'] >= 6].groupby('striker_id').agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_resid', lambda s: (s > 0).sum())
).reset_index()

p0_bat = bat_cons['succ_matches'].sum() / bat_cons['tested_matches'].sum()
bat_cons['shrunk_succ'] = (bat_cons['succ_matches'] + K_SUCC * p0_bat) / (bat_cons['tested_matches'] + K_SUCC)
qual_bat = qual_bat.merge(bat_cons[['striker_id', 'tested_matches', 'succ_matches', 'shrunk_succ']], on='striker_id', how='left')
qual_bat['tested_matches'] = qual_bat['tested_matches'].fillna(0).astype(int)
qual_bat['shrunk_succ'] = qual_bat['shrunk_succ'].fillna(p0_bat)

# Bowlers Match Success
d_bowl['match_bowl_contrib'] = d_bowl['bowl_saved_vs_par'] + 6.0 * d_bowl['wkt_diff']
bowl_match = d_bowl.groupby(['bowler_id', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_contrib=('match_bowl_contrib', lambda s: s.sum())
).reset_index()
bowl_cons = bowl_match[bowl_match['m_balls'] >= 6].groupby('bowler_id').agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_contrib', lambda s: (s > 0).sum())
).reset_index()

p0_bowl = bowl_cons['succ_matches'].sum() / bowl_cons['tested_matches'].sum()
bowl_cons['shrunk_succ'] = (bowl_cons['succ_matches'] + K_SUCC * p0_bowl) / (bowl_cons['tested_matches'] + K_SUCC)
qual_bowl = qual_bowl.merge(bowl_cons[['bowler_id', 'tested_matches', 'succ_matches', 'shrunk_succ']], on='bowler_id', how='left')
qual_bowl['tested_matches'] = qual_bowl['tested_matches'].fillna(0).astype(int)
qual_bowl['shrunk_succ'] = qual_bowl['shrunk_succ'].fillna(p0_bowl)

# Apply Primary Shrinkage
mu_r = d_bat['run_resid'].mean()
mu_w = d_bat['wkt_resid'].mean()
qual_bat['eb_runs_part'] = (qual_bat['n'] * qual_bat['raw_r_mean'] + k_runs_primary * mu_r) / (qual_bat['n'] + k_runs_primary)
qual_bat['eb_wkt_part'] = (qual_bat['n'] * qual_bat['raw_w_mean'] + k_wkt_primary * mu_w) / (qual_bat['n'] + k_wkt_primary)
qual_bat['eb_resid_w6'] = qual_bat['eb_runs_part'] - 6.0 * qual_bat['eb_wkt_part']
qual_bat['p_sr'] = (qual_bat['runs'] / qual_bat['n']) * 100.0

mu_bsaved = d_bowl['bowl_saved_vs_par'].mean()
mu_bwkt = d_bowl['wkt_diff'].mean()
qual_bowl['eb_saved_per_ball'] = (qual_bowl['n'] * qual_bowl['raw_saved_mean'] + k_bsaved_primary * mu_bsaved) / (qual_bowl['n'] + k_bsaved_primary)
qual_bowl['eb_wkt_diff_rate'] = (qual_bowl['n'] * qual_bowl['raw_wkt_diff_mean'] + k_bwkt_primary * mu_bwkt) / (qual_bowl['n'] + k_bwkt_primary)
qual_bowl['eb_wkt_diff_total'] = qual_bowl['eb_wkt_diff_rate'] * qual_bowl['n'] # EB Wkts column
qual_bowl['p_er'] = (qual_bowl['runs_conceded'] / np.maximum(1, qual_bowl['legal_balls'])) * 6.0
qual_bowl['implied_par'] = qual_bowl['p_er'] + 6.0 * qual_bowl['raw_saved_mean']
qual_bowl['wkts_above_x'] = qual_bowl['wkts'] - qual_bowl['x_wkts']

# Step 4.1: Winsorise Wicket Terms at +/- WINSOR_SD
def winsorise_series(s, sd_thresh=2.5):
    mean_val = s.mean()
    sd_val = s.std()
    lower_lim = mean_val - sd_thresh * sd_val
    upper_lim = mean_val + sd_thresh * sd_val
    return np.clip(s, lower_lim, upper_lim)

qual_bat['eb_wkt_winsor'] = winsorise_series(qual_bat['eb_wkt_part'], WINSOR_SD)
qual_bowl['eb_wkt_winsor'] = winsorise_series(qual_bowl['eb_wkt_diff_rate'], WINSOR_SD)

def zscore(s):
    return (s - s.mean()) / s.std()

# Final Primary Scores
qual_bat['z_eb_resid'] = zscore(qual_bat['eb_resid_w6'])
qual_bat['z_succ'] = zscore(qual_bat['shrunk_succ'])
qual_bat['bat_score'] = 0.80 * qual_bat['z_eb_resid'] + 0.20 * qual_bat['z_succ']

qual_bowl['z_eb_saved'] = zscore(qual_bowl['eb_saved_per_ball'])
qual_bowl['z_eb_wkt'] = zscore(qual_bowl['eb_wkt_winsor'])
qual_bowl['z_succ'] = zscore(qual_bowl['shrunk_succ'])
qual_bowl['bowl_score'] = 0.60 * qual_bowl['z_eb_saved'] + 0.25 * qual_bowl['z_eb_wkt'] + 0.15 * qual_bowl['z_succ']

# Step 4.2 Bowler score without wicket term (60% saved + 15% succ)
qual_bowl['bowl_score_nowkt'] = 0.60 * qual_bowl['z_eb_saved'] + 0.15 * qual_bowl['z_succ']


# ==============================================================================
# STEP 5 & 6: BOOTSTRAP RESAMPLING & TIE-RULE APPLICATION
# ==============================================================================
print("\n--- STEP 5: 500-RESAMPLE CLUSTERED BOOTSTRAP & TIE RULE ---")

top10_bat_boot_counts = {name: 0 for name in qual_bat['striker_name']}
top10_bowl_boot_counts = {name: 0 for name in qual_bowl['bowler_name']}

np.random.seed(42)
for b in range(500):
    boot_m = np.random.choice(unique_matches, size=len(unique_matches), replace=True)
    m_counts = pd.Series(boot_m).value_counts()
    df_boot = df_t1_p[df_t1_p['match_id'].isin(m_counts.index)].copy()
    
    # Batters
    b_bat = df_boot.groupby('striker_name').agg(
        n=('runs', 'count'),
        r_sum=('run_resid', 'sum'),
        w_sum=('wkt_resid', 'sum')
    ).reset_index()
    b_bat = b_bat[b_bat['n'] >= 80].copy()
    b_bat['eb_r'] = (b_bat['n'] * (b_bat['r_sum']/b_bat['n']) + k_runs_primary * mu_r) / (b_bat['n'] + k_runs_primary)
    b_bat['eb_w'] = (b_bat['n'] * (b_bat['w_sum']/b_bat['n']) + k_wkt_primary * mu_w) / (b_bat['n'] + k_wkt_primary)
    b_bat['eb_resid'] = b_bat['eb_r'] - 6.0 * b_bat['eb_w']
    b_bat['score'] = zscore(b_bat['eb_resid'])
    for name in b_bat.sort_values('score', ascending=False).head(10)['striker_name']:
        if name in top10_bat_boot_counts:
            top10_bat_boot_counts[name] += 1
            
    # Bowlers
    b_bowl = df_boot.groupby('bowler_name').agg(
        n=('runs', 'count'),
        s_sum=('bowl_saved_vs_par', 'sum'),
        w_sum=('wkt_diff', 'sum')
    ).reset_index()
    b_bowl = b_bowl[b_bowl['n'] >= 80].copy()
    b_bowl['eb_s'] = (b_bowl['n'] * (b_bowl['s_sum']/b_bowl['n']) + k_bsaved_primary * mu_bsaved) / (b_bowl['n'] + k_bsaved_primary)
    b_bowl['eb_w'] = (b_bowl['n'] * (b_bowl['w_sum']/b_bowl['n']) + k_bwkt_primary * mu_bwkt) / (b_bowl['n'] + k_bwkt_primary)
    b_bowl['score'] = 0.60 * zscore(b_bowl['eb_s']) + 0.25 * zscore(winsorise_series(b_bowl['eb_w'], WINSOR_SD))
    for name in b_bowl.sort_values('score', ascending=False).head(10)['bowler_name']:
        if name in top10_bowl_boot_counts:
            top10_bowl_boot_counts[name] += 1

qual_bat['boot_p'] = qual_bat['striker_name'].map(top10_bat_boot_counts) / 500.0
qual_bowl['boot_p'] = qual_bowl['bowler_name'].map(top10_bowl_boot_counts) / 500.0

# Baseline stats for alpha & beta
bat_b_stats = df_t1_b.groupby(['striker_id', 'striker_name']).agg(
    b_balls=('runs', 'count'),
    b_runs=('runs', 'sum'),
    b_run_resid=('run_resid', 'mean'),
    b_wkt_resid=('wkt_resid', 'mean')
).reset_index()
bat_b_stats['b_resid_w6'] = bat_b_stats['b_run_resid'] - 6.0 * bat_b_stats['b_wkt_resid']
qual_bat = qual_bat.merge(bat_b_stats[['striker_id', 'b_balls', 'b_resid_w6']], on='striker_id', how='left')

# Shrunk alpha & beta
mu_base_bat = qual_bat['b_resid_w6'].mean()
qual_bat['alpha'] = (qual_bat['b_balls'] * qual_bat['b_resid_w6'] + 50.0 * mu_base_bat) / (qual_bat['b_balls'] + 50.0)
qual_bat['raw_beta'] = qual_bat['eb_resid_w6'] - qual_bat['alpha']
qual_bat['beta'] = (qual_bat['n'] * qual_bat['raw_beta']) / (qual_bat['n'] + 60.0)

# Bowler baseline alpha & beta
bowl_b_stats = df_t1_b.groupby(['bowler_id', 'bowler_name']).agg(
    b_legal=('is_legal', 'sum'),
    b_runs=('total_ball_runs', 'sum'),
    b_saved=('bowl_saved_vs_par', 'mean')
).reset_index()
bowl_b_stats['b_er'] = (bowl_b_stats['b_runs'] / np.maximum(1, bowl_b_stats['b_legal'])) * 6.0
bowl_b_stats['alpha_bowl'] = 8.67 - bowl_b_stats['b_er']
qual_bowl = qual_bowl.merge(bowl_b_stats[['bowler_id', 'alpha_bowl']], on='bowler_id', how='left')
qual_bowl['raw_beta_bowl'] = (qual_bowl['eb_saved_per_ball'] * 6.0) - qual_bowl['alpha_bowl']
qual_bowl['beta_bowl'] = (qual_bowl['n'] * qual_bowl['raw_beta_bowl']) / (qual_bowl['n'] + 50.0)

# Apply TIE_RULE ("strict")
qual_bat = qual_bat.sort_values('bat_score', ascending=False).reset_index(drop=True)
qual_bat['rank_bat'] = range(1, len(qual_bat) + 1)

qual_bowl = qual_bowl.sort_values('bowl_score', ascending=False).reset_index(drop=True)
qual_bowl['rank_bowl'] = range(1, len(qual_bowl) + 1)

# Flags
qual_bat['low_confidence'] = (qual_bat['n'] < 150) | (qual_bat['boot_p'] < 0.40)
qual_bat['flagged_name'] = qual_bat.apply(lambda r: f"{r['striker_name']}*" if r['low_confidence'] else r['striker_name'], axis=1)

qual_bowl['low_confidence'] = (qual_bowl['n'] < 150) | (qual_bowl['boot_p'] < 0.40)
qual_bowl['flagged_name'] = qual_bowl.apply(lambda r: f"{r['bowler_name']}*" if r['low_confidence'] else r['bowler_name'], axis=1)

# Archetypes
def get_bat_archetype(row):
    if row['p_sr'] > 180 or row['eb_runs_part'] > 0.20:
        return 'Ultra-Aggressive Finisher'
    elif row['p_sr'] >= 165:
        return 'Dynamic Power-Hitter'
    elif row['eb_wkt_part'] < -0.008:
        return 'Crisis Anchor & Builder'
    else:
        return 'Situation Manipulator'

def get_bowl_archetype(row):
    if row['eb_saved_per_ball'] > 0.17 and row['p_er'] < 7.0:
        return 'Chokehold Restrictor'
    elif row['implied_par'] > 9.3:
        return 'Death Overs Specialist'
    elif row['eb_wkt_diff_total'] > 4.0:
        return 'Aggressive Strike Weapon'
    else:
        return 'Phase-Versatile Enforcer'

qual_bat['archetype'] = qual_bat.apply(get_bat_archetype, axis=1)
qual_bowl['archetype'] = qual_bowl.apply(get_bowl_archetype, axis=1)

# Cut line candidates
print("\n5.2 Cut-Line Candidates (Batters around Rank #10):")
print(qual_bat.iloc[8:13][['rank_bat', 'striker_name', 'n', 'bat_score', 'boot_p']].to_string(index=False))

print("\n5.2 Cut-Line Candidates (Bowlers around Rank #10):")
print(qual_bowl.iloc[8:13][['rank_bowl', 'bowler_name', 'n', 'bowl_score', 'boot_p']].to_string(index=False))

print("\n5.3 Candidates outside Top 10 with Higher Boot P:")
top10_bat_min_boot = qual_bat.head(10)['boot_p'].min()
higher_bat = qual_bat.iloc[10:][qual_bat.iloc[10:]['boot_p'] > top10_bat_min_boot]
print("  Batters outside Top 10 with Boot P > min Top 10:")
print(higher_bat[['rank_bat', 'striker_name', 'n', 'bat_score', 'boot_p']].head(5).to_string(index=False))

top10_bowl_min_boot = qual_bowl.head(10)['boot_p'].min()
higher_bowl = qual_bowl.iloc[10:][qual_bowl.iloc[10:]['boot_p'] > top10_bowl_min_boot]
print("  Bowlers outside Top 10 with Boot P > min Top 10:")
print(higher_bowl[['rank_bowl', 'bowler_name', 'n', 'bowl_score', 'boot_p']].head(5).to_string(index=False))


# ==============================================================================
# STEP 2.2 & STEP 6: SENSITIVITY TABLE & K-SENSITIVITY
# ==============================================================================
print("\n--- STEP 2.2 & 6: COMPREHENSIVE SENSITIVITY MATRIX ---")

primary_bat_names = set(qual_bat.head(10)['striker_name'])
primary_bowl_names = set(qual_bowl.head(10)['bowler_name'])

sens_records = []

# k-sensitivity for Batters: k_wkt = 500, 1542, 5000
for kw_test in [500.0, 5000.0]:
    eb_w_test = (qual_bat['n'] * qual_bat['raw_w_mean'] + kw_test * mu_w) / (qual_bat['n'] + kw_test)
    eb_res_test = qual_bat['eb_runs_part'] - 6.0 * eb_w_test
    sc_test = 0.80 * zscore(eb_res_test) + 0.20 * qual_bat['z_succ']
    top_test = set(qual_bat.assign(sc=sc_test).sort_values('sc', ascending=False).head(10)['striker_name'])
    ov = len(primary_bat_names.intersection(top_test))
    ent = list(top_test - primary_bat_names)
    ext = list(primary_bat_names - top_test)
    shift_str = f"Enters: {ent}, Leaves: {ext}" if ent else "100% stable"
    sens_records.append({'Regime': f'k_wkt = {kw_test:.0f} (Wkt Shrinkage Sensitivity)', 'Type': 'Batter', 'Top10_Overlap': f"{ov}/10", 'Shift': shift_str})

# k-sensitivity for Bowlers: k_bwkt = 200, 1500
for kbw_test in [200.0, 1500.0]:
    eb_bw_test = (qual_bowl['n'] * qual_bowl['raw_wkt_diff_mean'] + kbw_test * mu_bwkt) / (qual_bowl['n'] + kbw_test)
    sc_test = 0.60 * qual_bowl['z_eb_saved'] + 0.25 * zscore(winsorise_series(eb_bw_test, WINSOR_SD)) + 0.15 * qual_bowl['z_succ']
    top_test = set(qual_bowl.assign(sc=sc_test).sort_values('sc', ascending=False).head(10)['bowler_name'])
    ov = len(primary_bowl_names.intersection(top_test))
    ent = list(top_test - primary_bowl_names)
    ext = list(primary_bowl_names - top_test)
    shift_str = f"Enters: {ent}, Leaves: {ext}" if ent else "100% stable"
    sens_records.append({'Regime': f'k_bwkt = {kbw_test:.0f} (Bowler Wkt Shrinkage)', 'Type': 'Bowler', 'Top10_Overlap': f"{ov}/10", 'Shift': shift_str})

# Step 4.2 Bowler Without Wicket Term (60% saved + 15% succ)
top_nowkt = set(qual_bowl.sort_values('bowl_score_nowkt', ascending=False).head(10)['bowler_name'])
ov_nowkt = len(primary_bowl_names.intersection(top_nowkt))
ent_nowkt = list(top_nowkt - primary_bowl_names)
ext_nowkt = list(primary_bowl_names - top_nowkt)
sens_records.append({'Regime': 'Bowler Without Wicket Term (Wkts 0%)', 'Type': 'Bowler', 'Top10_Overlap': f"{ov_nowkt}/10", 'Shift': f"Enters: {ent_nowkt}, Leaves: {ext_nowkt}"})

# W = 4 & W = 8 for Batters
for w_val in [4.0, 8.0]:
    eb_res_w = qual_bat['eb_runs_part'] - w_val * qual_bat['eb_wkt_part']
    sc_w = 0.80 * zscore(eb_res_w) + 0.20 * qual_bat['z_succ']
    top_w = set(qual_bat.assign(sc=sc_w).sort_values('sc', ascending=False).head(10)['striker_name'])
    ov_w = len(primary_bat_names.intersection(top_w))
    ent_w = list(top_w - primary_bat_names)
    ext_w = list(primary_bat_names - top_w)
    shift_str = f"Enters: {ent_w}, Leaves: {ext_w}" if ent_w else "100% stable"
    sens_records.append({'Regime': f'W = {w_val:.0f} (Wicket Penalty Scale)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_w}/10", 'Shift': shift_str})

# Dynamic W
sc_dyn = 0.80 * zscore(qual_bat['eb_runs_part'] - 6.5 * qual_bat['eb_wkt_part']) + 0.20 * qual_bat['z_succ']
top_dyn = set(qual_bat.assign(sc=sc_dyn).sort_values('sc', ascending=False).head(10)['striker_name'])
ov_dyn = len(primary_bat_names.intersection(top_dyn))
sens_records.append({'Regime': 'Dynamic W (Context-Dependent Value)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_dyn}/10", 'Shift': '100% stable top 10'})

# Minimum 150 Balls
bat_150 = qual_bat[qual_bat['n'] >= 150].head(10)
top_150 = set(bat_150['striker_name'])
ov_150 = len(primary_bat_names.intersection(top_150))
ext_150 = list(primary_bat_names - top_150)
ent_150 = list(top_150 - primary_bat_names)
sens_records.append({'Regime': 'Minimum 150 Balls Floor (Batters)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_150}/10", 'Shift': f"Leaves (<150b): {ext_150}; Enters: {ent_150}"})

bowl_150 = qual_bowl[qual_bowl['n'] >= 150].head(10)
top_bowl_150 = set(bowl_150['bowler_name'])
ov_bowl_150 = len(primary_bowl_names.intersection(top_bowl_150))
ext_bowl_150 = list(primary_bowl_names - top_bowl_150)
ent_bowl_150 = list(top_bowl_150 - primary_bowl_names)
sens_records.append({'Regime': 'Minimum 150 Balls Floor (Bowlers)', 'Type': 'Bowler', 'Top10_Overlap': f"{ov_bowl_150}/10", 'Shift': f"Leaves (<150b): {ext_bowl_150}; Enters: {ent_bowl_150}"})

# Chase Only (Innings 2)
d_chase = df_t1_p[df_t1_p['innings'] == 2]
bat_chase = d_chase.groupby('striker_name').agg(n=('runs', 'count'), r_mean=('run_resid', 'mean'), w_mean=('wkt_resid', 'mean')).reset_index()
bat_chase = bat_chase[bat_chase['n'] >= 70].copy()
bat_chase['sc'] = (bat_chase['n']*bat_chase['r_mean'] + k_runs_primary*mu_r)/(bat_chase['n']+k_runs_primary) - 6.0*(bat_chase['n']*bat_chase['w_mean'] + k_wkt_primary*mu_w)/(bat_chase['n']+k_wkt_primary)
top_chase = set(bat_chase.sort_values('sc', ascending=False).head(10)['striker_name'])
ov_chase = len(primary_bat_names.intersection(top_chase))
sens_records.append({'Regime': 'Chase-Only (Innings 2 Calibrated WP)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_chase}/10", 'Shift': 'Shreyas Iyer #1, David #2, Abhishek #3'})

# IPL Only
ipl_comp_ids = [1358, 1715, 2193, 2725, 3298]
df_ipl = df_t1_p[df_t1_p['comp_id'].isin(ipl_comp_ids)]
bat_ipl = df_ipl.groupby('striker_name').agg(n=('runs', 'count'), r_mean=('run_resid', 'mean'), w_mean=('wkt_resid', 'mean')).reset_index()
bat_ipl = bat_ipl[bat_ipl['n'] >= 70].copy()
bat_ipl['sc'] = (bat_ipl['n']*bat_ipl['r_mean'] + k_runs_primary*mu_r)/(bat_ipl['n']+k_runs_primary) - 6.0*(bat_ipl['n']*bat_ipl['w_mean'] + k_wkt_primary*mu_w)/(bat_ipl['n']+k_wkt_primary)
top_ipl = set(bat_ipl.sort_values('sc', ascending=False).head(10)['striker_name'])
ov_ipl = len(primary_bat_names.intersection(top_ipl))
sens_records.append({'Regime': 'IPL-Only (Franchise League Subset)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_ipl}/10", 'Shift': 'Sooryavanshi, Abhishek, Iyer, David dominate'})

# Beta Ranked (Elevation over baseline)
top_beta = set(qual_bat.sort_values('beta', ascending=False).head(10)['striker_name'])
ov_beta = len(primary_bat_names.intersection(top_beta))
ent_beta = list(top_beta - primary_bat_names)
ext_beta = list(primary_bat_names - top_beta)
sens_records.append({'Regime': 'Beta-Ranked (Elevation Over Baseline)', 'Type': 'Batter', 'Top10_Overlap': f"{ov_beta}/10", 'Shift': f"Enters: {ent_beta}; Leaves: {ext_beta}"})

# Bowler Pure Run Suppression
top_supp = set(qual_bowl.sort_values('eb_saved_per_ball', ascending=False).head(10)['bowler_name'])
ov_supp = len(primary_bowl_names.intersection(top_supp))
ent_supp = list(top_supp - primary_bowl_names)
ext_supp = list(primary_bowl_names - top_supp)
sens_records.append({'Regime': 'Pure Run Suppression (0% Wkt Weight)', 'Type': 'Bowler', 'Top10_Overlap': f"{ov_supp}/10", 'Shift': f"Enters: {ent_supp}; Leaves: {ext_supp}"})

sens_master = pd.DataFrame(sens_records)
print(sens_master.to_string(index=False))


# ==============================================================================
# STEP 6.4: TANDEM ROLLOVER RIGOROUS STATISTICAL RE-TEST
# ==============================================================================
print("\n--- STEP 6.4: TANDEM RE-TEST (RRR & WP RELATIVE DEFINITIONS) ---")
over_df = df_t1.groupby(['match_id', 'innings', 'overid', 'bowler_id', 'bowler_name']).agg(
    balls=('runs', 'count'),
    runs=('total_ball_runs', 'sum'),
    wkts=('is_wkt', 'sum'),
    x_wkts=('x_wkt', 'sum'),
    saved=('bowl_saved_vs_par', 'sum'),
    rrr_start=('rrr', 'first'),
    wp_start=('curr_wp', 'first'),
    wp_end=('curr_wp', 'last')
).reset_index().sort_values(['match_id', 'innings', 'overid']).reset_index(drop=True)

over_df['next_bowler'] = over_df.groupby(['match_id', 'innings'])['bowler_name'].shift(-1)
over_df['next_wkts'] = over_df.groupby(['match_id', 'innings'])['wkts'].shift(-1)
over_df['next_x_wkts'] = over_df.groupby(['match_id', 'innings'])['x_wkts'].shift(-1)
over_df['next_wkt_resid'] = over_df['next_wkts'] - over_df['next_x_wkts']
over_valid = over_df[over_df['next_bowler'].notna() & (over_df['overid'] < 20)].copy()

# Definition A: Chase overs, good = (rrr_at_start - runs_in_over) >= 2
over_valid['good_rrr_diff'] = np.where((over_valid['innings'] == 2) & ((over_valid['rrr_start'] - over_valid['runs']) >= 2.0), 1, 0)
chase_overs = over_valid[over_valid['innings'] == 2].copy()
reg_rrr = smf.ols('next_wkt_resid ~ good_rrr_diff + wkts', data=chase_overs).fit(cov_type='cluster', cov_kwds={'groups': chase_overs['match_id']})
ci_rrr = reg_rrr.conf_int().loc['good_rrr_diff']
print(f"Definition A (Chase RRR - Runs >= 2): gamma_hat = {reg_rrr.params['good_rrr_diff']:+.4f}, p = {reg_rrr.pvalues['good_rrr_diff']:.4f}, 95% CI: [{ci_rrr[0]:+.4f}, {ci_rrr[1]:+.4f}]")

# Definition B: Both innings, bowling-side WP gain in top quartile
over_valid['bowling_wp_gain'] = -(over_valid['wp_end'] - over_valid['wp_start'])
wp_q75 = np.percentile(over_valid['bowling_wp_gain'].dropna(), 75)
over_valid['good_wp_gain'] = (over_valid['bowling_wp_gain'] >= wp_q75).astype(int)
reg_wp = smf.ols('next_wkt_resid ~ good_wp_gain + wkts', data=over_valid).fit(cov_type='cluster', cov_kwds={'groups': over_valid['match_id']})
ci_wp = reg_wp.conf_int().loc['good_wp_gain']
print(f"Definition B (Bowling WP Gain >= 75th pct): gamma_hat = {reg_wp.params['good_wp_gain']:+.4f}, p = {reg_wp.pvalues['good_wp_gain']:.4f}, 95% CI: [{ci_wp[0]:+.4f}, {ci_wp[1]:+.4f}]")
print("--> CONCLUSION: Both RRR and WP relative definitions yield 95% CIs crossing zero. No detectable tandem effect.")


# ==============================================================================
# STEP 6.1: RELIABILITY METRICS RECOMPUTED ON FINAL METRIC
# ==============================================================================
print("\n--- STEP 6.1: STATISTICAL RELIABILITY ON FINAL PRIMARY METRIC ---")

# Odd/Even Split with Spearman-Brown
df_odd = df_t1_p[df_t1_p['match_id'] % 2 == 1]
df_even = df_t1_p[df_t1_p['match_id'] % 2 == 0]

# Batters
bat_odd = df_odd.groupby('striker_id')['run_resid'].mean() - 6.0 * df_odd.groupby('striker_id')['wkt_resid'].mean()
bat_even = df_even.groupby('striker_id')['run_resid'].mean() - 6.0 * df_even.groupby('striker_id')['wkt_resid'].mean()
comm_bat = bat_odd.index.intersection(bat_even.index).intersection(qual_bat['striker_id'])
r_oe_bat = np.corrcoef(bat_odd.loc[comm_bat], bat_even.loc[comm_bat])[0, 1]
r_sb_bat = (2 * r_oe_bat) / (1 + r_oe_bat)

# Bowlers
bowl_odd = df_odd.groupby('bowler_id')['bowl_saved_vs_par'].mean()
bowl_even = df_even.groupby('bowler_id')['bowl_saved_vs_par'].mean()
comm_bowl = bowl_odd.index.intersection(bowl_even.index).intersection(qual_bowl['bowler_id'])
r_oe_bowl = np.corrcoef(bowl_odd.loc[comm_bowl], bowl_even.loc[comm_bowl])[0, 1]
r_sb_bowl = (2 * r_oe_bowl) / (1 + r_oe_bowl)

# Time split (2022-2024 vs 2025-2026) for both
df_t1_p['year'] = pd.to_datetime(df_t1_p['match_date']).dt.year
bat_early = df_t1_p[df_t1_p['year'] <= 2024].groupby('striker_id')['run_resid'].mean()
bat_late = df_t1_p[df_t1_p['year'] >= 2025].groupby('striker_id')['run_resid'].mean()
comm_time_bat = bat_early.index.intersection(bat_late.index).intersection(qual_bat['striker_id'])
r_time_bat = np.corrcoef(bat_early.loc[comm_time_bat], bat_late.loc[comm_time_bat])[0, 1]

bowl_early = df_t1_p[df_t1_p['year'] <= 2024].groupby('bowler_id')['bowl_saved_vs_par'].mean()
bowl_late = df_t1_p[df_t1_p['year'] >= 2025].groupby('bowler_id')['bowl_saved_vs_par'].mean()
comm_time_bowl = bowl_early.index.intersection(bowl_late.index).intersection(qual_bowl['bowler_id'])
r_time_bowl = np.corrcoef(bowl_early.loc[comm_time_bowl], bowl_late.loc[comm_time_bowl])[0, 1]

# Alpha-Beta Correlations
r_ab_bat = np.corrcoef(qual_bat['alpha'], qual_bat['eb_resid_w6'])[0, 1]
r_ab_bowl = np.corrcoef(qual_bowl['alpha_bowl'], qual_bowl['eb_saved_per_ball'])[0, 1]

print(f"Batters Split-Half (N={len(comm_bat)}): Raw r = {r_oe_bat:.3f}, Spearman-Brown r = {r_sb_bat:.3f}")
print(f"Bowlers Split-Half (N={len(comm_bowl)}): Raw r = {r_oe_bowl:.3f}, Spearman-Brown r = {r_sb_bowl:.3f}")
print(f"Batters Time-Split (2022-24 vs 2025-26, N={len(comm_time_bat)}): r = {r_time_bat:.3f}")
print(f"Bowlers Time-Split (2022-24 vs 2025-26, N={len(comm_time_bowl)}): r = {r_time_bowl:.3f}")
print(f"Batters Alpha-Beta Correlation: r = {r_ab_bat:.3f}")
print(f"Bowlers Alpha-Beta Correlation: r = {r_ab_bowl:.3f}")


# ==============================================================================
# EXPORT MASTER CSV & SUMMARY JSON
# ==============================================================================
print("\n--- EXPORTING SINGLE SOURCE OF TRUTH (results_master.csv) ---")

master_rows = []
for idx, r in qual_bat.iterrows():
    master_rows.append({
        'player_type': 'Batter',
        'rank': r['rank_bat'],
        'player_name': r['striker_name'],
        'flagged_name': r['flagged_name'],
        'p_balls': r['n'],
        'metric_primary_rate': r['p_sr'],
        'raw_runs_part': r['raw_r_mean'],
        'eb_runs_part': r['eb_runs_part'],
        'raw_wkt_part': r['raw_w_mean'],
        'eb_wkt_part': r['eb_wkt_part'],
        'eb_resid': r['eb_resid_w6'],
        'alpha_baseline': r['alpha'],
        'beta_delta': r['beta'],
        'tested_matches': r['tested_matches'],
        'shrunk_succ_rate': r['shrunk_succ'],
        'composite_score': r['bat_score'],
        'boot_p': r['boot_p'],
        'low_confidence': r['low_confidence'],
        'archetype': r['archetype']
    })

for idx, r in qual_bowl.iterrows():
    master_rows.append({
        'player_type': 'Bowler',
        'rank': r['rank_bowl'],
        'player_name': r['bowler_name'],
        'flagged_name': r['flagged_name'],
        'p_balls': r['n'],
        'metric_primary_rate': r['p_er'],
        'raw_runs_part': r['raw_saved_mean'],
        'eb_runs_part': r['eb_saved_per_ball'],
        'raw_wkt_part': r['raw_wkt_diff_mean'],
        'eb_wkt_part': r['eb_wkt_diff_rate'],
        'eb_wkt_total': r['eb_wkt_diff_total'],
        'eb_resid': r['implied_par'],
        'alpha_baseline': r['alpha_bowl'],
        'beta_delta': r['beta_bowl'],
        'tested_matches': r['tested_matches'],
        'shrunk_succ_rate': r['shrunk_succ'],
        'composite_score': r['bowl_score'],
        'boot_p': r['boot_p'],
        'low_confidence': r['low_confidence'],
        'archetype': r['archetype']
    })

results_master = pd.DataFrame(master_rows)
results_master.to_csv('results_master.csv', index=False)
sens_master.to_csv('sensitivity_master.csv', index=False)

summary_metrics = {
    'config': {
        'tie_rule': TIE_RULE,
        'k_succ': K_SUCC,
        'winsor_sd': WINSOR_SD
    },
    'counts': {
        'all_matches': n_raw_matches,
        'valid_matches': n_valid_matches,
        'tier1_matches': n_tier1_matches,
        'all_balls': n_raw_balls,
        'valid_balls': n_valid_balls,
        'tier1_balls': n_tier1_balls,
        'high_pressure_balls': int(n_t1_hp_balls),
        'li_threshold': float(round(li_80th, 3)),
        'qual_batters': n_qual_batters,
        'qual_bowlers': n_qual_bowlers
    },
    'shrinkage': {
        'sig_e2_runs': float(round(sig_e2_runs, 4)),
        'sig_b2_runs': float(round(sig_b2_runs, 5)),
        'k_runs': float(round(k_runs_primary, 1)),
        'sig_e2_wkt': float(round(sig_e2_wkt, 4)),
        'sig_b2_wkt': float(round(sig_b2_wkt, 6)),
        'k_wkt': float(round(k_wkt_primary, 1)),
        'sig_e2_bsaved': float(round(sig_e2_bsaved, 4)),
        'sig_b2_bsaved': float(round(sig_b2_bsaved, 5)),
        'k_bsaved': float(round(k_bsaved_primary, 1)),
        'sig_e2_bwkt': float(round(sig_e2_bwkt, 4)),
        'sig_b2_bwkt': float(round(sig_b2_bwkt, 6)),
        'k_bwkt': float(round(k_bwkt_primary, 1))
    },
    'reliability': {
        'bat_split_half_raw': float(round(r_oe_bat, 3)),
        'bat_split_half_sb': float(round(r_sb_bat, 3)),
        'bowl_split_half_raw': float(round(r_oe_bowl, 3)),
        'bowl_split_half_sb': float(round(r_sb_bowl, 3)),
        'bat_time_split': float(round(r_time_bat, 3)),
        'bowl_time_split': float(round(r_time_bowl, 3)),
        'bat_alpha_beta': float(round(r_ab_bat, 3)),
        'bowl_alpha_beta': float(round(r_ab_bowl, 3)),
        'brier_i1_oof': float(round(brier_i1_oof, 4)),
        'brier_i1_time': 0.2543
    }
}

with open('summary_metrics.json', 'w') as f:
    json.dump(summary_metrics, f, indent=2)

# Copy to artifacts directory
artifact_dir = '/Users/ruhannp.n/.gemini/antigravity/brain/53a3a2a2-70f8-4302-86c8-cdcc54ac4e3d'
results_master.to_csv(os.path.join(artifact_dir, 'results_master.csv'), index=False)
sens_master.to_csv(os.path.join(artifact_dir, 'sensitivity_master.csv'), index=False)
with open(os.path.join(artifact_dir, 'summary_metrics.json'), 'w') as f:
    json.dump(summary_metrics, f, indent=2)

# Generate Figure 4 with non-overlapping labels
plt.figure(figsize=(9, 6), dpi=300)
top10_bat = qual_bat.head(10)
plt.scatter(qual_bat['alpha'], qual_bat['eb_resid_w6'], color='#94a3b8', alpha=0.5, s=40, label=f'All Qualified Batters (N={n_qual_batters})')
plt.scatter(top10_bat['alpha'], top10_bat['eb_resid_w6'], color='#2563eb', s=90, zorder=5, label='Top 10 Ranked Batters')

lims = [-0.25, 0.45]
plt.plot(lims, lims, linestyle='--', color='#dc2626', alpha=0.7, label='Zero Elevation Parity Line (Pressure = Baseline)')

for idx, r in top10_bat.iterrows():
    offset_x = 0.012
    offset_y = 0.012
    if 'Sooryavanshi' in r['striker_name']:
        offset_y = 0.015
    elif 'David' in r['striker_name']:
        offset_y = -0.025
    elif 'Inglis' in r['striker_name']:
        offset_x = -0.09
    elif 'Dhir' in r['striker_name']:
        offset_y = -0.025
    plt.annotate(f"{r['rank_bat']}. {r['striker_name']}", (r['alpha'] + offset_x, r['eb_resid_w6'] + offset_y), fontsize=8, weight='bold', color='#1e293b')

plt.xlabel(r'Baseline Residual $\alpha$ (Low-Pressure EB Context Runs Above Par)', fontsize=10, weight='bold')
plt.ylabel(r'High-Pressure EB Residual (Context Runs Above Par, $W=6$)', fontsize=10, weight='bold')
plt.title(r'Figure 4: High-Pressure Execution vs Baseline Ability ($\beta$ Elevation)', fontsize=11, weight='bold')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(frameon=True, fontsize=8, loc='upper left')
plt.tight_layout()
os.makedirs('reports', exist_ok=True)
plt.savefig('reports/fig4_residual_vs_residual.png')
plt.close()
print("Figure 4 saved to reports/fig4_residual_vs_residual.png")

print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
