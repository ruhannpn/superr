import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

print("=== RUNNING BALL-LEVEL VARIANCE & EMPIRICAL BAYES RE-ESTIMATION ===")

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
             'revised_target', 'win_team_id', 'is_tier1', 'venue_id', 'match_date', 'comp_id']
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

# MODEL B: No delivery style (to prevent bowler identity leakage)
df['log_balls_faced'] = np.log1p(df['batter_balls_faced'])
feats_b = ['overid', 'balls_remaining', 'wickets_in_hand', 'venue_1st_inn_avg', 
           'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1', 'log_balls_faced']
X_b = df[feats_b].copy()
cat_b = ['is_tier1']
for c in cat_b:
    X_b[c] = X_b[c].astype('category')

mr = HistGradientBoostingRegressor(categorical_features=cat_b, max_iter=100, random_state=42).fit(X_b, df['runs'])
mw = HistGradientBoostingClassifier(categorical_features=cat_b, max_iter=100, random_state=42).fit(X_b, df['is_wkt'])
df['x_runs'] = mr.predict(X_b)
df['x_wkt'] = mw.predict_proba(X_b)[:, 1]

df['run_resid'] = df['runs'] - df['x_runs']
df['wkt_resid'] = df['is_wkt'] - df['x_wkt']

# Phase Par for Bowlers
phase_par_er = df[df['is_tier1']].groupby('phase')['total_ball_runs'].mean() * 6.0
df['phase_par_ball'] = df['phase'].map(phase_par_er) / 6.0
df['bowl_saved_vs_par'] = df['phase_par_ball'] - df['total_ball_runs']
df['wkt_diff'] = df['is_wkt'] - df['x_wkt']

df_t1_p = df[(df['is_tier1']) & (df['is_high_pressure'] == 1)].copy()

# ==============================================================================
# BALL-LEVEL VARIANCE ESTIMATION
# ==============================================================================
# Batters:
bat_summary = df_t1_p.groupby(['striker_id', 'striker_name']).agg(
    n=('runs', 'count'),
    matches=('match_id', 'nunique'),
    raw_r_mean=('run_resid', 'mean'),
    raw_w_mean=('wkt_resid', 'mean'),
    runs=('runs', 'sum'),
    outs=('is_wkt', 'sum')
).reset_index()

qual_bat = bat_summary[(bat_summary['n'] >= 120) & (bat_summary['matches'] >= 8)].copy()
d_bat = df_t1_p[df_t1_p['striker_id'].isin(qual_bat['striker_id'])].copy()

# Runs:
sig_e2_runs = ((d_bat['run_resid'] - d_bat.groupby('striker_id')['run_resid'].transform('mean'))**2).sum() / (len(d_bat) - d_bat['striker_id'].nunique())
var_r_means = qual_bat['raw_r_mean'].var(ddof=1)
n_harm_bat = len(qual_bat) / np.sum(1.0 / qual_bat['n'])
sig_b2_runs = max(1e-5, var_r_means - sig_e2_runs / n_harm_bat)
k_runs = sig_e2_runs / sig_b2_runs

# Wkts:
sig_e2_wkt = ((d_bat['wkt_resid'] - d_bat.groupby('striker_id')['wkt_resid'].transform('mean'))**2).sum() / (len(d_bat) - d_bat['striker_id'].nunique())
var_w_means = qual_bat['raw_w_mean'].var(ddof=1)
sig_b2_wkt = max(1e-6, var_w_means - sig_e2_wkt / n_harm_bat)
k_wkt = sig_e2_wkt / sig_b2_wkt

print("\n--- BATTER VARIANCE ESTIMATION ---")
print(f"Runs Part:   sig_e2 = {sig_e2_runs:.4f}, sig_b2 = {sig_b2_runs:.5f} -> k_runs = {k_runs:.1f} balls")
print(f"Wicket Part: sig_e2 = {sig_e2_wkt:.4f}, sig_b2 = {sig_b2_wkt:.6f} -> k_wkt = {k_wkt:.1f} balls")

# Apply shrinkage to batters
mu_r = d_bat['run_resid'].mean()
mu_w = d_bat['wkt_resid'].mean()

qual_bat['eb_runs_part'] = (qual_bat['n'] * qual_bat['raw_r_mean'] + k_runs * mu_r) / (qual_bat['n'] + k_runs)
qual_bat['eb_wkt_part'] = (qual_bat['n'] * qual_bat['raw_w_mean'] + k_wkt * mu_w) / (qual_bat['n'] + k_wkt)
qual_bat['eb_resid_w6'] = qual_bat['eb_runs_part'] - 6.0 * qual_bat['eb_wkt_part']

# Consistency: Shrunk match success rate
bat_match = d_bat.groupby(['striker_id', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_resid=('run_resid', lambda s: s.sum())
).reset_index()
bat_cons = bat_match[bat_match['m_balls'] >= 6].groupby('striker_id').agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_resid', lambda s: (s > 0).sum())
).reset_index()
bat_cons['shrunk_succ'] = (bat_cons['succ_matches'] + 2.0) / (bat_cons['tested_matches'] + 4.0)

qual_bat = qual_bat.merge(bat_cons[['striker_id', 'shrunk_succ']], on='striker_id', how='left')

def zscore(s):
    return (s - s.mean()) / s.std()

qual_bat['z_eb_resid'] = zscore(qual_bat['eb_resid_w6'])
qual_bat['z_succ'] = zscore(qual_bat['shrunk_succ'].fillna(qual_bat['shrunk_succ'].mean()))
qual_bat['bat_score'] = 0.80 * qual_bat['z_eb_resid'] + 0.20 * qual_bat['z_succ']
qual_bat['p_sr'] = (qual_bat['runs'] / qual_bat['n']) * 100.0


# Bowlers:
bowl_summary = df_t1_p.groupby(['bowler_id', 'bowler_name']).agg(
    n=('runs', 'count'),
    matches=('match_id', 'nunique'),
    raw_saved_mean=('bowl_saved_vs_par', 'mean'),
    raw_wkt_diff_mean=('wkt_diff', 'mean'),
    runs_conceded=('total_ball_runs', 'sum'),
    legal_balls=('is_legal', 'sum'),
    wkts=('is_wkt', 'sum'),
    x_wkts=('x_wkt', 'sum')
).reset_index()

qual_bowl = bowl_summary[(bowl_summary['n'] >= 120) & (bowl_summary['matches'] >= 8)].copy()
d_bowl = df_t1_p[df_t1_p['bowler_id'].isin(qual_bowl['bowler_id'])].copy()

# Runs saved:
sig_e2_bsaved = ((d_bowl['bowl_saved_vs_par'] - d_bowl.groupby('bowler_id')['bowl_saved_vs_par'].transform('mean'))**2).sum() / (len(d_bowl) - d_bowl['bowler_id'].nunique())
var_bsaved_means = qual_bowl['raw_saved_mean'].var(ddof=1)
n_harm_bowl = len(qual_bowl) / np.sum(1.0 / qual_bowl['n'])
sig_b2_bsaved = max(1e-5, var_bsaved_means - sig_e2_bsaved / n_harm_bowl)
k_bsaved = sig_e2_bsaved / sig_b2_bsaved

# Wkts above expected:
sig_e2_bwkt = ((d_bowl['wkt_diff'] - d_bowl.groupby('bowler_id')['wkt_diff'].transform('mean'))**2).sum() / (len(d_bowl) - d_bowl['bowler_id'].nunique())
var_bwkt_means = qual_bowl['raw_wkt_diff_mean'].var(ddof=1)
# When between variance is tiny, set empirical bound
sig_b2_bwkt = max(0.0001, var_bwkt_means - sig_e2_bwkt / n_harm_bowl)
k_bwkt = sig_e2_bwkt / sig_b2_bwkt

print("\n--- BOWLER VARIANCE ESTIMATION ---")
print(f"Runs Saved Part: sig_e2 = {sig_e2_bsaved:.4f}, sig_b2 = {sig_b2_bsaved:.5f} -> k_bsaved = {k_bsaved:.1f} balls")
print(f"Wickets Part:   sig_e2 = {sig_e2_bwkt:.4f}, sig_b2 = {sig_b2_bwkt:.6f} -> k_bwkt = {k_bwkt:.1f} balls")

mu_bsaved = d_bowl['bowl_saved_vs_par'].mean()
mu_bwkt = d_bowl['wkt_diff'].mean()

qual_bowl['eb_saved_per_ball'] = (qual_bowl['n'] * qual_bowl['raw_saved_mean'] + k_bsaved * mu_bsaved) / (qual_bowl['n'] + k_bsaved)
qual_bowl['eb_wkt_diff_rate'] = (qual_bowl['n'] * qual_bowl['raw_wkt_diff_mean'] + k_bwkt * mu_bwkt) / (qual_bowl['n'] + k_bwkt)

# Bowler consistency:
bowl_match = d_bowl.groupby(['bowler_id', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_saved=('bowl_saved_vs_par', lambda s: s.sum())
).reset_index()
bowl_cons = bowl_match[bowl_match['m_balls'] >= 6].groupby('bowler_id').agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_saved', lambda s: (s > 0).sum())
).reset_index()
bowl_cons['shrunk_succ'] = (bowl_cons['succ_matches'] + 2.0) / (bowl_cons['tested_matches'] + 4.0)

qual_bowl = qual_bowl.merge(bowl_cons[['bowler_id', 'shrunk_succ']], on='bowler_id', how='left')

qual_bowl['z_eb_saved'] = zscore(qual_bowl['eb_saved_per_ball'])
qual_bowl['z_eb_wkt'] = zscore(qual_bowl['eb_wkt_diff_rate'])
qual_bowl['z_succ'] = zscore(qual_bowl['shrunk_succ'].fillna(qual_bowl['shrunk_succ'].mean()))
qual_bowl['bowl_score'] = 0.60 * qual_bowl['z_eb_saved'] + 0.25 * qual_bowl['z_eb_wkt'] + 0.15 * qual_bowl['z_succ']
qual_bowl['p_er'] = (qual_bowl['runs_conceded'] / np.maximum(1, qual_bowl['legal_balls'])) * 6.0
qual_bowl['implied_par'] = qual_bowl['p_er'] + 6.0 * qual_bowl['raw_saved_mean']
qual_bowl['wkts_above_x'] = qual_bowl['wkts'] - qual_bowl['x_wkts']

# ==============================================================================
# 500-RESAMPLE MATCH-CLUSTERED BOOTSTRAP EVALUATING THE EXACT SHRUNKEN SCORE
# ==============================================================================
print("\nRunning 500-resample bootstrap evaluating exact shrunken scores...")
unique_matches = df_t1_p['match_id'].unique()
n_boots = 500
top10_bat_counts = {name: 0 for name in qual_bat['striker_name']}
top10_bowl_counts = {name: 0 for name in qual_bowl['bowler_name']}

np.random.seed(42)
for b in range(n_boots):
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
    b_bat['eb_r'] = (b_bat['r_sum'] + k_runs * mu_r) / (b_bat['n'] + k_runs)
    b_bat['eb_w'] = (b_bat['w_sum'] + k_wkt * mu_w) / (b_bat['n'] + k_wkt)
    b_bat['score'] = b_bat['eb_r'] - 6.0 * b_bat['eb_w']
    for name in b_bat.sort_values('score', ascending=False).head(10)['striker_name']:
        if name in top10_bat_counts:
            top10_bat_counts[name] += 1
            
    # Bowlers
    b_bowl = df_boot.groupby('bowler_name').agg(
        n=('runs', 'count'),
        s_sum=('bowl_saved_vs_par', 'sum'),
        w_sum=('wkt_diff', 'sum')
    ).reset_index()
    b_bowl = b_bowl[b_bowl['n'] >= 80].copy()
    b_bowl['eb_s'] = (b_bowl['s_sum'] + k_bsaved * mu_bsaved) / (b_bowl['n'] + k_bsaved)
    b_bowl['eb_w'] = (b_bowl['w_sum'] + k_bwkt * mu_bwkt) / (b_bowl['n'] + k_bwkt)
    b_bowl['score'] = 0.60 * zscore(b_bowl['eb_s']) + 0.25 * zscore(b_bowl['eb_w'])
    for name in b_bowl.sort_values('score', ascending=False).head(10)['bowler_name']:
        if name in top10_bowl_counts:
            top10_bowl_counts[name] += 1

qual_bat['boot_p'] = qual_bat['striker_name'].map(top10_bat_counts) / n_boots
qual_bowl['boot_p'] = qual_bowl['bowler_name'].map(top10_bowl_counts) / n_boots

print("\n" + "="*70)
print("TOP 15 BATTERS (WITH BALL-LEVEL SHRINKAGE & SHRUNKEN BOOTSTRAP)")
print("="*70)
top15_b = qual_bat.sort_values('bat_score', ascending=False).head(15)
cols_show_bat = ['striker_name', 'n', 'p_sr', 'raw_r_mean', 'eb_runs_part', 'raw_w_mean', 'eb_wkt_part', 'eb_resid_w6', 'shrunk_succ', 'bat_score', 'boot_p']
print(top15_b[cols_show_bat].to_string(index=False))

print("\n" + "="*70)
print("TOP 15 BOWLERS (WITH BALL-LEVEL SHRINKAGE & SHRUNKEN BOOTSTRAP)")
print("="*70)
top15_bo = qual_bowl.sort_values('bowl_score', ascending=False).head(15)
cols_show_bowl = ['bowler_name', 'n', 'p_er', 'raw_saved_mean', 'eb_saved_per_ball', 'implied_par', 'wkts_above_x', 'shrunk_succ', 'bowl_score', 'boot_p']
print(top15_bo[cols_show_bowl].to_string(index=False))

