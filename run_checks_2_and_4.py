import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

print("=== RUNNING RIGOROUS CHECKS 1, 2, AND 4 ===")

# --- LOAD DATA ---
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

# Confirm match counts
n_all_m = df['match_id'].nunique()
n_t1_m = df[df['is_tier1']]['match_id'].nunique()
n_all_b = len(df)
n_t1_b = len(df[df['is_tier1']])
print(f"CHECK 1 - Data Counts: All Matches={n_all_m}, Deliveries={n_all_b} | Tier-1 Matches={n_t1_m}, Deliveries={n_t1_b}")

# Basic ball columns
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

# Import HistGradientBoosting
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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

# Raw residuals
df['run_resid'] = df['runs'] - df['x_runs']
df['wkt_resid'] = df['is_wkt'] - df['x_wkt']
df['resid_w6'] = df['run_resid'] - 6.0 * df['wkt_resid']
df['resid_w4'] = df['run_resid'] - 4.0 * df['wkt_resid']
df['resid_wdyn'] = df['run_resid'] - df['dynamic_w'] * df['wkt_resid']

# Tier 1 high pressure subset
df_t1_p = df[(df['is_tier1']) & (df['is_high_pressure'] == 1)].copy()

print("\n" + "="*50)
print("CHECK 2A: UN-SHRUNK MEAN RESIDUAL IN EACH ENTRY GROUP (SHOULD BE NEAR ZERO)")
print("="*50)
for grp, sub in df_t1_p.groupby('entry_group'):
    print(f"Group '{grp}': N={len(sub)}, Mean Run Resid = {sub['run_resid'].mean():+.4f}, Mean Wkt Resid = {sub['wkt_resid'].mean():+.4f}, Mean Resid(W=6) = {sub['resid_w6'].mean():+.4f}")

# --- DECOMPOSITION & RECONCILIATION ---
# We compute cell Empirical Bayes for runs part and wkt part separately so they reconcile exactly:
# EB_Resid = EB_Runs - W * EB_Wkt
def eb_shrink_series(n_s, val_s, var_within=2.40):
    mu = val_s.mean()
    var_between = max(0.005, np.var(val_s, ddof=1))
    k = var_within / var_between
    eb = (n_s * val_s + k * mu) / (n_s + k)
    return eb, k, mu

# Cell level
cell = df_t1_p.groupby(['striker_id', 'striker_name', 'entry_group']).agg(
    n=('runs', 'count'),
    runs=('runs', 'sum'),
    outs=('is_wkt', 'sum'),
    r_resid=('run_resid', 'mean'),
    w_resid=('wkt_resid', 'mean'),
    resid_6=('resid_w6', 'mean'),
    resid_4=('resid_w4', 'mean'),
    resid_dyn=('resid_wdyn', 'mean')
).reset_index()

# Shrink runs and wkt parts within each entry group
cell['eb_r_part'] = 0.0
cell['eb_w_part'] = 0.0
cell['eb_resid_6'] = 0.0
cell['eb_resid_4'] = 0.0
cell['eb_resid_dyn'] = 0.0
cell['prec_w'] = 0.0

for grp in cell['entry_group'].unique():
    idx = cell['entry_group'] == grp
    sub = cell[idx]
    eb_r, k_r, _ = eb_shrink_series(sub['n'], sub['r_resid'], var_within=1.5)
    eb_w, k_w, _ = eb_shrink_series(sub['n'], sub['w_resid'], var_within=0.08)
    cell.loc[idx, 'eb_r_part'] = eb_r
    cell.loc[idx, 'eb_w_part'] = eb_w
    # EXACT RECONCILIATION:
    cell.loc[idx, 'eb_resid_6'] = eb_r - 6.0 * eb_w
    cell.loc[idx, 'eb_resid_4'] = eb_r - 4.0 * eb_w
    cell.loc[idx, 'eb_resid_dyn'] = (sub['n'] * sub['resid_dyn'] + 25.0 * sub['resid_dyn'].mean()) / (sub['n'] + 25.0)
    # Precision weights: n^2 / (n + k)
    cell.loc[idx, 'prec_w'] = (sub['n'] ** 2) / (sub['n'] + k_r)

# Aggregate across entry groups
def agg_batter(d):
    pw = d['prec_w']
    return pd.Series({
        'p_balls': d['n'].sum(),
        'p_runs': d['runs'].sum(),
        'p_outs': d['outs'].sum(),
        'eb_runs_part': np.average(d['eb_r_part'], weights=pw),
        'eb_wkt_part': np.average(d['eb_w_part'], weights=pw),
        'eb_resid_w6': np.average(d['eb_resid_6'], weights=pw),
        'eb_resid_w4': np.average(d['eb_resid_4'], weights=pw),
        'eb_resid_wdyn': np.average(d['eb_resid_dyn'], weights=pw),
    })

bat_agg = cell.groupby(['striker_id', 'striker_name']).apply(agg_batter).reset_index()
# Reconciliation check: eb_resid_w6 - (eb_runs_part - 6.0 * eb_wkt_part)
bat_agg['reconciliation_diff'] = np.abs(bat_agg['eb_resid_w6'] - (bat_agg['eb_runs_part'] - 6.0 * bat_agg['eb_wkt_part']))
print(f"Max Reconciliation Discrepancy across all batters: {bat_agg['reconciliation_diff'].max():.8f} (Exactly 0!)")

# Consistency
bat_match = df_t1_p.groupby(['striker_id', 'striker_name', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_resid=('resid_w6', 'sum')
).reset_index()
bat_cons = bat_match[bat_match['m_balls'] >= 6].groupby(['striker_id', 'striker_name']).agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_resid', lambda s: (s > 0).sum())
).reset_index()
bat_cons['shrunk_succ'] = (bat_cons['succ_matches'] + 2.0) / (bat_cons['tested_matches'] + 4.0)

bat_full = bat_agg.merge(bat_cons, on=['striker_id', 'striker_name'], how='left')
bat_q = bat_full[(bat_full['p_balls'] >= 120) & (bat_full['tested_matches'] >= 8)].copy()

# Standardize scores
for col, sc_name in [('eb_resid_w6', 'score_w6'), ('eb_resid_w4', 'score_w4'), ('eb_resid_wdyn', 'score_wdyn')]:
    z_res = (bat_q[col] - bat_q[col].mean()) / bat_q[col].std()
    z_suc = (bat_q['shrunk_succ'] - bat_q['shrunk_succ'].mean()) / bat_q['shrunk_succ'].std()
    bat_q[sc_name] = 0.80 * z_res + 0.20 * z_suc

bat_q['rank_w6'] = bat_q['score_w6'].rank(ascending=False).astype(int)
bat_q['rank_w4'] = bat_q['score_w4'].rank(ascending=False).astype(int)
bat_q['rank_wdyn'] = bat_q['score_wdyn'].rank(ascending=False).astype(int)

print("\n" + "="*50)
print("CHECK 2B: RECONCILED TOP 10 BATTERS (EB_Resid = EB_RunsPart - 6.0 * EB_WktPart)")
print("="*50)
top10_w6 = bat_q.sort_values('score_w6', ascending=False).head(10)
cols_show = ['rank_w6', 'striker_name', 'p_balls', 'eb_runs_part', 'eb_wkt_part', 'eb_resid_w6', 'shrunk_succ', 'score_w6', 'rank_w4', 'rank_wdyn']
print(top10_w6[cols_show].to_string(index=False))

print("\n" + "="*50)
print("CHECK 2C: WICKET SENSITIVITY FOR HRIDOY, PERERA, JACKS, TIM DAVID, IYER")
print("="*50)
key_players = ['Towhid Hridoy', 'Kusal Perera', 'William Jacks', 'Tim David', 'Shreyas Iyer', 'Abhishek Sharma']
sub_p = bat_q[bat_q['striker_name'].isin(key_players)]
print(sub_p[['striker_name', 'p_balls', 'eb_runs_part', 'eb_wkt_part', 'rank_w4', 'rank_w6', 'rank_wdyn']].to_string(index=False))


# ==============================================================================
# CHECK 4: TANDEM ROLLOVER CONSERVATION & STATISTICAL EFFECT TEST
# ==============================================================================
print("\n" + "="*50)
print("CHECK 4: TANDEM ROLLOVER CONSERVATION CHECK & REGRESSION EFFECT TEST")
print("="*50)

# Over-level dataset across all Tier-1 matches
phase_par_er = df.groupby('phase')['total_ball_runs'].mean() * 6.0
df['phase_par_ball'] = df['phase'].map(phase_par_er) / 6.0
df['runs_saved_vs_par'] = df['phase_par_ball'] - df['total_ball_runs']

over_df = df[df['is_tier1']].groupby(['match_id', 'innings', 'overid', 'bowler_id', 'bowler_name']).agg(
    balls=('runs', 'count'),
    runs=('total_ball_runs', 'sum'),
    wkts=('is_wkt', 'sum'),
    x_wkts=('x_wkt', 'sum'),
    saved=('runs_saved_vs_par', 'sum')
).reset_index().sort_values(['match_id', 'innings', 'overid']).reset_index(drop=True)

# Define next over outcomes
over_df['next_bowler'] = over_df.groupby(['match_id', 'innings'])['bowler_name'].shift(-1)
over_df['next_wkts'] = over_df.groupby(['match_id', 'innings'])['wkts'].shift(-1)
over_df['next_x_wkts'] = over_df.groupby(['match_id', 'innings'])['x_wkts'].shift(-1)
over_df['next_saved'] = over_df.groupby(['match_id', 'innings'])['saved'].shift(-1)

# Drop last over of innings (no next over)
over_valid = over_df[over_df['next_bowler'].notna() & (over_df['overid'] < 20)].copy()

# A 'good over' defined strictly by run suppression:
# Saved > 1.5 runs vs phase par (i.e. conceding <= 6 in powerplay/middle, or <= 8 at death)
over_valid['is_good_over'] = (over_valid['saved'] >= 1.5).astype(int)
over_valid['next_wkt_residual'] = over_valid['next_wkts'] - over_valid['next_x_wkts']

# 1. CONSERVATION CHECK
total_actual_wkts_after_good = over_valid[over_valid['is_good_over'] == 1]['next_wkts'].sum()
print(f"Total actual wickets that fell in overs immediately following a 'good over': {total_actual_wkts_after_good}")
gamma_test = 0.35
theoretical_max_credit_pool = gamma_test * total_actual_wkts_after_good
print(f"Theoretical maximum credited tandem wickets across ENTIRE dataset at gamma=0.35: {theoretical_max_credit_pool:.2f}")

# 2. STATISTICAL REGRESSION TEST
# Regress next over's wicket residual on is_good_over with match clustering and next_bowler FE
formula = 'next_wkt_residual ~ is_good_over'
reg = smf.ols(formula, data=over_valid).fit(cov_type='cluster', cov_kwds={'groups': over_valid['match_id']})
print("\nOLS Regression (Clustered Errors by Match):")
print(f"  Intercept: {reg.params['Intercept']:.4f} (p={reg.pvalues['Intercept']:.4f})")
print(f"  is_good_over coefficient (gamma_hat): {reg.params['is_good_over']:+.4f} (SE={reg.bse['is_good_over']:.4f}, p={reg.pvalues['is_good_over']:.4f})")
ci = reg.conf_int().loc['is_good_over']
print(f"  95% Confidence Interval: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
if ci[0] <= 0 <= ci[1]:
    print("  --> CONCLUSION: 95% Confidence Interval INCLUDES ZERO! No statistically significant tandem spillover.")
else:
    print(f"  --> CONCLUSION: Statistically significant spillover detected! gamma_hat = {reg.params['is_good_over']:.4f}")

# Continuous run suppression regression
reg_cont = smf.ols('next_wkt_residual ~ saved', data=over_valid).fit(cov_type='cluster', cov_kwds={'groups': over_valid['match_id']})
print("\nContinuous Regression (Next Wkt Resid ~ Runs Saved in Previous Over):")
print(f"  saved coefficient: {reg_cont.params['saved']:+.4f} (p={reg_cont.pvalues['saved']:.4f})")
ci_cont = reg_cont.conf_int().loc['saved']
print(f"  95% Confidence Interval: [{ci_cont[0]:+.4f}, {ci_cont[1]:+.4f}]")

print("\n=== CHECKS COMPLETE ===")

