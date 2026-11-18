import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== RUNNING V6 ADVANCED PIPELINE: 1 (DYNAMIC W), 2 (TANDEM SPILLOVER), 5 (OPPONENT ELO), 7 (ARCHETYPE CLUSTERING) ===")
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
matches['win_team_id'] = pd.to_numeric(matches['win_team_id'], errors='coerce')
matches['match_year'] = pd.to_datetime(matches['match_date']).dt.year

# --- ENHANCEMENT 5: CONTINUOUS OPPONENT QUALITY WEIGHTING ---
# Calculate rolling team win rate
teams = pd.concat([valid_matches['home_team'], valid_matches['away_team']]).unique()
team_quality_map = {}
for t in teams:
    m_h = valid_matches[valid_matches['home_team'] == t]
    m_a = valid_matches[valid_matches['away_team'] == t]
    tot = len(m_h) + len(m_a)
    if tot >= 8:
        w_h = len(m_h[m_h['win_team_id'] == m_h['home_team_id']])
        w_a = len(m_a[m_a['win_team_id'] == m_a['away_team_id']])
        team_quality_map[t] = (w_h + w_a) / tot
    else:
        team_quality_map[t] = 0.50

global_mean_q = np.mean(list(team_quality_map.values()))
matches['home_team_q'] = matches['home_team'].map(team_quality_map).fillna(global_mean_q)
matches['away_team_q'] = matches['away_team'].map(team_quality_map).fillna(global_mean_q)

meta_cols = ['match_id', 'stage', 'knock_out', 'first_inn_revised_overs', 'second_inn_revised_overs', 
             'revised_target', 'win_team_id', 'is_tier1', 'venue_id', 'match_date', 'match_year',
             'home_team_id', 'away_team_id', 'home_team_q', 'away_team_q']
df = bbb.merge(matches[meta_cols], on='match_id', how='left')
df = df[df['match_id'].isin(valid_match_ids) & df['innings'].isin([1, 2])].copy()

# Opponent quality for each delivery:
# If bowling team is home team, opponent quality for batter is home_team_q, and vice versa
df['bowling_team_q'] = np.where(df['batting_team_id'] == df['home_team_id'], df['away_team_q'], df['home_team_q'])
df['batting_team_q'] = np.where(df['batting_team_id'] == df['home_team_id'], df['home_team_q'], df['away_team_q'])
# Delivery weights scaled around 1.0 (range 0.65 to 1.35)
df['opp_weight_bat'] = 0.50 + 1.0 * df['bowling_team_q']
df['opp_weight_bowl'] = 0.50 + 1.0 * df['batting_team_q']

df['is_legal'] = ~df['extras_type'].isin(['wide', 'Overstep'])
df['total_ball_runs'] = (df['runs'].fillna(0).astype(int) + 
                         df['extras1_runs'].fillna(0).astype(int) + 
                         df['extras_2_runs'].fillna(0).astype(int))
df['is_wkt'] = (df['is_wicket'].astype(str).str.lower() == 'true').astype(int)
df['is_dot'] = ((df['runs'] == 0) & (df['extras1_runs'] == 0)).astype(int)
df = df.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)
df['bat_team_won'] = (df['batting_team_id'].astype(float) == df['win_team_id']).astype(int)

# Entry over & grouping
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

# Shifted rolling features
g_inn2 = df.groupby(['match_id', 'innings'])
df['runs_l12'] = g_inn2['total_ball_runs'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['wkts_l18'] = g_inn2['is_wkt'].transform(lambda s: s.shift(1).rolling(18, min_periods=1).sum()).fillna(0)
df['dots_l12'] = g_inn2['is_dot'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['phase'] = np.where(df['overid'] <= 6, 'Powerplay', np.where(df['overid'] <= 15, 'Middle', 'Death'))

# --- STEP 2: MODEL A (WIN PROBABILITY & CALIBRATION) ---
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

# --- ENHANCEMENT 1: DYNAMIC WICKET VALUE W(w, b) ---
# Calculate empirical expected runs remaining for each (wickets_in_hand, overid)
df['runs_rest_of_inn'] = g_inn['total_ball_runs'].transform('sum') - df['cum_runs_before']
exp_runs_table = df.groupby(['wickets_in_hand', 'overid'])['runs_rest_of_inn'].mean().unstack().bfill()

def get_dynamic_w(row):
    w = int(row['wickets_in_hand'])
    ov = int(row['overid'])
    if w <= 1:
        return 12.0 # Last wicket is game over
    try:
        curr_exp = exp_runs_table.loc[w, ov]
        next_exp = exp_runs_table.loc[w - 1, min(20, ov + 1)] if (w - 1) in exp_runs_table.index and (ov + 1) in exp_runs_table.columns else exp_runs_table.loc[w - 1, ov]
        w_val = np.clip(curr_exp - next_exp, 3.5, 14.0)
        return w_val
    except:
        return 6.0

df['dynamic_w'] = df.apply(get_dynamic_w, axis=1)

# --- STEP 4: MODEL B (ZERO LEAKAGE EXPECTATIONS) ---
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

# Residual with Dynamic W and Opponent Quality Weighting!
df['run_resid_bat'] = df['runs'] - df['x_runs']
df['wkt_resid_bat'] = df['is_wkt'] - df['x_wkt']
# Apply Dynamic W: (Runs - xRuns) - W(w, b) * (IsWkt - xWkt)
df['bat_resid_dyn'] = (df['run_resid_bat'] - df['dynamic_w'] * df['wkt_resid_bat']) * df['opp_weight_bat']

# Bowler Run Suppression (Phase Par) with Opponent Weighting
phase_par_er = df.groupby('phase')['total_ball_runs'].mean() * 6.0
df['phase_par_ball'] = df['phase'].map(phase_par_er) / 6.0
df['bowl_saved_vs_par'] = (df['phase_par_ball'] - df['total_ball_runs']) * df['opp_weight_bowl']
df['bowl_wkt_above_x'] = (df['is_wkt'] - df['x_wkt']) * df['opp_weight_bowl']

# --- ENHANCEMENT 2: TANDEM PRESSURE SPILLOVER ---
# Over-level aggregation
over_agg = df.groupby(['match_id', 'innings', 'overid', 'bowler_id', 'bowler_name']).agg(
    balls=('runs', 'count'),
    runs=('total_ball_runs', 'sum'),
    wkts=('is_wkt', 'sum'),
    saved=('bowl_saved_vs_par', 'sum'),
    is_hp=('is_high_pressure', 'max')
).reset_index().sort_values(['match_id', 'innings', 'overid']).reset_index(drop=True)

over_agg['next_wkts'] = over_agg.groupby(['match_id', 'innings'])['wkts'].shift(-1).fillna(0)
over_agg['next_saved'] = over_agg.groupby(['match_id', 'innings'])['saved'].shift(-1).fillna(0)

# Gamma = 0.35 tandem spillover attribution
gamma = 0.35
over_agg['tandem_wkt_credit'] = np.where(over_agg['saved'] > 1.5, gamma * over_agg['next_wkts'], 0.0)
over_agg['tandem_saved_credit'] = np.where(over_agg['saved'] > 1.5, gamma * np.maximum(0, over_agg['next_saved']), 0.0)

bowler_tandem = over_agg.groupby(['bowler_id', 'bowler_name']).agg(
    tandem_wkts=('tandem_wkt_credit', 'sum'),
    tandem_saved=('tandem_saved_credit', 'sum')
).reset_index()

# Filter Tier-1
df_t1 = df[df['is_tier1']].copy()
df_p = df_t1[df_t1['is_high_pressure'] == 1].copy()
df_b = df_t1[df_t1['is_high_pressure'] == 0].copy()

# BATTERS: ENTRY GROUP PRECISION-WEIGHTED EMPIRICAL BAYES WITH DYNAMIC W
cell = df_p.groupby(['striker_id', 'striker_name', 'entry_group']).agg(
    n=('runs', 'count'),
    runs=('runs', 'sum'),
    outs=('is_wkt', 'sum'),
    resid=('bat_resid_dyn', 'mean'),
    run_resid=('run_resid_bat', 'mean'),
    wkt_resid=('wkt_resid_bat', 'mean')
).reset_index()

def get_eb_params(sub):
    mu = sub['resid'].mean()
    var_between = max(0.01, np.var(sub['resid'], ddof=1))
    var_within = 2.40
    k = var_within / var_between
    return pd.Series({'mu': mu, 'k': k})

grp_params = cell.groupby('entry_group').apply(get_eb_params).reset_index()
cell = cell.merge(grp_params, on='entry_group')
cell['eb_resid'] = (cell['n'] * cell['resid'] + cell['k'] * cell['mu']) / (cell['n'] + cell['k'])
cell['prec_w'] = cell['n'] / (cell['n'] + cell['k'])
cell['ball_prec_w'] = cell['n'] * cell['prec_w']

bat_entry_agg = cell.groupby(['striker_id', 'striker_name']).apply(
    lambda d: pd.Series({
        'p_balls': d['n'].sum(),
        'p_runs': d['runs'].sum(),
        'p_outs': d['outs'].sum(),
        'eb_resid_entry_adj': np.average(d['eb_resid'], weights=d['ball_prec_w']),
        'raw_resid': np.average(d['resid'], weights=d['n']),
        'run_resid': np.average(d['run_resid'], weights=d['n']),
        'wkt_resid': np.average(d['wkt_resid'], weights=d['n']),
    })
).reset_index()

bat_b_stats = df_b.groupby(['striker_id', 'striker_name']).agg(
    b_balls=('runs', 'count'),
    b_runs=('runs', 'sum'),
    b_outs=('is_wkt', 'sum'),
    b_resid=('bat_resid_dyn', 'mean')
).reset_index()

bat_order = df_t1.groupby(['striker_id', 'striker_name'])['batting_order'].median().reset_index()
bat_order.columns = ['striker_id', 'striker_name', 'med_order']

bat_match = df_p.groupby(['striker_id', 'striker_name', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_resid=('bat_resid_dyn', 'sum')
).reset_index()
bat_cons = bat_match[bat_match['m_balls'] >= 6].groupby(['striker_id', 'striker_name']).agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_resid', lambda s: (s > 0).sum())
).reset_index()
bat_cons['shrunk_succ_rate'] = (bat_cons['succ_matches'] + 2.0) / (bat_cons['tested_matches'] + 4.0)

bat = bat_entry_agg.merge(bat_b_stats, on=['striker_id', 'striker_name']).merge(bat_order, on=['striker_id', 'striker_name']).merge(bat_cons, on=['striker_id', 'striker_name'], how='left')
bat = bat[(bat['p_balls'] >= 120) & (bat['tested_matches'] >= 8) & (bat['med_order'] <= 7)].copy()

bat['p_sr'] = (bat['p_runs'] / bat['p_balls']) * 100.0
bat['b_sr'] = (bat['b_runs'] / np.maximum(1, bat['b_balls'])) * 100.0
bat['z_eb_resid'] = (bat['eb_resid_entry_adj'] - bat['eb_resid_entry_adj'].mean()) / bat['eb_resid_entry_adj'].std()
bat['z_succ'] = (bat['shrunk_succ_rate'] - bat['shrunk_succ_rate'].mean()) / bat['shrunk_succ_rate'].std()
bat['bat_score'] = 0.80 * bat['z_eb_resid'] + 0.20 * bat['z_succ']
top10_batters = bat.sort_values('bat_score', ascending=False)

# BOWLERS: CLEAN RUN SUPPRESSION + WICKETS + TANDEM SPILLOVER
bowl_p = df_p.groupby(['bowler_id', 'bowler_name']).agg(
    p_balls=('runs', 'count'),
    p_legal_balls=('is_legal', 'sum'),
    p_runs_conceded=('total_ball_runs', 'sum'),
    p_wkts=('is_wkt', 'sum'),
    p_x_wkts=('x_wkt', 'sum'),
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

bowl = bowl_p.merge(bowl_b, on=['bowler_id', 'bowler_name']).merge(bowl_cons, on=['bowler_id', 'bowler_name'], how='left').merge(bowler_tandem, on=['bowler_id', 'bowler_name'], how='left')
bowl['tandem_wkts'] = bowl['tandem_wkts'].fillna(0.0)
bowl['tandem_saved'] = bowl['tandem_saved'].fillna(0.0)
bowl = bowl[(bowl['p_balls'] >= 120) & (bowl['tested_matches'] >= 8)].copy()

bowl['p_er'] = (bowl['p_runs_conceded'] / np.maximum(1, bowl['p_legal_balls'])) * 6.0
bowl['b_er'] = (bowl['b_runs_conceded'] / np.maximum(1, bowl['b_legal_balls'])) * 6.0
bowl['death_share'] = bowl['p_death_balls'] / bowl['p_balls']
bowl['runs_saved_per_ball'] = (bowl['p_runs_saved'] + bowl['tandem_saved']) / bowl['p_balls']
bowl['wkts_above_x'] = (bowl['p_wkts'] + bowl['tandem_wkts']) - bowl['p_x_wkts']

k_saved = 40.0
mu_saved = bowl['runs_saved_per_ball'].mean()
bowl['eb_runs_saved'] = (bowl['p_balls'] * bowl['runs_saved_per_ball'] + k_saved * mu_saved) / (bowl['p_balls'] + k_saved)
bowl['eb_wkts'] = bowl['wkts_above_x'] / np.sqrt(bowl['p_balls'])

bowl['z_eb_saved'] = (bowl['eb_runs_saved'] - bowl['eb_runs_saved'].mean()) / bowl['eb_runs_saved'].std()
bowl['z_eb_wkts'] = (bowl['eb_wkts'] - bowl['eb_wkts'].mean()) / bowl['eb_wkts'].std()
bowl['z_succ'] = (bowl['shrunk_succ_rate'] - bowl['shrunk_succ_rate'].mean()) / bowl['shrunk_succ_rate'].std()

# Primary Bowler Score with Tandem Credit:
bowl['bowl_score'] = 0.60 * bowl['z_eb_saved'] + 0.25 * bowl['z_eb_wkts'] + 0.15 * bowl['z_succ']
top10_bowlers = bowl.sort_values('bowl_score', ascending=False)

# --- ENHANCEMENT 7: ARCHETYPE CLUSTERING ---
# Batters Archetypes
b_features = ['p_sr', 'run_resid', 'wkt_resid', 'shrunk_succ_rate', 'med_order']
X_bat_cluster = StandardScaler().fit_transform(bat[b_features])
kmeans_bat = KMeans(n_clusters=4, random_state=42).fit(X_bat_cluster)
bat['cluster_id'] = kmeans_bat.labels_

def label_bat_archetype(row):
    sr = row['p_sr']
    run_r = row['run_resid']
    order = row['med_order']
    wkt_r = row['wkt_resid']
    if sr >= 170.0 or run_r >= 0.30:
        return 'Power Surge Aggressor'
    elif order <= 3.0 and wkt_r <= -0.01:
        return 'Crisis Anchor / Stabilizer'
    elif order >= 5.0:
        return 'Late Death Finisher'
    else:
        return 'Middle-Overs Rotator'

bat['archetype'] = bat.apply(label_bat_archetype, axis=1)

def label_bowl_archetype(row):
    d_share = row['death_share']
    tandem = row['tandem_wkts']
    saved = row['runs_saved_per_ball']
    er = row['p_er']
    if d_share >= 0.25:
        return 'Death Yorker Specialist'
    elif tandem >= 15.0 or saved >= 0.35:
        return 'Tandem Pressure Anchor'
    elif er <= 7.0:
        return 'Middle-Overs Choke Specialist'
    else:
        return 'Crisis Strike Enforcer'

bowl['archetype'] = bowl.apply(label_bowl_archetype, axis=1)

top10_batters = bat.sort_values('bat_score', ascending=False)
top10_bowlers = bowl.sort_values('bowl_score', ascending=False)

print("\n=== TOP 10 BATTERS (V6 with Dynamic W & Opponent Quality Weighting) ===")
cols_bat_print = ['striker_name', 'p_balls', 'p_sr', 'eb_resid_entry_adj', 'shrunk_succ_rate', 'bat_score', 'archetype']
print(top10_batters[cols_bat_print].head(10).to_string(index=False))

print("\n=== TOP 10 BOWLERS (V6 with Tandem Spillover & Opponent Quality Weighting) ===")
cols_bowl_print = ['bowler_name', 'p_balls', 'p_er', 'runs_saved_per_ball', 'wkts_above_x', 'tandem_wkts', 'bowl_score', 'archetype']
print(top10_bowlers[cols_bowl_print].head(10).to_string(index=False))

print(f"\nV6 Advanced Pipeline executed in {time.time()-t0:.1f}s!")
