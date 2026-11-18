import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("--- Step 1: Loading & Preprocessing ---")
t0 = time.time()
bbb = pd.read_csv('ball_by_ball.csv')
matches = pd.read_csv('matches.csv')

# Valid matches (exclude No Results)
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

# Keep standard innings 1 & 2 in valid matches
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

# Groupby match and innings
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

# Rolling features strictly shifted by 1 ball within match and innings
g_inn2 = df.groupby(['match_id', 'innings'])
df['runs_l12'] = g_inn2['total_ball_runs'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)
df['wkts_l18'] = g_inn2['is_wkt'].transform(lambda s: s.shift(1).rolling(18, min_periods=1).sum()).fillna(0)
df['dots_l12'] = g_inn2['is_dot'].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0)

df['phase'] = np.where(df['overid'] <= 6, 'Powerplay', np.where(df['overid'] <= 15, 'Middle', 'Death'))

speed = df['bowl_speed'].copy()
df['clean_bowl_speed'] = np.where((speed < 60) | (speed > 165), np.nan, speed)

print(f"Data prepared in {time.time()-t0:.2f}s. Total deliveries: {len(df)}")

print("--- Step 2: Model A — Win Probability & Out-of-Fold Cross Fitting ---")
feats_inn2 = ['runs_required', 'rrr', 'balls_remaining', 'wickets_in_hand', 'crr', 'rrr_minus_crr', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']
mono_cst_inn2 = [-1, -1, 1, 1, 0, -1, 0, -1, -1, 0]
feats_inn1 = ['cum_runs_before', 'cum_wkts_before', 'balls_remaining', 'wickets_in_hand', 'crr', 'projected_score', 'venue_1st_inn_avg', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']

df['wp'] = 0.5

gkf = GroupKFold(n_splits=5)
for tr_idx, te_idx in gkf.split(df[df['innings'] == 2], groups=df[df['innings'] == 2]['match_id']):
    idx_tr = df[df['innings'] == 2].iloc[tr_idx].index
    idx_te = df[df['innings'] == 2].iloc[te_idx].index
    
    m_inn2 = HistGradientBoostingClassifier(
        monotonic_cst=mono_cst_inn2, max_iter=100, learning_rate=0.08, min_samples_leaf=30, random_state=42
    )
    m_inn2.fit(df.loc[idx_tr, feats_inn2], df.loc[idx_tr, 'bat_team_won'])
    df.loc[idx_te, 'wp'] = m_inn2.predict_proba(df.loc[idx_te, feats_inn2])[:, 1]

for tr_idx, te_idx in gkf.split(df[df['innings'] == 1], groups=df[df['innings'] == 1]['match_id']):
    idx_tr = df[df['innings'] == 1].iloc[tr_idx].index
    idx_te = df[df['innings'] == 1].iloc[te_idx].index
    
    m_inn1 = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.08, min_samples_leaf=30, random_state=42
    )
    m_inn1.fit(df.loc[idx_tr, feats_inn1], df.loc[idx_tr, 'bat_team_won'])
    df.loc[idx_te, 'wp'] = m_inn1.predict_proba(df.loc[idx_te, feats_inn1])[:, 1]

brier = brier_score_loss(df['bat_team_won'], df['wp'])
logloss = log_loss(df['bat_team_won'], df['wp'])
print(f"Model A Out-of-Fold Brier Score: {brier:.4f}, Log-Loss: {logloss:.4f}")

train_mask = df['match_year'] <= 2024
test_mask = df['match_year'] >= 2025

m_time_inn2 = HistGradientBoostingClassifier(monotonic_cst=mono_cst_inn2, max_iter=100, random_state=42)
m_time_inn2.fit(df[train_mask & (df['innings'] == 2)][feats_inn2], df[train_mask & (df['innings'] == 2)]['bat_team_won'])
test_wp_inn2 = m_time_inn2.predict_proba(df[test_mask & (df['innings'] == 2)][feats_inn2])[:, 1]
brier_test_inn2 = brier_score_loss(df[test_mask & (df['innings'] == 2)]['bat_team_won'], test_wp_inn2)
print(f"Time-Split (Train 2022-2024 -> Test 2025-2026) Brier Score: {brier_test_inn2:.4f}")

print("--- Step 3: Computing Leverage Index (LI) & Outcomes ---")
closeness = 1.0 - np.abs(2.0 * df['wp'] - 1.0)
time_weight = 1.0 + 1.2 * ((df['scheduled_balls'] - df['balls_remaining']) / df['scheduled_balls'])
wicket_stress = (10.0 / (df['wickets_in_hand'] + 2.0)) ** 0.5
knockout_boost = np.where(df['knock_out'].astype(str) == '1', 1.15, 1.0)

raw_li = closeness * time_weight * wicket_stress * knockout_boost
raw_li = np.where((df['innings'] == 2) & (df['runs_required'] <= 0), 0.0, raw_li)

df['LI'] = raw_li / raw_li.mean()
df['LI_z'] = (df['LI'] - df['LI'].mean()) / df['LI'].std()

li_80th = np.percentile(df['LI'], 80)
df['is_high_pressure'] = (df['LI'] >= li_80th).astype(int)

print(f"LI 80th percentile threshold: {li_80th:.3f}")
print(f"High Pressure ball proportion: {df['is_high_pressure'].mean():.1%}")

ev_p = df[df['is_high_pressure'] == 1]
ev_n = df[df['is_high_pressure'] == 0]

print("\n=== EVIDENCE TABLE: High Pressure vs Normal Balls ===")
print(f"Run Rate: High Pressure = {ev_p['total_ball_runs'].mean()*6.0:.2f} rpo | Normal = {ev_n['total_ball_runs'].mean()*6.0:.2f} rpo")
print(f"Dot Ball %: High Pressure = {ev_p['is_dot'].mean():.1%} | Normal = {ev_n['is_dot'].mean():.1%}")
print(f"Wicket %: High Pressure = {ev_p['is_wkt'].mean():.2%} | Normal = {ev_n['is_wkt'].mean():.2%}")
print(f"Boundary %: High Pressure = {ev_p['is_boundary'].mean():.1%} | Normal = {ev_n['is_boundary'].mean():.1%}")

print("--- Step 4: Model B — Expected Runs & Wickets (Cross-Fitted) ---")
feats_b = ['overid', 'balls_remaining', 'wickets_in_hand', 'venue_1st_inn_avg', 
           'clean_bowl_speed', 'pace_or_spin', 'delivery_style', 'over_or_around', 
           'batting_type', 'bowling_type', 'runs_l12', 'wkts_l18', 'dots_l12', 'is_tier1']

X_b = df[feats_b].copy()
cat_b = ['pace_or_spin', 'delivery_style', 'over_or_around', 'batting_type', 'bowling_type', 'is_tier1']
for c in cat_b:
    X_b[c] = X_b[c].astype('category')

df['x_runs'] = 1.34
df['x_wkt'] = 0.054

gkf_b = GroupKFold(n_splits=5)
for tr_idx, te_idx in gkf_b.split(df, groups=df['match_id']):
    idx_tr = df.iloc[tr_idx].index
    idx_te = df.iloc[te_idx].index
    
    mr = HistGradientBoostingRegressor(categorical_features=cat_b, max_iter=100, random_state=42)
    mr.fit(X_b.loc[idx_tr], df.loc[idx_tr, 'runs'])
    df.loc[idx_te, 'x_runs'] = mr.predict(X_b.loc[idx_te])
    
    mw = HistGradientBoostingClassifier(categorical_features=cat_b, max_iter=100, random_state=42)
    mw.fit(X_b.loc[idx_tr], df.loc[idx_tr, 'is_wkt'])
    df.loc[idx_te, 'x_wkt'] = mw.predict_proba(X_b.loc[idx_te])[:, 1]

df['bat_resid'] = (df['runs'] - df['x_runs']) - 6.0 * (df['is_wkt'] - df['x_wkt'])
df['bowl_resid'] = (df['x_runs'] - df['total_ball_runs']) + 6.0 * (df['is_wkt'] - df['x_wkt'])

phase_par_er = df.groupby('phase')['total_ball_runs'].mean() * 6.0
df['phase_par_ball'] = df['phase'].map(phase_par_er) / 6.0
df['bowl_saved_vs_par'] = df['phase_par_ball'] - df['total_ball_runs']

print(f"Model B Cross-fitted in {time.time()-t0:.2f}s!")

print("--- Step 5: Layer C — Player Attribution & Empirical Bayes ---")
df_t1 = df[df['is_tier1']].copy()

bat_p = df_t1[df_t1['is_high_pressure'] == 1].groupby(['striker_id', 'striker_name']).agg(
    p_balls=('runs', 'count'),
    p_runs=('runs', 'sum'),
    p_outs=('is_wkt', 'sum'),
    p_resid=('bat_resid', 'mean'),
    p_matches=('match_id', 'nunique')
).reset_index()

bat_b = df_t1[df_t1['is_high_pressure'] == 0].groupby(['striker_id', 'striker_name']).agg(
    b_balls=('runs', 'count'),
    b_runs=('runs', 'sum'),
    b_outs=('is_wkt', 'sum'),
    b_resid=('bat_resid', 'mean')
).reset_index()

bat_order = df_t1.groupby(['striker_id', 'striker_name'])['batting_order'].median().reset_index()
bat_order.rename(columns={'batting_order': 'med_order'}, inplace=True)

bat_match = df_t1[df_t1['is_high_pressure'] == 1].groupby(['striker_id', 'striker_name', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_resid=('bat_resid', 'sum')
).reset_index()
bat_cons = bat_match[bat_match['m_balls'] >= 6].groupby(['striker_id', 'striker_name']).agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_resid', lambda s: (s > 0).sum())
).reset_index()
bat_cons['match_success_rate'] = bat_cons['succ_matches'] / bat_cons['tested_matches']

bat = bat_p.merge(bat_b, on=['striker_id', 'striker_name']).merge(bat_order, on=['striker_id', 'striker_name']).merge(bat_cons, on=['striker_id', 'striker_name'], how='left')

bat = bat[(bat['p_balls'] >= 120) & (bat['tested_matches'] >= 8) & (bat['med_order'] <= 7)].copy()

bat['p_sr'] = (bat['p_runs'] / bat['p_balls']) * 100.0
bat['b_sr'] = (bat['b_runs'] / bat['b_balls']) * 100.0
bat['sr_delta'] = bat['p_sr'] - bat['b_sr']

mu_bat = bat['p_resid'].mean()
var_between = np.var(bat['p_resid'], ddof=1)
var_within = 2.5
k_bat = var_within / max(0.01, var_between)
print(f"Empirical Bayes k for batters: {k_bat:.2f} balls (Pop Mean: {mu_bat:.3f})")

bat['p_resid_eb'] = (bat['p_balls'] * bat['p_resid'] + k_bat * mu_bat) / (bat['p_balls'] + k_bat)
bat['resid_delta'] = bat['p_resid_eb'] - bat['b_resid']

def zscore(s):
    return (s - s.mean()) / s.std()

bat['z_resid_eb'] = zscore(bat['p_resid_eb'])
bat['z_p_sr'] = zscore(bat['p_sr'])
bat['z_sr_delta'] = zscore(bat['sr_delta'])
bat['z_succ'] = zscore(bat['match_success_rate'].fillna(bat['match_success_rate'].mean()))

bat['bat_score'] = (
    0.40 * bat['z_resid_eb'] + 
    0.25 * bat['z_p_sr'] + 
    0.20 * bat['z_sr_delta'] + 
    0.15 * bat['z_succ']
)

top10_batters = bat.sort_values('bat_score', ascending=False).head(12)
print("\n=== TOP 12 BATTERS (Eradicated Weak Opposition, Strict EB, Z-scored) ===")
print(top10_batters[['striker_name', 'p_balls', 'p_sr', 'b_sr', 'sr_delta', 'p_resid_eb', 'match_success_rate', 'bat_score']].to_string(index=False))

# Bowlers
bowl_p = df_t1[df_t1['is_high_pressure'] == 1].groupby(['bowler_id', 'bowler_name']).agg(
    p_balls=('runs', 'count'),
    p_legal_balls=('is_legal', 'sum'),
    p_runs_conceded=('total_ball_runs', 'sum'),
    p_wkts=('is_wkt', 'sum'),
    p_x_wkts=('x_wkt', 'sum'),
    p_resid=('bowl_resid', 'mean'),
    p_runs_saved=('bowl_saved_vs_par', 'sum'),
    p_death_balls=('phase', lambda s: (s == 'Death').sum()),
    p_matches=('match_id', 'nunique')
).reset_index()

bowl_b = df_t1[df_t1['is_high_pressure'] == 0].groupby(['bowler_id', 'bowler_name']).agg(
    b_balls=('runs', 'count'),
    b_legal_balls=('is_legal', 'sum'),
    b_runs_conceded=('total_ball_runs', 'sum'),
    b_wkts=('is_wkt', 'sum'),
    b_resid=('bowl_resid', 'mean')
).reset_index()

bowl_match = df_t1[df_t1['is_high_pressure'] == 1].groupby(['bowler_id', 'bowler_name', 'match_id']).agg(
    m_balls=('runs', 'count'),
    m_resid=('bowl_resid', 'sum')
).reset_index()
bowl_cons = bowl_match[bowl_match['m_balls'] >= 6].groupby(['bowler_id', 'bowler_name']).agg(
    tested_matches=('match_id', 'count'),
    succ_matches=('m_resid', lambda s: (s > 0).sum())
).reset_index()
bowl_cons['match_success_rate'] = bowl_cons['succ_matches'] / bowl_cons['tested_matches']

bowl = bowl_p.merge(bowl_b, on=['bowler_id', 'bowler_name']).merge(bowl_cons, on=['bowler_id', 'bowler_name'], how='left')
bowl = bowl[(bowl['p_balls'] >= 120) & (bowl['tested_matches'] >= 8)].copy()

bowl['p_er'] = (bowl['p_runs_conceded'] / np.maximum(1, bowl['p_legal_balls'])) * 6.0
bowl['b_er'] = (bowl['b_runs_conceded'] / np.maximum(1, bowl['b_legal_balls'])) * 6.0
bowl['er_delta'] = bowl['b_er'] - bowl['p_er']

bowl['death_share'] = bowl['p_death_balls'] / bowl['p_balls']
bowl['wkts_above_x'] = bowl['p_wkts'] - bowl['p_x_wkts']
bowl['wkt_rate_above_x'] = bowl['wkts_above_x'] / bowl['p_balls']
bowl['runs_saved_per_ball'] = bowl['p_runs_saved'] / bowl['p_balls']

mu_bowl = bowl['p_resid'].mean()
var_b_between = np.var(bowl['p_resid'], ddof=1)
var_b_within = 2.8
k_bowl = var_b_within / max(0.01, var_b_between)
print(f"Empirical Bayes k for bowlers: {k_bowl:.2f} balls (Pop Mean: {mu_bowl:.3f})")

bowl['p_resid_eb'] = (bowl['p_balls'] * bowl['p_resid'] + k_bowl * mu_bowl) / (bowl['p_balls'] + k_bowl)
bowl['resid_delta'] = bowl['p_resid_eb'] - bowl['b_resid']

bowl['z_resid_eb'] = zscore(bowl['p_resid_eb'])
bowl['z_runs_saved'] = zscore(bowl['runs_saved_per_ball'])
bowl['z_wkt_above_x'] = zscore(bowl['wkt_rate_above_x'])
bowl['z_succ'] = zscore(bowl['match_success_rate'].fillna(bowl['match_success_rate'].mean()))

bowl['bowl_score'] = (
    0.35 * bowl['z_resid_eb'] + 
    0.30 * bowl['z_runs_saved'] + 
    0.20 * bowl['z_wkt_above_x'] + 
    0.15 * bowl['z_succ']
)

top10_bowlers = bowl.sort_values('bowl_score', ascending=False).head(12)
print("\n=== TOP 12 BOWLERS (Eradicated Weak Opposition, Phase-Par, Strict EB, Z-scored) ===")
print(top10_bowlers[['bowler_name', 'p_balls', 'p_wkts', 'p_er', 'death_share', 'runs_saved_per_ball', 'match_success_rate', 'bowl_score']].to_string(index=False))

r_bat = np.corrcoef(bat['b_resid'], bat['p_resid'])[0, 1]
r_bowl = np.corrcoef(bowl['b_resid'], bowl['p_resid'])[0, 1]
print(f"\n--- HONEST CHECK: Correlation Between General Skill (Baseline) & Pressure Performance ---")
print(f"Batters r(Baseline Skill, Pressure Performance) = {r_bat:.3f}")
print(f"Bowlers r(Baseline Skill, Pressure Performance) = {r_bowl:.3f}")

# Generate Figures
fig1, ax1 = plt.subplots(figsize=(7, 3.8))
ax1.hist(df['LI'], bins=60, color='#004BA0', edgecolor='white', alpha=0.85)
ax1.axvline(li_80th, color='#EA1A85', linestyle='--', linewidth=2, label=f'High Pressure Cutoff (80th Pct = {li_80th:.2f})')
ax1.set_title('Figure 1: Leverage Index (LI) Distribution Across 200,999 Deliveries', fontsize=11, fontweight='bold', color='#002B49')
ax1.set_xlabel('Leverage Index (LI / Mean(LI))', fontsize=9.5)
ax1.set_ylabel('Delivery Count', fontsize=9.5)
ax1.legend(frameon=True)
plt.tight_layout()
fig1.savefig('/tmp/fig1_leverage_distribution.png', dpi=200)
plt.close(fig1)

fig2, ax2 = plt.subplots(1, 3, figsize=(10, 3.4))
metrics = ['Run Rate (rpo)', 'Dot Ball %', 'Wicket %']
hp_vals = [ev_p['total_ball_runs'].mean()*6.0, ev_p['is_dot'].mean()*100, ev_p['is_wkt'].mean()*100]
norm_vals = [ev_n['total_ball_runs'].mean()*6.0, ev_n['is_dot'].mean()*100, ev_n['is_wkt'].mean()*100]

for i in range(3):
    ax2[i].bar(['Normal (80%)', 'High Pressure (20%)'], [norm_vals[i], hp_vals[i]], color=['#64748b', '#EA1A85'], width=0.55)
    ax2[i].set_title(metrics[i], fontsize=10, fontweight='bold', color='#002B49')
    for p in ax2[i].patches:
        ax2[i].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height() * 0.90),
                        ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
plt.suptitle('Figure 2: Empirical Match Outcomes — High Pressure vs Normal Balls', fontsize=11, fontweight='bold', color='#002B49')
plt.tight_layout()
fig2.savefig('/tmp/fig2_outcomes_pressure_vs_normal.png', dpi=200)
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(6, 3.8))
prob_true, prob_pred = calibration_curve(df['bat_team_won'], df['wp'], n_bins=10)
ax3.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#004BA0', label=f'Model A (Brier = {brier:.4f})')
ax3.plot([0, 1], [0, 1], linestyle='--', color='#94a3b8', label='Perfect Calibration')
ax3.set_title('Figure 3: Model A Win Probability Reliability Diagram', fontsize=11, fontweight='bold', color='#002B49')
ax3.set_xlabel('Predicted Win Probability', fontsize=9.5)
ax3.set_ylabel('Empirical Win Fraction', fontsize=9.5)
ax3.legend(frameon=True)
plt.tight_layout()
fig3.savefig('/tmp/fig3_wp_calibration.png', dpi=200)
plt.close(fig3)

fig4, ax4 = plt.subplots(1, 2, figsize=(11, 4.2))
ax4[0].scatter(bat['b_resid'], bat['p_resid_eb'], color='#004BA0', alpha=0.5, edgecolors='none', s=35)
for _, r in top10_batters.head(8).iterrows():
    ax4[0].scatter(r['b_resid'], r['p_resid_eb'], color='#EA1A85', s=65)
    ax4[0].annotate(r['striker_name'].split()[-1], (r['b_resid'], r['p_resid_eb']), fontsize=8, fontweight='bold')
ax4[0].set_title(f'Batters: Skill vs Pressure (r = {r_bat:.2f})', fontsize=10, fontweight='bold', color='#002B49')
ax4[0].set_xlabel('Baseline Residual (General Skill)', fontsize=9)
ax4[0].set_ylabel('Pressure Residual (Empirical Bayes)', fontsize=9)

ax4[1].scatter(bowl['b_resid'], bowl['p_resid_eb'], color='#004BA0', alpha=0.5, edgecolors='none', s=35)
for _, r in top10_bowlers.head(8).iterrows():
    ax4[1].scatter(r['b_resid'], r['p_resid_eb'], color='#EA1A85', s=65)
    ax4[1].annotate(r['bowler_name'].split()[-1], (r['b_resid'], r['p_resid_eb']), fontsize=8, fontweight='bold')
ax4[1].set_title(f'Bowlers: Skill vs Pressure (r = {r_bowl:.2f})', fontsize=10, fontweight='bold', color='#002B49')
ax4[1].set_xlabel('Baseline Residual (General Skill)', fontsize=9)
ax4[1].set_ylabel('Pressure Residual (Empirical Bayes)', fontsize=9)

plt.suptitle('Figure 4: General Skill vs Pressure Performance Attributions', fontsize=11, fontweight='bold', color='#002B49')
plt.tight_layout()
fig4.savefig('/tmp/fig4_pressure_vs_baseline_scatter.png', dpi=200)
plt.close(fig4)

print("All 4 figures generated successfully in /tmp/!")
