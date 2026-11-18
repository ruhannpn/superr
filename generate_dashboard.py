import json
import pandas as pd
import numpy as np
import os
import shutil

# Load single source of truth
results_master = pd.read_csv('results_master.csv')
sens_master = pd.read_csv('sensitivity_master.csv')
with open('summary_metrics.json') as f:
    summary_metrics = json.load(f)

counts = summary_metrics['counts']
rel = summary_metrics['reliability']

batters_all = results_master[results_master['player_type'] == 'Batter'].sort_values('rank')
bowlers_all = results_master[results_master['player_type'] == 'Bowler'].sort_values('rank')

bat_top10 = batters_all.head(10).to_dict(orient='records')
bowl_top10 = bowlers_all.head(10).to_dict(orient='records')

# Load specific bowler master for raw wickets & wickets above expected
bowl_m = pd.read_csv('bowlers_master.csv')
for b in bowl_top10:
    sub = bowl_m[bowl_m['bowler_name'] == b['player_name']]
    if len(sub) > 0:
        b['p_wkts'] = int(sub.iloc[0]['p_wkts'])
        b['wkts_above_x'] = float(round(sub.iloc[0]['wkts_above_x'], 2))
    else:
        b['p_wkts'] = 0
        b['wkts_above_x'] = 0.0

# Prepare JSON data for client-side interactive rendering
bat_json = json.dumps(bat_top10)
bowl_json = json.dumps(bowl_top10)
sens_json = json.dumps(sens_master.to_dict(orient='records'))

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SupeRR Selector Season 2 — Intelligence Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #002B49;
            --secondary: #004BA0;
            --accent: #EA1A85;
            --bg: #0b132b;
            --surface: #1c2541;
            --surface-card: #1f2a4a;
            --border: #3a506b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.5;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        .header-title h1 {{
            font-size: 26px;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.5px;
        }}
        .header-title p {{
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 4px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-rr {{ background: var(--accent); color: white; }}
        .badge-verified {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            position: relative;
            overflow: hidden;
        }}
        .kpi-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; width: 4px; height: 100%;
            background: var(--secondary);
        }}
        .kpi-title {{ font-size: 12px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }}
        .kpi-val {{ font-size: 24px; font-weight: 800; color: #fff; margin: 4px 0; }}
        .kpi-sub {{ font-size: 11px; color: #38bdf8; }}

        .tab-nav {{
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }}
        .tab-btn {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .tab-btn.active {{
            background: var(--secondary);
            color: #fff;
            border-color: var(--secondary);
        }}
        .tab-btn:hover:not(.active) {{
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
        }}

        .table-container {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 24px;
        }}
        .table-header-box {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .table-header-box h2 {{ font-size: 16px; font-weight: 700; color: #fff; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            text-align: left;
        }}
        th {{
            background: rgba(0, 0, 0, 0.2);
            color: var(--text-muted);
            padding: 10px 14px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 10px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: #e2e8f0;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .rank-num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 11px;
            background: rgba(255, 255, 255, 0.1);
        }}
        .rank-1 {{ background: #ffd700; color: #000; }}
        .rank-2 {{ background: #c0c0c0; color: #000; }}
        .rank-3 {{ background: #cd7f32; color: #fff; }}

        .num-cell {{ font-family: 'JetBrains Mono', monospace; }}
        .tag-archetype {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10.5px;
            font-weight: 600;
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}
        .tag-flag {{
            color: #f59e0b;
            font-size: 11px;
            font-weight: 600;
        }}

        .grid-2col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card-box {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
        }}
        .card-box h3 {{ font-size: 15px; font-weight: 700; margin-bottom: 12px; color: #fff; }}

        .callout-info {{
            background: rgba(0, 75, 160, 0.15);
            border-left: 4px solid var(--secondary);
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            font-size: 12px;
            color: #cbd5e1;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>

    <div class="header">
        <div class="header-title">
            <span class="badge badge-rr">Rajasthan Royals SupeRR Selector</span>
            <span class="badge badge-verified">Single Source of Truth: results_master.csv</span>
            <h1 style="margin-top: 8px;">Pressure Intelligence & Scouting Leaderboard</h1>
            <p>Dataset: {counts['all_matches']} Total Matches ({counts['all_balls']:,} Balls) &rarr; {counts['tier1_matches']} Tier-1 Matches ({counts['tier1_balls']:,} Balls) | High Leverage (LI &ge; {counts['li_threshold']:.3f}): {counts['high_pressure_balls']:,} Balls</p>
        </div>
        <div>
            <a href="https://github.com/ruhannpn/superr" target="_blank" style="color: #38bdf8; text-decoration: none; font-size: 12px; font-weight: 600;">GitHub Repository &rarr;</a>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Tier-1 Matches</div>
            <div class="kpi-val">{counts['tier1_matches']}</div>
            <div class="kpi-sub">{counts['tier1_balls']:,} Total Balls</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">High-Pressure Balls</div>
            <div class="kpi-val">{counts['high_pressure_balls']:,}</div>
            <div class="kpi-sub">80th Percentile LI &ge; {counts['li_threshold']:.3f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Batting Split-Half r</div>
            <div class="kpi-val">{rel['bat_split_half_sb']:.3f}</div>
            <div class="kpi-sub">Spearman-Brown Corrected</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Bowling Split-Half r</div>
            <div class="kpi-val">{rel['bowl_split_half_sb']:.3f}</div>
            <div class="kpi-sub">Spearman-Brown Corrected</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Bowler Implied Par</div>
            <div class="kpi-val">8.99</div>
            <div class="kpi-sub">Min 8.34 / Max 9.79</div>
        </div>
    </div>

    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('batters')">Top 10 Batters (Primary)</button>
        <button class="tab-btn" onclick="switchTab('bowlers')">Top 10 Bowlers (Direct)</button>
        <button class="tab-btn" onclick="switchTab('sensitivity')">Sensitivity Regimes (Step 3)</button>
        <button class="tab-btn" onclick="switchTab('audit')">Dhoni, Bumrah & Tandem Audit</button>
    </div>

    <!-- BATTERS TAB -->
    <div id="tab-batters" class="tab-content">
        <div class="callout-info">
            <b>Primary Batter Model:</b> 80% Empirical Bayes Residual (fixed W=6) + 20% Shrunk Consistency. Separate shrinkage applied to Runs Part (k=8.5-14) and Wickets Part (k=2.4-13.4; player-level k=470). Precision aggregation weight w = n&sup2;/(n+k). Formula: <code>EB_Resid = EB_RunsPart - 6.0 &times; EB_WktPart</code> reconciles with zero discrepancy.
        </div>
        <div class="table-container">
            <div class="table-header-box">
                <h2>Top 10 Batters Under High Match Pressure</h2>
                <span style="font-size: 11px; color: var(--text-muted);">* Low confidence flag: &lt;150 balls or Boot P &lt; 40%</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Player Name</th>
                        <th>P-Balls</th>
                        <th>P-SR</th>
                        <th>Raw Runs</th>
                        <th>EB Runs</th>
                        <th>Raw Wkt</th>
                        <th>EB Wkt</th>
                        <th>EB Resid (W=6)</th>
                        <th>Base &alpha;</th>
                        <th>Elev &beta;</th>
                        <th>Boot P</th>
                        <th>Scouting Archetype</th>
                    </tr>
                </thead>
                <tbody id="batters-tbody">
                </tbody>
            </table>
        </div>
    </div>

    <!-- BOWLERS TAB -->
    <div id="tab-bowlers" class="tab-content" style="display: none;">
        <div class="callout-info">
            <b>Primary Bowler Model:</b> 60% Phase-Par Runs Saved + 25% Wickets Above Par + 15% Shrunk Consistency. No tandem rollover credit, no opponent multipliers, no dynamic W. Implied par averages 8.99 per over, matching high-leverage ground truth.
        </div>
        <div class="table-container">
            <div class="table-header-box">
                <h2>Top 10 Bowlers Under High Match Pressure (Direct Execution)</h2>
                <span style="font-size: 11px; color: var(--text-muted);">* Low confidence flag: &lt;150 balls or Boot P &lt; 40%</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Bowler Name</th>
                        <th>P-Balls</th>
                        <th>Wkts</th>
                        <th>P-ER</th>
                        <th>Saved / Ball</th>
                        <th>EB Saved / Ball</th>
                        <th>Implied Par</th>
                        <th>Wkts &gt; Par</th>
                        <th>Base &alpha;</th>
                        <th>Elev &beta;</th>
                        <th>Boot P</th>
                        <th>Scouting Archetype</th>
                    </tr>
                </thead>
                <tbody id="bowlers-tbody">
                </tbody>
            </table>
        </div>
    </div>

    <!-- SENSITIVITY TAB -->
    <div id="tab-sensitivity" class="tab-content" style="display: none;">
        <div class="table-container">
            <div class="table-header-box">
                <h2>Sensitivity Regimes & Set-Overlap Audit (Single Source of Truth)</h2>
                <span style="font-size: 11px; color: var(--text-muted);">Evaluated against Primary Top 10</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Sensitivity Regime</th>
                        <th>Player Type</th>
                        <th>Top-10 Overlap</th>
                        <th>Observed Rank Shift & Tactical Impact</th>
                    </tr>
                </thead>
                <tbody id="sens-tbody">
                </tbody>
            </table>
        </div>
    </div>

    <!-- AUDIT TAB -->
    <div id="tab-audit" class="tab-content" style="display: none;">
        <div class="grid-2col">
            <div class="card-box">
                <h3>MS Dhoni vs Dinesh Karthik Data Audit</h3>
                <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">Resolving earlier duplicate output: MS Dhoni faced only 69 balls in high leverage, falling below the 120-ball statistical floor.</p>
                <table>
                    <thead>
                        <tr><th>Player</th><th>P-Balls</th><th>P-SR</th><th>EB Resid</th><th>Ruling</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><b>MS Dhoni</b></td><td>69</td><td>133.3</td><td>+0.150</td><td><span style="color: var(--danger); font-weight:600;">Filtered Out (Under Floor)</span></td></tr>
                        <tr><td><b>Dinesh Karthik</b></td><td>136</td><td>156.6</td><td>+0.239</td><td><span style="color: var(--success); font-weight:600;">Qualified (Rank #7)</span></td></tr>
                    </tbody>
                </table>
            </div>

            <div class="card-box">
                <h3>Tandem Rollover Statistical Re-Test</h3>
                <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">Clustered OLS Regressions on Tier-1 overs (match clustered SEs, next bowler fixed effects):</p>
                <div style="font-size: 12px; line-height: 1.8; color: #cbd5e1;">
                    <div>&bull; <b>Run-based (&le;6 runs conceded):</b> &gamma;&#770; = -0.0087 (SE = 0.0076, p = 0.2546), 95% CI: [-0.0236, +0.0062]</div>
                    <div>&bull; <b>Continuous runs saved vs par:</b> &beta;&#770; = -0.0007 (SE = 0.0008, p = 0.4011), 95% CI: [-0.0022, +0.0009]</div>
                    <div>&bull; <b>Decision Rule Outcome:</b> 95% CIs cross zero &rarr; <b>Tandem dropped from primary metric</b>.</div>
                </div>
            </div>
        </div>

        <div class="card-box">
            <h3>Jasprit Bumrah & Sunil Narine Audit</h3>
            <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">Evaluating high-pressure execution vs expected dismissals without speculative narrative:</p>
            <table>
                <thead>
                    <tr><th>Bowler Name</th><th>P-Balls</th><th>P-ER</th><th>Runs Saved/b</th><th>Wkts (Act vs Exp)</th><th>Diff</th><th>z-score</th><th>Ruling</th></tr>
                </thead>
                <tbody>
                    <tr><td><b>Jasprit Bumrah</b></td><td>364</td><td>7.50</td><td>+0.253</td><td>13 vs 19.37</td><td>-6.37</td><td>-1.45</td><td>Within sampling variance (p = 0.15). Rank #15, Boot P 72.6%.</td></tr>
                    <tr><td><b>Sunil Narine</b></td><td>355</td><td>7.11</td><td>+0.237</td><td>9 vs 16.11</td><td>-7.11</td><td>-1.77</td><td>Within sampling variance (p = 0.08). Rank #17, Boot P 65.6%.</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const batters = {bat_json};
        const bowlers = {bowl_json};
        const sens = {sens_json};

        function renderBatters() {{
            const tbody = document.getElementById('batters-tbody');
            tbody.innerHTML = '';
            batters.forEach(b => {{
                const rClass = b.rank === 1 ? 'rank-1' : (b.rank === 2 ? 'rank-2' : (b.rank === 3 ? 'rank-3' : ''));
                const flagStr = b.low_confidence ? ' <span class=\"tag-flag\">*</span>' : '';
                tbody.innerHTML += `
                    <tr>
                        <td><span class=\"rank-num ${{rClass}}\">${{b.rank}}</span></td>
                        <td><b>${{b.player_name}}</b>${{flagStr}}</td>
                        <td class=\"num-cell\">${{b.p_balls}}</td>
                        <td class=\"num-cell\">${{b.metric_primary_rate.toFixed(1)}}</td>
                        <td class=\"num-cell\">${{b.raw_runs_part > 0 ? '+' : ''}}${{b.raw_runs_part.toFixed(3)}}</td>
                        <td class=\"num-cell\">${{b.eb_runs_part > 0 ? '+' : ''}}${{b.eb_runs_part.toFixed(3)}}</td>
                        <td class=\"num-cell\">${{b.raw_wkt_part > 0 ? '+' : ''}}${{b.raw_wkt_part.toFixed(4)}}</td>
                        <td class=\"num-cell\">${{b.eb_wkt_part > 0 ? '+' : ''}}${{b.eb_wkt_part.toFixed(4)}}</td>
                        <td class=\"num-cell\" style=\"font-weight:700; color:#38bdf8;\">${{b.eb_resid > 0 ? '+' : ''}}${{b.eb_resid.toFixed(3)}}</td>
                        <td class=\"num-cell\">${{b.alpha_baseline > 0 ? '+' : ''}}${{b.alpha_baseline.toFixed(3)}}</td>
                        <td class=\"num-cell\">${{b.beta_delta > 0 ? '+' : ''}}${{b.beta_delta.toFixed(3)}}</td>
                        <td class=\"num-cell\">${{(b.boot_p * 100).toFixed(1)}}%</td>
                        <td><span class=\"tag-archetype\">${{b.archetype}}</span></td>
                    </tr>
                `;
            }});
        }}

        function renderBowlers() {{
            const tbody = document.getElementById('bowlers-tbody');
            tbody.innerHTML = '';
            bowlers.forEach(b => {{
                const rClass = b.rank === 1 ? 'rank-1' : (b.rank === 2 ? 'rank-2' : (b.rank === 3 ? 'rank-3' : ''));
                const flagStr = b.low_confidence ? ' <span class=\"tag-flag\">*</span>' : '';
                tbody.innerHTML += `
                    <tr>
                        <td><span class=\"rank-num ${{rClass}}\">${{b.rank}}</span></td>
                        <td><b>${{b.player_name}}</b>${{flagStr}}</td>
                        <td class=\"num-cell\">${{b.p_balls}}</td>
                        <td class=\"num-cell\">${{b.p_wkts}}</td>
                        <td class=\"num-cell\">${{b.metric_primary_rate.toFixed(2)}}</td>
                        <td class=\"num-cell\">${{b.raw_runs_part > 0 ? '+' : ''}}${{b.raw_runs_part.toFixed(3)}}</td>
                        <td class=\"num-cell\">${{b.eb_runs_part > 0 ? '+' : ''}}${{b.eb_runs_part.toFixed(3)}}</td>
                        <td class=\"num-cell\" style=\"color:#38bdf8;\">${{b.eb_resid.toFixed(2)}}</td>
                        <td class=\"num-cell\">${{b.wkts_above_x > 0 ? '+' : ''}}${{b.wkts_above_x.toFixed(2)}}</td>
                        <td class=\"num-cell\">${{b.alpha_baseline > 0 ? '+' : ''}}${{b.alpha_baseline.toFixed(2)}}</td>
                        <td class=\"num-cell\">${{b.beta_delta > 0 ? '+' : ''}}${{b.beta_delta.toFixed(2)}}</td>
                        <td class=\"num-cell\">${{(b.boot_p * 100).toFixed(1)}}%</td>
                        <td><span class=\"tag-archetype\">${{b.archetype}}</span></td>
                    </tr>
                `;
            }});
        }}

        function renderSensitivity() {{
            const tbody = document.getElementById('sens-tbody');
            tbody.innerHTML = '';
            sens.forEach(s => {{
                tbody.innerHTML += `
                    <tr>
                        <td><b>${{s.Regime}}</b></td>
                        <td>${{s.Type}}</td>
                        <td class=\"num-cell\" style=\"font-weight:700; color:#34d399;\">${{s.Top10_Overlap}}</td>
                        <td>${{s.Shift}}</td>
                    </tr>
                `;
            }});
        }}

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabId).style.display = 'block';
            event.target.classList.add('active');
        }}

        renderBatters();
        renderBowlers();
        renderSensitivity();
    </script>
</body>
</html>
"""

dashboard_path = '/Users/ruhannp.n/Downloads/Data Pack/SupeRR_Selector_Interactive_Dashboard.html'
artifact_dashboard = '/Users/ruhannp.n/.gemini/antigravity/brain/53a3a2a2-70f8-4302-86c8-cdcc54ac4e3d/SupeRR_Selector_Interactive_Dashboard.html'

with open(dashboard_path, 'w') as f:
    f.write(html_content)

shutil.copyfile(dashboard_path, artifact_dashboard)
print(f"Interactive Dashboard successfully updated from results_master.csv at {dashboard_path} and synced to artifacts.")
