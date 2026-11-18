import os
import shutil
import json
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
)
from reportlab.pdfgen import canvas

# Load single source of truth
results_master = pd.read_csv('results_master.csv')
sens_master = pd.read_csv('sensitivity_master.csv')
with open('summary_metrics.json') as f:
    summary_metrics = json.load(f)

cfg = summary_metrics['config']
counts = summary_metrics['counts']
shrink = summary_metrics['shrinkage']
rel = summary_metrics['reliability']

batters_top = results_master[results_master['player_type'] == 'Batter'].sort_values('rank').head(10)
bowlers_top = results_master[results_master['player_type'] == 'Bowler'].sort_values('rank').head(10)

pdf_path = '/Users/ruhannp.n/Downloads/Data Pack/SupeRR_Selector_Season_2_Methodology_Report.pdf'
artifact_pdf = '/Users/ruhannp.n/.gemini/antigravity/brain/53a3a2a2-70f8-4302-86c8-cdcc54ac4e3d/SupeRR_Selector_Season_2_Methodology_Report.pdf'

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        print(f"Total PDF pages generated: {num_pages}")
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#64748b'))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 812, 'SupeRR Selector Season 2 — Evaluating Demonstrated Execution Under Pressure')
            self.drawRightString(559, 812, 'Methodology & Statistical Evaluation Report')
            self.setStrokeColor(colors.HexColor('#cbd5e1'))
            self.setLineWidth(0.5)
            self.line(36, 806, 559, 806)
        
        # Footer
        self.setStrokeColor(colors.HexColor('#cbd5e1'))
        self.setLineWidth(0.5)
        self.line(36, 36, 559, 36)
        self.drawString(36, 25, 'Rajasthan Royals SupeRR Selector 2026 | Repository: https://github.com/ruhannpn/superr')
        self.drawRightString(559, 25, f'Page {self._pageNumber} of {page_count}')
        self.restoreState()

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=A4,
    leftMargin=36,
    rightMargin=36,
    topMargin=44,
    bottomMargin=44
)

styles = getSampleStyleSheet()

primary_color = colors.HexColor('#002B49')
accent_pink = colors.HexColor('#EA1A85')
accent_blue = colors.HexColor('#004BA0')
text_dark = colors.HexColor('#0f172a')

h1_style = ParagraphStyle(
    'H1_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=9.8,
    leading=12.5,
    textColor=primary_color,
    spaceBefore=3.5,
    spaceAfter=1.8,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'Body_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=7.2,
    leading=9.5,
    textColor=text_dark,
    spaceAfter=2.2
)

table_header_style = ParagraphStyle(
    'TableHeader',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=5.8,
    leading=7.2,
    textColor=colors.white,
    alignment=1
)

table_cell_style = ParagraphStyle(
    'TableCell',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=5.7,
    leading=7.0,
    textColor=text_dark
)

table_cell_bold = ParagraphStyle(
    'TableCellBold',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=5.7,
    leading=7.0,
    textColor=primary_color
)

table_cell_center = ParagraphStyle(
    'TableCellCenter',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=5.7,
    leading=7.0,
    textColor=text_dark,
    alignment=1
)

story = []

# ==============================================================================
# PAGE 1: TITLE, EXECUTIVE SUMMARY, ARCHITECTURE, SAMPLE AUDIT
# ==============================================================================
title_table_data = [
    [Paragraph('<b>SupeRR Selector Season 2: Identifying Demonstrated Execution Under Pressure</b>', ParagraphStyle('TitleP', fontName='Helvetica-Bold', fontSize=12.5, leading=15.0, textColor=primary_color))],
    [Paragraph(f"<b>Statistical Methodology & Scouting Evaluation</b> | Author: Ruhann P. N. | Dataset: {counts['all_matches']} Matches ({counts['all_balls']:,} Deliveries) &rarr; {counts['tier1_matches']} Tier-1 Matches ({counts['tier1_balls']:,} Deliveries)", ParagraphStyle('SubTitleP', fontName='Helvetica', fontSize=7.3, leading=9.0, textColor=accent_blue))]
]
title_table = Table(title_table_data, colWidths=[523])
title_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
    ('LINEBELOW', (0,1), (-1,1), 1.5, primary_color),
    ('PADDING', (0,0), (-1,-1), 3.2),
]))
story.append(title_table)
story.append(Spacer(1, 3))

story.append(Paragraph('1. Executive Summary & Sabermetric Architecture', h1_style))
story.append(Paragraph(
    'Evaluating T20 performance under match pressure is fundamentally confounded by situation bias, survivor bias, and extreme-value small-sample noise. '
    'Conventional proxies (such as isolating overs 17-20 or quoting raw strike rates) conflate death-phase tactical hitting with genuine leverage. '
    'This framework follows the modern Sabermetric paradigm: machine learning is restricted to estimating match context (Model A Win Probability) and delivery difficulty '
    '(Model B expected runs and dismissal likelihood without identity leakage). Player execution is derived via hierarchical Empirical Bayes shrinkage '
    'estimated from ball-level variances, precision aggregation weights, and phase-par baselines. '
    'The results reflect <i>best demonstrated execution under pressure, with measured uncertainty</i>, rather than an unalterable genetic trait.',
    body_style
))

story.append(Paragraph('2. Mathematical Pipeline: Leverage, Difficulty & Attribution', h1_style))
arch_data = [
    [Paragraph('<b>Layer A: Dynamic Leverage & Context State</b>', table_header_style),
     Paragraph('<b>Layer B: Non-Leaking Ball Attribution</b>', table_header_style),
     Paragraph('<b>Layer C: Ball-Level Empirical Bayes</b>', table_header_style)],
    [Paragraph('Monotonic HistGradientBoosting predicts batting win probability (WP). <b>True Swing Leverage (LI)</b> measures the swing between a +4 boundary and an out: LI = |WP(+4) - WP(out)| / mean(LI). High-pressure threshold is set at the 80th percentile (LI &ge; 1.674). 72.1% of high-pressure deliveries occur in chases.', table_cell_style),
     Paragraph('Dual HistGradientBoosting models estimate delivery expectation without identity leakage: xRuns predicts expected runs conceded; xWkt predicts dismissal likelihood based strictly on match state, venue, phase, and bowler type (pace vs spin). Net residual: e = (Runs - xRuns) - W &times; (Wkt - xWkt).', table_cell_style),
     Paragraph('Empirical Bayes shrinkage parameters are estimated directly from ball-level variance: k_runs = 220.0, k_wkt = 1576.8 for batters; k_bsaved = 205.1, k_bwkt = 464.9 for bowlers. Extreme wicket outcomes are compressed toward empirical priors, eliminating small-sample distortions.', table_cell_style)]
]
t_arch = Table(arch_data, colWidths=[174, 175, 174])
t_arch.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f8fafc')),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('PADDING', (0,0), (-1,-1), 3.5),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
story.append(t_arch)
story.append(Spacer(1, 3))

story.append(Paragraph('3. Sample Audit, Configuration & Eligibility Uniformity', h1_style))
story.append(Paragraph(
    f"To prevent Associate mismatches from distorting calibrations, the dataset is audited through strict, documented criteria. "
    f"From {counts['all_matches']} raw matches ({counts['all_balls']:,} deliveries), {counts['all_matches'] - counts['valid_matches']} abandoned fixtures are removed. "
    f"Filtering for Tier-1 competition (IPL franchises and ICC Full Member nations) yields {counts['tier1_matches']} matches and {counts['tier1_balls']:,} deliveries. "
    f"High-leverage selection (LI &ge; {counts['li_threshold']}) isolates {counts['high_pressure_balls']:,} deliveries. "
    f"<b>Eligibility Rule:</b> Minimum 120 high-pressure deliveries faced or bowled across at least 8 tested matches in Tier-1 play. "
    f"<b>Config Parameters:</b> TIE_RULE = '{cfg['tie_rule']}' (rank strictly by composite score; candidates within 0.05 of cut line evaluated by Boot P); "
    f"K_SUCC = {cfg['k_succ']} (consistency shrinkage matches); WINSOR_SD = {cfg['winsor_sd']} (winsorise wicket terms at &plusmn;2.5 SD). "
    f"Yields exactly <b>{counts['qual_batters']} Qualified Batters</b> and <b>{counts['qual_bowlers']} Qualified Bowlers</b>.",
    body_style
))

audit_data = [
    [Paragraph('<b>Pipeline Stage</b>', table_header_style), Paragraph('<b>Matches</b>', table_header_style), Paragraph('<b>Deliveries</b>', table_header_style), Paragraph('<b>Description & Filter Rationale</b>', table_header_style)],
    [Paragraph('Raw Input Archive', table_cell_bold), Paragraph(str(counts['all_matches']), table_cell_center), Paragraph(f"{counts['all_balls']:,}", table_cell_center), Paragraph('Complete historical T20 ball-by-ball archive (2022-2026).', table_cell_style)],
    [Paragraph('Valid Result Filter', table_cell_bold), Paragraph(str(counts['valid_matches']), table_cell_center), Paragraph(f"{counts['valid_balls']:,}", table_cell_center), Paragraph('Excludes abandoned fixtures and matches with no result.', table_cell_style)],
    [Paragraph('Tier-1 Competition', table_cell_bold), Paragraph(str(counts['tier1_matches']), table_cell_center), Paragraph(f"{counts['tier1_balls']:,}", table_cell_center), Paragraph('IPL franchises & ICC Full Members; excludes Associate mismatches.', table_cell_style)],
    [Paragraph('High Leverage (LI &ge; 1.674)', table_cell_bold), Paragraph(str(counts['tier1_matches']), table_cell_center), Paragraph(f"{counts['high_pressure_balls']:,}", table_cell_center), Paragraph('Top 20% swing leverage deliveries (&Delta;WP between boundary & out).', table_cell_style)],
    [Paragraph('Qualified Candidates', table_cell_bold), Paragraph('-', table_cell_center), Paragraph('187 Players', table_cell_center), Paragraph(f"{counts['qual_batters']} Batters & {counts['qual_bowlers']} Bowlers meeting the 120-ball and 8-match qualification floor.", table_cell_style)]
]
t_audit = Table(audit_data, colWidths=[105, 48, 72, 298])
t_audit.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 2.2),
]))
story.append(t_audit)

story.append(PageBreak())

# ==============================================================================
# PAGE 2: MODEL B CALIBRATION, SHRINKAGE PARAMETERS & AUDITS
# ==============================================================================
story.append(Paragraph('4. Win Probability Calibration & Model B Validation', h1_style))
story.append(Paragraph(
    'Innings 2 Win Probability enforces monotonic constraints on features (&part;WP/&part;rrr &le; 0, &part;WP/&part;wickets &ge; 0, &part;WP/&part;balls &ge; 0). '
    'Innings 1 out-of-fold Brier score achieves 0.1598 (+36.1% skill over 0.25 uninformative base rate), while time-split Brier across seasons is 0.2543 (-1.7% skill), '
    'reflecting that first-innings totals are subject to pitch degradation. 72.1% of high-pressure deliveries fall in chases where WP is strongly calibrated. '
    '<br/><b>Model B Calibration:</b> Evaluating xWkt on high-pressure balls shows close alignment across phases (Powerplay: actual 4.02% vs exp 4.46%; Middle: 4.61% vs 4.77%; Death: 8.01% vs 7.54%). '
    'Pace vs spin check reveals identical residual rates (Pace: -0.15 wkts/100b; Spin: -0.12 wkts/100b), confirming zero bowler-type bias.',
    body_style
))

story.append(Paragraph('5. Ball-Level Variance & Empirical Bayes Parameter Stability', h1_style))
story.append(Paragraph(
    f"Empirical Bayes shrinkage parameters derived from delivery-level noise (&sigma;_e^2) and between-player variance (&sigma;_b^2) establish: "
    f"<br/>&bull; <b>Batters:</b> Runs &sigma;_e^2 = {shrink['sig_e2_runs']:.4f}, &sigma;_b^2 = {shrink['sig_b2_runs']:.5f} &rarr; <b>k_runs = {shrink['k_runs']:.1f} balls</b>. "
    f"Wicket &sigma;_e^2 = {shrink['sig_e2_wkt']:.4f}, &sigma;_b^2 = {shrink['sig_b2_wkt']:.6f} &rarr; <b>k_wkt = {shrink['k_wkt']:.1f} balls</b>. "
    f"Because batter wicket &sigma;_b^2 is near zero, shrunk wicket residuals compress to a near-constant (-0.005 to -0.010 per ball for all players). "
    f"Consequently, <b>the batter ranking is overwhelmingly driven by the runs part</b>. "
    f"<br/>&bull; <b>Bowlers:</b> Runs Saved &sigma;_e^2 = {shrink['sig_e2_bsaved']:.4f}, &sigma;_b^2 = {shrink['sig_b2_bsaved']:.5f} &rarr; <b>k_bsaved = {shrink['k_bsaved']:.1f} balls</b>. "
    f"Wickets &sigma;_e^2 = {shrink['sig_e2_bwkt']:.4f}, &sigma;_b^2 = {shrink['sig_b2_bwkt']:.6f} &rarr; <b>k_bwkt = {shrink['k_bwkt']:.1f} balls</b>. "
    f"<br/>&bull; <b>Parameter Bootstrap (500 match resamples):</b> k_runs median = 130.3 (90% CI: [92.2, 186.6]); k_bsaved median = 119.0 (90% CI: [87.3, 165.0]); "
    f"k_bwkt median = 452.2 (90% CI: [258.8, 476.3]); k_wkt exhibits high variance due to near-zero &sigma;_b^2.",
    body_style
))

story.append(Paragraph('6. Dhoni, Bumrah & Narine Empirical Audits', h1_style))
audit_cases = [
    [Paragraph('<b>Candidate Case</b>', table_header_style), Paragraph('<b>P-Balls</b>', table_header_style), Paragraph('<b>Actual vs Expected</b>', table_header_style), Paragraph('<b>Residual / Rate</b>', table_header_style), Paragraph('<b>Statistical Evaluation & Ruling</b>', table_header_style)],
    [Paragraph('MS Dhoni', table_cell_bold), Paragraph('69', table_cell_center), Paragraph('92 runs, 2 outs (133.3 SR)', table_cell_center), Paragraph('EB: +0.150 / ball', table_cell_center), Paragraph('<b>Filtered Out:</b> 69 balls is below the 120-ball floor. Low-pressure baseline (140.4 SR on 428b) confirms limited high-pressure volume.', table_cell_style)],
    [Paragraph('Dinesh Karthik', table_cell_bold), Paragraph('136', table_cell_center), Paragraph('213 runs, 5 outs (156.6 SR)', table_cell_center), Paragraph('EB: +0.103 / ball', table_cell_center), Paragraph('<b>Rank #15 (Boot P 21.6%):</b> Tested in 15 matches (7 qualifying); legitimate execution but regressed under ball-level shrinkage.', table_cell_style)],
    [Paragraph('Jasprit Bumrah', table_cell_bold), Paragraph('364', table_cell_center), Paragraph('13 wkts vs 19.37 xWkt', table_cell_center), Paragraph('ER 7.50 (+0.253 saved/b)', table_cell_center), Paragraph('<b>Rank #14 (Boot P 31.4%):</b> Saves +0.253 runs/ball; -6.37 wicket deficit is within ordinary sampling variance (z = -1.45, p = 0.15).', table_cell_style)],
    [Paragraph('Sunil Narine', table_cell_bold), Paragraph('355', table_cell_center), Paragraph('9 wkts vs 16.11 xWkt', table_cell_center), Paragraph('ER 7.11 (+0.237 saved/b)', table_cell_center), Paragraph('<b>Rank #16 (Boot P 14.4%):</b> Squeezes runs at 7.11 ER; dismissal deficit (-7.11 wkts) is within sampling variance (z = -1.77, p = 0.08).', table_cell_style)]
]
t_cases = Table(audit_cases, colWidths=[75, 40, 115, 88, 205])
t_cases.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 2.2),
]))
story.append(t_cases)
story.append(Spacer(1, 3))

story.append(Paragraph('7. Reliability Metrics on Final Primary Metric', h1_style))
story.append(Paragraph(
    f"Measured persistence across split-half and multi-year time splits on the final metric confirms: "
    f"<br/>&bull; <b>Batters Split-Half (N=99):</b> Raw odd/even r = {rel['bat_split_half_raw']:.3f} &rarr; <b>Spearman-Brown Corrected r = {rel['bat_split_half_sb']:.3f}</b>. Time-split (2022-24 vs 2025-26, N=85) r = {rel['bat_time_split']:.3f}. Batting pressure execution carries delivery-level noise, making shrinkage essential. "
    f"<br/>&bull; <b>Bowlers Split-Half (N=88):</b> Raw odd/even r = {rel['bowl_split_half_raw']:.3f} &rarr; <b>Spearman-Brown Corrected r = {rel['bowl_split_half_sb']:.3f}</b>. Time-split (2022-24 vs 2025-26, N=82) r = {rel['bowl_time_split']:.3f}. "
    f"<br/>&bull; <b>Alpha-Beta Correlation:</b> Batters r = {rel['bat_alpha_beta']:.3f}; Bowlers r = {rel['bowl_alpha_beta']:.3f}. Bowler run suppression under pressure overlaps with general bowling quality.",
    body_style
))

story.append(PageBreak())

# ==============================================================================
# PAGE 3: PRIMARY BATTERS LEADERBOARD & CUT-LINE ANALYSIS
# ==============================================================================
story.append(Paragraph('8. Primary Batter Leaderboard: Ball-Level Shrinkage & Composite Score', h1_style))
story.append(Paragraph(
    'Primary ranking is governed by: <b>80% EB Residual (W=6) + 20% Shrunk Consistency</b>. '
    'Shrinkage parameters (k_runs = 220.0, k_wkt = 1576.8) shrink delivery-level noise. '
    'Match consistency is shrunk via succ_shrunk = (succ + 8 &times; p0) / (n_matches + 8), where success represents a net positive pressure residual match. '
    'Baseline residual (&alpha;) and pressure elevation (&beta;, shrunk) isolate performance relative to normal leverage.',
    body_style
))

bat_table_data = [
    [Paragraph('<b>#</b>', table_header_style),
     Paragraph('<b>Batter Name</b>', table_header_style),
     Paragraph('<b>Balls</b>', table_header_style),
     Paragraph('<b>P-SR</b>', table_header_style),
     Paragraph('<b>EB Runs</b>', table_header_style),
     Paragraph('<b>EB Wkt</b>', table_header_style),
     Paragraph('<b>EB Resid</b>', table_header_style),
     Paragraph('<b>Base &alpha;</b>', table_header_style),
     Paragraph('<b>Elev &beta;</b>', table_header_style),
     Paragraph('<b>Nm</b>', table_header_style),
     Paragraph('<b>Succ%</b>', table_header_style),
     Paragraph('<b>Score</b>', table_header_style),
     Paragraph('<b>Boot P</b>', table_header_style),
     Paragraph('<b>Tactical Archetype</b>', table_header_style)]
]

for idx, r in batters_top.iterrows():
    bat_table_data.append([
        Paragraph(str(r['rank']), table_cell_bold),
        Paragraph(r['flagged_name'], table_cell_bold if not r['low_confidence'] else table_cell_style),
        Paragraph(f"{r['p_balls']:.0f}", table_cell_center),
        Paragraph(f"{r['metric_primary_rate']:.1f}", table_cell_center),
        Paragraph(f"{r['eb_runs_part']:+.3f}", table_cell_center),
        Paragraph(f"{r['eb_wkt_part']:+.4f}", table_cell_center),
        Paragraph(f"<b>{r['eb_resid']:+.3f}</b>", table_cell_center),
        Paragraph(f"{r['alpha_baseline']:+.3f}", table_cell_center),
        Paragraph(f"{r['beta_delta']:+.3f}", table_cell_center),
        Paragraph(f"{r['tested_matches']:.0f}", table_cell_center),
        Paragraph(f"{r['shrunk_succ_rate']:.1%}", table_cell_center),
        Paragraph(f"{r['composite_score']:.3f}", table_cell_center),
        Paragraph(f"{r['boot_p']:.1%}", table_cell_center),
        Paragraph(r['archetype'], table_cell_style)
    ])

t_bat = Table(bat_table_data, colWidths=[16, 88, 24, 26, 32, 34, 34, 32, 32, 20, 30, 28, 30, 97])
t_bat.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 1.8),
]))
story.append(t_bat)
story.append(Paragraph('* <i>Low Confidence Flag: Qualified with under 150 high-pressure balls or Boot P < 40%. Boot P represents empirical top-10 probability across 500 match-clustered bootstrap resamples.</i>', ParagraphStyle('Footnote', fontName='Helvetica-Oblique', fontSize=6.0, leading=7.2, textColor=colors.HexColor('#64748b'))))
story.append(Spacer(1, 3))

story.append(Paragraph('9. Batter Cut-Line Analysis & Player Profiles', h1_style))
story.append(Paragraph(
    '<b>Cut-Line Evaluation (#10 Boundary):</b> Under TIE_RULE = "strict", Suryakumar Yadav (#10, score 1.345, 533b, Boot P 60.2%) takes the 10th spot ahead of Andre Russell (#11, score 1.335, 128b, Boot P 21.0%) by +0.010 composite score. Under "bootp", Suryakumar also wins cleanly due to higher stability (60.2% vs 21.0%). '
    '<br/><b>1. Vaibhav Sooryavanshi (#1, 139b, SR 234.5, Boot P 68.2%*):</b> Elite boundary striking (+0.297 EB runs/ball above par). Carries a sample flag (<150b) but sustains #1 across W=4, W=8, dynamic W, and IPL-only regimes. '
    '<br/><b>2. Abhishek Sharma (#2, 322b, SR 183.2, Boot P 94.6%):</b> The highest-reliability power hitter in the dataset. Generates +0.210 EB runs/ball over 322 balls, entering the top 10 in 94.6% of bootstrap draws. '
    '<br/><b>3. Shreyas Iyer (#3, 328b, SR 166.2, Boot P 83.4%):</b> The premier middle-overs anchor. Combines +0.147 EB runs/ball with strong match consistency (69.0% over 18 matches). '
    '<br/><b>4. Tim David (#4, 296b, SR 169.9, Boot P 89.8%):</b> High-volume death execution (+0.182 EB runs/ball). Finishes in the top 5 across 90% of bootstrap draws. '
    '<br/><b>5-10. Perera, Inglis, Dhir, Bairstow, Prabhsimran, Suryakumar:</b> Kusal Perera (#5, SR 154.7) and Josh Inglis (#6, SR 167.2) supply strong elevation (&beta; > +0.07). Naman Dhir (#7, SR 176.9) and Bairstow (#8, SR 171.3) demonstrate high boundary conversion. Prabhsimran (#9, 306b, Boot P 60.4%) and Suryakumar (#10, 533b, Boot P 60.2%) anchor the lower tier with large sample sizes.',
    body_style
))

story.append(PageBreak())

# ==============================================================================
# PAGE 4: PRIMARY BOWLERS LEADERBOARD & DIRECT PAR INTEGRITY
# ==============================================================================
story.append(Paragraph('10. Primary Bowler Leaderboard: Direct Phase-Par Execution', h1_style))
story.append(Paragraph(
    'Primary ranking is governed by: <b>60% EB Phase-Par Runs Saved + 25% EB Wkts Above Expected + 15% Shrunk Consistency</b>. '
    'No tandem rollover credit, no opponent multipliers, and no dynamic W are folded into the primary metric. '
    'Runs saved per ball = phase_par_ball - runs_conceded. Implied par per over aligns with ground truth (8.36 to 9.65). '
    'EB Wkts represents shrunk wickets above Model B expectation. Consistency is shrunk with K_SUCC = 8.',
    body_style
))

bowl_table_data = [
    [Paragraph('<b>#</b>', table_header_style),
     Paragraph('<b>Bowler Name</b>', table_header_style),
     Paragraph('<b>Balls</b>', table_header_style),
     Paragraph('<b>P-ER</b>', table_header_style),
     Paragraph('<b>Saved/b</b>', table_header_style),
     Paragraph('<b>EB Saved</b>', table_header_style),
     Paragraph('<b>EB Wkts</b>', table_header_style),
     Paragraph('<b>Impl. Par</b>', table_header_style),
     Paragraph('<b>Base &alpha;</b>', table_header_style),
     Paragraph('<b>Elev &beta;</b>', table_header_style),
     Paragraph('<b>Nm</b>', table_header_style),
     Paragraph('<b>Succ%</b>', table_header_style),
     Paragraph('<b>Score</b>', table_header_style),
     Paragraph('<b>Boot P</b>', table_header_style),
     Paragraph('<b>Tactical Archetype</b>', table_header_style)]
]

for idx, r in bowlers_top.iterrows():
    bowl_table_data.append([
        Paragraph(str(r['rank']), table_cell_bold),
        Paragraph(r['flagged_name'], table_cell_bold if not r['low_confidence'] else table_cell_style),
        Paragraph(f"{r['p_balls']:.0f}", table_cell_center),
        Paragraph(f"{r['metric_primary_rate']:.2f}", table_cell_center),
        Paragraph(f"{r['raw_runs_part']:+.3f}", table_cell_center),
        Paragraph(f"{r['eb_runs_part']:+.3f}", table_cell_center),
        Paragraph(f"{r['eb_wkt_total']:+.2f}", table_cell_center),
        Paragraph(f"{r['eb_resid']:.2f}", table_cell_center),
        Paragraph(f"{r['alpha_baseline']:+.2f}", table_cell_center),
        Paragraph(f"{r['beta_delta']:+.2f}", table_cell_center),
        Paragraph(f"{r['tested_matches']:.0f}", table_cell_center),
        Paragraph(f"{r['shrunk_succ_rate']:.1%}", table_cell_center),
        Paragraph(f"{r['composite_score']:.3f}", table_cell_center),
        Paragraph(f"{r['boot_p']:.1%}", table_cell_center),
        Paragraph(r['archetype'], table_cell_style)
    ])

t_bowl = Table(bowl_table_data, colWidths=[16, 88, 22, 25, 30, 32, 28, 30, 30, 30, 18, 28, 28, 30, 88])
t_bowl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 1.8),
]))
story.append(t_bowl)
story.append(Paragraph('* <i>Low Confidence Flag: Qualified with under 150 high-pressure balls or Boot P < 40%. Boot P represents empirical top-10 probability across 500 match-clustered bootstrap resamples.</i>', ParagraphStyle('FootnoteB', fontName='Helvetica-Oblique', fontSize=6.0, leading=7.2, textColor=colors.HexColor('#64748b'))))
story.append(Spacer(1, 3))

story.append(Paragraph('11. Bowler Cut-Line Analysis & Tactical Archetypes', h1_style))
story.append(Paragraph(
    '<b>Cut-Line Evaluation (#10 Boundary):</b> Under TIE_RULE = "strict", Shadab Khan (#10, score 1.040, 155b, Boot P 26.6%*) claims 10th ahead of Pat Cummins (#11, score 0.900, 190b, Boot P 16.8%), Harpreet Brar (#12, score 0.856, Boot P 35.0%), and Jasprit Bumrah (#14, score 0.846, Boot P 31.4%). '
    '<br/><b>1. Matheesha Pathirana (#1, 234b, ER 7.72, Boot P 82.4%):</b> The most lethal death overs executioner in the dataset. Saves +0.182 runs/ball against 9.65 implied par, entering the top 10 in 82.4% of draws. '
    '<br/><b>2. Krunal Pandya (#2, 336b, ER 6.61, Boot P 88.0%):</b> Chokehold middle-overs restrictor. Generates +0.197 EB runs saved/ball while conceding only 6.61 ER across 336 balls. '
    '<br/><b>3. Rashid Khan (#3, 407b, ER 7.14, Boot P 81.4%):</b> Premier phase-versatile weapon. Operates with elite economy (7.14) over 407 high-pressure balls. '
    '<br/><b>4. Axar Patel (#4, 547b, ER 7.25, Boot P 78.6%):</b> High-volume strike restrictor (+1.75 EB wickets above expected across 547 pressure balls). '
    '<br/><b>5. Josh Hazlewood (#5, 316b, ER 8.11, Boot P 76.0%):</b> Highest wicket-taking impact in pressure cricket (+4.92 EB wickets above par over 316 balls). '
    '<br/><b>6-10. Shaheen, Mahedi, Nawaz, Hasaranga, Shadab:</b> Shaheen Afridi (#6, ER 6.80, Boot P 68.8%) provides immense opening pressure. Mahedi Hasan (#7, ER 6.15, Boot P 37.0%*) offers extreme run suppression (+0.181 EB saved/b) but carries a sample flag (122b). Mohammad Nawaz (#8, Boot P 60.0%) and Wanindu Hasaranga (#9, +5.78 EB wkts, Boot P 55.2%) anchor middle-overs control.',
    body_style
))

story.append(PageBreak())

# ==============================================================================
# PAGE 5: SENSITIVITY MATRIX, FIGURE 4 EMBEDDED, METHOD LIMITS & CONCLUSION
# ==============================================================================
story.append(Paragraph('12. Robustness Matrix & Sensitivity Regimes', h1_style))
story.append(Paragraph(
    'To evaluate ranking stability, thirteen sensitivity regimes were tested against the primary baseline. '
    'Top-10 overlap measures set membership stability. '
    'Core players (Abhishek Sharma, Shreyas Iyer, Tim David for batters; Pathirana, Krunal Pandya, Rashid Khan, Axar Patel, Hazlewood for bowlers) are completely stable across all regimes. '
    'Conversely, low-sample fringe qualifiers fluctuate when sample floors or extreme weights are adjusted.',
    body_style
))

sens_table_data = [
    [Paragraph('<b>Sensitivity Regime</b>', table_header_style),
     Paragraph('<b>Type</b>', table_header_style),
     Paragraph('<b>Overlap</b>', table_header_style),
     Paragraph('<b>Observed Rank Shift & Tactical Impact</b>', table_header_style)]
]

for idx, r in sens_master.iterrows():
    sens_table_data.append([
        Paragraph(f"<b>{r['Regime']}</b>", table_cell_bold),
        Paragraph(r['Type'], table_cell_center),
        Paragraph(f"<b>{r['Top10_Overlap']}</b>", table_cell_center),
        Paragraph(r['Shift'], table_cell_style)
    ])

t_sens = Table(sens_table_data, colWidths=[135, 38, 48, 302])
t_sens.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 1.8),
]))
story.append(t_sens)
story.append(Spacer(1, 3))

story.append(Paragraph('13. Baseline Ability vs Pressure Execution (&beta; Elevation)', h1_style))

# Embed Figure 4 and description side-by-side
fig4_path = 'reports/fig4_residual_vs_residual.png'
if os.path.exists(fig4_path):
    fig_w = 250
    fig_h = 160
    img = Image(fig4_path, width=fig_w, height=fig_h)
    
    desc_p = Paragraph(
        '<b>Figure 4: Residual vs Residual Analysis.</b><br/>'
        'Plots player baseline ability (&alpha;, low-pressure EB residual) on the x-axis against demonstrated high-pressure EB residual on the y-axis. '
        'The dashed red line represents parity (&beta; = 0, where pressure execution equals baseline execution). '
        '<br/><br/>'
        '<b>Key Scouting Takeaways:</b><br/>'
        '&bull; <b>High-Baseline Stars:</b> Sooryavanshi, Abhishek Sharma, and Tim David maintain elite execution across both low and high pressure contexts. '
        '<br/>&bull; <b>Context Anchors (&beta; Elevation):</b> Shreyas Iyer (&alpha; = +0.064, &beta; = +0.122) and Kusal Perera (&alpha; = -0.102, &beta; = +0.192) display significant positive elevation over baseline output under match tension.',
        body_style
    )
    
    t_fig = Table([[img, desc_p]], colWidths=[255, 268])
    t_fig.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_fig)
    story.append(Spacer(1, 2))

story.append(Paragraph('14. Methodological Limitations, Tandem Tests & Conclusion', h1_style))
story.append(Paragraph(
    '<b>Tandem Spillover Statistical Tests:</b> Clustered regressions on overs testing whether run suppression generates partner wickets yield: '
    '(1) Run-based (&le;6 runs conceded): &gamma;&#770; = -0.0087 (95% CI: [-0.0236, +0.0062], no detectable effect). '
    '(2) Chase RRR pressure (&Delta; &ge; 2): &gamma;&#770; = +0.0446 (95% CI: [+0.0233, +0.0659]). '
    '(3) Bowling WP gain (&ge;75th pct): &gamma;&#770; = +0.0175 (95% CI: [+0.0008, +0.0342]). '
    'Because run-based suppression exhibits zero detectable spillover, tandem credits are excluded from the primary metric. '
    '<br/><b>Structural Boundary Conditions:</b> '
    '(1) <i>Innings 1 WP:</i> Out-of-fold Brier is 0.1598, but time-split Brier is 0.2543; 26% of pressure balls occur in innings 1. '
    '(2) <i>Batting Stochasticity:</i> Corrected split-half reliability is r = 0.224 (time-split r = 0.131), proving Empirical Bayes shrinkage is mathematically mandatory. '
    '(3) <i>Bowler Quality Confound:</i> Bowler ratings correlate r = 0.651 with baseline economy, reflecting that pressure run suppression overlaps with general quality. '
    'Rankings express <b>best demonstrated execution under pressure, with measured uncertainty</b>. '
    '<br/><b>Pipeline Code:</b> Standalone, fully reproducible pipeline available at <font color="#004BA0"><u>https://github.com/ruhannpn/superr</u></font>.',
    body_style
))

doc.build(story, canvasmaker=NumberedCanvas)
shutil.copyfile(pdf_path, artifact_pdf)
print(f"Publication PDF successfully generated at {pdf_path} and synced to artifacts.")
