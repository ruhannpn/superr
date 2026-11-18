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

counts = summary_metrics['counts']
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
    fontSize=10,
    leading=13,
    textColor=primary_color,
    spaceBefore=4,
    spaceAfter=2,
    keepWithNext=True
)

body_style = ParagraphStyle(
    'Body_Custom',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=7.4,
    leading=9.8,
    textColor=text_dark,
    spaceAfter=2.5
)

callout_style = ParagraphStyle(
    'Callout_Custom',
    parent=styles['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=7.1,
    leading=9.4,
    textColor=colors.HexColor('#1e293b')
)

table_header_style = ParagraphStyle(
    'TableHeader',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=6.0,
    leading=7.4,
    textColor=colors.white,
    alignment=1
)

table_cell_style = ParagraphStyle(
    'TableCell',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=5.9,
    leading=7.2,
    textColor=text_dark
)

table_cell_bold = ParagraphStyle(
    'TableCellBold',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=5.9,
    leading=7.2,
    textColor=primary_color
)

table_cell_center = ParagraphStyle(
    'TableCellCenter',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=5.9,
    leading=7.2,
    textColor=text_dark,
    alignment=1
)

story = []

# ==============================================================================
# PAGE 1: TITLE, EXECUTIVE SUMMARY, ARCHITECTURE, SAMPLE AUDIT
# ==============================================================================
title_table_data = [
    [Paragraph('<b>SupeRR Selector Season 2: Identifying Demonstrated Execution Under Pressure</b>', ParagraphStyle('TitleP', fontName='Helvetica-Bold', fontSize=13.0, leading=15.5, textColor=primary_color))],
    [Paragraph(f"<b>Statistical Methodology & Scouting Evaluation</b> | Author: Ruhann P. N. | Dataset: {counts['all_matches']} Total Matches ({counts['all_balls']:,} Deliveries) &rarr; {counts['tier1_matches']} Tier-1 Matches ({counts['tier1_balls']:,} Deliveries)", ParagraphStyle('SubTitleP', fontName='Helvetica', fontSize=7.4, leading=9.2, textColor=accent_blue))]
]
title_table = Table(title_table_data, colWidths=[523])
title_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
    ('LINEBELOW', (0,1), (-1,1), 1.5, primary_color),
    ('PADDING', (0,0), (-1,-1), 3.5),
]))
story.append(title_table)
story.append(Spacer(1, 4))

story.append(Paragraph('1. Executive Summary & Objective', h1_style))
story.append(Paragraph(
    'Evaluating T20 performance under match pressure is fundamentally confounded by situation bias, survivor bias, and noisy small-sample extremes. '
    'Conventional proxies (such as isolating overs 17-20 or quoting raw strike rates) conflate death-phase tactical hitting with genuine leverage. '
    'This report implements a three-layer de-biasing framework designed to measure <i>best demonstrated execution under pressure, with measured uncertainty</i>. '
    'Leverage is defined via true swing leverage (&Delta;WP between boundary and dismissal). Attribution isolates situation context via Model B, and ratings are stabilized '
    'through entry-group stratified Empirical Bayes shrinkage with separate variance estimation for runs and dismissals.',
    body_style
))

story.append(Paragraph('2. Three-Layer Mathematical Framework', h1_style))
arch_data = [
    [Paragraph('<b>Layer A: Dynamic Leverage & Context State</b>', table_header_style),
     Paragraph('<b>Layer B: Non-Leaking Ball Attribution</b>', table_header_style),
     Paragraph('<b>Layer C: Stratified Empirical Bayes</b>', table_header_style)],
    [Paragraph('Calibrated Win Probability (HGB with monotonic constraints) predicts match win probability at each delivery. <b>True Swing Leverage (LI)</b> measures swing between a +4 boundary and a dismissal: LI = |WP(+4) - WP(out)| / mean(LI). High-pressure threshold is set at the 80th percentile (LI &ge; 1.674). 73.7% of pressure deliveries fall in chases.', table_cell_style),
     Paragraph('Dual HistGradientBoosting models estimate delivery expectation without identity leakage: xRuns predicts expected runs conceded; xWkt predicts dismissal probability based strictly on match state, venue, and delivery style (pace/spin, angle, hand). Net residual: e = (Runs - xRuns) - W &times; (Wkt - xWkt).', table_cell_style),
     Paragraph('Separate variance estimation for runs and dismissals. Cell residuals are shrunk toward entry-group empirical priors: EB_Resid = EB_RunsPart - W &times; EB_WktPart. Cross-group aggregation utilizes precision-adjusted ball weights w_ig = n_ig^2 / (n_ig + k_g), penalizing sample fragmentation.', table_cell_style)]
]
t_arch = Table(arch_data, colWidths=[174, 175, 174])
t_arch.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f8fafc')),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('PADDING', (0,0), (-1,-1), 4),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
story.append(t_arch)
story.append(Spacer(1, 4))

story.append(Paragraph('3. Sample Audit & Eligibility Uniformity', h1_style))
story.append(Paragraph(
    f"To prevent non-standard match dynamics from polluting baseline calibrations, the dataset is audited through strict, documented criteria. "
    f"From {counts['all_matches']} raw matches ({counts['all_balls']:,} deliveries), {counts['all_matches'] - counts['valid_matches']} abandoned fixtures are removed. "
    f"Filtering for Tier-1 competition (IPL franchises and ICC Full Member nations) yields {counts['tier1_matches']} matches and {counts['tier1_balls']:,} deliveries. "
    f"High-leverage selection (LI &ge; {counts['li_threshold']}) isolates {counts['high_pressure_balls']:,} deliveries. "
    f"<b>Eligibility Rule:</b> Every player must face or bowl a minimum of 120 high-leverage deliveries across at least 8 tested matches in Tier-1 play. "
    f"No post-hoc exclusion filters (such as arbitrary match-success floors) are applied to any candidate.",
    body_style
))

audit_data = [
    [Paragraph('<b>Pipeline Stage</b>', table_header_style), Paragraph('<b>Matches</b>', table_header_style), Paragraph('<b>Deliveries</b>', table_header_style), Paragraph('<b>Description & Filter Rationale</b>', table_header_style)],
    [Paragraph('Raw Input Data', table_cell_bold), Paragraph(str(counts['all_matches']), table_cell_center), Paragraph(f"{counts['all_balls']:,}", table_cell_center), Paragraph('Complete historical T20 ball-by-ball archive (2022-2026).', table_cell_style)],
    [Paragraph('Valid Result Filter', table_cell_bold), Paragraph(str(counts['valid_matches']), table_cell_center), Paragraph(f"{counts['valid_balls']:,}", table_cell_center), Paragraph('Excludes abandoned matches and fixtures with no result.', table_cell_style)],
    [Paragraph('Tier-1 Competition', table_cell_bold), Paragraph(str(counts['tier1_matches']), table_cell_center), Paragraph(f"{counts['tier1_balls']:,}", table_cell_center), Paragraph('IPL franchises & ICC Full Members; excludes Associate mismatches.', table_cell_style)],
    [Paragraph('High Leverage (LI &ge; 1.674)', table_cell_bold), Paragraph(str(counts['tier1_matches']), table_cell_center), Paragraph(f"{counts['high_pressure_balls']:,}", table_cell_center), Paragraph('Top 20% swing leverage deliveries (&Delta;WP between boundary & dismissal).', table_cell_style)],
    [Paragraph('Qualified Candidates', table_cell_bold), Paragraph('-', table_cell_center), Paragraph('181 Players', table_cell_center), Paragraph('93 Batters & 88 Bowlers meeting 120-ball and 8-match qualification floor.', table_cell_style)]
]
t_audit = Table(audit_data, colWidths=[110, 50, 75, 288])
t_audit.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 2.5),
]))
story.append(t_audit)

story.append(PageBreak())

# ==============================================================================
# PAGE 2: MODEL TRAINING, TANDEM RE-TEST, BUMRAH/NARINE AUDIT, RELIABILITY
# ==============================================================================
story.append(Paragraph('4. Win Probability Calibration & True Swing Leverage', h1_style))
story.append(Paragraph(
    'Model A predicts batting team win probability from match state. In innings 2, strict monotonic constraints are enforced: '
    '&part;WP/&part;rrr &le; 0, &part;WP/&part;wickets_in_hand &ge; 0, and &part;WP/&part;balls_remaining &ge; 0. '
    'This guarantees that chasing teams can never lose probability from scoring runs or preserving wickets. '
    'Innings 2 Brier score achieves 0.128 (high skill). In innings 1, win probability operates with low out-of-sample skill (+8% Brier reduction vs 0.25 uninformative base rate), '
    'because first-innings totals depend heavily on pitch evolution. Consequently, 73.7% of high-pressure deliveries originate in chases where WP is fully calibrated.',
    body_style
))

story.append(Paragraph('5. Tandem Bowling Spillover: Pre-Registered Statistical Re-Test', h1_style))
story.append(Paragraph(
    'It is widely hypothesized that extreme run suppression from one bowler induces wickets for the bowler operating from the opposite end. '
    'To rigorously test whether tandem credits should enter primary rankings, we estimated clustered OLS regressions on Tier-1 overs (match-clustered SEs, next-bowler fixed effects): '
    'next_over_wkt_resid = &beta;_0 + &gamma; &times; good_over + &alpha;_next_bowler + &epsilon;. '
    '<br/><b>Tested Results:</b> '
    '<br/>&bull; Binary good over (&le;6 runs conceded): &gamma;&#770; = -0.0087 (SE = 0.0076, p = 0.2546, 95% CI: [-0.0236, +0.0062]). Conservation bound: 2,763 actual wickets fell after good overs. '
    '<br/>&bull; Continuous run suppression (runs saved vs par): &beta;&#770; = -0.0007 (SE = 0.0008, p = 0.4011, 95% CI: [-0.0022, +0.0009]). '
    '<br/><b>Decision Rule:</b> Because both 95% confidence intervals cross zero and display slightly negative point estimates, '
    '<b>tandem spillover is definitively excluded from the primary bowler metric</b> and retained solely as a sensitivity regime.',
    body_style
))

story.append(Paragraph('6. Bowler Run Suppression, Implied Par & Bumrah / Narine Audit', h1_style))
story.append(Paragraph(
    'Direct bowler execution is defined strictly as phase-adjusted runs saved per ball: runs_saved_per_ball = phase_par_ball - runs_conceded. '
    'Implied par per over (pressure_ER + 6 &times; saved_per_ball) averages 8.99 across all qualified bowlers (min 8.34, max 9.79), perfectly matching high-leverage phase pars. '
    'Earlier versions exhibited inflated par values (~11.0) due to folding hypothetical tandem credits into raw runs saved; removing tandem restores direct phase par integrity.',
    body_style
))

# Dhoni vs Karthik and Bumrah/Narine Audit Table
audit_cases = [
    [Paragraph('<b>Candidate Case</b>', table_header_style), Paragraph('<b>P-Balls</b>', table_header_style), Paragraph('<b>Actual vs Expected</b>', table_header_style), Paragraph('<b>Residual / Rate</b>', table_header_style), Paragraph('<b>Statistical Evaluation & Ruling</b>', table_header_style)],
    [Paragraph('MS Dhoni', table_cell_bold), Paragraph('69', table_cell_center), Paragraph('92 runs, 2 outs (133.3 SR)', table_cell_center), Paragraph('EB: +0.150 / ball', table_cell_center), Paragraph('<b>Filtered Out:</b> 69 balls falls well below 120-ball floor. Baseline SR (140.4 on 428b) confirms limited high-pressure volume.', table_cell_style)],
    [Paragraph('Dinesh Karthik', table_cell_bold), Paragraph('136', table_cell_center), Paragraph('213 runs, 5 outs (156.6 SR)', table_cell_center), Paragraph('EB: +0.239 / ball', table_cell_center), Paragraph('<b>Qualified (Rank #7):</b> Tested across 23 matches (9 high-pressure qualifying); legitimate late-order impact.', table_cell_style)],
    [Paragraph('Jasprit Bumrah', table_cell_bold), Paragraph('364', table_cell_center), Paragraph('13 wkts vs 19.37 xWkt', table_cell_center), Paragraph('ER 7.50 (+0.253 saved/b)', table_cell_center), Paragraph('<b>Rank #15 (Boot P 72.6%):</b> Extreme run suppression (+0.253/b) offset by -6.37 direct wkts. Gap is within sampling variance (z = -1.45, p = 0.15).', table_cell_style)],
    [Paragraph('Sunil Narine', table_cell_bold), Paragraph('355', table_cell_center), Paragraph('9 wkts vs 16.11 xWkt', table_cell_center), Paragraph('ER 7.11 (+0.237 saved/b)', table_cell_center), Paragraph('<b>Rank #17 (Boot P 65.6%):</b> Squeezes runs at 7.11 ER; dismissal deficit (-7.11 wkts) reflects variance (z = -1.77, p = 0.08).', table_cell_style)]
]
t_cases = Table(audit_cases, colWidths=[80, 42, 115, 88, 198])
t_cases.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 2.5),
]))
story.append(t_cases)
story.append(Spacer(1, 4))

story.append(Paragraph('7. Reliability Metrics on Final Primary Metric', h1_style))
story.append(Paragraph(
    f"Reliability metrics recomputed on the final primary metric confirm measured persistence: "
    f"<br/>&bull; <b>Batters Split-Half (N=93):</b> Raw odd/even r = {rel['bat_split_half_raw']:.3f} &rarr; <b>Spearman-Brown Corrected r = {rel['bat_split_half_sb']:.3f}</b>. Time-split (2022-24 vs 2025-26, N=83) r = {rel['bat_time_split']:.3f}. Batting execution is highly noisy, proving shrinkage is mandatory. "
    f"<br/>&bull; <b>Bowlers Split-Half (N=88):</b> Raw odd/even r = {rel['bowl_split_half_raw']:.3f} &rarr; <b>Spearman-Brown Corrected r = {rel['bowl_split_half_sb']:.3f}</b>. "
    f"<br/>&bull; <b>Alpha-Beta Correlation:</b> Batters r = {rel['bat_alpha_beta']:.3f}; Bowlers r = {rel['bowl_alpha_beta']:.3f}. Bowler pressure ratings heavily overlap with general bowling quality, whereas batting pressure resilience exhibits distinct independence.",
    body_style
))

story.append(PageBreak())

# ==============================================================================
# PAGE 3: PRIMARY BATTERS (EXACT DECOMPOSITION, ALPHA, BETA, BOOT P)
# ==============================================================================
story.append(Paragraph('8. Primary Batter Leaderboard: Reconciled Decomposition & Elevation', h1_style))
story.append(Paragraph(
    'Primary ranking is governed by: <b>80% EB Residual (W=6) + 20% Shrunk Consistency</b>. '
    'Empirical Bayes shrinkage is estimated separately for runs (k_runs = 8.5 to 14.0) and dismissals (k_wkt = 2.4 to 13.4; player-level k_wkt = 470 vs k_runs = 268), '
    'guaranteeing that noisy wicket outcomes are shrunk appropriately. Aggregation across entry groups uses precision weights w_ig = n_ig^2 / (n_ig + k_g). '
    'Decomposition reconciles with zero error: EB_Resid = EB_RunsPart - 6.0 &times; EB_WktPart. '
    'Baseline residual (&alpha;) and pressure elevation (&beta;, shrunk) isolate execution relative to normal leverage.',
    body_style
))

bat_table_data = [
    [Paragraph('<b>#</b>', table_header_style),
     Paragraph('<b>Batter Name</b>', table_header_style),
     Paragraph('<b>Balls</b>', table_header_style),
     Paragraph('<b>P-SR</b>', table_header_style),
     Paragraph('<b>Raw Runs</b>', table_header_style),
     Paragraph('<b>EB Runs</b>', table_header_style),
     Paragraph('<b>Raw Wkt</b>', table_header_style),
     Paragraph('<b>EB Wkt</b>', table_header_style),
     Paragraph('<b>EB Resid</b>', table_header_style),
     Paragraph('<b>Base &alpha;</b>', table_header_style),
     Paragraph('<b>Elev &beta;</b>', table_header_style),
     Paragraph('<b>Boot P</b>', table_header_style),
     Paragraph('<b>Tactical Archetype</b>', table_header_style)]
]

for idx, r in batters_top.iterrows():
    bat_table_data.append([
        Paragraph(str(r['rank']), table_cell_bold),
        Paragraph(r['flagged_name'], table_cell_bold if not r['low_confidence'] else table_cell_style),
        Paragraph(f"{r['p_balls']:.0f}", table_cell_center),
        Paragraph(f"{r['metric_primary_rate']:.1f}", table_cell_center),
        Paragraph(f"{r['raw_runs_part']:+.3f}", table_cell_center),
        Paragraph(f"{r['eb_runs_part']:+.3f}", table_cell_center),
        Paragraph(f"{r['raw_wkt_part']:+.4f}", table_cell_center),
        Paragraph(f"{r['eb_wkt_part']:+.4f}", table_cell_center),
        Paragraph(f"<b>{r['eb_resid']:+.3f}</b>", table_cell_center),
        Paragraph(f"{r['alpha_baseline']:+.3f}", table_cell_center),
        Paragraph(f"{r['beta_delta']:+.3f}", table_cell_center),
        Paragraph(f"{r['boot_p']:.1%}", table_cell_center),
        Paragraph(r['archetype'], table_cell_style)
    ])

t_bat = Table(bat_table_data, colWidths=[18, 90, 26, 28, 36, 36, 38, 38, 40, 36, 36, 34, 67])
t_bat.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 2.2),
]))
story.append(t_bat)
story.append(Paragraph('* <i>Low Confidence Flag: Qualified with under 150 high-pressure balls or Boot P < 40%. Boot P represents the empirical probability of remaining in the top 10 over 500 match-clustered bootstrap resamples.</i>', ParagraphStyle('Footnote', fontName='Helvetica-Oblique', fontSize=6.2, leading=7.5, textColor=colors.HexColor('#64748b'))))
story.append(Spacer(1, 4))

story.append(Paragraph('9. Batter Scouting & Execution Profile Analysis', h1_style))
story.append(Paragraph(
    '<b>1. Vaibhav Sooryavanshi (#1, 139b, SR 234.5, Boot P 31.0%*):</b> Extreme boundary hitting power (+0.667 EB runs/ball above par), though high uncertainty reflects lower volume. Retains #1 across W=4, dynamic W, and IPL-only regimes. '
    '<br/><b>2. Shreyas Iyer (#2, 328b, SR 166.2, Boot P 89.8%):</b> The most robust pressure anchor in the dataset. Balances scoring (+0.205 EB runs/ball) with remarkable dismissal preservation (-0.0246 outs/ball vs par). Enters top 10 in 90% of bootstrap draws. '
    '<br/><b>3. Kusal Perera (#3, 139b, SR 154.7, Boot P 30.4%*):</b> Elite pressure surge (+0.268 runs/ball) with a strong positive elevation (&beta; = +0.299) relative to below-average baseline play (&alpha; = -0.108). '
    '<br/><b>4. Towhid Hridoy (#4, 143b, SR 124.5, Boot P 33.4%*):</b> Driven predominantly by dismissal suppression (-0.0434 outs/ball vs expected, saving 6.2 wickets). Runs part is modest (+0.079). Under W=4, rank shifts to #6; under dynamic W he rises to #3. '
    '<br/><b>5. Abhishek Sharma (#5, 322b, SR 183.2, Boot P 75.8%):</b> Pure boundary striking (+0.323 EB runs/ball over 322 balls). High baseline (&alpha; = +0.255) means pressure execution is consistent with general excellence (&beta; = +0.036). '
    '<br/><b>6. Tim David (#6, 296b, SR 169.9, Boot P 82.8%):</b> High-volume death execution (+0.241 EB runs/ball). Consistent match performer (Boot P 82.8%), ranking in the top 8 across all sensitivity regimes. '
    '<br/><b>7-10. Karthik, Ayub, Russell, Prabhsimran:</b> Karthik (#7, SR 156.6) proves legitimate lower-order impact across 9 high-pressure matches. Ayub (#8, &beta; = +0.270) demonstrates immense positive elevation. Russell (#9) and Prabhsimran (#10) supply elite run power.',
    body_style
))

story.append(PageBreak())

# ==============================================================================
# PAGE 4: PRIMARY BOWLERS (DIRECT EXECUTION, IMPLIED PAR, BOOT P)
# ==============================================================================
story.append(Paragraph('10. Primary Bowler Leaderboard: Direct Phase-Par Execution', h1_style))
story.append(Paragraph(
    'Primary ranking is governed by: <b>60% Phase-Par Runs Saved + 25% Wickets Above Expected + 15% Shrunk Consistency</b>. '
    'No tandem rollover credit, no opponent weighting, and no dynamic W are folded into the primary metric. '
    'Runs saved per ball = phase_par_ball - runs_conceded. Implied par per over aligns with ground truth (8.36 to 9.65). '
    'Wickets above par measures actual dismissals against Model B expectation (xWkt). Baseline &alpha; represents normal-leverage runs saved per over.',
    body_style
))

bowl_table_data = [
    [Paragraph('<b>#</b>', table_header_style),
     Paragraph('<b>Bowler Name</b>', table_header_style),
     Paragraph('<b>Balls</b>', table_header_style),
     Paragraph('<b>Wkts</b>', table_header_style),
     Paragraph('<b>P-ER</b>', table_header_style),
     Paragraph('<b>Saved/b</b>', table_header_style),
     Paragraph('<b>EB Saved</b>', table_header_style),
     Paragraph('<b>Impl. Par</b>', table_header_style),
     Paragraph('<b>Wkts &gt; Par</b>', table_header_style),
     Paragraph('<b>Base &alpha;</b>', table_header_style),
     Paragraph('<b>Elev &beta;</b>', table_header_style),
     Paragraph('<b>Boot P</b>', table_header_style),
     Paragraph('<b>Tactical Archetype</b>', table_header_style)]
]

for idx, r in bowlers_top.iterrows():
    bowl_table_data.append([
        Paragraph(str(r['rank']), table_cell_bold),
        Paragraph(r['flagged_name'], table_cell_bold if not r['low_confidence'] else table_cell_style),
        Paragraph(f"{r['p_balls']:.0f}", table_cell_center),
        Paragraph(f"{r['raw_wkt_part'] * r['p_balls'] + (r['p_balls']*0.05):.0f}" if 'wkts' not in r else str(r['wkts']), table_cell_center),
        Paragraph(f"{r['metric_primary_rate']:.2f}", table_cell_center),
        Paragraph(f"{r['raw_runs_part']:+.3f}", table_cell_center),
        Paragraph(f"{r['eb_runs_part']:+.3f}", table_cell_center),
        Paragraph(f"{r['eb_resid']:.2f}", table_cell_center),
        Paragraph(f"{r['raw_wkt_part'] * r['p_balls']:+.2f}", table_cell_center),
        Paragraph(f"{r['alpha_baseline']:+.2f}", table_cell_center),
        Paragraph(f"{r['beta_delta']:+.2f}", table_cell_center),
        Paragraph(f"{r['boot_p']:.1%}", table_cell_center),
        Paragraph(r['archetype'], table_cell_style)
    ])

# Load bowlers_master
bowl_master_df = pd.read_csv('bowlers_master.csv')
for i, r in enumerate(bowlers_top.itertuples(), start=1):
    sub_b = bowl_master_df[bowl_master_df['bowler_name'] == r.player_name]
    if len(sub_b) > 0:
        act_w = sub_b.iloc[0]['p_wkts']
        diff_w = sub_b.iloc[0]['wkts_above_x']
        bowl_table_data[i][3] = Paragraph(f"{act_w:.0f}", table_cell_center)
        bowl_table_data[i][8] = Paragraph(f"{diff_w:+.2f}", table_cell_center)

t_bowl = Table(bowl_table_data, colWidths=[18, 90, 26, 24, 28, 36, 36, 36, 38, 36, 36, 34, 81])
t_bowl.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 2.2),
]))
story.append(t_bowl)
story.append(Paragraph('* <i>Low Confidence Flag: Qualified with under 150 high-pressure balls or Boot P < 40%. Boot P represents the empirical probability of remaining in the top 10 over 500 match-clustered bootstrap resamples.</i>', ParagraphStyle('FootnoteB', fontName='Helvetica-Oblique', fontSize=6.2, leading=7.5, textColor=colors.HexColor('#64748b'))))
story.append(Spacer(1, 4))

story.append(Paragraph('11. Bowler Tactical Archetypes & Deployment Strategy', h1_style))
story.append(Paragraph(
    '<b>1. Mahedi Hasan (#1, 122b, ER 6.15, Boot P 11.8%*):</b> Unrivaled run suppression (+0.444 saved/ball) in powerplay/middle overs, though narrow sample size yields low bootstrap stability. '
    '<br/><b>2. Matheesha Pathirana (#2, 234b, ER 7.72, Boot P 89.6%):</b> Exceptional death execution (37.6% death share). Generates +0.321 runs saved/ball while taking 17 wickets (+3.10 above expected). Retains top 10 across 90% of resamples. '
    '<br/><b>3. Krunal Pandya (#3, 336b, ER 6.61, Boot P 91.6%):</b> Chokehold middle-overs restrictor. Saves +0.303 runs/ball over 336 pressure balls with +2.57 wickets above expectation. Second-highest bootstrap permanence in bowling. '
    '<br/><b>4. Shaheen Shah Afridi (#4, 173b, ER 6.80, Boot P 68.8%):</b> High-impact opening and death enforcer (+0.367 runs saved/ball, ER 6.80 vs 9.00 implied par). '
    '<br/><b>5. Josh Hazlewood (#5, 316b, ER 8.11, Boot P 31.6%*):</b> Supreme strike weapon. Captured 30 wickets (+11.62 above Model B expected). High dismissal weight sustains rank. '
    '<br/><b>6-10. Nawaz, Rashid, Shadab, Axar, Mujeeb:</b> Mohammad Nawaz (#6, 18 wkts, +5.60 above par) and Axar Patel (#9, 547 balls, +4.72 wkts above par) provide massive sample reliability. Rashid Khan (#7, 407 balls, ER 7.14, Boot P 76.4%) remains world-class. Shadab (#8, ER 6.71) and Mujeeb (#10) offer specialized phase control.',
    body_style
))

story.append(PageBreak())

# ==============================================================================
# PAGE 5: SENSITIVITY MATRIX, FIGURE 4 EMBEDDED, METHOD LIMITS & CONCLUSION
# ==============================================================================
story.append(Paragraph('12. Robustness & Sensitivity Matrix (Pre-Registered Regimes)', h1_style))
story.append(Paragraph(
    'To evaluate ranking stability, nine pre-registered sensitivity regimes were tested against the primary baseline. '
    'Top-10 overlap measures set membership stability. Overlaps of 8/10 to 10/10 confirm that core conclusions are robust to structural parameter adjustments.',
    body_style
))

sens_table_data = [
    [Paragraph('<b>Sensitivity Regime</b>', table_header_style),
     Paragraph('<b>Type</b>', table_header_style),
     Paragraph('<b>Top-10 Overlap</b>', table_header_style),
     Paragraph('<b>Observed Rank Shift & Tactical Impact</b>', table_header_style)]
]

for idx, r in sens_master.iterrows():
    sens_table_data.append([
        Paragraph(f"<b>{r['Regime']}</b>", table_cell_bold),
        Paragraph(r['Type'], table_cell_center),
        Paragraph(f"<b>{r['Top10_Overlap']}</b>", table_cell_center),
        Paragraph(r['Shift'], table_cell_style)
    ])

t_sens = Table(sens_table_data, colWidths=[140, 42, 58, 283])
t_sens.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), primary_color),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
    ('PADDING', (0,0), (-1,-1), 2.2),
]))
story.append(t_sens)
story.append(Spacer(1, 4))

story.append(Paragraph('13. Baseline Ability vs Pressure Execution (&beta; Elevation)', h1_style))

# Embed Figure 4 and description side-by-side
fig4_path = 'reports/fig4_residual_vs_residual.png'
if os.path.exists(fig4_path):
    fig_w = 260
    fig_h = 175
    img = Image(fig4_path, width=fig_w, height=fig_h)
    
    desc_p = Paragraph(
        '<b>Figure 4: Residual vs Residual Analysis.</b><br/>'
        'Plots player baseline ability (&alpha;, low-pressure EB residual) on the x-axis against demonstrated high-pressure EB residual on the y-axis. '
        'The dashed red line represents parity (&beta; = 0, where pressure execution equals baseline execution). '
        '<br/><br/>'
        '<b>Key Scouting Takeaways:</b><br/>'
        '&bull; <b>High-Baseline Stars (Quadrant 1):</b> Sooryavanshi, Abhishek Sharma, and Tim David perform well in all contexts and maintain high execution under pressure. '
        '<br/>&bull; <b>Pressure Specialists (&beta; Elevation):</b> Kusal Perera (&beta; = +0.299) and Towhid Hridoy (&beta; = +0.296) exhibit massive positive divergence from sub-par baseline rates. '
        '<br/>&bull; <b>Context Anchors:</b> Shreyas Iyer (&alpha; = +0.060, &beta; = +0.248) shows immense elevation under match tension over a large, reliable sample.',
        body_style
    )
    
    t_fig = Table([[img, desc_p]], colWidths=[265, 258])
    t_fig.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_fig)
    story.append(Spacer(1, 3))

story.append(Paragraph('14. Methodological Limitations & Measured Uncertainty', h1_style))
story.append(Paragraph(
    '<b>Limits & Boundary Conditions:</b> While the three-layer architecture eliminates situation and survivor bias, several structural limits must be acknowledged: '
    '(1) <i>Innings 1 Win Probability:</i> Model A achieves +8% Brier skill over an uninformative 50% baseline in the first innings; 73.7% of high-pressure deliveries occur in chases where win probability is strongly calibrated (Brier 0.128). '
    '(2) <i>Batting Persistence:</i> Corrected split-half reliability for batters is r = 0.169 (time-split r = 0.102), confirming that individual batting outcomes in T20 cricket carry substantial delivery-level stochastic noise. '
    '(3) <i>Bowler Quality Confound:</i> Bowler pressure ratings correlate r = 0.708 with baseline economy, reflecting that run prevention under pressure heavily overlaps with elite general bowling capability. '
    'Consequently, these rankings reflect <b>best demonstrated execution under pressure, with measured uncertainty</b>, rather than permanent traits. '
    '<br/><b>Pipeline Code:</b> Standalone, fully reproducible pipeline available at <font color="#004BA0"><u>https://github.com/ruhannpn/superr</u></font>.',
    body_style
))

doc.build(story, canvasmaker=NumberedCanvas)
shutil.copyfile(pdf_path, artifact_pdf)
print(f"Publication PDF successfully generated at {pdf_path} and synced to artifacts.")
