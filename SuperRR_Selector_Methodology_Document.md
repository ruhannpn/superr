# SupeRR Selector Season 2: Evaluating Demonstrated Execution Under Pressure
## Rigorous Statistical Methodology & Player Attribution Report
**Challenge:** Rajasthan Royals – SupeRR Selector Season 2 (Round 1 Submission)  
**Author:** Ruhann P. N.  
**Dataset Scope:** 883 Total Matches (200,999 Deliveries) &rarr; 862 Valid Matches (199,466 Deliveries) &rarr; 681 Tier-1 Matches (161,323 Deliveries)  
**High-Leverage Subset (80th Percentile LI &ge; 1.674):** 31,601 Tier-1 Deliveries  
**Standalone Code Pipeline:** [build_full_solution_v5.py](file:///Users/ruhannp.n/Downloads/Data%20Pack/build_full_solution_v5.py) | [GitHub Repository](https://github.com/ruhannpn/superr)  
**Interactive Single-Page Dashboard:** [SupeRR_Selector_Interactive_Dashboard.html](file:///Users/ruhannp.n/Downloads/Data%20Pack/SupeRR_Selector_Interactive_Dashboard.html)  
**Publication-Grade PDF Report (Strictly 5 Pages):** [SupeRR_Selector_Season_2_Methodology_Report.pdf](file:///Users/ruhannp.n/Downloads/Data%20Pack/SupeRR_Selector_Season_2_Methodology_Report.pdf)  
**Single Source of Truth Results File:** [results_master.csv](file:///Users/ruhannp.n/Downloads/Data%20Pack/results_master.csv)  

---

## 1. Executive Summary & Framing

In competitive sports analytics, there is no pre-existing ground-truth label called a "clutch player". Attempting to train a black-box machine learning classifier directly on player labels produces circular, unscientific outputs that fail when deployed in live scouting environments.

Our framework strictly adheres to the modern Sabermetric paradigm:
1. **Context & Difficulty Estimation:** Machine learning is restricted to estimating match context (Model A Win Probability) and delivery-level difficulty (Model B expected runs conceded and dismissal likelihood).
2. **Attribution & Shrinkage:** Player execution under pressure is derived through a three-layer attribution pipeline utilizing **Entry-Group Stratification, Separate Empirical Bayes Shrinkage for Runs and Wickets, Precision Aggregation Weights ($w = \frac{n^2}{n+k}$), and Phase-Par Baselines**.
3. **Epistemological Framing:** Rankings reflect **best demonstrated execution under pressure, with measured uncertainty**, rather than a static athletic trait.

---

## 2. Three-Layer Mathematical Framework

```
Raw Ball-by-Ball Data (200,999 Deliveries across 883 Matches)
  │
  ├── 1. Filter Audit: Exclude abandoned matches & Associate mismatches
  │      └── 681 Tier-1 Matches (161,323 Deliveries)
  │
  ├── 2. Model A: Calibrated Monotonic Win Probability
  │      └── True Swing Leverage: LI = |WP(+4) - WP(out)| / mean(LI)
  │      └── High-Pressure Subset: LI >= 1.674 (Top 20% swing; 31,601 Tier-1 balls)
  │
  ├── 3. Model B: Non-Leaking Delivery Difficulty
  │      └── xRuns (Expected Runs Conceded) & xWkt (Expected Dismissal Probability)
  │      └── Context Covariates: match state, venue, phase, delivery style, set/cold balls
  │
  └── 4. Layer C: Empirical Bayes Shrinkage & Performance Attribution
         ├── Batters: Separate k for runs & wickets; Precision weights w = n^2 / (n + k)
         │     └── Score: 80% EB_Resid (W=6) + 20% Shrunk Match Consistency
         └── Bowlers: Phase-Par Runs Saved + Direct Wickets Above Expected + Shrunk Consistency
               └── Implied Par: ER + 6 * saved/ball = 8.99 (Matches ground-truth phase pars)
```

---

## 3. Sample Integrity, Eligibility & The Dhoni / Karthik Audit

### A. Stage-by-Stage Sample Counts
Every filter is documented upfront and applied uniformly to every candidate without unstated post-hoc exclusions:
- **Raw Input Archive:** 883 matches, 200,999 deliveries.
- **Valid Result Filter:** 862 matches, 199,466 deliveries (excludes 21 abandoned fixtures).
- **Tier-1 Match Filter:** 681 matches, 161,323 deliveries (IPL franchises and ICC Full Members; excludes 181 Associate mismatches).
- **High-Pressure Filter (LI &ge; 1.674):** 31,601 Tier-1 deliveries (39,894 across all cricket).
- **Qualification Floor:** Minimum 120 high-pressure deliveries faced or bowled across at least 8 tested matches in Tier-1 play. 181 players qualify (93 batters and 88 bowlers).

### B. MS Dhoni vs Dinesh Karthik Data Reconciliation
Earlier pipeline drafts had duplicate text erroneously copied between MS Dhoni and Dinesh Karthik. Auditing the raw ball-by-ball database clarifies the exact numbers:
- **Mahendra Singh Dhoni:** Faced only **69 high-pressure deliveries** (92 runs, 2 dismissals, SR 133.3, +0.150 EB residual) across 12 high-pressure matches. His low-pressure baseline comprises 428 balls (601 runs, 22 dismissals, SR 140.4). Because $69 < 120$, **MS Dhoni is filtered out strictly on sample size adequacy**.
- **Dinesh Karthik:** Faced **136 high-pressure deliveries** (213 runs, 5 dismissals, SR 156.6, raw runs part +0.106, raw wkt part -0.0278, EB residual +0.239) across 23 matches (9 high-pressure qualifying matches, 69.2% success rate). He comfortably qualifies and ranks **#7 overall**.

---

## 4. Bowler Execution: Direct Phase Par & Implied Par Integrity

### A. Upstream Fix: Resolving the Inflated Implied Par
In earlier iterations, bowler runs saved per ball appeared artificially doubled (+0.60 to +0.85), driving implied par ($ER + 6 \times \text{saved/ball}$) to ~11.0–11.7. 
* **Root Cause Identified:** Hypothetical tandem spillover credits (up to 21–31 phantom overs) were added into `p_runs_saved` alongside an opponent scaling factor.
* **Direct Phase Par Restored:** Removing tandem credits and opponent scaling establishes the single, unified definition:
$$\text{Runs\_Saved\_Per\_Ball} = \text{Phase\_Par\_Ball} - \text{Total\_Ball\_Runs}$$
where Tier-1 phase pars are 8.52 (Powerplay), 8.26 (Middle), and 9.74 (Death).

### B. Implied Par Distribution Across All 88 Qualified Bowlers
```
Pressure ER:       mean = 8.93, std = 1.22, min = 6.15, max = 12.57
Saved / Ball:      mean = +0.011, std = 0.167, min = -0.498, max = +0.444
Implied Par:       mean = 8.99, std = 0.36, min = 8.34, 25% = 8.68, 50% = 9.01, 75% = 9.30, max = 9.79
```
Implied par sits strictly inside the 8–10 range across all bowlers.

---

## 5. Tandem Bowling Spillover: Pre-Registered Statistical Re-Test

It has frequently been asserted in cricket commentary that severe run restriction from bowler A forces batters into false shots, creating "free wickets" for partner bowler B in over $t+1$. 
To determine whether tandem credit should be incorporated into the primary metric, we pre-registered and ran clustered OLS regressions on all Tier-1 overs (match-clustered SEs, next-bowler fixed effects):

$$\text{Next\_Over\_Wkt\_Residual} = \beta_0 + \hat{\gamma} \cdot \mathbf{1}_{\{\text{over } t \text{ is good}\}} + \alpha_{\text{next\_bowler}} + \varepsilon$$

### Tested Empirical Results:
1. **Binary Good Over ($\le 6$ runs conceded in over $t$):**
   * $\hat{\gamma} = -0.0087$ (SE = 0.0076, $p = 0.2546$)
   * **95% Confidence Interval:** $[-0.0236, +0.0062]$
   * **Conservation Bound Check:** 2,763 actual wickets fell in overs following good overs across the dataset. The upper bound of $+0.006$ wickets/over rules out double-digit spillover credits by two orders of magnitude.
2. **Continuous Run Suppression (Runs saved vs par in over $t$):**
   * $\hat{\beta} = -0.0007$ (SE = 0.0008, $p = 0.4011$)
   * **95% Confidence Interval:** $[-0.0022, +0.0009]$

> **Decision Rule Triggered:** Because both 95% confidence intervals cross zero and the point estimates are slightly negative, **tandem spillover credit is definitively excluded from the primary bowler rankings** and maintained strictly as a secondary sensitivity regime.

### Jasprit Bumrah & Sunil Narine Audit:
* **Jasprit Bumrah (Rank #15, 364 balls, ER 7.50, +0.253 saved/ball, Boot P 72.6%):** Generated elite run prevention, but captured 13 wickets against 19.37 Model B expected wickets ($-6.37$ wicket deficit). The difference corresponds to $z = -1.45$ ($p = 0.15$), confirming the gap is entirely within ordinary sampling variance.
* **Sunil Narine (Rank #17, 355 balls, ER 7.11, +0.237 saved/ball, Boot P 65.6%):** 9 wickets vs 16.11 expected ($-7.11$ wicket deficit, $z = -1.77$, $p = 0.08$). Also within sampling variance.

---

## 6. Separate Empirical Bayes Shrinkage for Runs and Wickets

Wicket events are binary and far noisier than continuous run scoring. Applying a single shrinkage parameter ($k \approx 65$) to the combined residual conflates high-variance wicket luck with genuine scoring skill.

### Empirical Variance & Shrinkage Parameters:
* **Group-by-Group Cell Shrinkage (Batters):**
  * Openers (`open`, $N=156$): $k_{\text{runs}} = 8.5$, $k_{\text{wkt}} = 13.4$
  * Early (`early`, $N=227$): $k_{\text{runs}} = 10.0$, $k_{\text{wkt}} = 3.6$
  * Middle (`mid`, $N=258$): $k_{\text{runs}} = 14.0$, $k_{\text{wkt}} = 2.6$
  * Late (`late`, $N=222$): $k_{\text{runs}} = 5.9$, $k_{\text{wkt}} = 2.4$
* **Overall Player-Level Aggregate Shrinkage:**
  * $k_{\text{runs}} = 268.3$ balls
  * $k_{\text{wkt}} = 470.1$ balls (wickets require $1.75\times$ more shrinkage than runs to eliminate noise!)

### Exact Reconciliation:
Shrinkage is performed on runs and dismissals independently, then aggregated across entry groups via precision weights $w_{ig} = \frac{n_{ig}^2}{n_{ig} + k_g}$:
$$\text{EB\_Resid}_{W=6} = \text{EB\_RunsPart} - 6.0 \times \text{EB\_WktPart}$$
Max discrepancy across all 93 qualified batters is **$0.00000000$ (exact zero)**.

---

## 7. Primary Top 10 Batters Leaderboard

*Ranked by 80% EB Residual ($W=6$) + 20% Shrunk Consistency.*  
*$\alpha$: Shrunk baseline EB residual in normal-leverage balls.*  
*$\beta$: Shrunk elevation ($\text{EB\_Resid}_{\text{pressure}} - \alpha$).*  
*\* Low Confidence Flag: Qualified with $<150$ balls or Boot P $<40\%$.*

| Rank | Batter Name | P-Balls | P-SR | Raw Runs | EB Runs | Raw Wkt | EB Wkt | EB Resid ($W=6$) | Base $\alpha$ | Elev $\beta$ | Boot P | Tactical Archetype |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Vaibhav Sooryavanshi\*** | 139 | 234.5 | $+0.740$ | $+0.667$ | $-0.0168$ | $-0.0128$ | **$+0.743$** | $+0.373$ | $+0.259$ | 31.0% | Ultra-Aggressive Finisher |
| **2** | **Shreyas Iyer** | 328 | 166.2 | $+0.226$ | $+0.205$ | $-0.0266$ | $-0.0246$ | **$+0.353$** | $+0.060$ | $+0.248$ | 89.8% | Crisis Anchor & Builder |
| **3** | **Kusal Perera\*** | 139 | 154.7 | $+0.307$ | $+0.268$ | $-0.0171$ | $-0.0087$ | **$+0.320$** | $-0.108$ | $+0.299$ | 30.4% | Dynamic Power-Hitter |
| **4** | **Towhid Hridoy\*** | 143 | 124.5 | $+0.060$ | $+0.079$ | $-0.0431$ | $-0.0434$ | **$+0.340$** | $-0.080$ | $+0.296$ | 33.4% | Crisis Anchor & Builder |
| **5** | **Abhishek Sharma** | 322 | 183.2 | $+0.343$ | $+0.323$ | $+0.0049$ | $+0.0042$ | **$+0.298$** | $+0.255$ | $+0.036$ | 75.8% | Ultra-Aggressive Finisher |
| **6** | **Tim David** | 296 | 169.9 | $+0.295$ | $+0.241$ | $-0.0096$ | $-0.0088$ | **$+0.294$** | $+0.298$ | $-0.003$ | 82.8% | Dynamic Power-Hitter |
| **7** | **Dinesh Karthik\*** | 136 | 156.6 | $+0.106$ | $+0.083$ | $-0.0278$ | $-0.0259$ | **$+0.239$** | $+0.128$ | $+0.077$ | 21.6% | Situation Manipulator |
| **8** | **Saim Ayub\*** | 148 | 154.1 | $+0.214$ | $+0.189$ | $-0.0049$ | $-0.0031$ | **$+0.207$** | $-0.172$ | $+0.270$ | 23.8% | Situation Manipulator |
| **9** | **Andre Russell\*** | 128 | 158.6 | $+0.238$ | $+0.224$ | $+0.0135$ | $+0.0037$ | **$+0.202$** | $+0.228$ | $-0.018$ | 5.8% | Dynamic Power-Hitter |
| **10** | **Prabhsimran Singh\*** | 306 | 168.0 | $+0.209$ | $+0.191$ | $-0.0000$ | $-0.0015$ | **$+0.200$** | $-0.005$ | $+0.172$ | 37.4% | Dynamic Power-Hitter |

---

## 8. Primary Top 10 Bowlers Leaderboard (Direct Execution)

*Ranked by 60% Phase-Par Runs Saved + 25% Wickets Above Expected + 15% Shrunk Consistency.*  
*No tandem credit, no opponent multipliers, no dynamic W.*  
*\* Low Confidence Flag: Qualified with $<150$ balls or Boot P $<40\%$.*

| Rank | Bowler Name | P-Balls | Wkts | P-ER | Saved/b | EB Saved/b | Implied Par | Wkts &gt; Par | Base $\alpha$ | Elev $\beta$ | Boot P | Tactical Archetype |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Mahedi Hasan\*** | 122 | 6 | 6.15 | $+0.444$ | $+0.329$ | 8.81 | $+0.25$ | $+1.45$ | $+0.37$ | 11.8% | Chokehold Restrictor |
| **2** | **Matheesha Pathirana** | 234 | 17 | 7.72 | $+0.321$ | $+0.268$ | 9.65 | $+3.10$ | $-0.74$ | $+1.93$ | 89.6% | Death Overs Specialist |
| **3** | **Krunal Pandya** | 336 | 18 | 6.61 | $+0.303$ | $+0.269$ | 8.43 | $+2.57$ | $+1.58$ | $+0.03$ | 91.6% | Chokehold Restrictor |
| **4** | **Shaheen Shah Afridi** | 173 | 9 | 6.80 | $+0.367$ | $+0.279$ | 9.00 | $+0.09$ | $-0.29$ | $+1.52$ | 68.8% | Chokehold Restrictor |
| **5** | **Josh Hazlewood\*** | 316 | 30 | 8.11 | $+0.145$ | $+0.133$ | 8.98 | $+11.62$ | $+0.53$ | $+0.23$ | 31.6% | Aggressive Strike Weapon |
| **6** | **Mohammad Nawaz** | 271 | 18 | 7.22 | $+0.189$ | $+0.169$ | 8.36 | $+5.60$ | $+1.34$ | $-0.27$ | 46.4% | Aggressive Strike Weapon |
| **7** | **Rashid Khan** | 407 | 22 | 7.14 | $+0.246$ | $+0.226$ | 8.61 | $+0.78$ | $+0.59$ | $+0.67$ | 76.4% | Phase-Versatile Enforcer |
| **8** | **Shadab Khan** | 155 | 7 | 6.71 | $+0.296$ | $+0.225$ | 8.48 | $-0.66$ | $+1.46$ | $-0.08$ | 50.2% | Chokehold Restrictor |
| **9** | **Axar Patel** | 547 | 30 | 7.25 | $+0.202$ | $+0.191$ | 8.47 | $+4.72$ | $+0.43$ | $+0.65$ | 50.8% | Aggressive Strike Weapon |
| **10** | **Mujeeb Ur Rahman\*** | 140 | 8 | 7.39 | $+0.206$ | $+0.165$ | 8.63 | $+1.80$ | $+1.65$ | $-0.49$ | 20.2% | Phase-Versatile Enforcer |

---

## 9. Comprehensive Robustness & Sensitivity Matrix

| Sensitivity Regime | Evaluation Type | Top-10 Overlap | Observed Rank Shift & Tactical Impact |
| :--- | :---: | :---: | :--- |
| **W = 4 (Lower Dismissal Weight)** | Batter | **9/10** | Towhid Hridoy shifts #4 &rarr; #6; Abhishek Sharma rises #5 &rarr; #4. |
| **W = 8 (Higher Dismissal Weight)** | Batter | **9/10** | Towhid Hridoy rises to #3; Tim David drops #6 &rarr; #8. |
| **Dynamic W (Situation Dependent)** | Batter | **10/10** | 100% membership stability across all top 10 batters. |
| **Opponent Elo Quality Covariate** | Both | **9/10** | Shaheen Afridi shifts #4 &rarr; #3; Sooryavanshi stays #1. |
| **Aggregation Weight: $w = n$ (Balls)** | Batter | **7/10** | Daryl Mitchell enters at #10; Andre Russell exits. |
| **Aggregation Weight: $w = \frac{n}{n+k}$** | Batter | **5/10** | Pure precision weight favors hyper-efficient small samples (Rinku Singh enters). |
| **Chase-Only (Innings 2 Calibrated WP)** | Batter | **7/10** | Shreyas Iyer rises to #1; Dinesh Karthik to #4; Tim David to #5. |
| **IPL-Only (Franchise League Subset)** | Batter | **4/10** | Sooryavanshi, Iyer, Abhishek, and Tim David dominate; international players filtered. |
| **Pressure Cutoff 85th Pct (Extreme)** | Both | **8/10** | Core top tier completely stable; fringe small samples fluctuate. |
| **Pressure Cutoff 75th Pct (Broader)** | Both | **9/10** | Smooths ratings; overlap rises to 90%. |
| **High Sample Floor ($\ge 150$ balls)** | Batter | **4/10** | Excludes 6 players ($<150$b): Sooryavanshi, Perera, Hridoy, Karthik, Ayub, Russell. |
| **Pure Run Suppression (0% Wkt Weight)** | Bowler | **7/10** | Josh Hazlewood falls to #11; Mahedi, Shaheen, and Krunal occupy top 3. |
| **Tandem Rollover Enabled** | Bowler | **10/10** | Bumrah rises #15 &rarr; #11; Top 8 remains identical. |
| **Beta Ranked (Elevation Over Baseline)** | Batter | **5/10** | Isolates pure surge performers: Perera, Hridoy, Ayub, Sooryavanshi, and Iyer. |

---

## 10. Statistical Reliability, Bootstrap & Boundary Conditions

### A. Measured Persistence on Final Metric
* **Batters Split-Half ($N=93$):** Raw odd/even $r = 0.092 \implies$ **Spearman-Brown Corrected $r = 0.169$**. Time-split (2022–24 vs 2025–26, $N=83$) $r = 0.102$.
* **Bowlers Split-Half ($N=88$):** Raw odd/even $r = 0.346 \implies$ **Spearman-Brown Corrected $r = 0.514$**.
* **$\alpha$-$\beta$ Correlations:** Batters $r = 0.246$; Bowlers $r = 0.708$. Bowler pressure run suppression strongly reflects elite baseline capability ($70.8\%$ overlap).

### B. 500-Resample Match-Clustered Bootstrap Stability
Boot P measures the proportion of bootstrap resamples where a player secures a top-10 rank:
* **High-Sample Solid Anchors:** Shreyas Iyer (**89.8%**), Matheesha Pathirana (**89.6%**), Krunal Pandya (**91.6%**), Tim David (**82.8%**), Rashid Khan (**76.4%**), Abhishek Sharma (**75.8%**), Shaheen Shah Afridi (**68.8%**).
* **High-Volume Fringe Players with High Boot P:**
  * **Bowlers:** Jasprit Bumrah (Rank #15, **72.6%** Boot P) and Sunil Narine (Rank #17, **65.6%** Boot P) have higher Boot P than Mahedi Hasan (11.8%) and Mujeeb (20.2%) because their large sample sizes keep them consistently near ranks 8–12.
  * **Batters:** Daryl Mitchell (Rank #12, **51.6%** Boot P) and William Jacks (Rank #17, **40.0%** Boot P) exceed small-sample qualifiers like Andre Russell (5.8%), Dinesh Karthik (21.6%), and Saim Ayub (23.8%).

### C. Methodological Limitations & Boundary Conditions:
1. **Innings 1 Win Probability:** Model A achieves $+8\%$ Brier skill over an uninformative $50\%$ base rate; $73.7\%$ of high-leverage deliveries occur in chases where win probability is strongly calibrated (Brier 0.128).
2. **Batting Stochasticity:** Individual ball outcomes in T20 cricket carry immense noise ($r_{\text{SB}} = 0.169$), proving that Empirical Bayes shrinkage is mathematically essential.
3. **Epistemological Claim:** These metrics describe **demonstrated execution under pressure, with measured uncertainty**, rather than unalterable genetic traits.

---

## 11. Final Unstop Submission Dropdown Lists (Exact Spellings)

### Batters (Rank 1 to 10):
1. `Vaibhav Sooryavanshi`
2. `Shreyas Iyer`
3. `Kusal Perera`
4. `Towhid Hridoy`
5. `Abhishek Sharma`
6. `Tim David`
7. `Dinesh Karthik`
8. `Saim Ayub`
9. `Andre Russell`
10. `Prabhsimran Singh`

### Bowlers (Rank 1 to 10):
1. `Mahedi Hasan`
2. `Matheesha Pathirana`
3. `Krunal Pandya`
4. `Shaheen Afridi`
5. `Josh Hazlewood`
6. `Mohammad Nawaz`
7. `Rashid Khan`
8. `Shadab Khan`
9. `Axar Patel`
10. `Mujeeb Ur Rahman`
