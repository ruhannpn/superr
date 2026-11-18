# SupeRR Selector Season 2 — Evaluating Demonstrated Execution Under Pressure

Official submission pipeline and statistical evaluation framework for **Rajasthan Royals – SupeRR Selector Season 2 (Round 1 Challenge)**.

---

## 🏆 Final Submission Dropdown Rankings

### Primary Batters (Rank 1 to 10)
1. **Vaibhav Sooryavanshi\*** (Ultra-Aggressive Finisher | P-SR 234.5 | EB Resid +0.743 | Boot P 31.0%)
2. **Shreyas Iyer** (Crisis Anchor & Builder | P-SR 166.2 | EB Resid +0.353 | Boot P 89.8%)
3. **Kusal Perera\*** (Dynamic Power-Hitter | P-SR 154.7 | EB Resid +0.320 | Boot P 30.4%)
4. **Towhid Hridoy\*** (Crisis Anchor & Builder | P-SR 124.5 | EB Resid +0.340 | Boot P 33.4%)
5. **Abhishek Sharma** (Ultra-Aggressive Finisher | P-SR 183.2 | EB Resid +0.298 | Boot P 75.8%)
6. **Tim David** (Dynamic Power-Hitter | P-SR 169.9 | EB Resid +0.294 | Boot P 82.8%)
7. **Dinesh Karthik\*** (Situation Manipulator | P-SR 156.6 | EB Resid +0.239 | Boot P 21.6%)
8. **Saim Ayub\*** (Situation Manipulator | P-SR 154.1 | EB Resid +0.207 | Boot P 23.8%)
9. **Andre Russell\*** (Dynamic Power-Hitter | P-SR 158.6 | EB Resid +0.202 | Boot P 5.8%)
10. **Prabhsimran Singh\*** (Dynamic Power-Hitter | P-SR 168.0 | EB Resid +0.200 | Boot P 37.4%)

*\* Flagged for low confidence: Qualified with <150 high-pressure deliveries or Boot P < 40%.*

---

### Primary Bowlers (Rank 1 to 10 — Direct Phase-Par Execution)
1. **Mahedi Hasan\*** (Chokehold Restrictor | P-ER 6.15 | Saved/b +0.444 | Implied Par 8.81)
2. **Matheesha Pathirana** (Death Overs Specialist | P-ER 7.72 | Saved/b +0.321 | Wkts > Par +3.10 | Boot P 89.6%)
3. **Krunal Pandya** (Chokehold Restrictor | P-ER 6.61 | Saved/b +0.303 | Wkts > Par +2.57 | Boot P 91.6%)
4. **Shaheen Shah Afridi** (Chokehold Restrictor | P-ER 6.80 | Saved/b +0.367 | Implied Par 9.00 | Boot P 68.8%)
5. **Josh Hazlewood\*** (Aggressive Strike Weapon | P-ER 8.11 | Wkts > Par +11.62 | Boot P 31.6%)
6. **Mohammad Nawaz** (Aggressive Strike Weapon | P-ER 7.22 | Saved/b +0.189 | Wkts > Par +5.60 | Boot P 46.4%)
7. **Rashid Khan** (Phase-Versatile Enforcer | P-ER 7.14 | Saved/b +0.246 | Implied Par 8.61 | Boot P 76.4%)
8. **Shadab Khan** (Chokehold Restrictor | P-ER 6.71 | Saved/b +0.296 | Implied Par 8.48 | Boot P 50.2%)
9. **Axar Patel** (Aggressive Strike Weapon | P-ER 7.25 | Saved/b +0.202 | Wkts > Par +4.72 | Boot P 50.8%)
10. **Mujeeb Ur Rahman\*** (Phase-Versatile Enforcer | P-ER 7.39 | Saved/b +0.206 | Implied Par 8.63)

---

## 📊 Core Architecture & Methodology

```
Raw Ball-by-Ball Data (200,999 Deliveries across 883 Matches)
  │
  ├── 1. Filter Audit: 681 Tier-1 Matches (161,323 Deliveries)
  │
  ├── 2. Model A: Calibrated Monotonic Win Probability
  │      └── True Swing Leverage: LI = |WP(+4) - WP(out)| / mean(LI)
  │      └── High-Pressure Cutoff: LI >= 1.674 (Top 20% swing; 31,601 Tier-1 balls)
  │
  ├── 3. Model B: Non-Leaking Delivery Difficulty
  │      └── Expected Runs (xRuns) & Dismissal Probability (xWkt)
  │
  └── 4. Layer C: Empirical Bayes Shrinkage & Attribution
         ├── Batters: Separate k for runs & wickets; Precision weights w = n^2 / (n + k)
         └── Bowlers: Direct Phase-Par Runs Saved + Direct Wkts Above Par (Implied Par = 8.99)
```

---

## 🚀 Running the Pipeline

### Prerequisites
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib reportlab
```

### Reproduce Everything
```bash
# 1. Run full statistical pipeline and generate results_master.csv
python3 build_full_solution_v5.py

# 2. Compile publication 5-page PDF report
python3 generate_final_pdf.py

# 3. Generate interactive single-page dashboard
python3 generate_dashboard.py
```

---

## 📁 Key Files & Deliverables
* `results_master.csv`: Single source of truth containing all metrics, residuals, and bootstrap stability rates.
* `SupeRR_Selector_Season_2_Methodology_Report.pdf`: Strictly 5-page publication-grade PDF report.
* `SupeRR_Selector_Interactive_Dashboard.html`: Standalone single-page interactive HTML dashboard.
* `SuperRR_Selector_Methodology_Document.md`: Exhaustive technical methodology and audit report.

