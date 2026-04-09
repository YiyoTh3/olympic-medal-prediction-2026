# 🥇 Predicting Canada's 2026 Winter Olympic Medal Performance
### Milan-Cortina 2026 | 120+ Years of Olympic History

> 📰 **This project was featured in regional media across online and television outlets.**

| Outlet | Format | Link |
|--------|--------|------|
| CityNews Edmonton | Online | [Edmonton college predicts gold medal win for Olympic men's hockey team using AI](https://edmonton.citynews.ca/2026/02/13/edmonton-college-predicts-gold-medal-win-for-olympic-mens-hockey-team-using-ai/) |
| CTV News Edmonton | Online | [Student AI-trained machine model predicts Olympic men's hockey gold for Canada](https://www.ctvnews.ca/edmonton/article/student-ai-trained-machine-model-predicts-olympic-mens-hockey-gold-for-canada/) |

---

## Overview

This project analyzes over 120 years of Olympic history to predict Canada's medal 
performance at the 2026 Winter Olympics in Milan-Cortina, Italy.

Two machine learning objectives are addressed:

| Objective | Type | Target |
|-----------|------|--------|
| How many total medals will Canada win in 2026? | Regression | `total` (medal count) |
| Will Canada win a medal in Men's Ice Hockey? | Classification | `medal_flag` (0/1) |

> ⚠️ **Disclaimer:** This was an academic project built for fun and learning purposes.  
> Predictions are data-driven and based on historical trends only. They do not account  
> for roster composition, NHL participation agreements, injuries, or other real-world  
> variables. **Do not use this information to make bets.**

---

## Key Results

### Regression — Total Medals (All Sports)

| Metric | Value |
|--------|-------|
| Model | Linear Regression (3 features) |
| Test MAE | 3.10 medals |
| Test R² | 0.798 |
| **2026 Prediction** | **~31 medals** |
| Confidence range | 28 — 34 medals |
| Recent average (2014–2022) | ~27 medals |

### Classification — Men's Ice Hockey Podium

| Medal | Predicted Team |
|-------|---------------|
| 🥇 Gold | Canada (CAN) |
| 🥈 Silver | Finland (FIN) |
| 🥉 Bronze | Germany (GER_ALL) |

### Tournament Simulation (10,000 runs)

| Metric | Value |
|--------|-------|
| Canada Gold Probability | 74.9% |
| 95% Confidence Interval | 74.0% — 75.7% |
| Canada Any Medal Probability | 96.7% |

---

## Notebooks

| File | Description |
|------|-------------|
| `olympic_eda_and_preprocessing.ipynb` | Data loading, filtering, feature engineering, EDA, visualizations |
| `olympic_regression_forecasting.ipynb` | Medal count regression — predicting Canada's total 2026 medals |
| `olympic_classification_ice_hockey.ipynb` | Ice hockey classification + 10,000-run tournament simulation |

---

## Datasets

| File | Description |
|------|-------------|
| `ath_base.csv` | Athlete-level historical Olympic records — used for training |
| `res_base.csv` | Official results table — used for 2018 & 2022 validation patching |
| `df_merged_clean.csv` | Cleaned merged dataset — used for regression modeling |

**Sources:**
- [Kaggle — Olympic Historical Dataset (Olympedia)](https://www.kaggle.com/datasets/josephcheng123456/olympic-historical-dataset-from-olympediaorg/data)
- GitHub Olympic Datasets (supplementary)

---

## Methodology

### Notebook 1 — EDA & Preprocessing
- Filtered dataset to Winter Olympics only
- Removed Youth Olympic Games years (2012, 2016, 2020) — different competition, not comparable
- Deduplicated team medals by `['Event', 'NOC', 'Year', 'Medal']` to avoid per-athlete inflation
- Engineered `IS_CHURNED` target variable using recency-based logic anchored to dataset ceiling
- Analyzed gender participation, delegation size, and churn rates by market segment
- Investigated geopolitical NOC changes over time (URS, GDR, TCH, YUG)
- Created host nation feature mapped to each Winter Olympics edition

### Notebook 2 — Regression: Total Medal Forecasting

**Approach:** Multiple regression models compared across engineered lag features.  
Training on years ≤ 2014, testing on 2018 & 2022, predicting 2026.

**Feature Engineering:**
- Lag features: previous and two-Olympics-ago medal counts
- Rolling 3-Olympics average
- Medal trend (improvement/decline signal)
- Delegation size and growth features
- Female participation ratio
- Host nation indicator

**Multicollinearity removal** (threshold > 0.85) and **dead weight pruning**  
(near-zero coefficients) reduced 15 candidate features down to 3 final features:

| Final Feature | Description |
|--------------|-------------|
| `rolling_avg_total_3` | Average medals over last 3 Olympics |
| `unique_female_count` | Female delegation size |
| `is_host_int` | Host nation indicator (0/1) |

**Models compared:**
Linear Regression, Ridge (α=1, α=10), Lasso (α=0.5, α=1), Random Forest.  
Linear Regression achieved the best performance (MAE=3.10, R²=0.798).

> **Data Limitation:** The athlete dataset was missing 2018 and 2022 delegation data.  
> Canada's female athlete counts were patched using official data from the  
> [Canadian Olympic Committee](https://www.olympic.ca):  
> 2014 (99), 2018 (103), 2022 (106).

### Notebook 3 — Classification: Men's Ice Hockey

**Two-model approach combined into a ranking score:**

| Model | Purpose |
|-------|---------|
| Logistic Regression | P(medal) — probability of winning any medal |
| Poisson Regression | Expected medal points (0–3 scale: Gold=3, Silver=2, Bronze=1) |
| Combined score | `score = p_medal × exp_points` |

**Lineage merging** applied to preserve historical performance continuity:

| Group | NOCs merged |
|-------|------------|
| `GER_ALL` | FRG, GDR, GER |
| `CZE_SVK_ALL` | TCH, CZE, SVK |
| `RUS_ALL` | URS, EUN, RUS, OAR, ROC |

**Lag features created:**
- `points_lag1`, `points_lag2` — previous 1 and 2 Olympics points
- `medal_lag1`, `medal_lag2` — previous medal flags
- `points_roll3_mean` — 3-Olympics rolling average
- `medal_roll3_sum` — medals won in last 3 Olympics

**Tournament Simulation (10,000 runs):**  
8-team seeded bracket with quarterfinals → semifinals → bronze/gold matches.  
Match outcomes weighted by team strength using Bradley-Terry win probability:  
`P(A beats B) = 1 / (1 + exp(-(rating_A - rating_B)))`

---

## Important Design Decisions

**Temporal split (no data leakage):**  
All models trained on past data only. Future years never seen during training.

| Split | Years | Purpose |
|-------|-------|---------|
| Train | ≤ 2014 | Model learning |
| Test | 2018 & 2022 | Validation |
| Predict | 2026 | Final forecast |

**Youth Olympics excluded:**  
2012, 2016, 2020 in the results dataset are Winter Youth Olympic Games —  
a separate competition not comparable to the main Games.

**Team medal deduplication:**  
Hockey teams have many players per medal. Deduplication on  
`['Event', 'NOC', 'Year', 'Medal']` ensures each medal counts once.

**Geopolitical boundaries:**  
For the regression model, countries are kept separate to preserve historical  
accuracy (particularly relevant given current geopolitical context).  
For ice hockey, lineage merging is applied to maintain performance continuity  
for countries that split or unified over time.

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Visualizations |
| scikit-learn | Linear Regression, Ridge, Lasso, Logistic Regression, Random Forest |
| PoissonRegressor | Expected medal strength modeling |
| Google Colab | Development environment |

---

## Team

**Members:**
- David Barahona
- Paula Frossard
- Mubarak Farah
- Dinsara Perera

---

## References

**Datasets**
- [Kaggle — Olympic Historical Dataset](https://www.kaggle.com/datasets/josephcheng123456/olympic-historical-dataset-from-olympediaorg/data)
- [Canadian Olympic Committee — Delegation Data](https://www.olympic.ca)

**Geopolitical Sources**
- [Collapse of the Soviet Union](https://history.state.gov/milestones/1989-1992/collapse-soviet-union)
- [Unified Team EUN — Olympedia](https://www.olympedia.org/countries/EUN)
- [German Reunification — Britannica](https://www.britannica.com/topic/German-reunification)
- [Velvet Divorce — Britannica](https://www.britannica.com/topic/Velvet-Divorce)
- [TCH — Olympedia](https://www.olympedia.org/countries/TCH)

---

*Academic project — Norquest College 2025–2026 🍁*  
*Predicting Canada's path to gold, one Olympic cycle at a time.*
