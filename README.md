# Hong Kong Mark Six Data Analysis

<p align="center">
  <img src="assets/cover_photo.png" alt="Hong Kong Mark Six Analysis Cover" width="600"/>
</p>

This project provides a comprehensive, data-driven analysis of the Hong Kong Mark Six lottery, focusing on the statistical trends of drawn numbers and their associated colors. It includes:

- **Historical Data Retrieval:** Fetches and processes all available Mark Six draw data (from 2002-07-04 to present).
- **Number & Color Trend Analysis:** Visualizes frequency, trends, and color distributions using Python notebooks.
- **Statistical Testing:** Applies statistical methods to assess randomness and identify short-term and long-term patterns.
- **Predictive Modeling:** Demonstrates advanced modeling (e.g., hierarchical EKF) for educational purposes.

## Project Structure

- `mark_six_number_color_trend_analysis.ipynb`: In-depth analysis of number and color trends, with visualizations and statistical explanations.
- `mark_six_pred.ipynb`: Demonstrates a hierarchical Extended Kalman Filter (EKF) model for prediction and evaluation.
- `get_lottery_data.py`: Utility for fetching and processing Mark Six draw data.

## Requirements

- Python 3.11+
- pandas, matplotlib, seaborn, numpy, scipy, statsmodels, scikit-learn

## Background

The Hong Kong Mark Six lottery draws 6 numbers (from 1 to 49) and an extra number. Each number is assigned a color (red, blue, or green) based on a systematic rule. Since 2010-11-09, each bet costs HK$10, and the prize structure was updated to increase fixed prizes and the jackpot guarantee. (2010 年 11 月 9 日，每注金額由港幣 5 元加倍至港幣 10 元，固定獎金提高一倍，頭獎獎金保證由港幣 500 萬增至 800 萬。)

- [RTHK News: Mark Six Prize Structure Update (Archived)](https://web.archive.org/web/20140808052241/http://rthk.hk/rthk/news/expressnews/20100928/news_20100928_55_701445.htm)

## Disclaimer

This project is for educational and research purposes only. Lottery outcomes are random, and past trends do not influence future draws.
