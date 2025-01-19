# Predicting Outcomes and Scores in Rugby World Cup 2023

## Technologies Used

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/R-276DC3?logo=r&logoColor=white" alt="R"/>
</p>

## Overview
This project focuses on leveraging machine learning techniques to predict match outcomes (win/loss) and score differences in rugby games from the Rugby World Cup 2023. Additionally, the study explores key performance indicators (KPIs) that influence these outcomes using principal component analysis (PCA).

The goal of this project is to provide valuable insights for rugby analysts, coaches, and enthusiasts by identifying patterns and relationships in the data that are not immediately apparent. By combining statistical rigor with machine learning, the project offers a comprehensive approach to understanding and predicting match outcomes.

## Features
- **Outcome Prediction**: Using logistic regression to predict whether a team wins or loses.
- **Score Difference Prediction**: Employing regularized regression methods to predict the margin of scores between teams.
- **KPI Analysis**: Analyzing the impact of different KPIs on match results using PCA.
- **Custom ELO System**: Development and evaluation of an ELO rating system tailored for rugby matches.

## Data Source
The dataset comprises match data from the Rugby World Cup 2023, including team performance metrics and match results. Additional data sources include scraped historical match results and betting odds collected from various platforms.

## Project Structure
1. **Final Report**: A comprehensive PDF report summarizing the findings and methodology.
2. **Notebooks**: Four Jupyter notebooks detailing the step-by-step analysis and modeling process.
    - **PCA_study**: The main analysis focusing on KPI analysis using PCA.
    - **elo_rugby_creation**: Creation of an ELO rating system from historical match results.
    - **elo_rugby_analysis**: Analysis and evaluation of the ELO rating system.
    - **wiki_scrap_all_games**: Data scraping from Wikipedia to collect historical match results.
    - **odds_import_df**: Collection and analysis of odds from a betting website to evaluate the ELO system.

## Key Findings
- Logistic regression achieved **X% accuracy** in predicting match outcomes.
- Regularized regression provided robust predictions of score differences with an RMSE of **Y**.
- PCA identified **Z key KPIs** as most influential in determining match results.
- The custom ELO rating system correlated strongly with observed match outcomes, showcasing its potential as a predictive tool.

Contribution of KPIs to Components

<img src="figures/figure_feature_loading.png" alt="loading figure">
## Contribution
Contributions are welcome! Feel free to fork the repository and submit a pull request with your improvements. Suggestions for additional analyses, new features, or enhanced visualizations are highly encouraged.

We hope this project inspires further research and collaboration in the exciting intersection of sports and machine learning!
