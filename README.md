# FIIT Knowledge Discovery Project

This project is elaborated as an assignment for the Knowledge Discovery course (SS 2019/2020) at FIIT STU by [Denis Mitana](https://github.com/dmitana/) and [Miroslav Sumega](https://github.com/xsumegam/).

Detailed description of our solution is in the attached [paper](paper.pdf).

## Task
Predicting electricity consumption is not a simple task. To do that, you have to train artificial intelligence models on lots of data to predict accurately. However these data are specific in terms of containing building types and climatic conditions and might not be available. Therefore it is important to build general models which are able to solve these issues. In order to tackle this problem, we use Support Vector Regression (SVR), Extremely Randomized Trees (EXRT) and Extreme Gradient Boosting (XGBoost) models to predict electricity consumption of multiple buildings.

## Data 
The dataset for this problem is publicly available on Kaggle and is associated with the [ASHRAE - Great Energy Predictor III competition](https://www.kaggle.com/c/ashrae-energy-prediction). It includes three years of hourly meter readings from over one thousand buildings at several different sites around the world. By problem definition only one year of these three years is intended for training and the remaining two years are intended for testing.

## Results
We use model trained on one climatic zone to predict electricity consumption of buildings from different climatic zones to prove that it is possible to achieve acceptable results when predicting for buildings which model has never seen. We also found out that most important features are values of previous consumption. With our best model, which is XGBoost, we achieved results of 18.67 RMSE, 9.38 \% NRMSE and 1.16 RMSLE.
