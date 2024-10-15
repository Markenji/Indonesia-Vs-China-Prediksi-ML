# Model Evaluation for Team Match Prediction

![Prediction Model](https://github.com/Markenji/Indonesia-Vs-China-Prediksi-ML/blob/main/Image/tiktok22.jpg)

## Overview

This repository contains a machine learning model developed to predict the outcomes of matches between various teams. The model uses different algorithms to evaluate performance based on historical match data.

## Model Evaluation Results

During the evaluation phase, several machine learning models were trained and tested. The results showed unusually high accuracy rates:

- **Logistic Regression:** 1.00
- **Decision Tree:** 1.00
- **Random Forest:** 1.00
- **Support Vector Machine:** 0.00
- **K-Nearest Neighbors:** 0.00

The model with the highest accuracy was **Logistic Regression**, achieving a perfect score. However, such results raise several concerns regarding the validity of the findings.

### Possible Reasons for High Accuracy

1. **Small Dataset**: The limited amount of data allows models to memorize training instances, resulting in inflated accuracy.
   
2. **Lack of Variability**: The dataset may lack sufficient variability in features and targets, leading to simplistic predictions.

3. **Target Leakage**: Features may correlate too closely with the target variable, causing overfitting.

4. **Evaluation Metrics**: Solely relying on accuracy can be misleading, especially in cases of class imbalance.

### Recommendations for Improvement

1. **Cross-Validation**: Implement k-fold cross-validation to assess model performance more robustly across different data splits.
   
   ```python
   from sklearn.model_selection import cross_val_score

   for model_name, model in models.items():
       cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
       print(f"{model_name}: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
