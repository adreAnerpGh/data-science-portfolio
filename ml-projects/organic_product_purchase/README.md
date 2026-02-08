# Organic Product Purchase Prediction

This project builds and **compares machine learning models** to predict whether a customer will purchase **organic products**, using structured loyalty-programme data.

The dataset contains a mix of **numerical and categorical features** and shows a **clear class imbalance** between organic buyers and non-buyers. As a result, model evaluation goes beyond accuracy and focuses on **precision, recall, F1-score, confusion matrices, and ROC-AUC**.

To ensure a **fair and controlled comparison**, all models are trained using the same preprocessing pipeline, **train/test split**, and evaluation metrics. Preprocessing steps (imputation, scaling, and encoding) are embedded within **scikit-learn pipelines** to prevent data leakage and ensure reproducibility.

## Modelling Approach

Three model families are evaluated:

- **Random Forest** – robust ensemble baseline  
- **XGBoost** – high-performance gradient boosting model  
- **Neural Network (MLP)** – non-linear alternative  

Hyperparameter tuning is performed using **GridSearchCV**, and conclusions are based exclusively on **held-out test-set performance**.

## Evaluation and Threshold Analysis

Given the marketing context, missing a potential organic buyer is more costly than contacting a non-buyer. For this reason, the analysis explores **decision-threshold adjustment**, showing how lowering the classification threshold improves **recall for organic buyers** at the cost of additional false positives.

Model performance is assessed using:
- Classification reports
- Confusion matrices
- ROC curves and ROC-AUC scores

## Final Model

The final recommended model is **XGBoost**, which achieves the strongest overall performance and good class separation. The trained model is saved using **joblib** to allow reuse without retraining.
