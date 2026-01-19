# Bank Marketing Campaign Classifier

This project builds and **compares** several machine learning models to predict the outcome of a bank marketing campaign using structured customer data.

This project focuses on exploration and comparison of models, rather than optimisation for a production setting.

The dataset has a strong class imbalance and a mix of numerical and categorical features, so accuracy alone is not sufficient and the analysis focuses on recall, precision, F1 score, and ROC-AUC. 

A range of common and well-understood models are tested to see how they behave under class imbalance, from simple baselines such as K-Nearest Neighbours and Naive Bayes to more flexible models like Random Forest, Support Vector Machines, and XGBoost. Different imbalance handling methods, including resampling and class weighting, are also explored to understand their impact on model performance and generalisation.

Model evaluation is performed using a clear train/test split. Training results are used to understand model behaviour and overfitting, while conclusions are based on test-set performance. For selected models, decision thresholds are explored to illustrate recall–precision trade-offs.

This notebook is intentionally more detailed than a production pipeline. In a real deployment, the workflow would typically be simplified by selecting fewer models, choosing a single imbalance-handling strategy, and adding monitoring and validation steps. All preprocessing, training, evaluation, and prediction steps are contained in the notebook, which can be run end-to-end in a standard Python environment.


## Technical Notes and Reflection

This project explored multiple models and imbalance-handling strategies in an academic setting. Some choices, such as testing K-Nearest Neighbours, Random Forest, SVM, and XGBoost with SMOTE and SMOTE-ENN, were intentionally exploratory to understand how models behave under class imbalance and how evaluation metrics change.

Training results were used diagnostically to observe overfitting and model sensitivity, but final assessments were always based on held-out test performance. Threshold tuning and multiple resampling strategies were included to demonstrate trade-offs, though in a production system these would typically be simplified.

Overall, the work is technically ambitious and shows broad coverage, experimentation, and understanding. In a production pipeline, the workflow would focus on fewer models, a single imbalance-handling method, and careful monitoring, while avoiding reliance on training-set performance to justify decisions.
