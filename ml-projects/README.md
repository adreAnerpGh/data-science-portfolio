# Bank Marketing Campaign Classifier

This project builds and **compares** several machine learning models to predict the outcome of a bank marketing campaign using structured customer data.

This project focuses on **exploration and comparison of models**, rather than optimisation for a production setting.

The dataset has a **strong class imbalance** and a mix of numerical and categorical features, so accuracy alone is not sufficient and the analysis focuses on **recall, precision, F1 score, and ROC-AUC**. 

To ensure a **fair and controlled comparison**, the same preprocessing steps, **train/test split**, imbalance strategies, and evaluation metrics are applied consistently across all models. This allows differences in results to be attributed to **model behaviour**, rather than changes in the pipeline.

A range of **common and well-understood models** are tested to see how they behave under class imbalance, from simple baselines such as **K-Nearest Neighbours** and **Naive Bayes** to more flexible models like **Random Forest**, **Support Vector Machines**, and **XGBoost**. Different imbalance handling methods, including **resampling** and **class weighting**, are also explored to understand their impact on model performance and generalisation.

Model evaluation is performed using a **clear train/test split**. Training results are used to understand model behaviour and **overfitting**, while conclusions are based on **test-set performance**. For selected models, **decision thresholds** are explored to illustrate **recall–precision trade-offs**.

This notebook is intentionally **more detailed than a production pipeline**. In a real deployment, the workflow would typically be simplified by selecting fewer models, choosing a **single imbalance-handling strategy**, and adding **monitoring and validation steps**. All preprocessing, training, evaluation, and prediction steps are contained in the notebook, which can be run **end-to-end** in a standard Python environment.


## Technical Notes and Reflection

This project was developed in an **academic and exploratory context**, with the goal of understanding how different models and imbalance-handling strategies behave on a **challenging, imbalanced dataset**. Techniques such as **SMOTE** and **SMOTE-ENN** were included deliberately to observe how resampling affects both model learning and evaluation metrics.

While **training-set results** are reported throughout the analysis, they are used only to diagnose **overfitting** and model sensitivity. It is important to note that **training performance (particularly after resampling)can be misleading**. Methods such as **SMOTE-ENN** alter the training distribution by removing difficult or borderline samples, which can lead to **artificially high training scores** that do not reflect real-world performance.

For this reason, **model comparisons and final assessments rely on test-set results**, which provide a **more reliable indication of generalisation**. Some exploratory choices add complexity that would not be appropriate in production, but they serve to highlight **important trade-offs and limitations** when working with imbalanced data.

## 

[**bank_marketing_xgboost_pipeline**](#ml-projects/bank_marketing_xgboost_pipeline)

This notebook is a **focused follow-up** to the original project, showcasing a **single, strong XGBoost pipeline** with the same preprocessing and evaluation standards. It removes the multiple model comparisons, emphasizes **class imbalance handling**, and demonstrates a **clean, reproducible workflow** suitable for a **practical application**.






