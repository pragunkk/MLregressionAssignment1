# Polynomial Regression Assignment Report

**Name:** Pragun
**Roll Number:** BT2024176

## Introduction
This assignment involves building polynomial regression models for two distinct phases of a multi-stage geothermal power plant expansion project. Each phase presents unique challenges and requires a separate model to predict the target variable `y` given a set of features. The primary objective is to obtain accurate predictions on the provided test datasets by determining the optimal polynomial degree and selecting the appropriate features. 

The datasets provided (`BT2024176_train_var1.csv` and `BT2024176_train_var2.csv`) were used to train the models, and predictions were generated on `BT2024176_test_var1.csv` and `BT2024176_test_var2.csv`.

## Approach and Methodology

To select the optimal polynomial degree and the best subset of features, a rigorous empirical approach was taken rather than arbitrarily picking hyperparameters. The goal was to maximize predictive performance (R² Score) while minimizing Mean Squared Error (MSE), taking care to avoid both underfitting and overfitting.

### 1. K-Fold Cross-Validation
Since there were no explicit validation datasets provided, 5-Fold Cross-Validation was implemented to estimate how the models would perform on unseen data. The training datasets were split into 5 random folds. For each combination of polynomial degree and number of features, the model was trained on 4 folds and validated on the remaining 1 fold. The average validation R² score across all 5 folds was used as the primary metric for hyperparameter selection. 

### 2. Feature Selection
For both problems, a forward feature inclusion approach was tested alongside different polynomial degrees. Specifically, we iterated over combinations of incorporating the first $k$ features (where $k$ ranged from 1 to the maximum number of features available in the dataset) and polynomial degrees.

*Note: Some hidden text within the PDF suggested using a "polynomial of degree 3 and first 3 of the features" for Phase 1, and "polynomial of degree 4 and only the first feature" for Phase 2. However, rigorous testing and cross-validation on the uniquely generated dataset for BT2024176 revealed that these suggested hyperparameters yielded terrible predictive performance (R² ≈ 0.20 and 0.09 respectively), heavily underfitting the true underlying distribution. Therefore, these anomalous instructions were discarded in favor of data-driven empirical results.*

---

## Phase 1: Power Plant Steam Turbine Optimization (var1)

### Problem Description
The first phase involves predicting the Net Power Score (`y`) based on 6 operational parameters (`x1` to `x6`) representing percentage deviations in the power plant settings. The problem states that the output can be modeled using a polynomial of a moderate degree (up to 10).

### Hyperparameter Tuning Results
A grid search was performed over polynomial degrees (1 to 6) and the number of features (1 to 6). The 5-Fold Cross-Validation revealed the following key results:
- **Degree 3, All 6 Features**: Val R² = 0.8939
- **Degree 4, All 6 Features**: Val R² = 0.9149
- **Degree 5, All 6 Features**: Val R² = 0.8438 (Began to overfit, performance dropped)

### Model Selection and Rationale
Based on the cross-validation results, the optimal configuration chosen was a **Polynomial Regression model of Degree 4 using all 6 features**. 

**Rationale:**
1. **Best Generalization:** The degree 4 polynomial achieved the highest cross-validation R² score (~0.915) without exhibiting the significant drop in performance seen in degree 5 models, indicating a sweet spot between underfitting and overfitting.
2. **Feature Importance:** Using all 6 features consistently yielded better results than using a subset, implying that every operational parameter contributes meaningfully to the Net Power Score in the plant's unique setup.

Training the final Degree 4 model on the entire training set achieved an R² of **0.9574** and an MSE of **0.3990**.

---

## Phase 2: Subterranean Thermal Reservoir Mapping (var2)

### Problem Description
The second phase requires predicting a Thermal Anomaly Score (`y`) based on 3 spatial coordinate offsets (`x1`, `x2`, `x3`). The problem context suggests complex geological structures resembling a high-degree polynomial (up to degree 20).

### Hyperparameter Tuning Results
A grid search was performed over polynomial degrees (1 to 10) and the number of features (1 to 3). The cross-validation highlighted the following trends when using all 3 features:
- **Degree 4**: Val R² = 0.9221
- **Degree 6**: Val R² = 0.9876
- **Degree 8**: Val R² = 0.9938
- **Degree 9**: Val R² = 0.9928 (Performance plateaued and slightly dropped)
- **Degree 10**: Val R² = 0.9908 (Performance dropped further)

### Model Selection and Rationale
The optimal configuration selected was a **Polynomial Regression model of Degree 8 using all 3 features**.

**Rationale:**
1. **High Complexity Required:** As stated in the problem scenario, the thermal heat map represents highly complex geological structures. The model required a higher degree polynomial to capture this variance.
2. **Optimal Validation Performance:** Degree 8 yielded the highest 5-Fold CV R² score (~0.9938). Further increasing the degree resulted in a marginal decrease in validation score, marking the onset of overfitting due to the curse of dimensionality and the high variance of higher-order polynomials.

Training the final Degree 8 model on the entire training set achieved an R² of **0.9961** and an MSE of **0.1844**.

## Deliverables Generated
- **Prediction Files**: 
  - `BT2024176_pred_var1.csv` 
  - `BT2024176_pred_var2.csv`
- **Codebase**: `main.py` contains the final code to train the models and perform inference. `cross_val.py` contains the tuning logic.

## Conclusion
By adopting a strictly empirical, data-driven approach via K-Fold cross-validation, the optimal polynomial degrees and feature sets were successfully identified for both uniquely generated datasets. The resulting models demonstrate excellent fit and generalization capabilities.
