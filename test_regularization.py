import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def evaluate_regularization(df, features, target_col, max_degree, prefix=""):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    degrees = list(range(1, max_degree + 1))
    
    alphas = np.logspace(-4, 4, 20)
    
    best_overall_r2 = -float('inf')
    best_overall_deg = None
    best_overall_alpha = None
    
    print(f"--- {prefix} Regularization Test ---")
    print(f"{'Degree':<10} | {'Best Alpha':<15} | {'Reg Val R2':<15} | {'OLS Val R2':<15}")
    print("-" * 65)
    
    X = df[features]
    y = df[target_col]
    
    for deg in degrees:
        # Cross-validation for OLS (to compare)
        ols_val_r2s = []
        # We will do a manual nested-like approach or just use RidgeCV inside manual CV.
        # It's better to do manual CV and use RidgeCV on the training folds.
        
        reg_val_r2s = []
        best_alpha_for_deg = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            try:
                X_train_poly = poly.fit_transform(X_train_fold)
                X_val_poly = poly.transform(X_val_fold)
                
                # OLS
                model_ols = LinearRegression().fit(X_train_poly, y_train_fold)
                ols_pred = model_ols.predict(X_val_poly)
                ols_val_r2s.append(r2_score(y_val_fold, ols_pred))
                
                # Ridge
                scaler = StandardScaler()
                X_train_poly_scaled = scaler.fit_transform(X_train_poly)
                X_val_poly_scaled = scaler.transform(X_val_poly)
                
                model_ridge = RidgeCV(alphas=alphas, cv=3).fit(X_train_poly_scaled, y_train_fold)
                ridge_pred = model_ridge.predict(X_val_poly_scaled)
                reg_val_r2s.append(r2_score(y_val_fold, ridge_pred))
                best_alpha_for_deg.append(model_ridge.alpha_)
                
            except Exception as e:
                pass
                
        if reg_val_r2s:
            mean_reg_r2 = np.mean(reg_val_r2s)
            mean_ols_r2 = np.mean(ols_val_r2s)
            mean_alpha = np.median(best_alpha_for_deg)
            
            if mean_reg_r2 > best_overall_r2:
                best_overall_r2 = mean_reg_r2
                best_overall_deg = deg
                best_overall_alpha = mean_alpha
                
            print(f"{deg:<10} | {mean_alpha:<15.4f} | {mean_reg_r2:<15.4f} | {mean_ols_r2:<15.4f}")
        else:
            print(f"{deg:<10} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
            
    print(f"\nBest Config for {prefix}: Degree {best_overall_deg}, Alpha ~ {best_overall_alpha:.4f} (Val R2: {best_overall_r2:.4f})\n")

def main():
    train_df_1 = pd.read_csv('BT2024176_train_var1.csv')
    train_df_2 = pd.read_csv('BT2024176_train_var2.csv')
    
    features_1 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    features_2 = ['x1', 'x2', 'x3']
    
    evaluate_regularization(train_df_1, features_1, 'y', max_degree=10, prefix="Var1")
    evaluate_regularization(train_df_2, features_2, 'y', max_degree=20, prefix="Var2")

if __name__ == "__main__":
    main()
