import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    print(f"{'Degree':<10} | {'Best Alpha':<15} | {'Reg Val MSE':<15} | {'Reg Val R2':<15}")
    print("-" * 65)
    
    X = df[features]
    y = df[target_col]
    
    mean_reg_mses = []
    mean_reg_r2s = []
    
    for deg in degrees:
        reg_val_mses = []
        reg_val_r2s = []
        best_alpha_for_deg = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            try:
                X_train_poly = poly.fit_transform(X_train_fold)
                X_val_poly = poly.transform(X_val_fold)
                
                # Ridge
                scaler = StandardScaler()
                X_train_poly_scaled = scaler.fit_transform(X_train_poly)
                X_val_poly_scaled = scaler.transform(X_val_poly)
                
                model_ridge = RidgeCV(alphas=alphas, cv=3).fit(X_train_poly_scaled, y_train_fold)
                ridge_pred = model_ridge.predict(X_val_poly_scaled)
                
                reg_val_mses.append(mean_squared_error(y_val_fold, ridge_pred))
                reg_val_r2s.append(r2_score(y_val_fold, ridge_pred))
                best_alpha_for_deg.append(model_ridge.alpha_)
                
            except Exception as e:
                pass
                
        if reg_val_r2s:
            mean_reg_mse = np.mean(reg_val_mses)
            mean_reg_r2 = np.mean(reg_val_r2s)
            mean_alpha = np.median(best_alpha_for_deg)
            
            mean_reg_mses.append(mean_reg_mse)
            mean_reg_r2s.append(mean_reg_r2)
            
            if mean_reg_r2 > best_overall_r2:
                best_overall_r2 = mean_reg_r2
                best_overall_deg = deg
                best_overall_alpha = mean_alpha
                
            print(f"{deg:<10} | {mean_alpha:<15.4f} | {mean_reg_mse:<15.4f} | {mean_reg_r2:<15.4f}")
        else:
            print(f"{deg:<10} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
            mean_reg_mses.append(np.nan)
            mean_reg_r2s.append(np.nan)
            
    print(f"\nBest Config for {prefix}: Degree {best_overall_deg}, Alpha ~ {best_overall_alpha:.4f} (Val R2: {best_overall_r2:.4f})\n")
    
    # Plotting Regularization Metrics (MSE and R2)
    plot_mses = np.clip(mean_reg_mses, a_min=0, a_max=np.nanmedian(mean_reg_mses) * 10)
    plot_r2s = np.clip(mean_reg_r2s, a_min=-1.0, a_max=1.0)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('Mean Squared Error (MSE)', color=color)
    ax1.plot(degrees, plot_mses, marker='o', color=color, label='Regularized MSE')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('R² Score', color=color)
    ax2.plot(degrees, plot_r2s, marker='s', color=color, label='Regularized R² Score')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(f'{prefix}: Regularized Validation Metrics vs Polynomial Degree')
    fig.tight_layout()
    
    plt.savefig(f'{prefix.lower()}_metrics_reg.png', dpi=300)
    plt.close()

def main():
    train_df_1 = pd.read_csv('BT2024176_train_var1.csv')
    train_df_2 = pd.read_csv('BT2024176_train_var2.csv')
    
    features_1 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    features_2 = ['x1', 'x2', 'x3']
    
    evaluate_regularization(train_df_1, features_1, 'y', max_degree=10, prefix="Var1")
    evaluate_regularization(train_df_2, features_2, 'y', max_degree=20, prefix="Var2")

if __name__ == "__main__":
    main()
