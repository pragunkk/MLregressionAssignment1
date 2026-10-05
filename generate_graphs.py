import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

def evaluate_degrees(df, features, target_col, max_degree=7, prefix=""):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    degrees = list(range(2, max_degree + 1))
    
    mean_mses = []
    mean_r2s = []
    
    X = df[features]
    y = df[target_col]
    
    for deg in degrees:
        val_mses = []
        val_r2s = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            try:
                X_train_poly = poly.fit_transform(X_train_fold)
                X_val_poly = poly.transform(X_val_fold)
                
                model = LinearRegression().fit(X_train_poly, y_train_fold)
                y_pred = model.predict(X_val_poly)
                
                val_mses.append(mean_squared_error(y_val_fold, y_pred))
                val_r2s.append(r2_score(y_val_fold, y_pred))
            except Exception as e:
                print(f"Error for degree {deg}: {e}")
                
        if val_mses:
            mean_mses.append(np.mean(val_mses))
            mean_r2s.append(np.mean(val_r2s))
        else:
            mean_mses.append(None)
            mean_r2s.append(None)
            
    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Polynomial Degree')
    ax1.set_ylabel('Mean Squared Error (MSE)', color=color)
    ax1.plot(degrees, mean_mses, marker='o', color=color, label='MSE')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('R² Score', color=color)
    ax2.plot(degrees, mean_r2s, marker='s', color=color, label='R² Score')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(f'{prefix}: Validation Metrics vs Polynomial Degree (2 to 7)')
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(f'{prefix.lower()}_metrics.png', dpi=300)
    plt.close()
    
    # Print table
    print(f"--- {prefix} Metrics Table ---")
    print(f"{'Degree':<10} | {'MSE':<15} | {'R² Score':<15}")
    print("-" * 45)
    for d, mse, r2 in zip(degrees, mean_mses, mean_r2s):
        print(f"{d:<10} | {mse:<15.4f} | {r2:<15.4f}")
    print("\n")
    
    return degrees, mean_mses, mean_r2s

def main():
    train_df_1 = pd.read_csv('BT2024176_train_var1.csv')
    train_df_2 = pd.read_csv('BT2024176_train_var2.csv')
    
    # Using all features for both based on previous findings
    features_1 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    features_2 = ['x1', 'x2', 'x3']
    
    evaluate_degrees(train_df_1, features_1, 'y', max_degree=7, prefix="Var1")
    evaluate_degrees(train_df_2, features_2, 'y', max_degree=7, prefix="Var2")

if __name__ == "__main__":
    main()
