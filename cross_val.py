import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

train_df_1 = pd.read_csv('BT2024176_train_var1.csv')
train_df_2 = pd.read_csv('BT2024176_train_var2.csv')

def find_best_model(df, max_features, max_degree, prefix):
    best_r2 = -float('inf')
    best_config = None
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for deg in range(1, max_degree + 1):
        for num_f in range(1, max_features + 1):
            features = [f'x{i}' for i in range(1, num_f+1)]
            X = df[features]
            y = df['y']
            
            val_r2_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                poly = PolynomialFeatures(degree=deg, include_bias=False)
                try:
                    X_train_poly = poly.fit_transform(X_train_fold)
                    X_val_poly = poly.transform(X_val_fold)
                    
                    model = LinearRegression().fit(X_train_poly, y_train_fold)
                    y_pred = model.predict(X_val_poly)
                    val_r2 = r2_score(y_val_fold, y_pred)
                    val_r2_scores.append(val_r2)
                except Exception as e:
                    pass # Memory error for high degree + many features
            
            if val_r2_scores:
                mean_r2 = np.mean(val_r2_scores)
                if mean_r2 > best_r2:
                    best_r2 = mean_r2
                    best_config = (deg, num_f)
                
                if mean_r2 > 0.8:
                    print(f"{prefix}: Degree {deg}, first {num_f} features -> Val R2: {mean_r2:.4f}")
                    
    print(f"\nBest Config for {prefix}: Degree {best_config[0]}, First {best_config[1]} features (Val R2: {best_r2:.4f})\n")
    return best_config

print("--- Phase 1 ---")
best_p1 = find_best_model(train_df_1, 6, 6, "Var1")

print("--- Phase 2 ---")
best_p2 = find_best_model(train_df_2, 3, 10, "Var2")
