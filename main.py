import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def phase_1():
    print("--- Phase 1: Power Plant Steam Turbine Optimization (var1) ---")
    train_df = pd.read_csv('BT2024176_train_var1.csv')
    test_df = pd.read_csv('BT2024176_test_var1.csv')
    
    # We found via CV that with Ridge Regularization, using all 6 features and degree 5 gives optimal validation R2
    features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    target = 'y'
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    # Polynomial features degree 5
    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    
    # Train model with Ridge (alpha = 11.2884)
    model = Ridge(alpha=11.2884)
    model.fit(X_train_poly_scaled, y_train)
    
    # Train metrics
    y_train_pred = model.predict(X_train_poly_scaled)
    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)
    print(f"Train MSE: {mse:.4f}")
    print(f"Train R2: {r2:.4f}")
    
    # Predict on test
    X_test = test_df[features]
    X_test_poly = poly.transform(X_test)
    X_test_poly_scaled = scaler.transform(X_test_poly)
    y_test_pred = model.predict(X_test_poly_scaled)
    
    # Save predictions
    pred_df = test_df.copy()
    pred_df['y'] = y_test_pred
    pred_df.to_csv('BT2024176_pred_var1.csv', index=False)
    print("Saved Phase 1 predictions to BT2024176_pred_var1.csv\n")

def phase_2():
    print("--- Phase 2: Subterranean Thermal Reservoir Mapping (var2) ---")
    train_df = pd.read_csv('BT2024176_train_var2.csv')
    test_df = pd.read_csv('BT2024176_test_var2.csv')
    
    # We found via CV that with Ridge Regularization, using all 3 features and degree 10 gives optimal validation R2
    features = ['x1', 'x2', 'x3']
    target = 'y'
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    # Polynomial features degree 10
    poly = PolynomialFeatures(degree=10, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    
    # Train model with Ridge (alpha = 1.6238)
    model = Ridge(alpha=1.6238)
    model.fit(X_train_poly_scaled, y_train)
    
    # Train metrics
    y_train_pred = model.predict(X_train_poly_scaled)
    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)
    print(f"Train MSE: {mse:.4f}")
    print(f"Train R2: {r2:.4f}")
    
    # Predict on test
    X_test = test_df[features]
    X_test_poly = poly.transform(X_test)
    X_test_poly_scaled = scaler.transform(X_test_poly)
    y_test_pred = model.predict(X_test_poly_scaled)
    
    # Save predictions
    pred_df = test_df.copy()
    pred_df['y'] = y_test_pred
    pred_df.to_csv('BT2024176_pred_var2.csv', index=False)
    print("Saved Phase 2 predictions to BT2024176_pred_var2.csv\n")

if __name__ == "__main__":
    phase_1()
    phase_2()
