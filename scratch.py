import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

train_df_1 = pd.read_csv('BT2024176_train_var1.csv')
features_1 = ['x1', 'x2', 'x3']
X1 = train_df_1[features_1]
y1 = train_df_1['y']
poly1 = PolynomialFeatures(degree=3, include_bias=False)
X1_poly = poly1.fit_transform(X1)
model1 = LinearRegression().fit(X1_poly, y1)
r2_1 = r2_score(y1, model1.predict(X1_poly))
print(f"Var1: Degree 3, first 3 features R2: {r2_1:.4f}")

train_df_2 = pd.read_csv('BT2024176_train_var2.csv')
features_2 = ['x1']
X2 = train_df_2[features_2]
y2 = train_df_2['y']
poly2 = PolynomialFeatures(degree=4, include_bias=False)
X2_poly = poly2.fit_transform(X2)
model2 = LinearRegression().fit(X2_poly, y2)
r2_2 = r2_score(y2, model2.predict(X2_poly))
print(f"Var2: Degree 4, first 1 feature R2: {r2_2:.4f}")
