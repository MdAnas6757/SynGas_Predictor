#start
# Basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Neural network imports
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# Load dataset
data = pd.read_csv('project.csv')

# Display the first few rows
print(data.head())

# Define feature columns and target variable
features = ['Moisture_Content', 'Fixed_Carbon', 'Volatile_Matter', 'Ash_Content', 'Carbon', 'Hydrogen', 'Oxygen', 'Nitrogen', 'Sulfur', 'LHV', 'Gasification_Temperature', 'Air_to_Steam_Ratio', 'Reactor_Type']
target = 'Syngas_Yield'

# Preprocess data (assuming categorical encoding if necessary)
X = pd.get_dummies(data[features])
y = data[target]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature Correlation Analysis
correlation_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Pair Plot
sns.pairplot(data, vars=features + [target])
plt.show()

# Feature Importance using Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances by Random Forest')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Scale data for KNN and ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def create_ann_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Initialize ANN model
ann_model = create_ann_model(X_train_scaled.shape[1])
# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    
    if name == 'KNN':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif name == 'ANN':
        ann_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
        y_pred = ann_model.predict(X_test_scaled).flatten()
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R^2': r2}

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("Model Performance Comparison:")
print(results_df)
# Plot MSE
plt.figure(figsize=(12, 6))
results_df['MSE'].plot(kind='bar', color='skyblue')
plt.title('Model Performance - Mean Squared Error')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.show()

# Plot R^2
plt.figure(figsize=(12, 6))
results_df['R^2'].plot(kind='bar', color='salmon')
plt.title('Model Performance - R^2 Score')
plt.ylabel('R^2')
plt.xticks(rotation=45)
plt.show()
# Define parameter grids for hyperparameter tuning
param_grids = {
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

# Function for hyperparameter tuning
def tune_hyperparameters(model_name, model, param_grid):
    print(f"Tuning {model_name}...")
    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42, scoring='neg_mean_squared_error')
    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_score = -search.best_score_
    print(f"Best Params for {model_name}: {best_params}")
    print(f"Best Score for {model_name}: {best_score}")
    return best_params

# Tune hyperparameters
for name, model in models.items():
    if name in param_grids:
        best_params = tune_hyperparameters(name, model, param_grids[name])
# Feature Importance using Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances by Random Forest')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# 3D Plots
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['Carbon'], X_test['Hydrogen'], y_test, c='r', marker='o', label='Actual')
ax.scatter(X_test['Carbon'], X_test['Hydrogen'], y_pred, c='b', marker='x', label='Predicted')
ax.set_xlabel('Carbon')
ax.set_ylabel('Hydrogen')
ax.set_zlabel('Syngas Yield')
plt.legend()
plt.title('3D Plot of Actual vs Predicted Syngas Yield')
plt.show()

# Contour Plots
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Generate grid to evaluate the model
x_grid = np.linspace(X_test['Carbon'].min(), X_test['Carbon'].max(), 100)
y_grid = np.linspace(X_test['Hydrogen'].min(), X_test['Hydrogen'].max(), 100)
x_grid, y_grid = np.meshgrid(x_grid, y_grid)

# Predict on the grid
grid = np.c_[x_grid.ravel(), y_grid.ravel()]
grid_predictions = rf_model.predict(grid)
grid_predictions = grid_predictions.reshape(x_grid.shape)

plt.figure(figsize=(12, 8))
plt.contourf(x_grid, y_grid, grid_predictions, cmap='coolwarm', alpha=0.8)
plt.scatter(X_test['Carbon'], X_test['Hydrogen'], c=y_test, edgecolor='k', marker='o', cmap='coolwarm')
plt.colorbar()
plt.xlabel('Carbon')
plt.ylabel('Hydrogen')
plt.title('Contour Plot of Syngas Yield')
plt.show()

# Model Predictions vs Actual
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, c='blue', marker='o', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Syngas Yield')
plt.ylabel('Predicted Syngas Yield')
plt.title('Model Predictions vs Actual Syngas Yield')
plt.legend()
plt.show()
# Residual Analysis for each model
plt.figure(figsize=(14, 10))
for i, (name, model) in enumerate(models.items(), 1):
    plt.subplot(2, 4, i)
    
    if name == 'KNN':
        y_pred = model.predict(X_test_scaled)
    elif name == 'ANN':
        y_pred = ann_model.predict(X_test_scaled).flatten()
    else:
        y_pred = model.predict(X_test)
    
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, c='blue', marker='o', edgecolor='k')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {name}')

plt.tight_layout()
plt.show()
# Error Distribution
plt.figure(figsize=(14, 7))
for name, model in models.items():
    if name == 'KNN':
        y_pred = model.predict(X_test_scaled)
    elif name == 'ANN':
        y_pred = ann_model.predict(X_test_scaled).flatten()
    else:
        y_pred = model.predict(X_test)
    
    errors = y_test - y_pred
    plt.hist(errors, bins=30, alpha=0.5, label=name)

plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.legend()
plt.show()
import lime
import lime.lime_tabular

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=['Syngas_Yield'],
    mode='regression'
)

# Choose an instance to explain
i = 0  # Index of the instance
exp = explainer.explain_instance(X_test.iloc[i], model.predict, num_features=10)

# Visualize explanation
exp.show_in_notebook(show_table=True, show_all=False)
import shap

# SHAP values for Random Forest
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)

# Plot SHAP values
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type='bar')
from sklearn.svm import SVR

# Initialize and train SVR model
svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
svr_model.fit(X_train, y_train)

# Evaluate SVR model
y_pred = svr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
results['SVR'] = {'MSE': mse, 'R^2': r2}

# Update results DataFrame
results_df = pd.DataFrame(results).T
print("Model Performance Comparison:")
print(results_df)
from sklearn.ensemble import StackingRegressor

# Define base models and meta-model
base_models = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
]

meta_model = LinearRegression()
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train and evaluate stacking model
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
results['Stacking'] = {'MSE': mse, 'R^2': r2}

# Update results DataFrame
results_df = pd.DataFrame(results).T
print("Model Performance Comparison:")
print(results_df)
import plotly.express as px

# Plotly scatter plot for Model Predictions vs Actual
fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Syngas Yield', 'y': 'Predicted Syngas Yield'})
fig.add_scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Fit', line=dict(color='red', width=2))
fig.update_layout(title='Model Predictions vs Actual Syngas Yield')
fig.show()
# Example: Feature importance over different subsets (e.g., different years or data segments)
# This assumes you have a time-based column to split your data
time_column = 'Year'  # Replace with your actual time column

# Example segmentation
data['Year'] = pd.to_datetime(data['Date']).dt.year
years = data['Year'].unique()

plt.figure(figsize=(14, 10))
for year in years:
    subset = data[data['Year'] == year]
    X_subset = subset[features]
    y_subset = subset[target]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_subset, y_subset)
    importances = rf_model.feature_importances_
    plt.plot(features, importances, label=f'Year {year}')
    
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Over Time')
plt.legend()
plt.xticks(rotation=90)
plt.show()
# Example: Feature importance over different subsets (e.g., different years or data segments)
# This assumes you have a time-based column to split your data
time_column = 'Year'  # Replace with your actual time column

# Example segmentation
data['Year'] = pd.to_datetime(data['Date']).dt.year
years = data['Year'].unique()

plt.figure(figsize=(14, 10))
for year in years:
    subset = data[data['Year'] == year]
    X_subset = subset[features]
    y_subset = subset[target]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_subset, y_subset)
    importances = rf_model.feature_importances_
    plt.plot(features, importances, label=f'Year {year}')
    
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance Over Time')
plt.legend()
plt.xticks(rotation=90)
plt.show()
