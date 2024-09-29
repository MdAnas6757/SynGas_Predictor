import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Load dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Define features and target
features = ['MC', 'FC', 'VM', 'Ash', 'C', 'H', 'O', 'N', 'S', 'LHV', 'T', 'AtoS', 'Reactor']
target = 'SY'

X = pd.get_dummies(data[features])
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Linear Regression': LinearRegression(),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
}

# Train models
for model in models.values():
    model.fit(X_train, y_train)

def predict_syngas_yield_all(input_data):
    input_df = pd.DataFrame([input_data])
    input_transformed = pd.get_dummies(input_df)
    missing_cols = set(X.columns) - set(input_transformed.columns)
    
    for col in missing_cols:
        input_transformed[col] = 0
    input_transformed = input_transformed[X.columns]  # Ensure same order
    
    predictions = {name: model.predict(input_transformed)[0] for name, model in models.items()}
    
    # Find the model with the highest yield
    best_model = max(predictions, key=predictions.get)
    best_yield = predictions[best_model]
    
    return {
        "predictions": predictions,
        "best_model": best_model,
        "best_yield": best_yield
    }
