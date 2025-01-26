import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

'''
# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

data = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
data['target'] = diabetes.target

# Save to CSV
data.to_csv("data/diabetes.csv", index=False)

# Split into training and testing data
'''
dataset_path = "data/diabetes.csv"

# Load the dataset
data = pd.read_csv(dataset_path)
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Experiment Setup
mlflow.set_experiment("load_diabetes")

# Define a function to log an experiment
def log_experiment(model, model_name, params):
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        print(f"Experiment for {model_name} logged successfully!")

# Run 1: Linear Regression
params_lr = {"fit_intercept": True, "copy_X": True, "n_jobs": -1}
model_lr = LinearRegression(**params_lr)
log_experiment(model_lr, "linear_regression_model", params_lr)

# Run 2: Ridge Regression
params_ridge = {"alpha": 1.0, "fit_intercept": True, "solver": "auto"}
model_ridge = Ridge(**params_ridge)
log_experiment(model_ridge, "ridge_regression_model", params_ridge)

# Run 3: Lasso Regression
params_lasso = {"alpha": 0.1, "fit_intercept": True, "max_iter": 1000}
model_lasso = Lasso(**params_lasso)
log_experiment(model_lasso, "lasso_regression_model", params_lasso)

print("All experiments logged. Check the MLflow UI for details.")