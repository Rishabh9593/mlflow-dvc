import pandas as pd

dataset_path = "data/diabetes.csv"

# Load the dataset
data = pd.read_csv(dataset_path)
X = data.drop(columns=["target"])
y = data["target"]

# Example: Remove one column
data.drop(columns=["bmi"], inplace=True)
data.to_csv("data/diabetes.csv", index=False)


