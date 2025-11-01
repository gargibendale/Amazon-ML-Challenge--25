import pandas as pd
import numpy as np

# Load the file (replace with your actual file path)
df = pd.read_csv("../dataset/dump_submission.csv")

# Ensure the column names are correct
actual = df["price"]
pred = df["pred_price"]

# Define SMAPE function
def smape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    diff = np.abs(actual - predicted) / denominator
    diff[denominator == 0] = 0  # Handle divide-by-zero
    return np.mean(diff) * 100  # Convert to percentage

# Calculate SMAPE
smape_value = smape(actual, pred)

print(f"SMAPE: {smape_value:.2f}%")
