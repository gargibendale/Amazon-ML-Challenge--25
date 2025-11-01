import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- CONFIG ----------
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# ---------- LOAD DATA ----------
df = pd.read_csv("../dataset/processed/final_cleaned_dataset.csv")  # change to cleaned_test.csv if needed

print("✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ---------- BASIC INFO ----------
print("🔹 Basic Information:")
print(df.info())
print("\n")

# ---------- MISSING VALUE ANALYSIS ----------
print("🔹 Missing Value Summary:")
missing_summary = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)
missing_df = pd.DataFrame({'Missing Values': missing_summary, 'Missing %': missing_percent})
print(missing_df)
print("\nTotal Columns with Missing Values:", (missing_summary > 0).sum(), "\n")

# ---------- DESCRIPTIVE STATS FOR NUMERIC COLUMNS ----------
print("🔹 Numerical Columns Summary:")
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.describe().T)
print("\n")

# ---------- CATEGORICAL / TEXTUAL COLUMN ANALYSIS ----------
print("🔹 Categorical/Text Columns Summary:")
categorical_df = df.select_dtypes(exclude=[np.number])
for col in categorical_df.columns:
    if col == 'image_link':
        print(f"\n▶ Column: {col}")
        print(f"Unique Values: {categorical_df[col].nunique()}")
        print(f"Most Frequent Value: {categorical_df[col].mode()[0] if not categorical_df[col].mode().empty else 'N/A'}")
        print(f"Top 5 Values:\n{categorical_df[col].value_counts().head(30)}\n")

# # ---------- TEXT LENGTH ANALYSIS ----------
# text_cols = [col for col in df.columns if df[col].dtype == "object"]
# print("🔹 Text Length and Word Count Analysis:")
# for col in text_cols:
#     df[f"{col}_len"] = df[col].astype(str).apply(len)
#     df[f"{col}_word_count"] = df[col].astype(str).apply(lambda x: len(x.split()))
#     print(f"{col}: mean_len={df[f'{col}_len'].mean():.2f}, mean_words={df[f'{col}_word_count'].mean():.2f}")

# # ---------- CORRELATION ANALYSIS ----------
# if len(numeric_df.columns) > 1:
#     print("\n🔹 Correlation Matrix:")
#     corr = numeric_df.corr()
#     print(corr)
    
#     plt.figure(figsize=(8, 5))
#     sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
#     plt.title("Correlation Heatmap (Numeric Columns)")
#     plt.show()

# # ---------- OUTLIER DETECTION ----------
# print("\n🔹 Outlier Summary (using IQR):")
# for col in numeric_df.columns:
#     Q1 = numeric_df[col].quantile(0.25)
#     Q3 = numeric_df[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
#     outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)]
#     print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

# # ---------- DISTRIBUTION PLOTS ----------
# print("\n🔹 Plotting Numeric Distributions...")
# for col in numeric_df.columns:
#     plt.figure(figsize=(6, 3))
#     sns.histplot(df[col], kde=True)
#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.show()

print("\n✅ EDA Completed Successfully!")
