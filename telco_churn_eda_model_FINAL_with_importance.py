
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Drop customerID column
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric and drop missing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# EDA Visualizations
sns.set(style='whitegrid')

# Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# Tenure vs Churn
plt.figure(figsize=(8, 4))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Monthly Charges vs Churn
plt.figure(figsize=(8, 4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Churn by Contract Type
df_original = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
plt.figure(figsize=(6, 4))
sns.countplot(x='Contract', hue='Churn', data=df_original)
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.show()

# One-hot encoding
categorical_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Correlation Heatmap (now after encoding)
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# === Feature Importance ===
import numpy as np

# Combine feature names and coefficients
feature_names = X.columns
coefficients = model.coef_[0]

# Create DataFrame for feature impact
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
coef_df['AbsImpact'] = np.abs(coef_df['Coefficient'])
coef_df_sorted = coef_df.sort_values(by='AbsImpact', ascending=False)

# Print Top 10
print("\nTop 10 Influential Features on Churn:")
print(coef_df_sorted[['Feature', 'Coefficient']].head(10))

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df_sorted.head(10), palette='coolwarm')
plt.title('Top 10 Influential Features on Customer Churn')
plt.xlabel('Coefficient Value (Impact)')
plt.ylabel('Feature')
plt.grid(True)
plt.tight_layout()
plt.show()
