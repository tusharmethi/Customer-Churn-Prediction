"""Customer Churn Prediction & Analysis
Dataset: IBM Telco Customer Churn (~7K records)
Model: Logistic Regression | Accuracy: ~82%
Tech: Python, Pandas, Scikit-learn, Matplotlib
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATASET
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Dataset loaded successfully!")
print(f"Total records: {df.shape[0]}")

# 2. EDA — VISUALIZATION 1: Churn by Contract Type
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn.plot(kind='bar', stacked=False, figsize=(8, 5), color=['steelblue', 'tomato'])
plt.title('Customer Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.legend(title='Churn')
plt.tight_layout()
plt.show()

# 3. EDA — VISUALIZATION 2: Churn by Tenure
plt.figure(figsize=(10, 5))
df.groupby('tenure')['Churn'].apply(
    lambda x: (x == 'Yes').sum()
).plot(kind='line', color='darkorange', linewidth=2)

plt.title('Customer Churn Count by Tenure (Months)')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Churned Customers')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 4. EDA — VISUALIZATION 3: Churn by Monthly Charges
plt.figure(figsize=(8, 5))
churn_yes = df[df['Churn'] == 'Yes']['MonthlyCharges']
churn_no  = df[df['Churn'] == 'No']['MonthlyCharges']

plt.hist(churn_no,  bins=30, alpha=0.6, label='No Churn',  color='steelblue')
plt.hist(churn_yes, bins=30, alpha=0.6, label='Churned',   color='tomato')
plt.title('Monthly Charges Distribution by Churn Status')
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Number of Customers')
plt.legend()
plt.tight_layout()
plt.show()

# 5. DATA CLEANING & FEATURE ENGINEERING
# Handle hidden missing values in TotalCharges 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)  # Drop ~11 rows with missing data

# Drop customerID — no predictive power
df.drop('customerID', axis=1, inplace=True)
# Encode categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if len(df[col].unique()) == 2:
        # Binary columns (Yes/No, Male/Female) → Label Encode to 0/1
        df[col] = le.fit_transform(df[col])
    else:
        # Multi-category columns → One-Hot Encode
        df = pd.get_dummies(df, columns=[col], drop_first=True)

print("Data cleaning and feature engineering complete.")

# 6. MODEL TRAINING
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
print("Model training complete.")

# 7. EVALUATION
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n--- Model Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Performance Report:")
print(classification_report(y_test, predictions))

# 8. VISUALIZATION 4: Feature Importance
feature_names = X.columns
coefficients  = model.coef_[0]

feature_importance = pd.DataFrame({
    'Feature':    feature_names,
    'Importance': coefficients
}).sort_values(by='Importance', ascending=True)

top_features = pd.concat([
    feature_importance.head(5),   
    feature_importance.tail(5)    
])

plt.figure(figsize=(10, 6))
colors = ['green' if x < 0 else 'red' for x in top_features['Importance']]
plt.barh(top_features['Feature'], top_features['Importance'], color=colors)
plt.title('What Drives Customer Churn? (Logistic Regression Coefficients)')
plt.xlabel('Impact on Churn  |  Negative = Stays · Positive = Leaves')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
