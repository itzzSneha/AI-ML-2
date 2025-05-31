import matplotlib.pyplot as plt
import seaborn as sns
# ...existing code...
import pandas as pd

# Load the dataset
df = pd.read_csv("c:/Users/sneha/Downloads/Titanic-Dataset.csv")

# Show info and first few rows
print(df.info())
print(df.head())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode 'Sex' column: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Encode 'Embarked' column: S=0, C=1, Q=2
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
# Save cleaned data

# Descriptive statistics for numerical columns
print("\nDescriptive statistics:")
print(df.describe())

# Descriptive statistics for categorical columns
print("\nValue counts for 'Survived':")
print(df['Survived'].value_counts())
print("\nValue counts for 'Sex':")
print(df['Sex'].value_counts())
print("\nValue counts for 'Embarked':")
print(df['Embarked'].value_counts())

# Visualize Age distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Visualize survival count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Multiple Regression (example: predicting Fare based on Age, Sex, Pclass, SibSp, Parch, Embarked)
from sklearn.linear_model import LinearRegression

features = ['Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
X = df[features]
y = df['Fare']

reg = LinearRegression()
reg.fit(X, y)

print("\nRegression coefficients:")
for feat, coef in zip(features, reg.coef_):
    print(f"{feat}: {coef:.4f}")
print(f"Intercept: {reg.intercept_:.4f}")

# Model interpretation
print("\nModel Interpretation:")
print("Positive coefficients mean the feature increases the predicted Fare; negative means it decreases it.")
print("The magnitude shows the strength of the effect (after standardization).")

# Add evaluation metrics here
from sklearn.metrics import mean_squared_error, r2_score

y_pred = reg.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Save cleaned data to CSV
df.to_csv("Titanic-Dataset-clean.csv", index=False)
# Visualize Age distribution

print("Data cleaned and saved as Titanic-Dataset-clean.csv")