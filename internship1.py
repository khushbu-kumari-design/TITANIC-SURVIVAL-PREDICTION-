import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the Dataset Using (Kaggle's hosted raw URL)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("--- Dataset Preview ---")
print(df.head(), "\n")

# 2. Data Cleaning & Preprocessing
# Fill missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the most common port (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' (too many missing values), 'PassengerId', 'Name', and 'Ticket' (for basic model)
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Convert categorical variables into numerical dummy variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 3. Split Data into Features (X) and Target (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build and Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-- Model Performance --")
print(f"Accuracy Score: {accuracy:.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=importances.index, palette="viridis")
plt.title("Feature Importance in Titanic Survival Prediction")
plt.xlabel("Score")
plt.ylabel("Features")
plt.show()