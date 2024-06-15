# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the datasets
train_df = pd.read_csv('/mnt/data/train.csv')
test_df = pd.read_csv('/mnt/data/test.csv')
gender_submission_df = pd.read_csv('/mnt/data/gender_submission.csv')

# 1. Data Exploration
# Check the structure of the train dataset
print(train_df.info())
print(train_df.describe())
print(train_df.head())

# Check for missing data
print(train_df.isnull().sum())

# Visualize missing data
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap - Train Dataset')
plt.show()

# Analyze categorical variables
print(train_df['Sex'].value_counts())
print(train_df['Embarked'].value_counts())

# Visualize distributions
sns.countplot(x='Survived', data=train_df)
plt.title('Survived Count')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title('Pclass vs Survived')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=train_df)
plt.title('Sex vs Survived')
plt.show()

# 2. Data Preprocessing
# Combine train and test datasets for consistent preprocessing
combined_df = pd.concat([train_df.drop(columns=['Survived']), test_df])

# Handle missing values
imputer = SimpleImputer(strategy='median')
combined_df['Age'] = imputer.fit_transform(combined_df[['Age']])
combined_df['Fare'] = imputer.fit_transform(combined_df[['Fare']])
combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numeric
label_encoder = LabelEncoder()
combined_df['Sex'] = label_encoder.fit_transform(combined_df['Sex'])
combined_df['Embarked'] = label_encoder.fit_transform(combined_df['Embarked'])

# Split the combined dataset back into train and test sets
train_df_processed = combined_df.iloc[:len(train_df), :]
test_df_processed = combined_df.iloc[len(train_df):, :]

# Add the 'Survived' column back to the train dataset
train_df_processed['Survived'] = train_df['Survived']

# Drop irrelevant columns
train_df_processed.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
test_df_processed.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Feature scaling
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_df_processed.drop(columns=['Survived'])), columns=train_df_processed.columns[:-1])
test_scaled = pd.DataFrame(scaler.transform(test_df_processed), columns=test_df_processed.columns)

# Split the dataset
X = train_scaled
y = train_df_processed['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Building
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Decision Tree Classifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# Random Forest Classifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)

# 4. Model Evaluation
def evaluate_model(y_test, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    return metrics

logreg_metrics = evaluate_model(y_test, y_pred_logreg)
tree_metrics = evaluate_model(y_test, y_pred_tree)
forest_metrics = evaluate_model(y_test, y_pred_forest)

# Print the evaluation results for each model
print(f"Logistic Regression:\nAccuracy: {logreg_metrics['accuracy']}\nPrecision: {logreg_metrics['precision']}\nRecall: {logreg_metrics['recall']}\nF1-Score: {logreg_metrics['f1']}\n")
print(f"Decision Tree:\nAccuracy: {tree_metrics['accuracy']}\nPrecision: {tree_metrics['precision']}\nRecall: {tree_metrics['recall']}\nF1-Score: {tree_metrics['f1']}\n")
print(f"Random Forest:\nAccuracy: {forest_metrics['accuracy']}\nPrecision: {forest_metrics['precision']}\nRecall: {forest_metrics['recall']}\nF1-Score: {forest_metrics['f1']}\n")

# 5. Model Tuning
# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_forest = grid_search.best_estimator_
y_pred_best_forest = best_forest.predict(X_test)

# Evaluate the tuned Random Forest model
best_forest_metrics = evaluate_model(y_test, y_pred_best_forest)
print(f"Tuned Random Forest:\nAccuracy: {best_forest_metrics['accuracy']}\nPrecision: {best_forest_metrics['precision']}\nRecall: {best_forest_metrics['recall']}\nF1-Score: {best_forest_metrics['f1']}\n")

# 6. Make Predictions
# Make predictions on the test set
test_predictions = best_forest.predict(test_scaled)

# 7. Submission
submission = pd.DataFrame({
    'PassengerId': gender_submission_df['PassengerId'],
    'Survived': test_predictions
})

submission.to_csv('/mnt/data/submission.csv', index=False)
print("Submission file created successfully.")
