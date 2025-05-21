  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# EDA: Churn distribution
sns.countplot(x='churn', data=train_df)
plt.title("Churn Distribution")
plt.savefig("churn_distribution.png")
plt.clf()

# Correlation heatmap
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
corr = train_df[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.clf()

# Boxplot: Customer service calls
train_df['churn'] = train_df['churn'].astype(str)
sns.boxplot(x='churn', y='number_customer_service_calls', data=train_df)
plt.title("Customer Service Calls vs. Churn")
plt.savefig("customer_service_calls_boxplot.png")
plt.clf()

# Prepare data for model
df = train_df.copy()
df['churn'] = df['churn'].map({'yes': 1, 'no': 0})
X = df.drop(columns=['churn'])
y = df['churn']

categorical_features = ['state', 'area_code', 'international_plan', 'voice_mail_plan']
numerical_features = X.drop(columns=categorical_features).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)[:, 1]
print("Classification Report:\n", classification_report(y_val, y_pred))
print("ROC AUC Score:", roc_auc_score(y_val, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Predict test data and generate submission
X_test = test_df.drop(columns=['id'])
test_preds = pipeline.predict(X_test)
test_labels = ['yes' if p == 1 else 'no' for p in test_preds]
submission_df = pd.DataFrame({'id': test_df['id'], 'churn': test_labels})
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to 'submission.csv'")
