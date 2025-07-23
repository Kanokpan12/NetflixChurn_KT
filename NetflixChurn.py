import pandas as pd
import numpy as np
from scipy.stats import pearsonr, chi2_contingency, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Load dataset
df = pd.read_csv('C:/Users/ksriwong/Desktop/NetflixChurn.csv')

#Handling missing value and cleaning data
#impute missing value in watch_hours with median
df['watch_hours'].fillna(df['watch_hours'].median(), inplace=True)

#replace missing genre in 'favorite_genre' with "unknown"
df['favorite_genre'].fillna('Unknown', inplace=True)

# Using median to fill 'age' column
df['age'].fillna(df['age'].median(), inplace=True)

#standardized region
df['region'] = df['region'].replace(['EUROPE', 'europa'], 'Europe')

df = df.dropna(subset=['churned'])

#EDA process and feature engineering
#churn rate overall
churn_rate = df['churned'].value_counts(normalize=True)[1]
print(f"Overall churn rate: {churn_rate:.2%}")

# Bar plot: Churn by subscription type
ax = sns.countplot(x='subscription_type', hue='churned', data=df)
plt.title("Churn Rate by Subscription Type")
for container in ax.containers:
    ax.bar_label(container)
plt.show()

#Churn Percentage by Region
# Step 1: Filter churned users
churned_df = df[df['churned'] == 1]
# Step 2: Calculate churn percentage by region
churn_percent = (
    churned_df['region'].value_counts(normalize=True) * 100
).round(2).reset_index()
churn_percent.columns = ['region', 'churn_percent']
# Step 3: Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='region', y='churn_percent', data=churn_percent, palette='Reds')
# Add percentages on top of bars
for index, row in churn_percent.iterrows():
    plt.text(index, row['churn_percent'] + 1, f"{row['churn_percent']}%", ha='center')
plt.title("Churn Percentage by Region")
plt.ylabel("Churn Percentage (%)")
plt.xlabel("Region")
plt.ylim(0, churn_percent['churn_percent'].max() + 10)
plt.tight_layout()
plt.show()

# Feature Engineering
# Create new features
df['estimated_tenure'] = 365 - df['last_login_days']
df['engagement_ratio'] = df['watch_hours'] / df['estimated_tenure']
df['payment_frequency'] = df['estimated_tenure'] / df['number_of_profiles']
df['activity_score'] = df['watch_hours'] * df['avg_watch_time_per_day']
# Apply log transform
df['log_activity_score'] = np.log(df['activity_score'] + 1)

# Engagement Ratio by churn
plt.figure(figsize=(8, 6))
sns.boxplot(x='churned', y='engagement_ratio', data=df)
plt.title('Engagement Ratio by Churn Status')
plt.xlabel('Churned')
plt.ylabel('Engagement Ratio')
plt.show()

# Activity Score by churn
plt.figure(figsize=(8, 6))
sns.boxplot(x='churned', y='log_activity_score', data=df)
plt.title('Activity Score by Churn Status')
plt.xlabel('Churned')
plt.ylabel('Activity Score')
plt.show()

#2. Preprocessing data
# Drop the target variable and non-numeric columns
numeric_df = df.drop(columns=['churned'])

# Define features set to numerical and categorical features
numeric_features = ['age', 'monthly_fee', 'estimated_tenure', 'engagement_ratio', 'payment_frequency', 'activity_score']

categorical_features = ['gender', 'subscription_type', 'region', 'device', 'payment_method', 'favorite_genre']

# Numerical correlation with spearman heatmap
corr_matrix = df[numeric_features].corr(method='spearman')
# Plot the heatmap with annotations
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", square=True)
plt.title("Numerical Feature Correlation")
plt.tight_layout()
plt.show()

#categorical features correlation with Cramer V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

#categorical features
categorical_features = ['gender', 'subscription_type', 'region', 'device', 'payment_method', 'favorite_genre']

# Compute Cramér’s V for each pair
cramer_matrix = pd.DataFrame(np.zeros((len(categorical_features), len(categorical_features))),
                             index=categorical_features, columns=categorical_features)

for col1, col2 in itertools.combinations(categorical_features, 2):
    val = cramers_v(df[col1], df[col2])
    cramer_matrix.loc[col1, col2] = val
    cramer_matrix.loc[col2, col1] = val

np.fill_diagonal(cramer_matrix.values, 1.0)  # set diagonal to 1
plt.figure(figsize=(8, 6))
sns.heatmap(cramer_matrix, annot=True, cmap="Reds")
plt.title("Cramér's V Correlation")
plt.show()

#ML process
# Features and target
X = df[numeric_features + categorical_features]
y = df['churned']

# Automatically detect numeric and categorical features from X
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Define transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine into preprocessor with updated features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Evaluation function
def evaluate_model(model, name, param_grid=None):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    if param_grid:
        print(f"Tuning hyperparameters for {name}...")
        search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")
    else:
        best_pipeline = pipeline
        best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    print(f"\n{name} Results:")
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return {
        'Model': name,
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'Pipeline': best_pipeline
    }

# Initialize results list
results = []

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

lr_params = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['liblinear', 'lbfgs']
}
lr_result = evaluate_model(lr_model, "Logistic Regression", lr_params)
results.append({k: v for k, v in lr_result.items() if k != 'Pipeline'})
lr_pipeline = lr_result['Pipeline']

# Get feature names from the pipeline
encoded_cat_features = lr_pipeline.named_steps['preprocessor'].transformers_[1][1] \
    .get_feature_names_out(categorical_features)
feature_names = numeric_features + list(encoded_cat_features)

# Get logistic regression coefficients
coefficients = lr_pipeline.named_steps['classifier'].coef_[0]

# Create a DataFrame of feature names and coefficients
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()

# Show top 5 features by absolute effect on churn
top5 = coef_df.sort_values(by='Abs_Coefficient', ascending=False).head(5)
print("\nTop 5 Features Affecting Churn (Logistic Regression):")
print(top5[['Feature', 'Coefficient']])

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_params = {
    'classifier__n_neighbors': [3, 5, 7],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}
results.append(evaluate_model(knn_model, "K-Nearest Neighbors", knn_params))

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__max_features': ['sqrt', 'log2']
}
rf_result = evaluate_model(rf_model, "Random Forest", rf_params)
results.append({k: v for k, v in rf_result.items() if k != 'Pipeline'})
rf_pipeline = rf_result['Pipeline']

# Get full feature names from pipeline
encoded_cat_features = rf_pipeline.named_steps['preprocessor'].transformers_[1][1] \
    .get_feature_names_out(categorical_features)
feature_names = numeric_features + list(encoded_cat_features)

# Feature importances from Random Forest
rf_importances = rf_pipeline.named_steps['classifier'].feature_importances_
rf_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_importances})
top5_rf = rf_df.sort_values(by='Importance', ascending=False).head(5)

print("\nTop 5 Features Affecting Churn (Random Forest):")
print(top5_rf)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_params = {
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.3],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__n_estimators': [100, 200]
}
xgb_result = evaluate_model(xgb_model, "XGBoost", xgb_params)
results.append({k: v for k, v in xgb_result.items() if k != 'Pipeline'})
xgb_pipeline = xgb_result['Pipeline']

# Get full feature names from pipeline
encoded_cat_features = xgb_pipeline.named_steps['preprocessor'].transformers_[1][1] \
    .get_feature_names_out(categorical_features)
feature_names = numeric_features + list(encoded_cat_features)

# Feature importances from XGBoost
xgb_importances = xgb_pipeline.named_steps['classifier'].feature_importances_
xgb_df = pd.DataFrame({'Feature': feature_names, 'Importance': xgb_importances})
top5_xgb = xgb_df.sort_values(by='Importance', ascending=False).head(5)

print("\nTop 5 Features Affecting Churn (XGBoost):")
print(top5_xgb)