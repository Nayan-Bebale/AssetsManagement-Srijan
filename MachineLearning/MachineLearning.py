import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('assets_data.csv')

# Features and target for regression
features = ['condition_score', 'criticality_score', 'usage', 'failure_history', 'age', 'environment_factor']
target_cost = 'maintenance_cost'

# Train-test split for regression
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target_cost], test_size=0.2, random_state=42)

# Train regression model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Save the model
joblib.dump(regressor, 'regressor_model.pkl')

# Create priority labels (1 for high priority, 0 for low priority)
df['priority'] = (df['criticality_score'] * 2 + df['condition_score']) > 12

# Train-test split for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(df[features], df['priority'], test_size=0.2, random_state=42)

# Train classification model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_cls, y_train_cls)

# Save the model
joblib.dump(classifier, 'classifier_model.pkl')