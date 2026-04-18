import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
print("Loading dataset...")
data = pd.read_csv("heart.csv")
# Note: The original dataset uses 'HeartDisease' as the target column
target_col = 'HeartDisease'

print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")

# Check for missing values
if data.isnull().sum().sum() > 0:
    print("Found missing values! Dropping rows with NaN...")
    data = data.dropna()

# Categorical columns that need encoding:
# Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
print(f"Encoding categorical features: {categorical_cols}")

# One-hot encoding for categorical variables
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Split features (X) and target (y)
if target_col in data_encoded.columns:
    X = data_encoded.drop(target_col, axis=1)
    y = data_encoded[target_col]
else:
    # Fallback if column name is different
    print(f"Error: Target column '{target_col}' not found!")
    X = data_encoded.iloc[:, :-1]
    y = data_encoded.iloc[:, -1]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42      # For reproducible results
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Train the model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate on test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "model.pkl")
# Save the feature names to ensure prediction input matches training input
joblib.dump(list(X.columns), "features.pkl")
print("\nModel and features saved successfully.")

# Show feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head(5))
