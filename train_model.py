import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SPACESHIP TITANIC - MACHINE LEARNING MODEL")
print("="*60)

# Load data
print("\n1. Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Store PassengerId for submission
test_ids = test_df['PassengerId']

# Separate target
y = train_df['Transported'].astype(int)
train_df = train_df.drop('Transported', axis=1)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Feature Engineering Function
def engineer_features(df):
    """Create new features from existing ones"""
    
    # Group features from PassengerId
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    
    # Cabin features
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['CabinNum'] = pd.to_numeric(cabin_split[1], errors='coerce')
    df['Side'] = cabin_split[2]
    
    # Spending features
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    
    # Luxury spending (RoomService + Spa)
    df['LuxurySpending'] = df['RoomService'] + df['Spa']
    
    # Basic spending (FoodCourt + ShoppingMall)
    df['BasicSpending'] = df['FoodCourt'] + df['ShoppingMall']
    
    # Age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 65, 100], 
                            labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior', 'Elder'],
                            include_lowest=True)
    df['AgeGroup'] = df['AgeGroup'].astype(str)  # Convert to string to avoid categorical issues
    
    # Is alone (group size = 1)
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    
    # Family size categories
    df['FamilySize'] = pd.cut(df['GroupSize'], bins=[0, 1, 3, 5, 20], 
                              labels=['Solo', 'Small', 'Medium', 'Large'],
                              include_lowest=True)
    df['FamilySize'] = df['FamilySize'].astype(str)  # Convert to string to avoid categorical issues
    
    return df

print("\n2. Engineering features...")
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# Preprocessing Function
def preprocess_data(train_df, test_df):
    """Handle missing values and encode categorical features"""
    
    # Combine for consistent preprocessing
    combined = pd.concat([train_df, test_df], axis=0, sort=False)
    
    # Handle boolean columns
    bool_cols = ['VIP', 'CryoSleep']
    for col in bool_cols:
        combined[col] = combined[col].map({True: 1, False: 0, 'True': 1, 'False': 0})
    
    # Categorical columns to encode
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'AgeGroup', 'FamilySize']
    
    # Label encode categorical features
    le_dict = {}
    for col in categorical_cols:
        if col in combined.columns:
            le = LabelEncoder()
            # Handle missing values before encoding
            combined[col] = combined[col].fillna('Unknown')
            combined[col] = le.fit_transform(combined[col].astype(str))
            le_dict[col] = le
    
    # Numerical columns
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 
                     'VRDeck', 'CabinNum', 'GroupSize', 'TotalSpending', 
                     'LuxurySpending', 'BasicSpending']
    
    # Impute numerical features
    for col in numerical_cols:
        if col in combined.columns:
            median_val = combined[col].median()
            combined[col] = combined[col].fillna(median_val)
    
    # Impute boolean features
    for col in bool_cols:
        if col in combined.columns:
            mode_val = combined[col].mode()[0] if len(combined[col].mode()) > 0 else 0
            combined[col] = combined[col].fillna(mode_val)
    
    # Drop unnecessary columns
    drop_cols = ['PassengerId', 'Name', 'Cabin', 'Group']
    combined = combined.drop(columns=[col for col in drop_cols if col in combined.columns])
    
    # Split back
    train_processed = combined.iloc[:len(train_df)]
    test_processed = combined.iloc[len(train_df):]
    
    return train_processed, test_processed

print("\n3. Preprocessing data...")
X_train_full, X_test = preprocess_data(train_df, test_df)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

print(f"Feature count: {X_train_scaled.shape[1]}")

# Train multiple models
print("\n5. Training models...")
print("-"*40)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                           min_samples_split=5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                                   max_depth=5, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
}

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train on training set
    model.fit(X_train_scaled, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results[name] = {
        'val_accuracy': val_accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'model': model
    }
    
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    if val_accuracy > best_score:
        best_score = val_accuracy
        best_model = model
        best_model_name = name

print("\n6. Model Comparison")
print("-"*40)
results_df = pd.DataFrame({
    'Model': results.keys(),
    'Val Accuracy': [r['val_accuracy'] for r in results.values()],
    'CV Mean': [r['cv_mean'] for r in results.values()],
    'CV Std': [r['cv_std'] for r in results.values()]
})
print(results_df.sort_values('Val Accuracy', ascending=False))

print(f"\nBest Model: {best_model_name} with {best_score:.4f} validation accuracy")

# Feature importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print("\n7. Top 10 Feature Importances")
    print("-"*40)
    feature_importance = pd.DataFrame({
        'feature': X_train_full.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))

# Train best model on full training data
print("\n8. Training best model on full dataset...")
best_model.fit(X_train_full_scaled, y)

# Make predictions on test set
print("\n9. Generating predictions for test set...")
test_predictions = best_model.predict(X_test_scaled)

# Create submission file
print("\n10. Creating submission file...")
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': test_predictions.astype(bool)
})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved to 'submission.csv'")
print(f"Predictions: {test_predictions.sum()} True, {len(test_predictions) - test_predictions.sum()} False")

# Final validation metrics
print("\n11. Final Validation Metrics")
print("-"*40)
y_val_pred = best_model.predict(X_val_scaled)
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Not Transported', 'Transported']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE")
print(f"Best Model: {best_model_name}")
print(f"Validation Accuracy: {best_score:.4f}")
print("Submission file: submission.csv")
print("="*60)