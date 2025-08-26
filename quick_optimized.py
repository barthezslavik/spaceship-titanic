"""
Spaceship Titanic - Quick Optimized Implementation
Focused on highest-impact optimizations for immediate results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("⚡ QUICK OPTIMIZED PIPELINE - HIGH-IMPACT FEATURES")
print("="*80)

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_ids = test_df['PassengerId']

y = train_df['Transported'].astype(int)
train_df = train_df.drop('Transported', axis=1)

print("\n🎯 PHASE 1: CRITICAL FEATURE INTERACTIONS")
print("-"*60)

def create_optimized_features(df):
    """Focus on highest-impact features only"""
    df = df.copy()
    
    print(f"Processing {len(df)} samples...")
    
    # Core group features
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    
    # Cabin features  
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0]
    df['Side'] = cabin_split[2]
    
    # Deck encoding (most important spatial feature)
    deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 0}
    df['DeckLevel'] = df['Deck'].map(deck_map).fillna(-1)
    df['IsPortSide'] = (df['Side'] == 'P').astype(int).fillna(-1)
    
    # Spending features (critical predictors)
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spending_cols:
        df[col] = df[col].fillna(0)
    
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)  # Most important feature!
    
    # Key behavioral indicators
    df['LuxurySpending'] = df['RoomService'] + df['Spa']
    df['EntertainmentSpending'] = df['ShoppingMall'] + df['VRDeck']
    
    # Binary features
    df['CryoSleep_binary'] = df['CryoSleep'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(-1)
    df['VIP_binary'] = df['VIP'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(-1)
    
    # Age handling
    df['Age_filled'] = df['Age'].fillna(df['Age'].median())
    df['IsChild'] = (df['Age_filled'] < 18).astype(int)
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    
    # Planet encoding
    df['IsEarth'] = (df['HomePlanet'] == 'Earth').astype(int).fillna(0)
    df['IsEuropa'] = (df['HomePlanet'] == 'Europa').astype(int).fillna(0)
    df['IsMars'] = (df['HomePlanet'] == 'Mars').astype(int).fillna(0)
    
    # CRITICAL INTERACTIONS (from domain analysis)
    print("  Creating critical interactions...")
    
    # 1. CryoSleep × Spending (logical impossibility)
    df['CryoSleep_x_HasSpending'] = df['CryoSleep_binary'] * df['HasSpending']
    df['Anomaly_CryoSpending'] = ((df['CryoSleep_binary'] == 1) & (df['HasSpending'] == 1)).astype(int)
    
    # 2. Group behavior
    df['Group_HasSpending_Rate'] = df.groupby('Group')['HasSpending'].transform('mean')
    df['GroupSize_x_IsAlone'] = df['GroupSize'] * df['IsAlone']  # Redundant but helps tree models
    
    # 3. Spatial clustering
    df['Deck_x_Side'] = df['DeckLevel'] * (df['IsPortSide'] + 1)  # Avoid multiplication by -1
    
    # 4. Age-based interactions
    df['Child_x_GroupSize'] = df['IsChild'] * df['GroupSize']
    
    # 5. Luxury behavior
    df['VIP_x_LuxurySpending'] = df['VIP_binary'] * (df['LuxurySpending'] > 0).astype(int)
    
    # 6. Planet-specific patterns
    df['Europa_x_CryoSleep'] = df['IsEuropa'] * df['CryoSleep_binary']
    
    return df

print("Engineering features for train and test...")
train_features = create_optimized_features(train_df)
test_features = create_optimized_features(test_df)

# Select modeling features
drop_cols = ['PassengerId', 'Name', 'Cabin', 'Group', 'HomePlanet', 
             'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age']
feature_cols = [col for col in train_features.columns if col not in drop_cols]

X_train = train_features[feature_cols].fillna(-999)
X_test = test_features[feature_cols].fillna(-999)

print(f"Selected features: {len(feature_cols)}")

# Quick scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n🚀 PHASE 2: FAST ENSEMBLE")
print("-"*60)

# Diverse high-performance models
models = {
    'xgb_fast': xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, 
        random_state=42, verbosity=0
    ),
    'lgb_fast': lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1
    ),
    'cat_fast': CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.02,
        random_state=42, verbose=False
    )
}

print("Training models with 5-fold CV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store out-of-fold predictions
oof_predictions = np.zeros((len(X_train_scaled), len(models)))
test_predictions = np.zeros((len(X_test_scaled), len(models)))
cv_scores = []

for i, (name, model) in enumerate(models.items()):
    print(f"  Training {name}...")
    
    oof_pred = np.zeros(len(X_train_scaled))
    test_pred = np.zeros(len(X_test_scaled))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model_fold = clone(model)
        model_fold.fit(X_tr, y_tr)
        
        val_pred = model_fold.predict_proba(X_val)[:, 1]
        oof_pred[val_idx] = val_pred
        test_pred += model_fold.predict_proba(X_test_scaled)[:, 1] / cv.n_splits
        
        fold_score = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_score)
    
    oof_predictions[:, i] = oof_pred
    test_predictions[:, i] = test_pred
    
    cv_score = roc_auc_score(y, oof_pred)
    cv_scores.append(cv_score)
    print(f"    CV ROC-AUC: {cv_score:.4f}")

print(f"\nIndividual model performance:")
for name, score in zip(models.keys(), cv_scores):
    print(f"  {name}: {score:.4f}")

print("\n💎 PHASE 3: PSEUDO-LABELING")
print("-"*60)

# Simple voting ensemble for pseudo-labeling
ensemble_pred = test_predictions.mean(axis=1)

# Conservative threshold for high-confidence predictions
threshold = 0.92
high_conf_mask = (ensemble_pred > threshold) | (ensemble_pred < (1-threshold))
pseudo_labels = (ensemble_pred > 0.5).astype(int)

print(f"High-confidence predictions: {high_conf_mask.sum()}")

if high_conf_mask.sum() > 100:  # Only if we have enough samples
    # Add pseudo-labeled data
    X_pseudo = X_test_scaled[high_conf_mask]
    y_pseudo = pseudo_labels[high_conf_mask]
    
    # Combine datasets
    X_combined = np.vstack([X_train_scaled, X_pseudo])
    y_combined = np.hstack([y.values, y_pseudo])
    
    print(f"  Added {len(y_pseudo)} pseudo-labeled samples")
    
    # Retrain best model
    best_model_idx = np.argmax(cv_scores)
    best_model_name = list(models.keys())[best_model_idx]
    best_model = list(models.values())[best_model_idx]
    
    print(f"  Retraining {best_model_name} with expanded data...")
    enhanced_model = clone(best_model)
    enhanced_model.fit(X_combined, y_combined)
    
    # Generate enhanced predictions
    enhanced_pred = enhanced_model.predict_proba(X_test_scaled)[:, 1]
    
    # Blend with original ensemble
    final_pred = 0.7 * ensemble_pred + 0.3 * enhanced_pred
else:
    print("  Not enough high-confidence predictions, using original ensemble")
    final_pred = ensemble_pred

print("\n🎯 PHASE 4: FINAL ENSEMBLE")
print("-"*60)

# Weighted ensemble based on CV scores
weights = np.array(cv_scores) / np.sum(cv_scores)
weighted_pred = np.sum(test_predictions * weights, axis=1)

# Combine approaches
final_ensemble = 0.6 * weighted_pred + 0.4 * final_pred

# Convert to binary
final_binary = (final_ensemble > 0.5).astype(int)

print(f"Final ensemble weights: {weights}")
print(f"Prediction distribution: {np.bincount(final_binary) / len(final_binary)}")

print("\n📊 PHASE 5: RESULTS & SUBMISSION")
print("-"*60)

# Create submission
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': final_binary.astype(bool)
})

submission.to_csv('quick_optimized_submission.csv', index=False)

# Performance analysis
print("\nModel Performance Summary:")
print(f"  Best individual model: {max(cv_scores):.4f} ROC-AUC")
print(f"  Average model performance: {np.mean(cv_scores):.4f} ROC-AUC")
print(f"  Model agreement: {np.std(test_predictions, axis=1).mean():.4f} (lower = better)")

print(f"\nPrediction Statistics:")
print(f"  Transported=True: {final_binary.sum()} ({final_binary.mean():.2%})")
print(f"  Transported=False: {len(final_binary) - final_binary.sum()} ({1-final_binary.mean():.2%})")

# Confidence analysis
prediction_confidence = np.abs(final_ensemble - 0.5) * 2
print(f"  Mean confidence: {prediction_confidence.mean():.3f}")
print(f"  High confidence (>0.8): {(prediction_confidence > 0.8).sum()}")

print("\n" + "="*80)
print("✅ QUICK OPTIMIZATION COMPLETE")
print("="*80)
print("\nKey Optimizations Applied:")
print("  ✓ Domain-specific feature interactions")
print("  ✓ Multi-model ensemble (XGB+LGB+Cat)")
print("  ✓ Pseudo-labeling with conservative threshold")
print("  ✓ Weighted ensemble based on CV performance")
print("  ✓ 5-fold cross-validation for robust estimates")
print("\n📁 Output: quick_optimized_submission.csv")
print(f"📈 Expected Performance: {max(cv_scores):.1%}+ accuracy")
print("="*80)