"""
Spaceship Titanic - Best Models Optimized Implementation
Key techniques from comprehensive data science framework
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, tree, linear_model, naive_bayes, neighbors, svm
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, metrics
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🏆 BEST MODELS OPTIMIZED FRAMEWORK")
print("="*60)

# Load data
data_raw = pd.read_csv('train.csv')
data_val = pd.read_csv('test.csv')

print(f"Train: {data_raw.shape}, Test: {data_val.shape}")

# Create processing copies
data1 = data_raw.copy(deep=True)
data_cleaner = [data1, data_val]

print("\n🔧 KEY FEATURE ENGINEERING")
print("-"*40)

# Core feature groups
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

print("1. Age Groups & Expenditure...")
for dataset in data_cleaner:
    # Age groups (best-models approach)
    dataset['Age_group'] = 'Age_31-50'  # default
    dataset.loc[dataset['Age'] <= 12, 'Age_group'] = 'Age_0-12'
    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] < 18), 'Age_group'] = 'Age_13-17'
    dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] <= 25), 'Age_group'] = 'Age_18-25'
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age_group'] = 'Age_26-30'
    dataset.loc[dataset['Age'] > 50, 'Age_group'] = 'Age_51+'
    
    # Expenditure features
    dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
    dataset['No_spending'] = (dataset['Expenditure'] == 0).astype(int)

print("2. Group Dynamics...")
for dataset in data_cleaner:
    # Group features from PassengerId
    dataset['Group'] = dataset['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
    dataset['Group_size'] = dataset['Group'].map(lambda x: dataset['Group'].value_counts()[x])
    dataset['Solo'] = (dataset['Group_size'] == 1).astype(int)

print("3. Advanced Cabin Engineering...")
for dataset in data_cleaner:
    # Cabin parsing with temporary handling
    dataset['Cabin'].fillna('Z/9999/Z', inplace=True)
    dataset['Cabin_deck'] = dataset['Cabin'].apply(lambda x: x.split('/')[0])
    dataset['Cabin_number'] = dataset['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
    dataset['Cabin_side'] = dataset['Cabin'].apply(lambda x: x.split('/')[2])
    
    # Clean up temp values
    dataset.loc[dataset['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
    dataset.loc[dataset['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
    dataset.loc[dataset['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
    
    # Key innovation: Cabin regions (7 distinct regions)
    dataset['Cabin_region1'] = (dataset['Cabin_number'] < 300).astype(int)
    dataset['Cabin_region2'] = ((dataset['Cabin_number'] >= 300) & (dataset['Cabin_number'] < 600)).astype(int)
    dataset['Cabin_region3'] = ((dataset['Cabin_number'] >= 600) & (dataset['Cabin_number'] < 900)).astype(int)
    dataset['Cabin_region4'] = ((dataset['Cabin_number'] >= 900) & (dataset['Cabin_number'] < 1200)).astype(int)
    dataset['Cabin_region5'] = ((dataset['Cabin_number'] >= 1200) & (dataset['Cabin_number'] < 1500)).astype(int)
    dataset['Cabin_region6'] = ((dataset['Cabin_number'] >= 1500) & (dataset['Cabin_number'] < 1800)).astype(int)
    dataset['Cabin_region7'] = (dataset['Cabin_number'] >= 1800).astype(int)

print("4. Family Features...")
for dataset in data_cleaner:
    if 'Name' in dataset.columns:
        dataset['Name'].fillna('Unknown Unknown', inplace=True)
        dataset['Surname'] = dataset['Name'].str.split().str[-1]
        dataset['Family_size'] = dataset['Surname'].map(lambda x: dataset['Surname'].value_counts()[x])
        dataset.loc[dataset['Surname'] == 'Unknown', 'Surname'] = np.nan
        dataset.loc[dataset['Family_size'] > 100, 'Family_size'] = 0  # Handle outliers

print("\n🧹 SMART MISSING VALUE HANDLING")
print("-"*40)

print("Missing value imputation...")
for dataset in data_cleaner:
    # HomePlanet rules (key insight from best-models)
    if 'HomePlanet' in dataset.columns:
        dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
        dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'] == 'G'), 'HomePlanet'] = 'Earth'
        dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'] == 'D'), 'HomePlanet'] = 'Mars'
        dataset['HomePlanet'].fillna('Earth', inplace=True)
    
    # Other categorical features
    if 'Destination' in dataset.columns:
        dataset['Destination'].fillna('TRAPPIST-1e', inplace=True)
    if 'VIP' in dataset.columns:
        dataset['VIP'].fillna(False, inplace=True)
    if 'CryoSleep' in dataset.columns:
        dataset['CryoSleep'].fillna(dataset['No_spending'] == 1, inplace=True)
    if 'Cabin_side' in dataset.columns:
        dataset['Cabin_side'].fillna('S', inplace=True)  # Most common
    if 'Cabin_deck' in dataset.columns:
        dataset['Cabin_deck'].fillna('F', inplace=True)  # Most common
    
    # Numerical features
    if 'Age' in dataset.columns:
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    if 'Cabin_number' in dataset.columns:
        dataset['Cabin_number'].fillna(dataset['Cabin_number'].median(), inplace=True)
        # Recalculate regions
        dataset['Cabin_region1'] = (dataset['Cabin_number'] < 300).astype(int)
        dataset['Cabin_region2'] = ((dataset['Cabin_number'] >= 300) & (dataset['Cabin_number'] < 600)).astype(int)
        dataset['Cabin_region3'] = ((dataset['Cabin_number'] >= 600) & (dataset['Cabin_number'] < 900)).astype(int)
        dataset['Cabin_region4'] = ((dataset['Cabin_number'] >= 900) & (dataset['Cabin_number'] < 1200)).astype(int)
        dataset['Cabin_region5'] = ((dataset['Cabin_number'] >= 1200) & (dataset['Cabin_number'] < 1500)).astype(int)
        dataset['Cabin_region6'] = ((dataset['Cabin_number'] >= 1500) & (dataset['Cabin_number'] < 1800)).astype(int)
        dataset['Cabin_region7'] = (dataset['Cabin_number'] >= 1800).astype(int)
    
    # Expenditure features
    for col in exp_feats:
        if col in dataset.columns:
            dataset.loc[(dataset[col].isna()) & (dataset['CryoSleep'] == True), col] = 0
            dataset[col].fillna(0, inplace=True)
    
    # Recalculate expenditure
    dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
    dataset['No_spending'] = (dataset['Expenditure'] == 0).astype(int)
    
    # Log transform (reduces skewness)
    for col in exp_feats + ['Expenditure']:
        if col in dataset.columns:
            dataset[col] = np.log1p(dataset[col])

print("Label encoding...")
# Convert categorical to numeric
label = LabelEncoder()
for dataset in data_cleaner:
    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Age_group', 'Cabin_deck', 'Cabin_side']:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype(str)
            dataset[f'{col}_Code'] = label.fit_transform(dataset[col])

print("\n🎯 MODEL SELECTION")
print("-"*40)

# Feature set (optimized)
feature_cols = [
    'Age', 'Expenditure', 'No_spending', 'Group_size', 'Solo',
    'Cabin_region1', 'Cabin_region2', 'Cabin_region3', 'Cabin_region4',
    'Cabin_region5', 'Cabin_region6', 'Cabin_region7',
    'HomePlanet_Code', 'CryoSleep_Code', 'Destination_Code',
    'VIP_Code', 'Age_group_Code', 'Cabin_deck_Code', 'Cabin_side_Code'
]

# Add Family_size if available
if 'Family_size' in data1.columns:
    feature_cols.append('Family_size')

# Clean features
available_features = [col for col in feature_cols if col in data1.columns]
print(f"Features: {len(available_features)}")

# Prepare target
if 'Transported' in data1.columns:
    data1['Transported'] = data1['Transported'].astype(int)

# Final cleanup
for dataset in data_cleaner:
    for col in available_features:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(0)

print("Evaluating key models...")
# Best performing models from comprehensive analysis
models = {
    'RandomForest': ensemble.RandomForestClassifier(n_estimators=100, random_state=0),
    'ExtraTrees': ensemble.ExtraTreesClassifier(n_estimators=100, random_state=0),
    'GradientBoosting': ensemble.GradientBoostingClassifier(random_state=0),
    'XGBoost': XGBClassifier(random_state=0, verbosity=0),
    'AdaBoost': ensemble.AdaBoostClassifier(random_state=0),
}

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
results = {}

for name, model in models.items():
    try:
        scores = model_selection.cross_val_score(model, data1[available_features], data1['Transported'], 
                                               cv=cv, scoring='accuracy')
        results[name] = scores.mean()
        print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    except Exception as e:
        print(f"  {name}: FAILED")

# Create ensemble of top performers
top_models = sorted(results.items(), key=lambda x: x[1], reverse=True)[:4]
print(f"\nTop 4 models for ensemble:")
for name, score in top_models:
    print(f"  {name}: {score:.4f}")

# Voting ensemble
estimators = [(name.lower(), models[name]) for name, score in top_models]
voting_clf = ensemble.VotingClassifier(estimators=estimators, voting='soft')

# Evaluate ensemble
ensemble_scores = model_selection.cross_val_score(voting_clf, data1[available_features], data1['Transported'],
                                                cv=cv, scoring='accuracy')

print(f"\nEnsemble Performance:")
print(f"  Accuracy: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")

# Train and predict
print("\n🚀 FINAL PREDICTIONS")
print("-"*40)

voting_clf.fit(data1[available_features], data1['Transported'])
predictions = voting_clf.predict(data_val[available_features])

# Create submission
submission = pd.DataFrame({
    'PassengerId': data_val['PassengerId'],
    'Transported': predictions.astype(bool)
})
submission.to_csv('best_models_optimized_submission.csv', index=False)

accuracy_estimate = ensemble_scores.mean() * 100

print("📊 RESULTS")
print("-"*40)
print(f"Best Models Optimized Results:")
print(f"  Accuracy: {accuracy_estimate:.1f}%")
print(f"  Ensemble: {len(estimators)} top algorithms")
print(f"  Features: {len(available_features)}")

if accuracy_estimate >= 95:
    print("  🏆 GOLD TIER!")
elif accuracy_estimate >= 90:
    print("  🥈 SILVER TIER!")
else:
    print("  📈 SOLID PERFORMANCE!")

print(f"\nPredictions: True={predictions.sum()} ({predictions.mean():.1%})")

print(f"\n💡 BEST-MODELS KEY TECHNIQUES:")
print("-"*40)
print("✅ 7-region cabin classification system")
print("✅ Smart age group binning")
print("✅ Group dynamics from PassengerId")
print("✅ Rule-based missing value imputation")
print("✅ Log-transformed expenditure features")
print("✅ Ensemble of top-performing models")

print(f"\n📁 Output: best_models_optimized_submission.csv")
print(f"🎯 Performance: {accuracy_estimate:.1f}%")
print("="*60)