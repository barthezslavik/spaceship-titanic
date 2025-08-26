"""
Spaceship Titanic - Best Models Framework Implementation
Based on comprehensive data science framework with advanced feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.preprocessing import LabelEncoder
from sklearn import feature_selection, model_selection, metrics
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🏆 BEST MODELS COMPREHENSIVE FRAMEWORK")
print("="*70)

# Load data
data_raw = pd.read_csv('train.csv')
data_val = pd.read_csv('test.csv')

print(f"Train shape: {data_raw.shape}, Test shape: {data_val.shape}")

# Create copies for processing
data1 = data_raw.copy(deep=True)
data_cleaner = [data1, data_val]

print("\n🔧 ADVANCED FEATURE ENGINEERING")
print("-"*50)

# Define feature groups (as per best-models framework)
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
qual_feats = ['PassengerId', 'Cabin', 'Name']

print("1. Creating Age Groups...")
# Age groups with specific bins
for dataset in data_cleaner:
    dataset['Age_group'] = np.nan
    dataset.loc[dataset['Age'] <= 12, 'Age_group'] = 'Age_0-12'
    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] < 18), 'Age_group'] = 'Age_13-17'
    dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] <= 25), 'Age_group'] = 'Age_18-25'
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age_group'] = 'Age_26-30'
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 50), 'Age_group'] = 'Age_31-50'
    dataset.loc[dataset['Age'] > 50, 'Age_group'] = 'Age_51+'

print("2. Creating Expenditure Features...")
# Total expenditure and spending indicator
for dataset in data_cleaner:
    dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
    dataset['No_spending'] = (dataset['Expenditure'] == 0).astype(int)

print("3. Group Features from PassengerId...")
# Group dynamics
for dataset in data_cleaner:
    dataset['Group'] = dataset['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
    dataset['Group_size'] = dataset['Group'].map(lambda x: dataset['Group'].value_counts()[x])
    dataset['Solo'] = (dataset['Group_size'] == 1).astype(int)

print("4. Advanced Cabin Engineering...")
# Cabin feature extraction and region binning
for dataset in data_cleaner:
    # Handle missing cabins temporarily
    dataset['Cabin'].fillna('Z/9999/Z', inplace=True)
    
    # Extract cabin components
    dataset['Cabin_deck'] = dataset['Cabin'].apply(lambda x: x.split('/')[0])
    dataset['Cabin_number'] = dataset['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
    dataset['Cabin_side'] = dataset['Cabin'].apply(lambda x: x.split('/')[2])
    
    # Put NaNs back for proper handling
    dataset.loc[dataset['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
    dataset.loc[dataset['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
    dataset.loc[dataset['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
    
    # Drop original cabin
    dataset.drop('Cabin', axis=1, inplace=True)
    
    # Cabin regions (key innovation from best-models)
    dataset['Cabin_region1'] = (dataset['Cabin_number'] < 300).astype(int)
    dataset['Cabin_region2'] = ((dataset['Cabin_number'] >= 300) & (dataset['Cabin_number'] < 600)).astype(int)
    dataset['Cabin_region3'] = ((dataset['Cabin_number'] >= 600) & (dataset['Cabin_number'] < 900)).astype(int)
    dataset['Cabin_region4'] = ((dataset['Cabin_number'] >= 900) & (dataset['Cabin_number'] < 1200)).astype(int)
    dataset['Cabin_region5'] = ((dataset['Cabin_number'] >= 1200) & (dataset['Cabin_number'] < 1500)).astype(int)
    dataset['Cabin_region6'] = ((dataset['Cabin_number'] >= 1500) & (dataset['Cabin_number'] < 1800)).astype(int)
    dataset['Cabin_region7'] = (dataset['Cabin_number'] >= 1800).astype(int)

print("5. Family Features from Names...")
# Family size from surname
for dataset in data_cleaner:
    if 'Name' in dataset.columns:
        dataset['Name'].fillna('Unknown Unknown', inplace=True)
        dataset['Surname'] = dataset['Name'].str.split().str[-1]
        dataset['Family_size'] = dataset['Surname'].map(lambda x: dataset['Surname'].value_counts()[x])
        # Clean up
        dataset.loc[dataset['Surname'] == 'Unknown', 'Surname'] = np.nan
        dataset.loc[dataset['Family_size'] > 100, 'Family_size'] = np.nan
        dataset.drop('Name', axis=1, inplace=True)

print("\n🧹 ADVANCED MISSING VALUE IMPUTATION")
print("-"*50)

# Advanced imputation strategies from best-models framework

print("Imputing HomePlanet using group information...")
for dataset in data_cleaner:
    if 'HomePlanet' in dataset.columns and dataset['HomePlanet'].isna().any():
        # Group-based imputation
        try:
            GHP_gb = dataset.groupby(['Group', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
            GHP_index = dataset[dataset['HomePlanet'].isna()][(dataset[dataset['HomePlanet'].isna()]['Group']).isin(GHP_gb.index)].index
            dataset.loc[GHP_index, 'HomePlanet'] = dataset.iloc[GHP_index]['Group'].map(lambda x: GHP_gb.idxmax(axis=1)[x])
        except:
            pass
        
        # Cabin-based rules
        dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
        dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'] == 'G'), 'HomePlanet'] = 'Earth'
        
        # Final fallback
        dataset.loc[(dataset['HomePlanet'].isna()) & ~(dataset['Cabin_deck'] == 'D'), 'HomePlanet'] = 'Earth'
        dataset.loc[(dataset['HomePlanet'].isna()) & (dataset['Cabin_deck'] == 'D'), 'HomePlanet'] = 'Mars'

print("Imputing other categorical features...")
# Destination imputation
for dataset in data_cleaner:
    if 'Destination' in dataset.columns:
        dataset['Destination'].fillna('TRAPPIST-1e', inplace=True)

# VIP imputation
for dataset in data_cleaner:
    if 'VIP' in dataset.columns:
        dataset['VIP'].fillna(False, inplace=True)

# CryoSleep imputation based on spending
for dataset in data_cleaner:
    if 'CryoSleep' in dataset.columns and dataset['CryoSleep'].isna().any():
        # Use No_spending to infer CryoSleep
        dataset.loc[dataset['CryoSleep'].isna(), 'CryoSleep'] = dataset.loc[dataset['CryoSleep'].isna(), 'No_spending'] == 1

print("Advanced cabin imputation...")
# Cabin side imputation
for dataset in data_cleaner:
    if 'Cabin_side' in dataset.columns:
        dataset['Cabin_side'].fillna('Z', inplace=True)

# Age imputation using group statistics
for dataset in data_cleaner:
    if 'Age' in dataset.columns and dataset['Age'].isna().any():
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
        # Recalculate age groups
        dataset.loc[dataset['Age'] <= 12, 'Age_group'] = 'Age_0-12'
        dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] < 18), 'Age_group'] = 'Age_13-17'
        dataset.loc[(dataset['Age'] >= 18) & (dataset['Age'] <= 25), 'Age_group'] = 'Age_18-25'
        dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age_group'] = 'Age_26-30'
        dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 50), 'Age_group'] = 'Age_31-50'
        dataset.loc[dataset['Age'] > 50, 'Age_group'] = 'Age_51+'

# Expenditure features imputation
for dataset in data_cleaner:
    for col in exp_feats:
        if col in dataset.columns:
            dataset.loc[(dataset[col].isna()) & (dataset['CryoSleep'] == True), col] = 0
            dataset[col].fillna(0, inplace=True)
    
    # Recalculate expenditure features
    dataset['Expenditure'] = dataset[exp_feats].sum(axis=1)
    dataset['No_spending'] = (dataset['Expenditure'] == 0).astype(int)
    
    # Log transform expenditure (reduces skewness)
    for col in exp_feats + ['Expenditure']:
        if col in dataset.columns:
            dataset[col] = np.log1p(dataset[col])

print("\n🏷️ LABEL ENCODING")
print("-"*50)

# Convert categorical features to numeric
label = LabelEncoder()
for dataset in data_cleaner:
    for col in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Age_group', 'Cabin_deck', 'Cabin_side']:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype(str)
            dataset[f'{col}_Code'] = label.fit_transform(dataset[col])

print("\n📊 FEATURE SELECTION AND PREPARATION")
print("-"*50)

# Define feature sets as per best-models framework
target = ['Transported']

# Comprehensive feature set with all engineered features
data1_x_calc = [
    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 
    'Expenditure', 'No_spending', 'Group', 'Group_size', 'Solo',
    'Cabin_number', 'Cabin_region1', 'Cabin_region2', 'Cabin_region3',
    'Cabin_region4', 'Cabin_region5', 'Cabin_region6', 'Cabin_region7',
    'Family_size', 'HomePlanet_Code', 'CryoSleep_Code', 'Destination_Code',
    'VIP_Code', 'Age_group_Code', 'Cabin_deck_Code', 'Cabin_side_Code'
]

# Clean feature set (remove features with high missing rates)
available_features = [col for col in data1_x_calc if col in data1.columns and data1[col].isna().sum() < len(data1) * 0.5]
print(f"Available features: {len(available_features)}")

if 'Transported' in data1.columns:
    data1['Transported'] = data1['Transported'].astype(int)

# Fill any remaining missing values
for dataset in data_cleaner:
    for col in available_features:
        if col in dataset.columns:
            if dataset[col].dtype in ['object', 'bool']:
                dataset[col] = dataset[col].fillna('Unknown')
            else:
                dataset[col] = dataset[col].fillna(dataset[col].median())

print(f"Final feature set: {available_features}")

print("\n🤖 COMPREHENSIVE MODEL EVALUATION")
print("-"*50)

# Machine Learning Algorithm Selection (from best-models framework)
MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(random_state=0),
    ensemble.BaggingClassifier(random_state=0),
    ensemble.ExtraTreesClassifier(random_state=0),
    ensemble.GradientBoostingClassifier(random_state=0),
    ensemble.RandomForestClassifier(random_state=0),
    
    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(random_state=0),
    
    # GLM
    linear_model.LogisticRegressionCV(random_state=0, max_iter=1000),
    linear_model.RidgeClassifierCV(),
    
    # Naive Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    # SVM
    svm.SVC(probability=True, random_state=0),
    
    # Trees
    tree.DecisionTreeClassifier(random_state=0),
    
    # XGBoost
    XGBClassifier(random_state=0, verbosity=0)
]

# Cross-validation setup
cv_split = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Model comparison
MLA_columns = ['MLA Name', 'MLA Train Accuracy', 'MLA Test Accuracy', 'MLA Test Std']
MLA_compare = pd.DataFrame(columns=MLA_columns)

print("Evaluating models...")
row_index = 0
for alg in MLA:
    MLA_name = alg.__class__.__name__
    
    try:
        cv_results = model_selection.cross_validate(
            alg, data1[available_features], data1[target], 
            cv=cv_split, return_train_score=True, scoring='accuracy'
        )
        
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Train Accuracy'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy'] = cv_results['test_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Std'] = cv_results['test_score'].std()
        
        row_index += 1
        print(f"  {MLA_name}: {cv_results['test_score'].mean():.4f}")
        
    except Exception as e:
        print(f"  {MLA_name}: FAILED ({str(e)[:50]})")

# Sort by performance
MLA_compare = MLA_compare.sort_values(by=['MLA Test Accuracy'], ascending=False)
print(f"\nTop 5 Models:")
print(MLA_compare.head())

print("\n🗳️ ENSEMBLE VOTING CLASSIFIER")
print("-"*50)

# Select best performing models for ensemble
top_models = MLA_compare.head(8)  # Top 8 models
vote_est = []

model_mapping = {
    'AdaBoostClassifier': ensemble.AdaBoostClassifier(random_state=0),
    'BaggingClassifier': ensemble.BaggingClassifier(random_state=0),
    'ExtraTreesClassifier': ensemble.ExtraTreesClassifier(random_state=0),
    'GradientBoostingClassifier': ensemble.GradientBoostingClassifier(random_state=0),
    'RandomForestClassifier': ensemble.RandomForestClassifier(random_state=0),
    'LogisticRegressionCV': linear_model.LogisticRegressionCV(random_state=0, max_iter=1000),
    'BernoulliNB': naive_bayes.BernoulliNB(),
    'GaussianNB': naive_bayes.GaussianNB(),
    'KNeighborsClassifier': neighbors.KNeighborsClassifier(),
    'SVC': svm.SVC(probability=True, random_state=0),
    'XGBClassifier': XGBClassifier(random_state=0, verbosity=0),
    'DecisionTreeClassifier': tree.DecisionTreeClassifier(random_state=0)
}

for idx, row in top_models.iterrows():
    model_name = row['MLA Name']
    if model_name in model_mapping:
        vote_est.append((model_name.lower()[:3], model_mapping[model_name]))

print(f"Ensemble with {len(vote_est)} models")

# Soft voting ensemble (weighted probabilities)
if len(vote_est) > 1:
    vote_soft = ensemble.VotingClassifier(estimators=vote_est, voting='soft')
    vote_soft_cv = model_selection.cross_validate(
        vote_soft, data1[available_features], data1[target], 
        cv=cv_split, return_train_score=True, scoring='accuracy'
    )
    
    print("Ensemble Results:")
    print(f"  Training Accuracy: {vote_soft_cv['train_score'].mean():.4f}")
    print(f"  Test Accuracy: {vote_soft_cv['test_score'].mean():.4f}")
    print(f"  Std: ±{vote_soft_cv['test_score'].std():.4f}")
    
    # Train final model
    print("\n🚀 FINAL PREDICTIONS")
    print("-"*50)
    
    vote_soft.fit(data1[available_features], data1[target])
    predictions = vote_soft.predict(data_val[available_features])
    
    # Create submission
    submission = pd.DataFrame({
        'PassengerId': data_val['PassengerId'],
        'Transported': predictions.astype(bool)
    })
    submission.to_csv('best_models_submission.csv', index=False)
    
    accuracy_estimate = vote_soft_cv['test_score'].mean() * 100
    
    print("📊 RESULTS ANALYSIS")
    print("-"*50)
    print(f"Best Models Framework Results:")
    print(f"  Cross-validation Accuracy: {accuracy_estimate:.1f}%")
    print(f"  Model: Soft Voting Ensemble ({len(vote_est)} algorithms)")
    print(f"  Features used: {len(available_features)}")
    
    if accuracy_estimate >= 95:
        print("  🏆 GOLD TIER ACHIEVED!")
    elif accuracy_estimate >= 90:
        print("  🥈 SILVER TIER ACHIEVED!")
    else:
        print("  📈 SOLID PERFORMANCE!")
    
    print(f"\nPrediction Distribution:")
    print(f"  Transported=True: {predictions.sum()} ({predictions.mean():.1%})")
    print(f"  Transported=False: {len(predictions) - predictions.sum()} ({1-predictions.mean():.1%})")
    
    print(f"\n💡 BEST-MODELS KEY INNOVATIONS:")
    print("-"*50)
    print("✅ Comprehensive age group binning")
    print("✅ Advanced cabin region classification (7 regions)")
    print("✅ Group dynamics and family size features")
    print("✅ Sophisticated missing value imputation strategies")
    print("✅ Log transformation of expenditure features")
    print("✅ Multi-model ensemble with soft voting")
    print("✅ Systematic feature engineering pipeline")
    
    print(f"\n📁 Output: best_models_submission.csv")
    print(f"🎯 Performance: {accuracy_estimate:.1f}% accuracy")
    print("="*70)

else:
    print("Not enough models for ensemble. Using best single model.")
    best_model = model_mapping[top_models.iloc[0]['MLA Name']]
    best_model.fit(data1[available_features], data1[target])
    predictions = best_model.predict(data_val[available_features])
    
    submission = pd.DataFrame({
        'PassengerId': data_val['PassengerId'],
        'Transported': predictions.astype(bool)
    })
    submission.to_csv('best_models_submission.csv', index=False)
    
    print(f"Single best model: {top_models.iloc[0]['MLA Name']}")
    print(f"Accuracy: {top_models.iloc[0]['MLA Test Accuracy']:.1%}")

print("\n✅ BEST-MODELS FRAMEWORK COMPLETE")