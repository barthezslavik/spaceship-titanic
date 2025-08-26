import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Load datasets
print("="*60)
print("SPACESHIP TITANIC DATA ANALYSIS")
print("="*60)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print("\n1. DATASET OVERVIEW")
print("-"*40)
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

print("\n2. TRAINING DATA - First 5 rows")
print("-"*40)
print(train_df.head())

print("\n3. COLUMN INFORMATION")
print("-"*40)
print("\nTraining columns:")
print(train_df.columns.tolist())
print("\nData types:")
print(train_df.dtypes)

print("\n4. TARGET VARIABLE ANALYSIS")
print("-"*40)
if 'Transported' in train_df.columns:
    print("Target: 'Transported'")
    print(train_df['Transported'].value_counts())
    print(f"\nTransported ratio: {train_df['Transported'].mean():.2%}")

print("\n5. MISSING VALUES ANALYSIS")
print("-"*40)
missing_train = train_df.isnull().sum()
missing_train_pct = (missing_train / len(train_df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_train,
    'Percentage': missing_train_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print("\nMissing values in training data:")
print(missing_df)

print("\n6. NUMERICAL FEATURES STATISTICS")
print("-"*40)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
if 'Transported' in numeric_cols:
    numeric_cols.remove('Transported')
print(f"Numerical columns: {numeric_cols}")
if numeric_cols:
    print("\nStatistical summary:")
    print(train_df[numeric_cols].describe())

print("\n7. CATEGORICAL FEATURES ANALYSIS")
print("-"*40)
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")
for col in categorical_cols[:5]:  # Show first 5 categorical columns
    print(f"\n{col} value counts:")
    print(train_df[col].value_counts().head())

print("\n8. FEATURE ENGINEERING OPPORTUNITIES")
print("-"*40)
# Check for ID patterns
if 'PassengerId' in train_df.columns:
    print("\nPassengerId structure analysis:")
    sample_ids = train_df['PassengerId'].head(10)
    print(sample_ids)
    # Try to parse the ID
    if train_df['PassengerId'].str.contains('_').any():
        print("\nPassengerId contains group and individual identifiers")
        train_df['Group'] = train_df['PassengerId'].str.split('_').str[0]
        train_df['GroupSize'] = train_df.groupby('Group')['PassengerId'].transform('count')
        print(f"Number of unique groups: {train_df['Group'].nunique()}")
        print(f"Average group size: {train_df['GroupSize'].mean():.2f}")
        print("Group size distribution:")
        print(train_df['GroupSize'].value_counts().sort_index())

# Check for cabin structure
if 'Cabin' in train_df.columns:
    print("\nCabin structure analysis:")
    cabin_sample = train_df['Cabin'].dropna().head(10)
    print(cabin_sample)
    if train_df['Cabin'].str.contains('/').any():
        print("\nCabin contains deck/room/side information")
        cabin_split = train_df['Cabin'].str.split('/', expand=True)
        train_df['Deck'] = cabin_split[0]
        train_df['CabinNum'] = cabin_split[1]
        train_df['Side'] = cabin_split[2]
        print(f"Unique decks: {train_df['Deck'].dropna().unique()}")
        print(f"Cabin sides: {train_df['Side'].dropna().unique()}")

# Check for name patterns  
if 'Name' in train_df.columns:
    print("\nName patterns:")
    print(f"Unique names: {train_df['Name'].nunique()}")
    print(f"Duplicate names: {len(train_df) - train_df['Name'].nunique()}")

print("\n9. SPENDING FEATURES ANALYSIS")
print("-"*40)
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
spending_present = [col for col in spending_cols if col in train_df.columns]
if spending_present:
    print(f"Spending columns found: {spending_present}")
    train_df['TotalSpending'] = train_df[spending_present].sum(axis=1)
    print("\nTotal spending statistics:")
    print(train_df['TotalSpending'].describe())
    print(f"\nPassengers with zero spending: {(train_df['TotalSpending'] == 0).sum()} ({(train_df['TotalSpending'] == 0).mean():.2%})")

print("\n10. CORRELATIONS WITH TARGET")
print("-"*40)
if 'Transported' in train_df.columns:
    # Convert boolean target to numeric
    train_df['Transported_num'] = train_df['Transported'].astype(int)
    
    # Calculate correlations for numeric features
    numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Transported_num' in numeric_features:
        correlations = train_df[numeric_features].corr()['Transported_num'].sort_values(ascending=False)
        print("Correlations with Transported:")
        print(correlations[correlations.index != 'Transported_num'])
    
    # Check categorical associations
    print("\nCategorical feature associations with target:")
    for col in categorical_cols[:5]:
        if train_df[col].nunique() < 20:  # Only for low cardinality
            crosstab = pd.crosstab(train_df[col], train_df['Transported'], normalize='index')
            print(f"\n{col} vs Transported:")
            print(crosstab)

print("\n11. DATA QUALITY INSIGHTS")
print("-"*40)
print(f"Duplicate rows in training: {train_df.duplicated().sum()}")
print(f"Duplicate PassengerIds: {train_df['PassengerId'].duplicated().sum() if 'PassengerId' in train_df.columns else 'N/A'}")

# Check if test has same features as train
train_features = set(train_df.columns)
test_features = set(test_df.columns)
print(f"\nFeatures in train but not in test: {train_features - test_features}")
print(f"Features in test but not in train: {test_features - train_features}")

print("\n12. KEY INSIGHTS & RECOMMENDATIONS")
print("-"*40)
print("""
Based on the analysis:
1. This is a binary classification problem (Transported: True/False)
2. Several features have missing values that need imputation
3. PassengerId contains group information - families/groups travel together
4. Cabin structure provides deck/side information which might be predictive
5. Spending features could be aggregated for total spending patterns
6. Mix of numerical and categorical features requires appropriate encoding
7. Consider feature engineering from ID and Cabin structures
8. Group behavior might be important - people in same group likely have same fate
""")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)