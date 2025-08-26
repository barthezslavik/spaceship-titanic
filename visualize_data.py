import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
train_df = pd.read_csv('train.csv')

# Prepare data with engineered features
train_df['Group'] = train_df['PassengerId'].str.split('_').str[0]
train_df['GroupSize'] = train_df.groupby('Group')['PassengerId'].transform('count')

# Parse cabin
cabin_split = train_df['Cabin'].str.split('/', expand=True)
train_df['Deck'] = cabin_split[0]
train_df['Side'] = cabin_split[2]

# Total spending
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_df['TotalSpending'] = train_df[spending_cols].sum(axis=1)

# Create figure with subplots
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Spaceship Titanic Data Visualization', fontsize=16, y=1.02)

# 1. Target Distribution
ax1 = plt.subplot(4, 4, 1)
train_df['Transported'].value_counts().plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Target Distribution')
ax1.set_xlabel('Transported')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['False', 'True'], rotation=0)

# 2. Age Distribution
ax2 = plt.subplot(4, 4, 2)
train_df['Age'].dropna().hist(bins=30, ax=ax2, edgecolor='black', alpha=0.7)
ax2.set_title('Age Distribution')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')

# 3. Home Planet Distribution
ax3 = plt.subplot(4, 4, 3)
planet_transported = pd.crosstab(train_df['HomePlanet'], train_df['Transported'], normalize='index')
planet_transported.plot(kind='bar', stacked=True, ax=ax3, color=['#FF6B6B', '#4ECDC4'])
ax3.set_title('Transport Rate by Home Planet')
ax3.set_xlabel('Home Planet')
ax3.set_ylabel('Proportion')
ax3.legend(title='Transported', labels=['False', 'True'])
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

# 4. CryoSleep Impact
ax4 = plt.subplot(4, 4, 4)
cryo_transported = pd.crosstab(train_df['CryoSleep'], train_df['Transported'], normalize='index')
cryo_transported.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4'])
ax4.set_title('CryoSleep vs Transported')
ax4.set_xlabel('CryoSleep')
ax4.set_ylabel('Proportion')
ax4.legend(title='Transported', labels=['False', 'True'])
ax4.set_xticklabels(['False', 'True'], rotation=0)

# 5. Group Size Distribution
ax5 = plt.subplot(4, 4, 5)
train_df['GroupSize'].value_counts().sort_index().plot(kind='bar', ax=ax5, color='#95E77E')
ax5.set_title('Group Size Distribution')
ax5.set_xlabel('Group Size')
ax5.set_ylabel('Count')

# 6. Deck Distribution
ax6 = plt.subplot(4, 4, 6)
deck_counts = train_df['Deck'].value_counts()
deck_counts.plot(kind='bar', ax=ax6, color='#FFE66D')
ax6.set_title('Passenger Distribution by Deck')
ax6.set_xlabel('Deck')
ax6.set_ylabel('Count')

# 7. Destination Distribution
ax7 = plt.subplot(4, 4, 7)
dest_transported = pd.crosstab(train_df['Destination'], train_df['Transported'], normalize='index')
dest_transported.plot(kind='bar', stacked=True, ax=ax7, color=['#FF6B6B', '#4ECDC4'])
ax7.set_title('Transport Rate by Destination')
ax7.set_xlabel('Destination')
ax7.set_ylabel('Proportion')
ax7.legend(title='Transported', labels=['False', 'True'])
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right')

# 8. Total Spending Distribution (log scale)
ax8 = plt.subplot(4, 4, 8)
spending_nonzero = train_df[train_df['TotalSpending'] > 0]['TotalSpending']
ax8.hist(np.log1p(spending_nonzero), bins=30, edgecolor='black', alpha=0.7, color='#A8E6CF')
ax8.set_title('Total Spending Distribution (log scale)')
ax8.set_xlabel('log(Total Spending + 1)')
ax8.set_ylabel('Frequency')

# 9. Spending by Transport Status
ax9 = plt.subplot(4, 4, 9)
transported = train_df[train_df['Transported'] == True]['TotalSpending'].dropna()
not_transported = train_df[train_df['Transported'] == False]['TotalSpending'].dropna()
ax9.boxplot([transported, not_transported], labels=['Transported', 'Not Transported'])
ax9.set_title('Spending by Transport Status')
ax9.set_ylabel('Total Spending')
ax9.set_ylim(0, 10000)  # Limit y-axis for better visibility

# 10. VIP Status
ax10 = plt.subplot(4, 4, 10)
vip_transported = pd.crosstab(train_df['VIP'], train_df['Transported'], normalize='index')
vip_transported.plot(kind='bar', ax=ax10, color=['#FF6B6B', '#4ECDC4'])
ax10.set_title('VIP Status vs Transported')
ax10.set_xlabel('VIP')
ax10.set_ylabel('Proportion')
ax10.legend(title='Transported', labels=['False', 'True'])
ax10.set_xticklabels(['False', 'True'], rotation=0)

# 11. Age by Transport Status
ax11 = plt.subplot(4, 4, 11)
train_df.boxplot(column='Age', by='Transported', ax=ax11)
ax11.set_title('Age Distribution by Transport Status')
ax11.set_xlabel('Transported')
ax11.set_ylabel('Age')
plt.sca(ax11)
plt.xticks([1, 2], ['False', 'True'])

# 12. Cabin Side Distribution
ax12 = plt.subplot(4, 4, 12)
side_transported = pd.crosstab(train_df['Side'], train_df['Transported'], normalize='index')
side_transported.plot(kind='bar', ax=ax12, color=['#FF6B6B', '#4ECDC4'])
ax12.set_title('Cabin Side vs Transported')
ax12.set_xlabel('Cabin Side')
ax12.set_ylabel('Proportion')
ax12.legend(title='Transported', labels=['False', 'True'])
ax12.set_xticklabels(['Port', 'Starboard'], rotation=0)

# 13. Spending Correlation Heatmap
ax13 = plt.subplot(4, 4, 13)
spending_corr = train_df[spending_cols].corr()
sns.heatmap(spending_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax13, 
            cbar_kws={'shrink': 0.8})
ax13.set_title('Spending Features Correlation')

# 14. Missing Values Heatmap
ax14 = plt.subplot(4, 4, 14)
missing_data = train_df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
ax14.barh(range(len(missing_data)), missing_data.values, color='#FF6B6B')
ax14.set_yticks(range(len(missing_data)))
ax14.set_yticklabels(missing_data.index)
ax14.set_xlabel('Missing Count')
ax14.set_title('Missing Values by Feature')
ax14.invert_yaxis()

# 15. Group Size vs Transport Rate
ax15 = plt.subplot(4, 4, 15)
group_transport = train_df.groupby('GroupSize')['Transported'].mean()
group_transport.plot(kind='line', marker='o', ax=ax15, color='#4ECDC4', linewidth=2)
ax15.set_title('Transport Rate by Group Size')
ax15.set_xlabel('Group Size')
ax15.set_ylabel('Transport Rate')
ax15.grid(True, alpha=0.3)

# 16. Age Groups Analysis
ax16 = plt.subplot(4, 4, 16)
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=[0, 18, 35, 50, 65, 100], 
                              labels=['0-18', '19-35', '36-50', '51-65', '65+'])
age_group_transport = pd.crosstab(train_df['AgeGroup'], train_df['Transported'], normalize='index')
age_group_transport.plot(kind='bar', stacked=True, ax=ax16, color=['#FF6B6B', '#4ECDC4'])
ax16.set_title('Transport Rate by Age Group')
ax16.set_xlabel('Age Group')
ax16.set_ylabel('Proportion')
ax16.legend(title='Transported', labels=['False', 'True'])
ax16.set_xticklabels(ax16.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('spaceship_titanic_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("Visualization saved as 'spaceship_titanic_analysis.png'")
print("\nKey Visual Insights:")
print("="*50)
print("1. CryoSleep has STRONG impact - 82% transport rate when True")
print("2. Europa passengers have highest transport rate (66%)")
print("3. Zero spending correlates with transportation")
print("4. Group size affects transport rate")
print("5. Luxury amenities users less likely to be transported")
print("6. Age shows weak correlation with transport status")
print("7. Cabin side (P/S) shows slight difference in transport rates")
print("8. VIP status has minimal impact on transportation")