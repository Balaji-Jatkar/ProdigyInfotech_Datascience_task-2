import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = sns.load_dataset('titanic')
print("Dataset loaded successfully!")


print("\n=== BASIC DATASET INFO ===")
print(f"Dataset shape: {train.shape}")
print(f"Columns: {list(train.columns)}")


print("\n=== DATA TYPES & MISSING VALUES ===")
print(train.info())
print("\nMissing values:")
print(train.isnull().sum())


print("\n=== DATA CLEANING ===")

print("Before cleaning - Missing values:")
print(train.isnull().sum())

train['age'].fillna(train['age'].median(), inplace=True)

train['embarked'].fillna(train['embarked'].mode()[0], inplace=True)

train.dropna(subset=['fare'], inplace=True)

print("\nAfter cleaning - Missing values:")
print(train.isnull().sum())

train['family_size'] = train['sibsp'] + train['parch'] + 1
train['alone'] = (train['family_size'] == 1).astype(int)

print("\n=== EXPLORATORY DATA ANALYSIS ===")

print("\nBasic Statistics:")
print(train.describe())

print(f"\nOverall survival rate: {train['survived'].mean():.2%}")

plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Titanic Dataset - Exploratory Data Analysis', fontsize=16)

ax1 = axes[0, 0]
survival_counts = train['survived'].value_counts()
ax1.pie(survival_counts.values, labels=['Died', 'Survived'], autopct='%1.1f%%')
ax1.set_title('Overall Survival Rate')

ax2 = axes[0, 1]
survival_gender = train.groupby('sex')['survived'].mean()
survival_gender.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightblue'])
ax2.set_title('Survival Rate by Gender')
ax2.set_ylabel('Survival Rate')
ax2.tick_params(axis='x', rotation=0)

ax3 = axes[0, 2]
survival_class = train.groupby('class')['survived'].mean()
survival_class.plot(kind='bar', ax=ax3, color='green', alpha=0.7)
ax3.set_title('Survival Rate by Class')
ax3.set_ylabel('Survival Rate')
ax3.tick_params(axis='x', rotation=45)

ax4 = axes[1, 0]
ax4.hist(train['age'], bins=20, color='skyblue', alpha=0.7)
ax4.set_title('Age Distribution')
ax4.set_xlabel('Age')
ax4.set_ylabel('Count')

ax5 = axes[1, 1]
ax5.hist(train['fare'], bins=20, color='orange', alpha=0.7)
ax5.set_title('Fare Distribution')
ax5.set_xlabel('Fare')
ax5.set_ylabel('Count')

ax6 = axes[1, 2]
family_survival = train.groupby('family_size')['survived'].mean()
family_survival.plot(kind='bar', ax=ax6, color='purple', alpha=0.7)
ax6.set_title('Survival Rate by Family Size')
ax6.set_ylabel('Survival Rate')
ax6.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

print("\n=== CORRELATION ANALYSIS ===")

numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size', 'alone']

if 'pclass' not in train.columns:
    class_map = {'First': 1, 'Second': 2, 'Third': 3}
    train['pclass'] = train['class'].map(class_map)

correlation_matrix = train[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

print("\n=== KEY FINDINGS ===")
print(f"1. Women had {train[train['sex']=='female']['survived'].mean():.1%} survival rate")
print(f"2. Men had {train[train['sex']=='male']['survived'].mean():.1%} survival rate")
print(f"3. First class survival rate: {train[train['class']=='First']['survived'].mean():.1%}")
print(f"4. Third class survival rate: {train[train['class']=='Third']['survived'].mean():.1%}")
print(f"5. Average age of survivors: {train[train['survived']==1]['age'].mean():.1f} years")
print(f"6. Average age of non-survivors: {train[train['survived']==0]['age'].mean():.1f} years")

print("\n=== SURVIVAL SUMMARY ===")
print("Survivors vs Non-survivors comparison:")
summary = train.groupby('survived')[['age', 'fare', 'family_size']].agg(['mean', 'median'])
print(summary)

print("\n=== ANALYSIS COMPLETE ===")
print("Data cleaning and EDA completed successfully!")