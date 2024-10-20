import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt

file_path = "/Users/aaditmehta/Desktop/Developer Projects/Red-Ventures-Case/Red_Ventures_Dataset.csv" 
data = pd.read_csv(file_path)

#Create Correlation Matrix
correlation_matrix = data[['Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment', 'Approved']].corr()

#Visualize Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix - Loan Approval Analysis")
plt.show()

# Employment Status
approval_by_employment = data.groupby('Employment_Status')['Approved'].mean().reset_index()
sns.barplot(data=approval_by_employment, x='Employment_Status', y='Approved')
plt.title('Approval Rate by Employment Status')
plt.show()

# Loan Purpose
approval_by_reason = data.groupby('Reason')['Approved'].mean().reset_index()
sns.barplot(data=approval_by_reason, x='Reason', y='Approved')
plt.title('Approval Rate by Loan Purpose')
plt.xticks(rotation=45)
plt.show()

#Approval Rated by FICO Category
fico_order = ["excellent", "very_good", "good", "fair", "poor"]
data['Fico_Score_group'] = pd.Categorical(data['Fico_Score_group'], categories=fico_order, ordered=True)
approval_by_fico_category = data.groupby('Fico_Score_group')['Approved'].mean().reset_index()
sns.barplot(data=approval_by_fico_category, x='Fico_Score_group', y='Approved', order=fico_order)
plt.title('Approval Rate by FICO Category')
plt.show()

# Approval Rates by Lender
lender_approval = data.groupby('Lender')['Approved'].mean().reset_index()
sns.barplot(data=lender_approval, x='Lender', y='Approved')
plt.title('Approval Rate by Lender')
plt.show()

#Approval Rates by Reason
approval_by_lender_reason = data.groupby(['Lender', 'Reason'])['Approved'].mean().unstack()
approval_by_lender_reason.plot(kind='bar', stacked=False)
plt.title('Approval Rate by Lender and Loan Purpose')
plt.ylabel('Approval Rate')
plt.show()

#Approval Rated by FICO Category by lender
approval_by_fico_category_lender = data.groupby(['Lender','Fico_Score_group'])['Approved'].mean().unstack()
approval_by_fico_category_lender = approval_by_fico_category_lender[fico_order]
approval_by_fico_category_lender.plot(kind='bar', stacked=False)
plt.title('Approval Rate by FICO Category by Lender')
plt.show()

#Approval Rates by Employment Status
approval_by_lender_employment = data.groupby(['Lender', 'Employment_Status'])['Approved'].mean().unstack()
approval_by_lender_employment.plot(kind='bar', stacked=False)
plt.title('Approval Rate by Lender and Employment Status')
plt.ylabel('Approval Rate')
plt.show()

#Average Bounty per Lender
average_bounty = data.groupby('Lender')['bounty'].mean().reset_index()
sns.barplot(data=average_bounty, x='Lender', y='bounty')
plt.title('Average Bounty per Lender')
plt.show()

#Correlation between Monthly Gross Income and Approval
income_bins = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
income_labels = ["0-2000", "2000-4000", "4000-6000", "6000-8000", "8000-10000", "10000-12000", "12000-14000", "14000-16000", "16000-18000", "18000-20000"]

data['Income_Range'] = pd.cut(data['Monthly_Gross_Income'], bins=income_bins, labels=income_labels, include_lowest=True)

approval_by_income = data.groupby('Income_Range')['Approved'].mean().reset_index()

approval_by_income['Income_Range'] = pd.Categorical(approval_by_income['Income_Range'], categories=income_labels, ordered=True)

plt.figure(figsize=(10, 6))
sns.lineplot(data=approval_by_income, x='Income_Range', y='Approved', marker='o')
plt.title('Approval Rate by Monthly Gross Income Range')
plt.xlabel('Monthly Gross Income Range')
plt.ylabel('Approval Rate')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#Correlation between Monthly Gross Income and Approval per Lender
income_bins = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
income_labels = ["0-2000", "2000-4000", "4000-6000", "6000-8000", "8000-10000", "10000-12000", "12000-14000", "14000-16000", "16000-18000", "18000-20000"]

data['Income_Range'] = pd.cut(data['Monthly_Gross_Income'], bins=income_bins, labels=income_labels, include_lowest=True)

approval_by_income_lender = data.groupby(['Lender','Income_Range'])['Approved'].mean().reset_index()

approval_by_income_lender['Income_Range'] = pd.Categorical(approval_by_income_lender['Income_Range'], categories=income_labels, ordered=True)

plt.figure(figsize=(10, 6))
sns.lineplot(data=approval_by_income_lender, x='Income_Range', y='Approved', hue='Lender', marker='o')
plt.title('Approval Rate by Monthly Gross Income Range per Lender')
plt.xlabel('Monthly Gross Income Range')
plt.ylabel('Approval Rate')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Lender')
plt.show()

#Correlation between FICO Scores and Approval
approval_by_fico_score = data.groupby('FICO_score')['Approved'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=approval_by_fico_score, x='FICO_score', y='Approved', marker='o')
plt.title('Approval Rate by FICO Score')
plt.xlabel('FICO Score')
plt.ylabel('Approval Rate')
plt.grid(True)
plt.show()

#Correlation between FICO Scores and Approval per Lender
approval_by_fico_per_lender = data.groupby(['Lender', 'FICO_score'])['Approved'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=approval_by_fico_per_lender, x='FICO_score', y='Approved', hue='Lender', marker='o')
plt.title('Approval Rate by FICO Score per Lender')
plt.xlabel('FICO Score')
plt.ylabel('Approval Rate')
plt.grid(True)
plt.legend(title='Lender')
plt.show()

#Correlation between Loan Amount and Approval per Lender
approval_by_loan_amount_per_lender = data.groupby(['Lender', 'Loan_Amount'])['Approved'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=approval_by_loan_amount_per_lender, x='Loan_Amount', y='Approved', hue='Lender', marker='o')
plt.title('Approval Rate by Loan Amount per Lender')
plt.xlabel('Loan Amount')
plt.ylabel('Approval Rate')
plt.grid(True)
plt.legend(title='Lender')
plt.show()

#Correlation between Monthly Housing Payment and Approval per Lender
housing_payment_bins = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300]
housing_payment_labels = ["0-300", "300-600", "600-900", "900-1200", "1200-1500", "1500-1800", "1800-2100", "2100-2400", "2400-2700", "2700-3000", "3000-3300"]

data['Housing_Payment_Range'] = pd.cut(data['Monthly_Housing_Payment'], bins=housing_payment_bins, labels=housing_payment_labels, include_lowest=True)

approval_by_housing_payment_per_lender = data.groupby(['Lender', 'Housing_Payment_Range'])['Approved'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=approval_by_housing_payment_per_lender, x='Housing_Payment_Range', y='Approved', hue='Lender', marker='o')
plt.title('Approval Rate by Monthly Housing Payment Range per Lender')
plt.xlabel('Monthly Housing Payment Range')
plt.ylabel('Approval Rate')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Lender')
plt.show()

#Logistic Regression of Variables

data = pd.get_dummies(data, columns=['Fico_Score_group', 'Employment_Status'], drop_first=True)

features = ['Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment', 'Fico_Score_group_good', 'Fico_Score_group_fair', 'Fico_Score_group_poor', 'Fico_Score_group_very_good']
target = 'Approved'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Feature Importance by Coefficient Magnitudes
coefficients = np.abs(log_reg.coef_[0])
feature_importance = pd.DataFrame({'Feature': features, 'Importance': coefficients})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display data
print(feature_importance)

# Visualize data
plt.figure(figsize=(8, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature Importance Based on Logistic Regression')
plt.xlabel('Coefficient Magnitude (Importance)')
plt.ylabel('Features')
plt.show()

# Model Accuracy
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to analyze variables for each lender separately
def analyze_lender(data, lender_name):
    # Filter data for the specific lender
    lender_data = data[data['Lender'] == lender_name]
    
    # Check if there is any data for the lender
    if lender_data.empty:
        print(f"No data available for {lender_name}. Skipping analysis.")
        return
    
    # Define features dynamically based on available columns
    features = [
        'Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 
        'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose'
    ]
    features += [col for col in data.columns if 'Fico_Score_group_' in col or 'Employment_Status_' in col]
    
    # Ensure only valid columns are used in case any are missing
    features = [f for f in features if f in lender_data.columns]
    
    target = 'Approved'
    
    # Prepare the data
    X = lender_data[features]
    y = lender_data[target]
    
    # Check if there are enough samples to proceed
    if len(X) == 0:
        print(f"No data points available for {lender_name} after filtering. Skipping analysis.")
        return
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train a logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    
    # Extract Coefficients
    coefficients = np.abs(log_reg.coef_[0])
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': coefficients})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    # Display Results
    print(f"Feature importance for {lender_name}:")
    print(feature_importance)
    
    # Visualize Feature Importance
    plt.figure(figsize=(8, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title(f'Feature Importance for {lender_name}')
    plt.xlabel('Coefficient Magnitude (Importance)')
    plt.ylabel('Features')
    plt.grid(True)
    plt.show()
    
    # Model Accuracy
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy for {lender_name}: {accuracy:.2f}\n")

# Example Usage: Analyze for each lender
analyze_lender(data, 'A')
analyze_lender(data, 'B')
analyze_lender(data, 'C')
