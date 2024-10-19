import pandas as pd
import numpy as np
import seaborn as sns
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
