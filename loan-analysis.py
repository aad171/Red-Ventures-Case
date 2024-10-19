import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "/Users/aaditmehta/Desktop/Developer Projects/Red-Ventures-Case/Red_Ventures_Dataset.xlsx" 
df = pd.read_excel(file_path)

def explore_approvability(df):
    # Correlation Analysis
    corr_matrix = df[['Loan_Amount', 'FICO_score', 'Monthly_Gross_Income', 
                      'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose', 'Approved']].corr()

    # Correlation Matrix Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix - Loan Approval Analysis")
    plt.show()

    # Explore categorical variables' relationship with approval
    approval_by_reason = df.groupby('Reason')['Approved'].mean().sort_values(ascending=False)
    approval_by_employment = df.groupby('Employment_Status')['Approved'].mean().sort_values(ascending=False)
    approval_by_sector = df.groupby('Employment_Sector')['Approved'].mean().sort_values(ascending=False)

    return approval_by_reason, approval_by_employment, approval_by_sector

# Step 2: Lender Approval Rates
def analyze_lender_approval(df):
    # Calculate average approval rate per lender
    lender_approval_rates = df.groupby('Lender')['Approved'].mean().sort_values(ascending=False)

    # Analyze approval rates across different variables for each lender
    approval_by_lender_and_reason = df.groupby(['Lender', 'Reason'])['Approved'].mean().unstack()
    approval_by_lender_and_employment = df.groupby(['Lender', 'Employment_Status'])['Approved'].mean().unstack()

    return lender_approval_rates, approval_by_lender_and_reason, approval_by_lender_and_employment

# Revenue Optimization
def revenue_optimization(df):
    # Average bounty per lender
    average_bounty_per_lender = df.groupby('Lender')['bounty'].mean()

    # Revenue opportunities by customer characteristics and potential matching strategies
    revenue_by_fico_and_employment = df.groupby(['Fico_Score_group', 'Employment_Status', 'Lender'])['bounty'].mean().unstack()

    # Revenue maximization
    best_lender_match_by_segment = revenue_by_fico_and_employment.idxmax(axis=1)
    best_lender_match_by_segment_revenue = revenue_by_fico_and_employment.max(axis=1)

    return average_bounty_per_lender, best_lender_match_by_segment, best_lender_match_by_segment_revenue

# Main function to run the analysis
def main():
    # Analysis for approvability
    reason, employment, sector = explore_approvability(df)
    print("Approval by Loan Reason:")
    print(reason)
    print("\nApproval by Employment Status:")
    print(employment)
    print("\nApproval by Employment Sector:")
    print(sector)

    # Analysis for lender approval
    lender_rates, lender_reason, lender_employment = analyze_lender_approval(df)
    print("\nLender Approval Rates:")
    print(lender_rates)
    print("\nApproval by Lender and Reason:")
    print(lender_reason)
    print("\nApproval by Lender and Employment Status:")
    print(lender_employment)

    # Revenue optimization analysis
    avg_bounty, best_match, best_revenue = revenue_optimization(df)
    print("\nAverage Bounty per Lender:")
    print(avg_bounty)
    print("\nBest Lender Matches for Segments:")
    print(best_match)
    print("\nBest Revenue per Segment:")
    print(best_revenue)

if __name__ == "__main__":
    main()

