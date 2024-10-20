import pandas as pd

# Load processed dataset
file_path = "/Users/aaditmehta/Desktop/Developer Projects/Red-Ventures-Case/Red_Ventures_Dataset.csv" 
data = pd.read_csv(file_path)

# Part (b): Calculate Incremental Revenue from Adjusted Matching with Debugging
def calculate_incremental_revenue(data):
    # Step 1: Calculate Current Revenue (only include approved loans with non-zero bounty)
    current_revenue = data[(data['Approved'] == 1) & (data['bounty'] > 0)]['bounty'].sum()
    
    # Step 2: Adjust Lender Matching based on analysis insights
    adjusted_data = data.copy()
    
    # Expanded conditions:
    # Rule 1: Customers with FICO scores <= 640 and income under $5000 denied by Lender B might be a better fit for Lender C
    condition_1 = (adjusted_data['Lender'] == 'B') & (adjusted_data['FICO_score'] <= 640) & (adjusted_data['Monthly_Gross_Income'] < 5000) & (adjusted_data['Approved'] == 0)
    print(f"Number of customers reallocated from B to C: {condition_1.sum()}")
    
    adjusted_data.loc[condition_1, 'Lender'] = 'C'
    adjusted_data.loc[condition_1, 'Approved'] = 1  # Assume they would be approved under Lender C's criteria
    
    # Rule 2: Customers with FICO scores >= 620 and income above $4000 denied by Lender C might be a better fit for Lender B
    condition_2 = (adjusted_data['Lender'] == 'C') & (adjusted_data['FICO_score'] >= 620) & (adjusted_data['Monthly_Gross_Income'] >= 4000) & (adjusted_data['Approved'] == 0)
    print(f"Number of customers reallocated from C to B: {condition_2.sum()}")
    
    adjusted_data.loc[condition_2, 'Lender'] = 'B'
    adjusted_data.loc[condition_2, 'Approved'] = 1  # Assume they would be reconsidered and approved by Lender B
    
    # Step 3: Calculate Adjusted Revenue (exclude loans that still have zero bounty)
    adjusted_revenue = adjusted_data[(adjusted_data['Approved'] == 1) & (adjusted_data['bounty'] > 0)]['bounty'].sum()
    
    # Step 4: Calculate Incremental Revenue
    incremental_revenue = adjusted_revenue - current_revenue
    print(f"Current Revenue: {current_revenue}")
    print(f"Adjusted Revenue: {adjusted_revenue}")
    print(f"Incremental Revenue from Adjusted Matching: {incremental_revenue}")
    
    # Optional: Inspect adjusted data
    print("\nSample Adjusted Data:")
    print(adjusted_data[(condition_1 | condition_2) & (adjusted_data['bounty'] > 0)].head(10))
    
    # Additional Information:
    print("\nBreakdown of Adjusted Loan Approvals by Lender:")
    print(adjusted_data.groupby('Lender')['Approved'].sum())

# Run the function
calculate_incremental_revenue(data)
