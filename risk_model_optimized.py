import pandas as pd
import numpy as np

def create_delinquency_index(df_path='monthly_avg_data.csv'):
    all_df = pd.read_csv(df_path)
    
    # Safety: Fill NaN to avoid unexpected errors
    all_df['avg_days_to_delinquency'] = all_df['avg_days_to_delinquency'].fillna(0)
    all_df['num_loans'] = all_df['num_loans'].fillna(1)
    all_df['pct_delinquent'] = all_df['pct_delinquent'].fillna(0)

    # Step 1: Calculate raw avg_delinquency_index
    all_df['avg_delinquency_index'] = all_df['avg_days_to_delinquency'] / (all_df['num_loans'] * (all_df['pct_delinquent'] + 1e-6))
    
    # Step 2: Adjust infinities and scaling
    all_df['avg_delinquency_index'] = (9999 - all_df['avg_delinquency_index'].replace([np.inf, -np.inf], 9999)) * (1 - all_df['pct_delinquent'])

    # Step 3: Iteratively adjust delinquency index
    epsilon = 1e-6
    raw_index = all_df['avg_delinquency_index'].values
    num_loans = all_df['num_loans'].values

    adjusted_index = np.zeros_like(raw_index)
    adjusted_index[0] = raw_index[0]

    for t in range(1, len(raw_index)):
        prev = adjusted_index[t-1]
        curr = raw_index[t]

        if abs(prev) < epsilon:
            adjusted_index[t] = curr
        else:
            adjusted_index[t] = prev + num_loans[t] * ((curr - prev) / (prev + epsilon))

    all_df['avg_delinquency_index'] = adjusted_index

    # Step 4: Final scaling - log1p
    all_df['avg_adjusted_delinquency_index'] = np.log1p(adjusted_index)

    return all_df

def main():
    result_df = create_delinquency_index()
    result_df.to_csv('monthly_avg_data_with_index.csv', index=False)
    print("Saved: monthly_avg_data_with_index.csv")

if __name__ == '__main__':
    main()