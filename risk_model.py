import pandas as pd
import numpy as np
all_df = pd.read_csv('monthly_avg_data.csv')

def create_delinquency_index(df='monthly_avg_data.csv'):
    all_df = pd.read_csv()
    all_df['avg_delinquency_index'] = all_df['avg_days_to_delinquency'] / all_df['num_loans'] / all_df['pct_delinquent']
    all_df['avg_delinquency_index'] = all_df['avg_delinquency_index'].replace([np.inf, -np.inf], 9999)
    all_df['avg_delinquency_index'] = (9999 - all_df['avg_delinquency_index'])*(1-all_df['pct_delinquent'])

    epsilon = 1e-6
    raw_index = all_df['avg_delinquency_index']
    adjusted_index = [raw_index.iloc[0]]  # start from the first value

    # Step 3: Iteratively compute adjusted values
    for t in range(1, len(raw_index)):
        prev = adjusted_index[-1]
        curr = raw_index.iloc[t]
        loans = all_df['num_loans'].iloc[t]

        if abs(prev) < epsilon:
            # avoid division by zero
            adjusted_value = curr
        else:
            adjustment = loans * ((curr - prev) / prev)
            adjusted_value = prev + adjustment
        
        adjusted_index.append(adjusted_value)

    # Step 4: Save to DataFrame
    all_df['avg_delinquency_index'] = adjusted_index


    all_df['avg_adjusted_delinquency_index'] = np.log1p(
        adjusted_index
    )

    return all_df

def main():
    create_delinquency_index()

if __name__ == '__main__':
    main()

