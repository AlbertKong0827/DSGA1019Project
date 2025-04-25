import pandas as pd
import numpy as np
import gdown
import zipfile
import os

IDS = ['1obUqshqVQR9mzKnUUOwOXELo2CeyDsYU','1ujt3l4HIiT6fMeeUlwX7r_pYIgAqsaKj','1teP3qIrxGK7Eh4CUYGAStADMVSnOm1mt',\
       '1n32-K_WKPaHnLDbuPG1Ke7rH-0a4nJJA','1LEeng4_shu_MX4Wz3iFsbCYp2OTLOnCV','1voganvmbmogpmOsyLX4zekHoJs2gnUQU',\
        '1JufHZvdLcwa-xQEunbgI-SQbZYd2XD99','12MLtLYbCNyjQ5i6gLsaqwai8jm3pXGXv','16GwxjwEor4XlOEVcQjnzDT11uBI2TnKe',\
        '17xs04GDzUC-c8ON120fItgxRLdhGYwzK','1ZbbW35Hib9jFA__L9d7TdFpWz9QRYCDf','1YkZOXCn6VQipQ63WUeM36FwZnAYBC0Nz',\
        '19ARzaT4Vvutk50lQY5H_fsuizbbBIkeb','1LFDXqeUx3mHlsyH_uq0s3KMzMYQ8OAgC','1jd9jvQX_Xv0LpRZH-Llqx3hHYgP1ETeG',\
            '1GQNmYtmk7d24o1Ylk33BzbbBBsmDRbZI','1IyjUlSzY0WIjpV3TJyJt2FT5qN6FoMdR','1-4HSTqCt-LQ1aiPWUWVsnPLQt5R_Dnxg',\
                '1zbE4vuRW2VZZcCEr1iyyXwezA1IWEz6G','1jaknuNTex_MnxhI_5pQ5LWnAMAShGdRM','1ZnmgZnRRbY8oW8Dfxt-3rV6a9o54TJvj',\
                    '1zZfh1vpnQE7iJrcOmt5vkZTytk1mIxfi','1uESSFgodNIqAibWDy_i0ZYJxdQCpAVpn','1_q3f5PEe0YwEtikTc1RVK_Q0ElcYMLeB',\
                        '1UWENCUVJfTgvxb8zx9zeYVTt7iGEnVos','13dymLaCeSeu0ysOq-M374QQPf8V8j1t2']
def download_data(ids):
    # Download the zip
    for file_id in ids:
        zip_path = 'data.zip'
        gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_path, quiet=False)

    # Extract the contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data_extracted')

        # List contents
        print("Files extracted:", os.listdir('data_extracted'))

def monthly_avg_data(year):
    file1 = f'data_extracted/sample_orig_{str(year)}.txt'
    file2 = f'data_extracted/sample_svcg_{str(year)}.txt'
    df1_cols = ['CREDIT SCORE', 'FIRST PAYMENT DATE', 'FIRST TIME HOMEBUYER FLAG', 'MATURITY DATE', 'METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION', 'MORTGAGE INSURANCE PERCENTAGE (MI %)', 'NUMBER OF UNITS', 'OCCUPANCY STATUS', 'ORIGINAL COMBINED LOAN-TO-VALUE(CLTV)', 'ORIGINAL DEBT-TO-INCOME (DTI) RATIO', 'ORIGINAL UPB', 'ORIGINAL LOAN-TO-VALUE(LTV)', 'ORIGINAL INTEREST RATE', 'CHANEL','REPAYMENT PENALTY MORTGAGE (PPM) FLAG', 'AMORTIZATION TYPE','PROPERTY STATE','PROPERTY TYPE', 'POSTAL CODE', 'LOAN SEQUENCE NUMBER', 'LOAN PURPOSE', 'ORIGINAL LOAN TERM', 'NUMBER OF BORROWERS','SELLER NAME','SERVICER NAME', 'SUPER CONFORMING FLAG','PRE-RELIEF REFINANCE LOAN SEQUENCE NUMBER', 'PROGRAM INDICATOR', 'RELIEF REFINANCE INDICATOR', 'PROPERTY VALUATION METHOD', 'INTEREST ONLY INDICATOR (I/O INDICATOR)', 'MI CANCELLATION INDICATOR']
    df2_cols = ['LOAN SEQUENCE NUMBER', 'MONTHLY REPORTING PERIOD', 'CURRENT ACTUAL UPB', 'CURRENT LOAN DELINQUENCY STATUS', 'LOAN AGE','REMAINING MONTHS TO LEGAL MATURITY', 'DEFECT SETTLEMENT DATE', 'MODIFICATION FLAG', 'ZERO BALANCE CODE', 'ZERO BALANCE EFFECTIVE DATE', 'CURRENT INTEREST RATE', 'CURRENT NON-INTEREST BEARING UPB', 'DUE DATE OF LAST PAID INSTALLMENT (DDLPI)', 'MI RECOVERIES', 'NET SALE PROCEEDS', 'NON MI RECOVERIES', 'TOTAL EXPENSES', 'LEGAL COSTS', 'MAINTENANCE AND PRESERVATION COSTS', 'TAXES AND INSURANCE', 'MISCELLANEOUS EXPENSES', 'ACTUAL LOSS CALCULATION', 'CUMULATIVE MODIFICATION COST', 'STEP MODIFICATION FLAG', 'PAYMENT DEFERRAL', 'ESTIMATED LOAN TO VALUE (ELTV)', 'ZERO BALANCE REMOVAL UPB', 'DELINQUENT ACCRUED INTEREST', 'DELINQUENCY DUE TO DISASTER', 'BORROWER ASSISTANCE STATUS CODE', 'CURRENT MONTH MODIFICATION COST', 'INTEREST BEARING UPB']
    df1 = pd.read_csv(file1, names=df1_cols, delimiter='|')  # or delimiter=',' if comma-separated
    df2 = pd.read_csv(file2, names=df2_cols, delimiter='|')

    df1['CREDIT SCORE'] = df1['CREDIT SCORE'].replace(9999, -1)

    # Encode FIRST TIME HOMEBUYER FLAG
    df1['FIRST TIME HOMEBUYER FLAG'] = df1['FIRST TIME HOMEBUYER FLAG'].map({'Y': 1, 'N': 0, '9': -1})

    # Drop rows with invalid MI %
    df1 = df1[df1['MORTGAGE INSURANCE PERCENTAGE (MI %)'] != 999]

    # Drop rows with invalid NUMBER OF UNITS
    df1 = df1[df1['NUMBER OF UNITS'] != 99]

    # Drop rows with invalid ORIGINAL CLTV
    df1 = df1[df1['ORIGINAL COMBINED LOAN-TO-VALUE(CLTV)'] != 999]

    # Drop rows with invalid DTI
    df1 = df1[df1['ORIGINAL DEBT-TO-INCOME (DTI) RATIO'] != 999]

    # Drop rows with invalid LTV
    df1 = df1[df1['ORIGINAL LOAN-TO-VALUE(LTV)'] != 999]

    # Encode PPM FLAG
    df1['REPAYMENT PENALTY MORTGAGE (PPM) FLAG'] = df1['REPAYMENT PENALTY MORTGAGE (PPM) FLAG'].map({'Y': 1, 'N': 0})

    # Drop rows with invalid NUMBER OF BORROWERS
    df1 = df1[df1['NUMBER OF BORROWERS'] != 99]

    # Encode I/O INDICATOR
    df1['INTEREST ONLY INDICATOR (I/O INDICATOR)'] = df1['INTEREST ONLY INDICATOR (I/O INDICATOR)'].map({'Y': 1, 'N': 0})

    df1_columns_to_drop = ['METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION', 'PROPERTY VALUATION METHOD', 'MI CANCELLATION INDICATOR']
    df1 = df1.drop(columns=df1_columns_to_drop)

    threshold = 0.9  # drop if >90% of values are NaN
    df2 = df2.loc[:, df2.isna().mean() < threshold]
    df2['CURRENT LOAN DELINQUENCY STATUS'] = df2['CURRENT LOAN DELINQUENCY STATUS'].apply(
    lambda x: -1 if str(x).strip().upper() == 'RA' else int(x)
    )

    df_combined = pd.merge(df2, df1, on='LOAN SEQUENCE NUMBER', how='left')

    # If MONTHLY REPORTING PERIOD is integer or string like 200209, convert it to datetime
    df_combined['MONTHLY REPORTING PERIOD'] = pd.to_datetime(
        df_combined['MONTHLY REPORTING PERIOD'].astype(str), format='%Y%m'
    )
    
    # Sort data
    df_combined.sort_values(['LOAN SEQUENCE NUMBER', 'MONTHLY REPORTING PERIOD'], inplace=True)
    
    # Define delinquency: status is a number ≥ 1 (exclude RA and 0)
    def is_delinquent(x):
        try:
            return int(x) >= 1
        except:
            return False
    
    df_combined['IS_DELINQUENT'] = df_combined['CURRENT LOAN DELINQUENCY STATUS'].apply(is_delinquent)
    
    # First delinquency date per loan
    first_dq = df_combined[df_combined['IS_DELINQUENT']].groupby(
        'LOAN SEQUENCE NUMBER'
    )['MONTHLY REPORTING PERIOD'].min().reset_index().rename(
        columns={'MONTHLY REPORTING PERIOD': 'FIRST_DELINQUENCY_DATE'}
    )
    
    # First reporting date per loan
    first_obs = df_combined.groupby(
        'LOAN SEQUENCE NUMBER'
    )['MONTHLY REPORTING PERIOD'].min().reset_index().rename(
        columns={'MONTHLY REPORTING PERIOD': 'FIRST_REPORTING_DATE'}
    )
    
    # Merge both
    merged = first_obs.merge(first_dq, on='LOAN SEQUENCE NUMBER', how='left')
    
    # Calculate months to first delinquency
    merged['TIME_TO_FIRST_DELINQUENCY_DAYS'] = (
        merged['FIRST_DELINQUENCY_DATE'] - merged['FIRST_REPORTING_DATE']
    ).dt.days

    merged['OBSERVATION_MONTH'] = merged['FIRST_REPORTING_DATE'].dt.to_period('M').dt.to_timestamp()

    # Mask to keep only loans that ever went delinquent
    delinquent_loans = merged[merged['TIME_TO_FIRST_DELINQUENCY_DAYS'].notna()]
    
    # Metric 1: Total cohort size per month
    cohort_sizes = merged.groupby('OBSERVATION_MONTH')['LOAN SEQUENCE NUMBER'].count().rename("num_loans")
    print(len(cohort_sizes))
    
    # Metric 2: % of loans that ever became delinquent
    delinquency_counts = delinquent_loans.groupby('OBSERVATION_MONTH')['LOAN SEQUENCE NUMBER'].count().rename("num_delinquent_loans")
    print(len(delinquency_counts))
    delinquency_rate = (delinquency_counts / cohort_sizes).rename("pct_delinquent")
    print(len(delinquency_rate))
    # Metric 3: Among delinquent loans only — how quickly did they go delinquent?
    ttd_stats = delinquent_loans.groupby('OBSERVATION_MONTH')['TIME_TO_FIRST_DELINQUENCY_DAYS'].agg(
        avg_days_to_delinquency='mean',
        median_days_to_delinquency='median'
    )
    print(len(ttd_stats))

    # Combine all metrics
    market_risk_by_month = pd.concat([cohort_sizes, delinquency_counts, delinquency_rate, ttd_stats], axis=1).reset_index()
    
    # Fill NaNs where there were no delinquencies
    market_risk_by_month['num_delinquent_loans'] = market_risk_by_month['num_delinquent_loans'].fillna(0).astype(int)
    market_risk_by_month['pct_delinquent'] = market_risk_by_month['pct_delinquent'].fillna(0)

    monthly_avg = df_combined.groupby('MONTHLY REPORTING PERIOD').mean(numeric_only=True)
    monthly_avg = monthly_avg.dropna(axis=1, how='all')
    monthly_avg = monthly_avg.loc[:, ~monthly_avg.columns.str.contains('date', case=False)]
    monthly_avg = monthly_avg.loc[:, ~monthly_avg.columns.str.contains('code', case=False)]
    monthly_avg = monthly_avg.reset_index()
    monthly_avg = monthly_avg.dropna()

    all_df = pd.merge(
        monthly_avg,
        market_risk_by_month,
        how='left',
        left_on='MONTHLY REPORTING PERIOD',
        right_on='OBSERVATION_MONTH'
    )

    return all_df

def main():
    #download_data(IDS)
    combined_df = []
    for year in range(1999, 2001):
        monthly_avg = monthly_avg_data(year)
        combined_df.append(monthly_avg)

    df_combined = pd.concat(combined_df, ignore_index=True)
    df_combined = df_combined.groupby('MONTHLY REPORTING PERIOD').mean(numeric_only=True)
    df_combined = df_combined.reset_index()
    print(df_combined.columns[0])
    df_combined.to_csv('monthly_avg_data.csv', index=False)

if __name__ == '__main__':
    main()