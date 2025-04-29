import pandas as pd
import numpy as np
import gdown
import zipfile
import os
from concurrent.futures import ProcessPoolExecutor

IDS = ['1obUqshqVQR9mzKnUUOwOXELo2CeyDsYU','1ujt3l4HIiT6fMeeUlwX7r_pYIgAqsaKj','1teP3qIrxGK7Eh4CUYGAStADMVSnOm1mt',\
       '1n32-K_WKPaHnLDbuPG1Ke7rH-0a4nJJA','1LEeng4_shu_MX4Wz3iFsbCYp2OTLOnCV','1voganvmbmogpmOsyLX4zekHoJs2gnUQU',\
        '1JufHZvdLcwa-xQEunbgI-SQbZYd2XD99','12MLtLYbCNyjQ5i6gLsaqwai8jm3pXGXv','16GwxjwEor4XlOEVcQjnzDT11uBI2TnKe',\
        '17xs04GDzUC-c8ON120fItgxRLdhGYwzK','1ZbbW35Hib9jFA__L9d7TdFpWz9QRYCDf','1YkZOXCn6VQipQ63WUeM36FwZnAYBC0Nz',\
        '19ARzaT4Vvutk50lQY5H_fsuizbbBIkeb','1LFDXqeUx3mHlsyH_uq0s3KMzMYQ8OAgC','1jd9jvQX_Xv0LpRZH-Llqx3hHYgP1ETeG',\
            '1GQNmYtmk7d24o1Ylk33BzbbBBsmDRbZI','1IyjUlSzY0WIjpV3TJyJt2FT5qN6FoMdR','1-4HSTqCt-LQ1aiPWUWVsnPLQt5R_Dnxg',\
                '1zbE4vuRW2VZZcCEr1iyyXwezA1IWEz6G','1jaknuNTex_MnxhI_5pQ5LWnAMAShGdRM','1ZnmgZnRRbY8oW8Dfxt-3rV6a9o54TJvj',\
                    '1zZfh1vpnQE7iJrcOmt5vkZTytk1mIxfi','1uESSFgodNIqAibWDy_i0ZYJxdQCpAVpn','1_q3f5PEe0YwEtikTc1RVK_Q0ElcYMLeB',\
                        '1UWENCUVJfTgvxb8zx9zeYVTt7iGEnVos','13dymLaCeSeu0ysOq-M374QQPf8V8j1t2']

def download_and_extract(file_id):
    zip_path = f'data_{file_id}.zip'
    output_dir = f'data_extracted/{file_id}'
    os.makedirs(output_dir, exist_ok=True)

    gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_path, quiet=False)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)


def monthly_avg_data(year):
    path_prefix = f'data_extracted/{year}/'
    file1 = f'{path_prefix}sample_orig_{year}.txt'
    file2 = f'{path_prefix}sample_svcg_{year}.txt'

    df1 = pd.read_csv(file1, delimiter='|', low_memory=False)
    df2 = pd.read_csv(file2, delimiter='|', low_memory=False)

    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Clean and preprocess df1
    replacements = {
        'CREDIT SCORE': {9999: -1},
        'FIRST TIME HOMEBUYER FLAG': {'Y': 1, 'N': 0, '9': -1},
        'REPAYMENT PENALTY MORTGAGE (PPM) FLAG': {'Y': 1, 'N': 0},
        'INTEREST ONLY INDICATOR (I/O INDICATOR)': {'Y': 1, 'N': 0}
    }
    for col, mapping in replacements.items():
        if col in df1.columns:
            df1[col] = df1[col].map(mapping).fillna(df1[col])

    invalid_value_cols = {
        'MORTGAGE INSURANCE PERCENTAGE (MI %)': 999,
        'NUMBER OF UNITS': 99,
        'ORIGINAL COMBINED LOAN-TO-VALUE(CLTV)': 999,
        'ORIGINAL DEBT-TO-INCOME (DTI) RATIO': 999,
        'ORIGINAL LOAN-TO-VALUE(LTV)': 999,
        'NUMBER OF BORROWERS': 99
    }
    for col, inval in invalid_value_cols.items():
        if col in df1.columns:
            df1 = df1[df1[col] != inval]

    df1 = df1.drop(columns=[col for col in df1.columns if 'MSA' in col or 'VALUATION METHOD' in col or 'MI CANCELLATION' in col], errors='ignore')

    threshold = 0.9
    df2 = df2.loc[:, df2.isna().mean() < threshold]
    df2['CURRENT LOAN DELINQUENCY STATUS'] = df2['CURRENT LOAN DELINQUENCY STATUS'].replace('RA', -1).astype(int, errors='ignore')

    df_combined = df2.merge(df1, on='LOAN SEQUENCE NUMBER', how='left')
    df_combined['MONTHLY REPORTING PERIOD'] = pd.to_datetime(df_combined['MONTHLY REPORTING PERIOD'].astype(str), format='%Y%m', errors='coerce')
    df_combined.sort_values(['LOAN SEQUENCE NUMBER', 'MONTHLY REPORTING PERIOD'], inplace=True)

    df_combined['IS_DELINQUENT'] = df_combined['CURRENT LOAN DELINQUENCY STATUS'].ge(1)

    first_obs = df_combined.groupby('LOAN SEQUENCE NUMBER')['MONTHLY REPORTING PERIOD'].min()
    first_dq = df_combined[df_combined['IS_DELINQUENT']].groupby('LOAN SEQUENCE NUMBER')['MONTHLY REPORTING PERIOD'].min()

    merged = pd.DataFrame({
        'FIRST_REPORTING_DATE': first_obs,
        'FIRST_DELINQUENCY_DATE': first_dq
    })

    merged['TIME_TO_FIRST_DELINQUENCY_DAYS'] = (merged['FIRST_DELINQUENCY_DATE'] - merged['FIRST_REPORTING_DATE']).dt.days.fillna(9999)
    merged['OBSERVATION_MONTH'] = merged['FIRST_REPORTING_DATE'].dt.to_period('M').dt.to_timestamp()

    cohort_sizes = merged.groupby('OBSERVATION_MONTH').size().rename('num_loans')
    delinquency_counts = merged['TIME_TO_FIRST_DELINQUENCY_DAYS'].lt(9999).groupby(merged['OBSERVATION_MONTH']).sum().rename('num_delinquent_loans')
    delinquency_rate = (delinquency_counts / cohort_sizes).fillna(0).rename('pct_delinquent')
    ttd_stats = merged.groupby('OBSERVATION_MONTH')['TIME_TO_FIRST_DELINQUENCY_DAYS'].agg(
        avg_days_to_delinquency='mean',
        median_days_to_delinquency='median'
    )

    monthly_avg = df_combined.groupby('MONTHLY REPORTING PERIOD').mean(numeric_only=True).dropna(axis=1, how='all').reset_index()

    return cohort_sizes, delinquency_counts, delinquency_rate, ttd_stats, monthly_avg


def combine_metrics(cohort_sizes, delinquency_counts, delinquency_rate, ttd_stats, monthly_avg):
    metrics = pd.concat([cohort_sizes, delinquency_counts, delinquency_rate, ttd_stats], axis=1).reset_index()
    all_df = monthly_avg.merge(metrics, how='left', left_on='MONTHLY REPORTING PERIOD', right_on='OBSERVATION_MONTH')
    return all_df


def main():
    # with ProcessPoolExecutor() as executor:
    #     executor.map(download_and_extract, IDS)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(monthly_avg_data, range(1999, 2025)))

    cohort_sizes, delinquency_counts, delinquency_rate, ttd_stats, monthly_avg = zip(*results)

    df_combined_all = pd.concat(monthly_avg)
    df_combined_cohort_sizes = pd.concat(cohort_sizes)
    df_combined_delinquency_counts = pd.concat(delinquency_counts)
    df_combined_delinquency_rate = pd.concat(delinquency_rate)
    df_combined_ttd_stats = pd.concat(ttd_stats)

    df_combined_all = combine_metrics(
        df_combined_cohort_sizes,
        df_combined_delinquency_counts,
        df_combined_delinquency_rate,
        df_combined_ttd_stats,
        df_combined_all
    )

    df_combined_all = df_combined_all.groupby('MONTHLY REPORTING PERIOD').mean(numeric_only=True).reset_index()
    df_combined_all.to_csv('monthly_avg_data.csv', index=False)


if __name__ == '__main__':
    main()