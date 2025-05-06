import pandas as pd
import os
import gdown
import zipfile

# ---------------------------
# Step 1: Download Data Safely 
# ---------------------------

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
    working_dir = os.getcwd()
    extract_folder = os.path.join(working_dir, 'data_extracted')

    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    for i, file_id in enumerate(ids):
        zip_path = os.path.join(working_dir, f"data_{i}.zip")

        if not os.path.exists(zip_path):
            print(f"Downloading file {i+1}/{len(ids)}...")
            gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_path, quiet=False)
        else:
            print(f"File {zip_path} already exists — skipping download.")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
    
    print(f"✅ All files extracted to: {extract_folder}")


# ---------------------------
# Step 2: Data Cleaning for Default Rate by State
# ---------------------------

def clean_data_for_default_rate(year):
    file1 = os.path.join(os.getcwd(), "data_extracted", f"sample_orig_{year}.txt")
    file2 = os.path.join(os.getcwd(), "data_extracted", f"sample_svcg_{year}.txt")

    print("\n🔍 Looking for:")
    print(" -", file1)
    print(" -", file2)

    try:
        df1_cols = ['CREDIT SCORE', 'FIRST PAYMENT DATE', 'FIRST TIME HOMEBUYER FLAG', 'MATURITY DATE',
                'METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION', 'MORTGAGE INSURANCE PERCENTAGE (MI %)',
                'NUMBER OF UNITS', 'OCCUPANCY STATUS', 'ORIGINAL COMBINED LOAN-TO-VALUE(CLTV)',
                'ORIGINAL DEBT-TO-INCOME (DTI) RATIO', 'ORIGINAL UPB', 'ORIGINAL LOAN-TO-VALUE(LTV)',
                'ORIGINAL INTEREST RATE', 'CHANEL', 'REPAYMENT PENALTY MORTGAGE (PPM) FLAG', 'AMORTIZATION TYPE',
                'PROPERTY STATE', 'PROPERTY TYPE', 'POSTAL CODE', 'LOAN SEQUENCE NUMBER', 'LOAN PURPOSE',
                'ORIGINAL LOAN TERM', 'NUMBER OF BORROWERS', 'SELLER NAME', 'SERVICER NAME',
                'SUPER CONFORMING FLAG', 'PRE-RELIEF REFINANCE LOAN SEQUENCE NUMBER', 'PROGRAM INDICATOR',
                'RELIEF REFINANCE INDICATOR', 'PROPERTY VALUATION METHOD', 'INTEREST ONLY INDICATOR (I/O INDICATOR)',
                'MI CANCELLATION INDICATOR']

        df2_cols = ['LOAN SEQUENCE NUMBER', 'MONTHLY REPORTING PERIOD', 'CURRENT ACTUAL UPB',
                'CURRENT LOAN DELINQUENCY STATUS', 'LOAN AGE', 'REMAINING MONTHS TO LEGAL MATURITY',
                'DEFECT SETTLEMENT DATE', 'MODIFICATION FLAG', 'ZERO BALANCE CODE',
                'ZERO BALANCE EFFECTIVE DATE', 'CURRENT INTEREST RATE', 'CURRENT NON-INTEREST BEARING UPB',
                'DUE DATE OF LAST PAID INSTALLMENT (DDLPI)', 'MI RECOVERIES', 'NET SALE PROCEEDS',
                'NON MI RECOVERIES', 'TOTAL EXPENSES', 'LEGAL COSTS', 'MAINTENANCE AND PRESERVATION COSTS',
                'TAXES AND INSURANCE', 'MISCELLANEOUS EXPENSES', 'ACTUAL LOSS CALCULATION',
                'CUMULATIVE MODIFICATION COST', 'STEP MODIFICATION FLAG', 'PAYMENT DEFERRAL',
                'ESTIMATED LOAN TO VALUE (ELTV)', 'ZERO BALANCE REMOVAL UPB', 'DELINQUENT ACCRUED INTEREST',
                'DELINQUENCY DUE TO DISASTER', 'BORROWER ASSISTANCE STATUS CODE',
                'CURRENT MONTH MODIFICATION COST', 'INTEREST BEARING UPB']

        # Load data
        df1 = pd.read_csv(file1, names=df1_cols, delimiter='|')
        df2 = pd.read_csv(file2, names=df2_cols, delimiter='|')

    except Exception as e:
        print(f"❌ Error loading files for year {year}: {e}")
        return

    # Clean delinquency status
    df2["CURRENT LOAN DELINQUENCY STATUS"] = df2["CURRENT LOAN DELINQUENCY STATUS"].apply(
        lambda x: -1 if str(x).strip().upper() == 'RA' else int(x)
    )

    # Convert MONTHLY REPORTING PERIOD
    df2["MONTHLY REPORTING PERIOD"] = pd.to_datetime(df2["MONTHLY REPORTING PERIOD"].astype(str), format="%Y%m")

    # Merge with origination data
    df_combined = pd.merge(
        df2,
        df1[["LOAN SEQUENCE NUMBER", "PROPERTY STATE"]],
        on="LOAN SEQUENCE NUMBER",
        how="left"
    )

    # Mark delinquents
    df_combined["IS_DELINQUENT"] = df_combined["CURRENT LOAN DELINQUENCY STATUS"] >= 1

    # Save cleaned file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f"loan_level_data_{year}.csv")

    df_combined.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")


# ---------------------------
# Step 3: Main
# ---------------------------

if __name__ == "__main__":
    # Uncomment to download data (if needed)
    download_data(IDS)

    for year in range(2014, 2025):  
        clean_data_for_default_rate(year)
