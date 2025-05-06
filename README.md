# Mortgage Data Analysis Project

This project analyzes mortgage data from 1999 to 2024, focusing on loan origination and servicing data. The project includes data cleaning, processing, and exploratory data analysis (EDA) components.

## Project Structure

- `data_cleaning.py`: Main script for downloading, cleaning, and processing mortgage data
- `eda.ipynb`: Jupyter notebook containing exploratory data analysis
- `monthly_avg_data.csv`: Processed monthly average data
- `data_extracted/`: Directory containing raw data files
- `data.zip`: Compressed data file (not tracked in git)
- `Geographic_Risk_Analysis_Default Rate by State.ipynb`: Clustering and visualization of mortgage default rates
- `Geographic_Risk_Analysis_Average Actual Loss.ipynb` : Analysis of financial impact by state

## Data Sources

The project uses mortgage data from multiple sources, including:
- Loan origination data (`sample_orig_*.txt`)
- Loan servicing data (`sample_svcg_*.txt`)

## Features

- Data downloading and extraction from Google Drive
- Comprehensive data cleaning and preprocessing
- Monthly aggregation of loan metrics
- Exploratory data analysis of mortgage trends
- State-level risk segmentation using default rate and actual loss
- Clustering states using KMeans to categorize mortgage risk
- Visualization of regional risk using maps and bar charts

## Data Processing

The data processing pipeline includes:
1. Downloading and extracting raw data files
2. Cleaning and standardizing data formats
3. Handling missing values and invalid entries
4. Computing monthly averages for key metrics
5. Merging origination and servicing data

## Key Metrics

The analysis includes various mortgage-related metrics such as:
- Credit scores
- Loan-to-value ratios
- Interest rates
- Delinquency status
- Property types
- Borrower information
- KMeans clustering for risk categorization

## Requirements

- Python 3.x
- pandas
- numpy
- gdown
- Jupyter Notebook

## Usage

1. Clone the repository
2. Install required dependencies
3. Run `data_cleaning.py` and 'data_cleaning_geographic risk analysis_default rate.py' to process the data
4. Open `eda.ipynb` to explore the analysis
5. Open `Geographic_Risk_Analysis_Default Rate by State.ipynb` for Default Rate by State
6. Open `Geographic_Risk_Analysis_Average Actual Loss.ipynb` for Average Actual Loss

## License

This project is licensed under the terms included in the LICENSE file.