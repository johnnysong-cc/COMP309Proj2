# region: Import dependencies

import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# endregion

# region: Data collection and general cleansing

filename = 'Bicycle_Thefts_Open_Data.csv'  # change to parameter input later
filepath = os.getcwd() + '/data/' + filename
raw_data_df = pd.read_csv(filepath)

# region: pd display options
pd.set_option('display.max_columns', None)  # show all columns
pd.option_context('display.max_rows', None)  # show all rows
pd.set_option('display.max_colwidth', None)  # show full column contents
# pd.set_option('display.width', None) # use the size of the console window returned from os.get_terminal_size()
# use the size of the console window returned from os.get_terminal_size()
pd.set_option('display.width', 240)
# suppress scientific notation as this dataset is not that large
pd.set_option('display.float_format', lambda x: '%.4f' % x)
# endregion

# region: print data description in the console
# print(f'\nShape of Data: \n  - Number of records: {raw_data_df.shape[0]}\n  - Number of Columns (Features): {raw_data_df.shape[1]}\n')
# print(f'\nHead and Tail Rows: \n {raw_data_df.head()} \n\n {raw_data_df.tail()}\n')
# [print(f'Column Name: {colName} \t Data Type: {colType}') for colName,colType in zip(raw_data_df.columns, raw_data_df.dtypes)]
# print(f'\nDescribe Data: \n{raw_data_df.describe(include="all")}\n')
# endregion

# region: data cleansing - general cleansing phase
columns_to_drop = []
"""
As per the requirements, we are not analyzing the masked location data
However, the types of locations are still useful
"""
columns_to_drop_positional = ['X', 'Y', 'HOOD_158', 'HOOD_140',
                              'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140', 'DIVISION', 'LONG_WGS84', 'LAT_WGS84']
columns_to_drop.extend(columns_to_drop_positional)
# data_cleansed_general: pd.DataFrame = raw_data_df.drop(
#     columns=['X', 'Y', 'HOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140', 'DIVISION', 'LONG_WGS84', 'LAT_WGS84'])

"""
Convert date/time columns to datetime type of unix timestamps
"""
# Convert months to numbers
month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
raw_data_df['OCC_MONTH'] = raw_data_df['OCC_MONTH'].map(month_mapping)
raw_data_df['REPORT_MONTH'] = raw_data_df['REPORT_MONTH'].map(month_mapping)
# data_cleansed_general['OCC_MONTH'] = data_cleansed_general['OCC_MONTH'].map(
#     month_mapping)  # ; print(data_cleansed1['OCC_MONTH'].head())
# data_cleansed_general['REPORT_MONTH'] = data_cleansed_general['REPORT_MONTH'].map(
#     month_mapping)  # ; print(data_cleansed1['REPORT_MONTH'].head())

# Convert the OCC_TIMESTAMP column to datetime type of unix timestamps
raw_data_df['OCC_TIMESTAMP'] = pd.to_datetime(
    raw_data_df['OCC_YEAR'].astype(str)
    + '-' + raw_data_df['OCC_MONTH'].astype(str)
    + '-' + raw_data_df['OCC_DAY'].astype(str)
    + ' ' + raw_data_df['OCC_HOUR'].astype(str),
    format='%Y-%m-%d %H', errors='coerce')
raw_data_df['OCC_TIMESTAMP'] = raw_data_df['OCC_TIMESTAMP'].astype(
    np.int64) / 10**9       # convert to unix timestamp

raw_data_df['REPORT_TIMESTAMP'] = pd.to_datetime(
    raw_data_df['REPORT_YEAR'].astype(str)
    + '-' + raw_data_df['REPORT_MONTH'].astype(str)
    + '-' + raw_data_df['REPORT_DAY'].astype(str)
    + ' ' + raw_data_df['REPORT_HOUR'].astype(str),
    format='%Y-%m-%d %H', errors='coerce')
raw_data_df['REPORT_TIMESTAMP'] = raw_data_df['REPORT_TIMESTAMP'].astype(
    np.int64) / 10**9  # convert to unix timestamp

# Drop the original date/time columns
columns_to_drop_temporal = ['OCC_DATE', 'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_HOUR', 'OCC_DOW', 'OCC_DOY',
                            'REPORT_DATE', 'REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY', 'REPORT_HOUR', 'REPORT_DOW', 'REPORT_DOY']
columns_to_drop.extend(columns_to_drop_temporal)

"""
OBJECTID is simply a sequence from 1 to the number of records, which can be replaced with the index, so we drop it
"""
columns_to_drop_misc = ['OBJECTID']
columns_to_drop.extend(columns_to_drop_misc)

"""
TODO: merge similar PRIMARY_OFFENCES, e.g., 'B&E', 'B&E OUT' & 'B&E W'INTENT' into 'B&E'
"""


# At the end of general cleansing, drop the columns
data_cleansed_general: pd.DataFrame = raw_data_df.drop(columns=columns_to_drop)
data_cleansed_general.reset_index(drop=True, inplace=True)

# endregion

# region: print cleansed data description in the console

print(
    f'\nShape of the cleansed dataset:\n  - Number of records: {data_cleansed_general.shape[0]}\n  - Number of Columns (Features): {data_cleansed_general.shape[1]}\n')
print(
    f'\nHead and tail rows of the cleansed dataset:\n{data_cleansed_general.head()}\n\n{data_cleansed_general.tail()}\n')
[print(f'Column Name: {colName} \t Data Type: {colType}') for colName, colType in zip(
    data_cleansed_general.columns, data_cleansed_general.dtypes)]
print(f'\nDescribe Data: \n{data_cleansed_general.describe(include="all")}\n')
print(f'\nCheck for missing data:\n{data_cleansed_general.isnull().sum()}\n')

# endregion

# region: get high-level statistics for later use

stat_values_raw = raw_data_df.describe(
    include="all").drop(columns=["EVENT_UNIQUE_ID"])
stat_values_cleansed_general = data_cleansed_general.describe(
    include="all").drop(columns=["EVENT_UNIQUE_ID"])

means: pd.Series = stat_values_cleansed_general.loc["mean"].astype(float).to_frame()\
    .select_dtypes(include=[np.number]).drop(['OCC_TIMESTAMP', 'REPORT_TIMESTAMP']).dropna()  # ; print(f'\nMeans:\n{means}\n')

stds: pd.Series = stat_values_cleansed_general.loc["std"].astype(float).to_frame()\
    .select_dtypes(include=[np.number]).drop(['OCC_TIMESTAMP', 'REPORT_TIMESTAMP']).dropna()  # ; print(f'\nStd:\n{stds}\n')

mins: pd.Series = stat_values_cleansed_general.loc["min", [
    "BIKE_SPEED", "BIKE_COST"]].astype(float).to_frame()  # ; print(f'\nMin:\n{mins}\n')

iqr25s: pd.Series = stat_values_cleansed_general.loc["25%", [
    "BIKE_SPEED", "BIKE_COST"]].astype(float).to_frame()  # ; print(f'\n25%:\n{iqr25s}\n')

medians: pd.Series = stat_values_cleansed_general.loc["50%", [
    "BIKE_SPEED", "BIKE_COST"]].astype(float).to_frame()  # ; print(f'\n50%:\n{medians}\n')

iqr75s: pd.Series = stat_values_cleansed_general.loc["75%", [
    "BIKE_SPEED", "BIKE_COST"]].astype(float).to_frame()  # ; print(f'\n75%:\n{iqr75s}\n')

maxes: pd.Series = stat_values_cleansed_general.loc["max", [
    "BIKE_SPEED", "BIKE_COST"]].astype(float).to_frame()
print(f'\nMax:\n{maxes}\n')

tops: pd.DataFrame = pd.concat([stat_values_raw.loc["top"], stat_values_raw.loc["freq"]], axis=1, keys=[
                               'Top case', 'Freq']).dropna()  # ; print(f'\nTop:\n{tops}\n')

# ; print(f'\nUnique:\n{nr_unique_values_per_column}\n')
nr_unique_values_per_column: pd.Series = data_cleansed_general.nunique()

unique_values_per_column: pd.DataFrame = data_cleansed_general.drop(columns=['EVENT_UNIQUE_ID', 'OCC_TIMESTAMP', 'REPORT_TIMESTAMP']).apply(
    lambda x: x.unique())  # ; print(f'\nUnique:\n{unique_values_per_column}\n')

# endregion

# endregion

# region: Preprocessing data

# region: Imputation for missing values

# ; print(f'\nMissing:\n{missing_counts_dict}\n')
missing_counts_dict = data_cleansed_general.isnull().sum().to_dict()
missing_rowIndexes_dict = {k: data_cleansed_general[data_cleansed_general[k].isnull()].index.to_list(
) for k in data_cleansed_general.columns if data_cleansed_general[k].isnull().any()}

imputer_freq = SimpleImputer(strategy='most_frequent')
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_unknown = SimpleImputer(strategy='constant', fill_value='Unknown')

data_cleansed_general['BIKE_MAKE'] = imputer_freq.fit_transform(
    data_cleansed_general[['BIKE_MAKE']]).ravel()       # For BIKE_MAKE: Most frequent
data_cleansed_general['BIKE_MODEL'] = imputer_unknown.fit_transform(
    data_cleansed_general[['BIKE_MODEL']]).ravel()  # For BIKE_MODEL: Fill with a new category 'Unknown'
data_cleansed_general['BIKE_SPEED'] = imputer_median.fit_transform(
    data_cleansed_general[['BIKE_SPEED']]).ravel()   # For BIKE_SPEED: Median
data_cleansed_general['BIKE_COLOUR'] = imputer_freq.fit_transform(
    data_cleansed_general[['BIKE_COLOUR']]).ravel()   # For BIKE_COLOUR: Most frequent
data_cleansed_general['BIKE_COST'] = imputer_mean.fit_transform(
    data_cleansed_general[['BIKE_COST']]).ravel()       # For BIKE_TYPE: Most frequent

"""
check for missing data again
"""
print('Imputation is successful' if data_cleansed_general.isnull(
).sum().all() == 0 else 'Imputation is not successful')

# endregion

# region: Normalize numeric data ("BIKE_COST" to "BIKE_COST_NORMALIZED", "BIKE_SPEED" to "BIKE_SPEED_NORMALIZED")

scaler = preprocessing.StandardScaler()
data_cleansed_general['BIKE_COST_NORMALIZED'] = scaler.fit_transform(
    data_cleansed_general[['BIKE_COST']])
data_cleansed_general['BIKE_SPEED_NORMALIZED'] = scaler.fit_transform(
    data_cleansed_general[['BIKE_SPEED']])
# data_cleansed_general.drop(columns=['BIKE_COST','BIKE_SPEED','EVENT_UNIQUE_ID'], inplace=True)
"""
check normalized data
"""
print(
    f'\nDescribe Normalized Data: \n{data_cleansed_general.describe(include="all")}\n')

# 578, 0.5%. Can be dropped when predicting if the bike is stolen or not
print((data_cleansed_general[data_cleansed_general['STATUS'] == 'UNKNOWN'])[
      'STATUS'].count())

# endregion

# endregion

# region: Fundamental visualizations

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# fig.tight_layout(pad=3.0)
fig.suptitle('Fundamental Visualizations')

# region: histogram of bike cost

bike_cost_log = np.log(data_cleansed_general[data_cleansed_general['BIKE_COST'] > 0]['BIKE_COST'])/np.log(10)
sns.histplot(bike_cost_log, bins=20, color='blue', alpha=0.7, kde=True, ax=axes[0, 0])
axes[0, 0].set_xticks(np.arange(0, 6, 1))
axes[0, 0].set_title('Bike Cost Distribution (Imputed))')
axes[0, 0].set_xlabel('Log10 of Bike Cost')
axes[0, 0].set_ylabel('Frequency')

# endregion

# region: histogram of bike speed

sns.histplot(data_cleansed_general[data_cleansed_general['BIKE_SPEED'] > 0]['BIKE_SPEED'], bins=20, color='green', alpha=0.7, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Bike Speed Distribution (Imputed)')
axes[0, 1].set_xlabel('Bike Speed')
axes[0, 1].set_ylabel('Frequency')

# endregion

# region: line plot of occurrence over time (year)

data_temp = data_cleansed_general.copy()
data_temp['case_year'] = pd.to_datetime(data_temp['OCC_TIMESTAMP'], unit='s').dt.year
# cases_per_year = data_temp['case_year'].value_counts()  # Found some outlier years with very few cases
# cases_per_year = data_temp[data_temp['case_year']>=2013].value_counts() # .value_counts() applies to Series only
cases_per_year = data_temp[data_temp['case_year'] >= 2013]['case_year'].value_counts().sort_index()
print(cases_per_year)
sns.lineplot(x=cases_per_year.index, y=cases_per_year.values, color='red', markers='o', ax=axes[1, 0])
axes[1, 0].set_title('Number of Cases Per Year')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Number of Cases')

# endregion

# region: line plot of occurrence over time (month)

data_temp = data_cleansed_general.copy()
data_temp['case_month'] = pd.to_datetime(data_temp['OCC_TIMESTAMP'], unit='s').dt.month
cases_per_month = data_temp[data_temp['case_month']>=1]['case_month'].value_counts().sort_index()
print(cases_per_month)
sns.barplot(x=cases_per_month.index, y=cases_per_month.values, ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title('Number of Cases Per Month')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Number of Cases')

# endregion

plt.show()

# endregion

# region: Correlation Analysis of Numeric Data

print(data_cleansed_general.columns)
correlation_matrix = data_cleansed_general.select_dtypes(include=[np.number]).corr().where(
    (data_cleansed_general.select_dtypes(include=[np.number]).corr() > 0.5) | (data_cleansed_general.select_dtypes(include=[np.number]).corr() < -0.5))
print(correlation_matrix)

# endregion

# region: Chi-Square Analysis of Independence for Categorical Data

cat_columns = data_cleansed_general.select_dtypes(
    include=['object']).drop(columns=['EVENT_UNIQUE_ID']).columns
print(cat_columns)
chi_square_results = {}

for i in range(len(cat_columns)):
    for j in range(i+1, len(cat_columns)):
        col1_name = cat_columns[i]
        col2_name = cat_columns[j]

        contingency_table = pd.crosstab(
            data_cleansed_general[col1_name], data_cleansed_general[col2_name])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        chi_square_results[(col1_name, col2_name)] = {
            'chi2': chi2, 'p': p, 'dof': dof}

chi_square_results_df = pd.DataFrame(chi_square_results).T
chi_square_results_df['p'] = chi_square_results_df['p'].apply(
    lambda x: round(x, 4))
print(f'\nChi-Square Results: \n{chi_square_results_df}')

"""
Conclusion from Chi-Square Test of Independence:
                               chi2      p           dof
BIKE_MODEL      STATUS   23533.5318 0.0000    21092.0000 (Too big to be useful, due to too many categories in BIKE_MODEL)
BIKE_MAKE       STATUS    6031.1831 0.0000     2278.0000 
PREMISES_TYPE   STATUS     139.1673 0.0000       12.0000
LOCATION_TYPE   STATUS    2083.3365 0.0000       90.0000
BIKE_TYPE       STATUS     164.9892 0.0000       24.0000
PRIMARY_OFFENCE STATUS   23654.4881 0.0000      150.0000 
BIKE_COLOUR     STATUS    1992.7366 0.0000      562.0000
"""

# endregion

# region: Label encoding for categorical data ("BIKE_MAKE", "BIKE_MODEL", "BIKE_COLOUR", "BIKE_TYPE", "PREMISES_TYPE", "LOCATION_TYPE", "PRIMARY_OFFENCE")

categorical_columns = ['BIKE_MAKE', 'BIKE_MODEL', 'BIKE_COLOUR',
                       'BIKE_TYPE', 'PREMISES_TYPE', 'LOCATION_TYPE', 'PRIMARY_OFFENCE', 'STATUS']
bicycle_thefts_data_regr = data_cleansed_general.copy()
# data_encoded = pd.get_dummies(data_cleansed_general, columns=categorical_columns) # this would create 12119 columns

# For BIKE_MAKE:
# 1. map as specified:
type_map = {"UNKNOWN MAKE": "UNKNOWN", "KONA\\": "KONA", "GI": "GIANT", "GIAN": "GIANT", "EM": "EMMO",
            "CC": "CCM", "CA": "CANNONDALE", "BI": "BIANCHI", "FJ": "FUJI", "IN": "INFINITY", "KH": "KHS",
            "MARIN": "MARIN OR MARINO", "MO": "MONGOOSE", "NO": "NORCO", "PE": "PEUGEOT", "RA": "RALEIGH",
            "RM": "ROCKY MOUNTAIN", "SC": "SCHWINN", "SP": "SPECIALIZED", "SPEC": "SPECIALIZED", "SU": "SUPERCYCLE", "TR": "TREK",
            "OT": "Other", "OTHE": "Other", "UNKNOWN": "Other", "Unknown": "Other", "UK": "Other", "UNK": "Other", "UNKNOWN MAKE": "Other", "OTHER": "Other", }
# There are still OTHER in the data, so we need to convert to upper case first
bicycle_thefts_data_regr['BIKE_MAKE'] = bicycle_thefts_data_regr['BIKE_MAKE'].str.upper()
bicycle_thefts_data_regr['BIKE_MAKE'] = bicycle_thefts_data_regr['BIKE_MAKE'].map(
    type_map).fillna(bicycle_thefts_data_regr['BIKE_MAKE'])
# 2. Get the top 10 most frequent values and group the rest into 'Other'
bicycle_thefts_data_regr['BIKE_MAKE'] = bicycle_thefts_data_regr['BIKE_MAKE'].apply(lambda x: 'Other' if
                                                                                    x not in bicycle_thefts_data_regr['BIKE_MAKE'].value_counts().index[:10] and (x != 'Other') else x)

# For BIKE_MODEL:
# 1. map as specified:
model_map = {"HARD ROCK": "HARDROCK", "UNKNOWN": "Other", "NONE": "Other",
             "U/K": "Other", "UNK": "Other", "Unknown": "Other", "UNKN": "Other", "OTHER": "Other"}
bicycle_thefts_data_regr['BIKE_MODEL'] = bicycle_thefts_data_regr['BIKE_MODEL'].map(
    model_map).fillna(bicycle_thefts_data_regr['BIKE_MODEL'])
# 2. Get the top 10 most frequent values and group the rest into 'Other'
bicycle_thefts_data_regr['BIKE_MODEL'] = bicycle_thefts_data_regr['BIKE_MODEL'].apply(lambda x: 'Other' if
                                                                                      x not in bicycle_thefts_data_regr['BIKE_MODEL'].value_counts().index[:10] else x)

# For BIKE_COLOUR:
# 1. map as specified:
color_map = {"TEAL": "BLU", "TURQ": "BLU", "TURQUOISE": "BLU", "DBLLBL": "BLU", "WHT": "WHI", "DARK": "BLK", "GREEN": "GRN", "DGR": "GREEN",
             "OTH": "Other", "UNKNOWN": "Other", "Unknown": "Other", 18: "Other", "OTHER": "Other"}
bicycle_thefts_data_regr['BIKE_COLOUR'] = bicycle_thefts_data_regr['BIKE_COLOUR'].map(
    color_map).fillna(bicycle_thefts_data_regr['BIKE_COLOUR'])
# 2. trim color code
bicycle_thefts_data_regr['BIKE_COLOUR'] = bicycle_thefts_data_regr['BIKE_COLOUR'].astype(str).apply(lambda x: x[0:3] if
                                                                                                    (len(x) > 3) and (x != 'Other') else x)
# map again after trimming
bicycle_thefts_data_regr['BIKE_COLOUR'] = bicycle_thefts_data_regr['BIKE_COLOUR'].map(
    color_map).fillna(bicycle_thefts_data_regr['BIKE_COLOUR'])
# 3. get the top 10 most frequent values and group the rest into 'Other'
bicycle_thefts_data_regr['BIKE_COLOUR'] = bicycle_thefts_data_regr['BIKE_COLOUR'].apply(lambda x: 'Other' if
                                                                                        x not in bicycle_thefts_data_regr['BIKE_COLOUR'].value_counts().nlargest(10).index else x)

# For BIKE_TYPE: 13 types seems manageable, so we will keep them all for now
# For PREMISES_TYPE: Educational and Commercial are both public places, and Transit can also be seen as Outside given its small number, so we will merge them
premises_map = {"Commercial": "Public Places",
                "Educational": "Public Places", "Transit": "Outside"}
bicycle_thefts_data_regr['PREMISES_TYPE'] = bicycle_thefts_data_regr['PREMISES_TYPE'].map(
    premises_map).fillna(bicycle_thefts_data_regr['PREMISES_TYPE'])

# For LOCATION_TYPE: 90 types seems too many, and they are derived from PREMISES_TYPE, so we give up on this column for now as the accuracy does not have to be this granular


# For PRIMARY_OFFENCE: 150 types seems too many, and a lot of them are similar, so we will merge them
"""
1. index that contains:
   - "THEFT" and "OVER" belong to "THEFT OVER", 
   - "THEFT" and "UNDER" belong to all "THEFT UNDER", 
   - "DRUG" belong to all "DRUGS",
   - "INCIDENT" belong to "INCIDENT", 
   - "MISCHIEF" belong to "MISCHIEF", 
   - "POSSESSION" or "TENANT" or "HOUSE" belong to "POSSESSION",
   - "PROPERTY" or "DAMAGE" belong to all "PROPERTY",
   - "ROBBERY" or "THREAT" or "ASSAULT" or "ARMED" or "FIRE" or "WEAPON" belong to "ROBBERY", 
   - "FRAUD" or "FORGERY" belong to "FRAUD", 
   - "B&E" or "TRESPASS" belong to all "B&E",
"""
replacement_dict = {
    r'.*THEFT.*OVER.*': 'THEFT OVER',
    r'.*THEFT OF MOTOR VEHICLE.*': 'THEFT OVER',
    r'.*THEFT.*UNDER.*': 'THEFT UNDER',
    r'.*DRUG.*': 'DRUGS',
    r'.*INCIDENT.*': 'INCIDENT',
    r'.*MISCHIEF.*': 'MISCHIEF',
    r'.*(POSSESSION|TENANT|HOUSE).*': 'POSSESSION',
    r'.*(PROPERTY|DAMAGE).*': 'PROPERTY',
    r'.*(ROBBERY|THREAT|ASSAULT|ARMED|WEAPON|FIRE).*': 'ROBBERY',
    r'.*(FRAUD|FORGERY).*': 'FRAUD',
    r'.*(B&E|TRESPASS).*': 'B&E',
    # Failure to Comply does not imply a specific offence type
    r'.*(FTC|OTHER).*': 'Other',
}
bicycle_thefts_data_regr['PRIMARY_OFFENCE'] = bicycle_thefts_data_regr['PRIMARY_OFFENCE'].replace(
    replacement_dict, regex=True)
# 2. Manual mapping
# pri_offence_map = {"THEFT OF MOTOR VEHICLE":"THEFT OVER",""}
# 3. Get the top 10 most frequent values and group the rest into 'Other'
bicycle_thefts_data_regr['PRIMARY_OFFENCE'] = bicycle_thefts_data_regr['PRIMARY_OFFENCE'].apply(lambda x: 'Other' if
                                                                                                x not in bicycle_thefts_data_regr['PRIMARY_OFFENCE'].value_counts().nlargest(10).index else x)

# For STATUS: 3 types seems manageable, it is the prediction target, so we convert it to numeric values

# FINAL CHECK
"""
sort_values() is used to sort the values in descending order by default
sort_index() is used to sort the index in ascending order by default
# demo = bicycle_thefts_data_regr['PRIMARY_OFFENCE'].value_counts().head(500)
# demo.sort_values(inplace=True)
# demo.sort_index(ascending=True, inplace=True)
"""
print(f'\n{bicycle_thefts_data_regr["BIKE_MAKE"].value_counts()}')
print(f'\n{bicycle_thefts_data_regr["BIKE_MODEL"].value_counts()}')
print(f'\n{bicycle_thefts_data_regr["BIKE_COLOUR"].value_counts()}')
print(f'\n{bicycle_thefts_data_regr["BIKE_TYPE"].value_counts()}')
print(f'\n{bicycle_thefts_data_regr["PRIMARY_OFFENCE"].value_counts()}')
print(f'\n{bicycle_thefts_data_regr["PREMISES_TYPE"].value_counts()}')
print(f'\n{bicycle_thefts_data_regr["STATUS"].value_counts()}')

print(bicycle_thefts_data_regr.info())
print(bicycle_thefts_data_regr.head(50).to_string())

# endregion

# region: export cleansed data to csv

data_cleansed_general.to_csv('data/bicycle_thefts_data_cleansed_general.csv', index=False) # index=False to avoid adding the index column
bicycle_thefts_data_regr.to_csv('data/bicycle_thefts_data_regr.csv', index=False)

# endregion