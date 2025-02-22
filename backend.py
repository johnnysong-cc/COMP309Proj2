#region: import dependencies
import os, random, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.impute import SimpleImputer
#endregion


# region: Load and describe data elements (columns), providing descriptions, types, ranges, and values.

filename = 'Bicycle_Thefts_Open_Data.csv'       # change to parameter input later
filepath = os.getcwd() + '/data/' + filename
raw_data_df = pd.read_csv(filepath)

pd.set_option('display.max_columns', None)      # show all columns
# pd.set_option('display.max_rows', None)         # show all rows
pd.set_option('display.max_colwidth', None)     # show full column contents

# print(f'Head and Tail Rows: \n {raw_data_df.head()} \n {raw_data_df.tail()} \n\n')
# print(f'Shape of Data: \n{raw_data_df.shape}\n\n')
# print(f'Column Names: \n{raw_data_df.columns}\n\n')
# print(f'Data Types: \n{raw_data_df.dtypes}\n\n')
# print(f'Describe Data: \n{raw_data_df.describe(include="all")}\n\n')

# something interesting:
# df_grouped_by_bike_make = raw_data_df.groupby(['BIKE_MAKE','BIKE_MODEL','BIKE_SPEED','BIKE_COLOUR','BIKE_COST'])
# for name, group in df_grouped_by_bike_make:
#     print(f'BIKE_MAKE: {name}')
#     print((group.head(5)))


#endregion

# region: Perform statistical assessments, including means, averages, and correlations.

# Retrieve the statistic values for subsequent use
stat_values_raw = raw_data_df.describe(include="all")
means:pd.Series = stat_values_raw.loc["mean"]
stds:pd.Series = stat_values_raw.loc["std"]
qs: pd.Series = stat_values_raw.loc[["25%", "50%", "75%"]]
iqrs:pd.Series = stat_values_raw.loc["75%"] - stat_values_raw.loc["25%"]    # ;print(iqrs[iqrs.notna()],"\n")
min_values:pd.Series = stat_values_raw.loc["min"]
max_values:pd.Series = stat_values_raw.loc["max"]
ranges:pd.Series = max_values - min_values    # ;print(ranges[ranges.notna()],"\n")

# Correlation Matrix of Numeric Data
df_dropped = raw_data_df.drop(columns=['X','Y','OBJECTID','LONG_WGS84','LAT_WGS84'])      # the requirements suggest not analyzing the masked location data
corr_matrix_num_to_num:pd.DataFrame = df_dropped.select_dtypes(include=[np.number]).corr()
# print(corr_matrix_num_to_num)
corr_matrix_num_to_num_filtered = corr_matrix_num_to_num.where((corr_matrix_num_to_num > 0.5) | (corr_matrix_num_to_num < -0.5))
print(corr_matrix_num_to_num_filtered)

# corr_matrix_cat_to_num:pd.DataFrame = df_dropped.select_dtypes(include=[np.object]).corr()
# print(corr_matrix_cat_to_num)
# corr_matrix_cat_to_num_filtered = corr_matrix_cat_to_num.where((corr_matrix_cat_to_num > 0.5) | (corr_matrix_cat_to_num < -0.5))
# print(corr_matrix_cat_to_num_filtered)

# TODO: Correlation Matrix of Categorical Data? (even possible or necessary?)
# TODO: Correlation Matrix of Categorical-Numerical Data? (even possible or necessary?)

#endregion

# region: Evaluate missing data

# The number of missing data per column
missing_data_df = raw_data_df.isnull().sum().to_frame(name='missing_count')            # both sns and plt.imshow() require a dataframe
missing_data_df_filtered = missing_data_df[missing_data_df['missing_count']>0]
print(f'\nMissing Data: \n{missing_data_df_filtered}\n\n')

# Get the row indices of the missing data
missing_rows_bike_make = raw_data_df[raw_data_df['BIKE_MAKE'].isnull()].index.to_list()       # TODO: return the non-zero rows to avoid hard-coding
missing_rows_bike_model = raw_data_df[raw_data_df['BIKE_MODEL'].isnull()].index.to_list()
missing_rows_bike_speed = raw_data_df[raw_data_df['BIKE_SPEED'].isnull()].index.to_list()
missing_rows_bike_colour = raw_data_df[raw_data_df['BIKE_COLOUR'].isnull()].index.to_list()
missing_rows_bike_cost = raw_data_df[raw_data_df['BIKE_COST'].isnull()].index.to_list()
# print(f'\nMissing Rows in BIKE_MAKE: \n{missing_rows_bike_make}\n\n')
# print(f'\nMissing Rows in BIKE_MODEL: \n{missing_rows_bike_model}\n\n')
# print(f'\nMissing Rows in BIKE_SPEED: \n{missing_rows_bike_speed}\n\n')
# print(f'\nMissing Rows in BIKE_COLOUR: \n{missing_rows_bike_colour}\n\n')
# print(f'\nMissing Rows in BIKE_COST: \n{missing_rows_bike_cost}\n\n')

# Visualize the missing data
# sns.heatmap(missing_data_df, cmap='viridis', cbar=False)
plt.imshow(missing_data_df, cmap='viridis', aspect='auto', interpolation='nearest')
plt.show()

# Imputation for missing data
imputer_freq = SimpleImputer(strategy='most_frequent')
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_unknown = SimpleImputer(strategy='constant', fill_value='Unknown')

# For BIKE_MAKE: Most frequent
raw_data_df['BIKE_MAKE'] = imputer_freq.fit_transform(raw_data_df[['BIKE_MAKE']]).ravel()
# For BIKE_MODEL: Fill with a new category 'Unknown'
raw_data_df['BIKE_MODEL'] = imputer_unknown.fit_transform(raw_data_df[['BIKE_MODEL']]).ravel()  
# For BIKE_SPEED: Median
raw_data_df['BIKE_SPEED'] = imputer_median.fit_transform(raw_data_df[['BIKE_SPEED']]).ravel()     # TODO: discussion: median vs mean? which is better?
# For BIKE_COLOUR: Most frequent
raw_data_df['BIKE_COLOUR'] = imputer_freq.fit_transform(raw_data_df[['BIKE_COLOUR']]).ravel()
# For BIKE_COST: Mean
raw_data_df['BIKE_COST'] = imputer_mean.fit_transform(raw_data_df[['BIKE_COST']]).ravel()         # TODO: discussion: median vs mean? which is better?

missing_data_df_imputed = raw_data_df.isnull().sum().to_frame(name='missing_count')
plt.imshow(missing_data_df_imputed, cmap='viridis', aspect='auto', interpolation='nearest')
plt.show()  # All missing data have been imputed and the heatmap is now empty

# TODO: Transformation 1: De-duplicate the data ("OCC_DATE" vs "OCC_YEAR", "OCC_MONTH", ...; "REPORT_DATE" vs "REPORT_YEAR", "REPORT_MONTH", ...)
# TODO: Transformation 2: Convert "OCC_DATE" and "REPORT_DATE" to Unix timestamp
# TODO: Transformation 3: Convert categorical data to numerical data ("BIKE_MAKE" to "BIKE_MAKE_ID", "BIKE_MODEL" to "BIKE_MODEL_ID", "BIKE_COLOUR" to "BIKE_COLOUR_ID", ...)
# TODO: Transformation 4: Normalize numeric data ("BIKE_COST" to "BIKE_COST_NORMALIZED", "BIKE_SPEED" to "BIKE_SPEED_NORMALIZED")
# TODO: Transformation 5: Bin numeric data ("BIKE_COST" to "BIKE_COST_BIN", "BIKE_SPEED" to "BIKE_SPEED_BIN")

# endregion

# region: Perform data transformations, including normalization, binning, and discretization.

# endregion

# region: feature elimination

# endregion

# region: split data into training and testing sets

# endregion

# region: linear regression modeling

# endregion

# region: logistic regression modeling

# endregion

# region: decision tree modeling

# endregion

# region: clustering analysis

# endregion