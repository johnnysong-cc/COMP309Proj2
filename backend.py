#region: import dependencies
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, random
#endregion


# region: Load and describe data elements (columns), providing descriptions, types, ranges, and values.
filename = 'Bicycle_Thefts_Open_Data.csv'     # change to parameter input later
filepath = os.getcwd() + '/data/' + filename
raw_data_df = pd.read_csv(filepath)

print(f'Head and Tail Rows: \n {raw_data_df.head()} \n {raw_data_df.tail()} \n\n')
print(f'Shape of Data: \n{raw_data_df.shape}\n\n')
print(f'Column Names: \n{raw_data_df.columns}\n\n')
print(f'Data Types: \n{raw_data_df.dtypes}\n\n')
print(f'Describe Data: \n{raw_data_df.describe(include="all")}\n\n')

# a little more details for overview
# print(raw_data_df[['LOCATION_TYPE', 'STATUS', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140', 'NEIGHBOURHOOD_140', 'LONG_WGS84', 'LAT_WGS84']].head())

# ranges of each column


#endregion
