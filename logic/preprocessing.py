# region: Import dependencies
import numpy as np, pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
# endregion: Import dependencies

def data_cleansing_general(data: np.DataFrame, save: bool = False, filename: str = ''):
  columns_to_drop = []

  """
  Avoid masked location data
  """
  columns_to_drop_positional = ['X', 'Y', 'HOOD_158', 'HOOD_140',
                                'NEIGHBOURHOOD_158', 'NEIGHBOURHOOD_140', 'DIVISION', 'LONG_WGS84', 'LAT_WGS84']

  """
  Convert months to numbers
  """
  month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                   'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
  data['OCC_MONTH'] = data['OCC_MONTH'].map(month_mapping)
  data['REPORT_MONTH'] = data['REPORT_MONTH'].map(month_mapping)

  """
  Convert date/time columns to datetime type of unix timestamps
  """
  data['OCC_TIMESTAMP'] = pd.to_datetime(
      data['OCC_YEAR'].astype(str)
      + '-' + data['OCC_MONTH'].astype(str)
      + '-' + data['OCC_DAY'].astype(str)
      + ' ' + data['OCC_HOUR'].astype(str),
      format='%Y-%m-%d %H', errors='coerce').astype(np.int64) // 10 ** 9

  data['REPORT_TIMESTAMP'] = pd.to_datetime(
      data['REPORT_YEAR'].astype(str)
      + '-' + data['REPORT_MONTH'].astype(str)
      + '-' + data['REPORT_DAY'].astype(str)
      + ' ' + data['REPORT_HOUR'].astype(str),
      format='%Y-%m-%d %H', errors='coerce').astype(np.int64) // 10 ** 9

  columns_to_drop_temporal = ['OCC_DATE', 'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_HOUR', 'OCC_DOW', 'OCC_DOY',
                              'REPORT_DATE', 'REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY', 'REPORT_HOUR', 'REPORT_DOW', 'REPORT_DOY']

  """
  Miscellaneous dropped columns
  """
  columns_to_drop_misc = ['OBJECTID']

  """
  Drop columns
  """
  columns_to_drop = columns_to_drop_positional + \
      columns_to_drop_temporal + columns_to_drop_misc
  data.drop(columns=columns_to_drop, inplace=True)
  data.reset_index(drop=True, inplace=True)

  if save and filename:
    data.to_csv(filename, index=False)
  elif save:
    print('Data cleaning successful but no filename provided for saving')
  else:
    pass

  return data

def impute_missing_values(data: np.DataFrame, save: bool = False, filename: str = ''):
  imputer_freq = SimpleImputer(strategy='most_frequent')
  imputer_mean = SimpleImputer(strategy='mean')
  imputer_median = SimpleImputer(strategy='median')
  imputer_unknown = SimpleImputer(strategy='constant', fill_value='Unknown')

  data['BIKE_MAKE'] = imputer_freq.fit_transform(data[['BIKE_MAKE']]).ravel()
  data['BIKE_MODEL'] = imputer_unknown.fit_transform(data[['BIKE_MODEL']]).ravel()
  data['BIKE_SPEED'] = imputer_median.fit_transform(data[['BIKE_SPEED']]).ravel()
  data['BIKE_COLOUR'] = imputer_freq.fit_transform(data[['BIKE_COLOUR']]).ravel()
  data['BIKE_COST'] = imputer_mean.fit_transform(data[['BIKE_COST']]).ravel()

  if data.isnull().values.any():
    raise Exception('Imputation failed')
  elif save and filename:
    data.to_csv(filename, index=False)
    print('Imputation successful and saved to file')
  elif save:
    print('Imputation successful but no filename provided for saving')
  else:
    print('Imputation successful')

  return data

def normalize_data(data: np.DataFrame, save: bool = False, filename: str = ''):
  scaler = preprocessing.MinMaxScaler()
  data['BIKE_COST_NORMALIZED'] = scaler.fit_transform(data[['BIKE_COST']])
  data['BIKE_SPEED_NORMALIZED'] = scaler.fit_transform(data[['BIKE_SPEED']])

  if save and filename:
    data.to_csv(filename, index=False)
    print('Normalization successful and saved to file')
  elif save:
    print('Normalization successful but no filename provided for saving')
  else:
    print('Normalization successful')

  return data

def label_encoding(data: np.DataFrame, save: bool = False, filename: str = ''):
  categorical_columns = ['BIKE_MAKE', 'BIKE_MODEL', 'BIKE_COLOUR', 'BIKE_TYPE', 'PREMISES_TYPE', 'LOCATION_TYPE', 'PRIMARY_OFFENCE', 'STATUS']

  # For BIKE_MAKE:
  type_map = {"UNKNOWN MAKE": "UNKNOWN", "KONA\\": "KONA", "GI": "GIANT", "GIAN": "GIANT", "EM": "EMMO",
              "CC": "CCM", "CA": "CANNONDALE", "BI": "BIANCHI", "FJ": "FUJI", "IN": "INFINITY", "KH": "KHS",
              "MARIN": "MARIN OR MARINO", "MO": "MONGOOSE", "NO": "NORCO", "PE": "PEUGEOT", "RA": "RALEIGH",
              "RM": "ROCKY MOUNTAIN", "SC": "SCHWINN", "SP": "SPECIALIZED", "SPEC": "SPECIALIZED", "SU": "SUPERCYCLE", "TR": "TREK",
              "OT": "Other", "OTHE": "Other", "UNKNOWN": "Other", "Unknown": "Other", "UK": "Other", "UNK": "Other", "UNKNOWN MAKE": "Other", "OTHER": "Other", }
  data['BIKE_MAKE'] = data['BIKE_MAKE'].map(type_map).fillna(data['BIKE_MAKE'])
  data['BIKE_MAKE'] = data['BIKE_MAKE'].apply(lambda x: 'Other' if (
      x not in data['BIKE_MAKE'].value_counts().nlargest(10).index) and (x != 'Other') else x)

  # For BIKE_MODEL:
  model_map = {"HARD ROCK": "HARDROCK", "UNKNOWN": "Other", "NONE": "Other",
               "U/K": "Other", "UNK": "Other", "Unknown": "Other", "UNKN": "Other", "OTHER": "Other"}
  data['BIKE_MODEL'] = data['BIKE_MODEL'].map(model_map).fillna(data['BIKE_MODEL'])
  data['BIKE_MODEL'] = data['BIKE_MODEL'].apply(lambda x: 'Other' if (
      x not in data['BIKE_MODEL'].value_counts().nlargest(10).index) and (x != 'Other') else x)
  
  # For BIKE_COLOUR:
  color_map = {"TEAL": "BLU", "TURQ": "BLU", "TURQUOISE": "BLU", "DBLLBL": "BLU", "WHT": "WHI", "DARK": "BLK", "GREEN": "GRN", "DGR": "GREEN",
               "OTH": "Other", "UNKNOWN": "Other", "Unknown": "Other", 18: "Other", "OTHER": "Other"}
  data['BIKE_COLOUR'] = data['BIKE_COLOUR'].map(color_map).fillna(data['BIKE_COLOUR'])
  data['BIKE_COLOUR'] = data['BIKE_COLOUR'].apply(lambda x: x[:3] if len(x) > 3 and (x != 'Other') else x)
  data['BIKE_COLOUR'] = data['BIKE_COLOUR'].map(color_map).fillna(data['BIKE_COLOUR'])
  data['BIKE_COLOUR'] = data['BIKE_COLOUR'].apply(lambda x: 'Other' if (
      x not in data['BIKE_COLOUR'].value_counts().nlargest(10).index) and (x != 'Other') else x)
  
  # For PREMISES_TYPE:
  premises_map = {"Commercial": "Public Places", "Educational": "Public Places", "Transit": "Outside"}
  data['PREMISES_TYPE'] = data['PREMISES_TYPE'].map(premises_map).fillna(data['PREMISES_TYPE'])

  # For PRIMARY_OFFENCE:
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
      r'.*(FTC|OTHER).*': 'Other', # Failure to Comply does not imply a specific offence type
  }
  data['PRIMARY_OFFENCE'] = data['PRIMARY_OFFENCE'].replace(replacement_dict, regex=True)
  data['PRIMARY_OFFENCE'] = data['PRIMARY_OFFENCE'].apply(lambda x: 'Other' if (
      x not in data['PRIMARY_OFFENCE'].value_counts().nlargest(10).index) and (x != 'Other') else x)

  if save and filename:
    data.to_csv(filename, index=False)
    print('Label encoding successful and saved to file')
  elif save:
    print('Label encoding successful but no filename provided for saving')
  else:
    print('Label encoding successful')
  
  return data

