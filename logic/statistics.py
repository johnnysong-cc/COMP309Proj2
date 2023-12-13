# region: Import dependencies
import numpy as np
import pandas as pd
from scipy import stats
# endregion: Import dependencies

def describe_data(data:pd.DataFrame):
  # region: pd display options
  pd.set_option('display.max_columns', None)  # show all columns
  pd.option_context('display.max_rows', None)  # show all rows
  pd.set_option('display.max_colwidth', None)  # show full column contents
  pd.set_option('display.width', 240)
  pd.set_option('display.float_format', lambda x: '%.4f' % x)
  # endregion: pd display options

  # region: data description info
  shape_info = f'\nShape of Data: \n  - Number of records: {data.shape[0]}\n  - Number of Columns (Features): {data.shape[1]}\n'
  head_tail_info = f'\nHead and Tail Rows: \n {data.head()} \n\n {data.tail()}\n'
  column_info = f'\nColumn Names: \n {data.columns}\n'
  describe_info = f'\nDescribe Data: \n{data.describe(include="all")}\n'
  # endregion: data description

  return {
    'shape': shape_info,
    'head_tail': head_tail_info,
    'columns': column_info,
    'describe': describe_info,
  }

def explorative_assessment(data:pd.DataFrame, drop_columns:list = []):
  stats_raw = data.describe(include='all').drop(columns=drop_columns)
  means: pd.Series = stats_raw.loc["mean"].astype(float).to_frame().select_dtypes(include=[np.number]).drop(columns=drop_columns).dropna()
  stds: pd.Series = stats_raw.loc["std"].astype(float).to_frame().select_dtypes(include=[np.number]).drop(columns=drop_columns).dropna()
  mins: pd.Series = stats_raw.loc["min"].astype(float).to_frame().select_dtypes(include=[np.number]).drop(columns=drop_columns).dropna()
  maxs: pd.Series = stats_raw.loc["max"].astype(float).to_frame().select_dtypes(include=[np.number]).drop(columns=drop_columns).dropna()
  ranges: pd.Series = (maxs - mins).astype(float).to_frame().select_dtypes(include=[np.number]).drop(columns=drop_columns).dropna()
  iqr25s: pd.Series = stats_raw.loc["25%"].astype(float).to_frame().select_dtypes(include=[np.number]).drop(columns=drop_columns).dropna()
  medians: pd.Series = stats_raw.loc["50%"].astype(float).to_frame().select_dtypes(include=[np.number]).drop(columns=drop_columns).dropna()
  iqr75s: pd.Series = stats_raw.loc["75%"].astype(float).to_frame().select_dtypes(include=[np.number]).drop(columns=drop_columns).dropna()
  unique_counts: pd.Series = data.nunique().astype(float).to_frame().select_dtypes(include=[np.number]).drop(columns=drop_columns).dropna()
  tops: pd.DataFrame = pd.concat([stats_raw.loc["top"], stats_raw.loc["freq"]], axis=1, keys=['Top case', 'Freq']).dropna()
  unique_values: pd.DataFrame = data.value_counts().to_frame().dropna()

  return {
    'means': means,
    'stds': stds,
    'mins': mins,
    'maxs': maxs,
    'ranges': ranges,
    'iqr25s': iqr25s,
    'medians': medians,
    'iqr75s': iqr75s,
    'unique_counts': unique_counts,
    'tops': tops,
    'unique_values': unique_values,
  }

def correlation_assessment(data:pd.DataFrame):
  correlation_matrix = data.select_dtypes(include=[np.number]).corr()\
                       .where((data.select_dtypes(include=[np.number]).corr() > 0.5)
                               | (data.select_dtypes(include=[np.number]).corr() < -0.5))
  return {
    'correlation_matrix': correlation_matrix,
  }

def chi2_assessment(data: pd.DataFrame, drop_columns: list = []):
  cat_columns = data.select_dtypes(include=['object']).drop(columns=drop_columns).columns
  chi2_results = {}

  for i in range(len(cat_columns)):
    for j in range(i + 1, len(cat_columns)):
      col1_name = cat_columns[i]; col2_name = cat_columns[j]

      contingency_table = pd.crosstab(data[col1_name], data[col2_name])
      chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
      chi2_results[(col1_name, col2_name)] = {
        'chi2': chi2,
        'p': p,
        'dof': dof,
      }
  
  chi2_results_df = pd.DataFrame(chi2_results).T
  chi2_results_df['p'] = chi2_results_df['p'].astype(float).apply(lambda x: round(x, 4))

  return {
      'chi2_results': f'\nChi-Square Results: \n{chi2_results_df}'
  }

