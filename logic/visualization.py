# region: Import dependencies
import os,sys,random,io
import numpy as np,pandas as pd,seaborn as sns,matplotlib.pyplot as plt
# endregion

def explorative_vis(data: np.DataFrame, save: bool = False, filename: str = ''):
  fig, axes = plt.subplots(2, 2, figsize=(10, 10))
  fig.suptitle('Fundamental Visualizations')

  # region: histogram of bike cost
  data_log = np.log(data[data['BIKE_COST'] > 0]['BIKE_COST'])
  sns.histplot(data_log, bins=20, color='blue', alpha=0.7, kde=True, ax=axes[0, 0])
  axes[0, 0].set_xticks(np.arange(0, 6, 1))
  axes[0, 0].set_title('Bike Cost Distribution (Imputed))')
  axes[0, 0].set_xlabel('Log10 of Bike Cost')
  axes[0, 0].set_ylabel('Frequency')
  # endregion

  # region: histogram of bike speed
  sns.histplot(data[data['BIKE_SPEED'] > 0]['BIKE_SPEED'], bins=20, color='green', alpha=0.7, kde=True, ax=axes[0, 1])
  axes[0, 1].set_title('Bike Speed Distribution (Imputed)')
  axes[0, 1].set_xlabel('Bike Speed')
  axes[0, 1].set_ylabel('Frequency')
  # endregion

  # region: line plot of occurrence over time (since year 2013)
  data_temp = data.copy()
  data_temp['case_year'] = pd.to_datetime(data_temp['OCC_TIMESTAMP'], unit='s').dt.year
  cases_per_year = data_temp[data_temp['case_year'] >= 2013]['case_year'].value_counts().sort_index()
  print(cases_per_year)
  sns.lineplot(x=cases_per_year.index, y=cases_per_year.values, color='red', markers='o', ax=axes[1, 0])
  axes[1, 0].set_title('Number of Cases Per Year')
  axes[1, 0].set_xlabel('Year')
  axes[1, 0].set_ylabel('Number of Cases')
  # endregion

  # region: line plot of occurrence over time (month)
  data_temp = data.copy()
  data_temp['case_month'] = pd.to_datetime(data_temp['OCC_TIMESTAMP'], unit='s').dt.month
  cases_per_month = data_temp[data_temp['case_month']>=1]['case_month'].value_counts().sort_index()
  print(cases_per_month)
  sns.barplot(x=cases_per_month.index, y=cases_per_month.values, ax=axes[1, 1], palette='viridis')
  axes[1, 1].set_title('Number of Cases Per Month')
  axes[1, 1].set_xlabel('Month')
  axes[1, 1].set_ylabel('Number of Cases')
  # endregion

  # region: generate and save the plot
  buffer = io.BytesIO()
  plt.savefig(buffer, format='png')
  if save and filename:
    plt.savefig(filename, format='png')
    print(f'Saved visualization to {filename}')
  elif save:
    print('No filename provided for saving')
  else:
    plt.show()
    
  buffer.seek(0)
  return buffer
  # endregion