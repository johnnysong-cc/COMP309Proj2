# COMP309 Project 2: Supervised Predictive Modeling APIs for Bicycle Theft in Toronto

## Group Members:

- Johnny Z. Song (301167073)


## Project Objectives:
- [ ] Obtain and prepare data:
  - [ ] Load and explore the dataset referenced in Section 4 of this document using techniques from the course.
  - [ ] Visualize and describe the data, identify correlations, and clean and transform categorical data.
  - [ ] Build a supervised predictive model using a suitable classification algorithm in Python with scikit-learn, pandas, numpy, etc.
- [ ] Validate and evaluate models, selecting the best one.
- [ ] Create an API for the model using the Python Flask framework.
- [ ] Develop a simple front end to access the API and input new feature values for predictions.

## Project Dataset:
-  [Download link](https://opendata.arcgis.com/api/v3/datasets/a89d10d5e28444ceb0c8d1d4c0ee39cc_0/downloads/data?format=csv&spatialRefId=3857&where=1%3D1)

## Project Requirements:

Provide the following deliverables:

### Data exploration

- [x] Load and describe data elements (columns), providing descriptions, types, ranges, and values.
- [ ] Perform statistical assessments, including means, averages, and correlations.
- [ ] Evaluate missing data.
- [ ] Create graphs and visualizations.

### Data modeling

- [ ] Perform data transformations, including handling missing data, managing categorical data, and data normalization.
- [ ] Select features.
- [ ] Split data into training and testing sets.
- [ ] Handle imbalanced classes if needed.

### Predictive model building

- [ ] Use logistic regression and decision trees as a **minimum**.
- [ ] etc.

### Model scoring and evaluation

- [ ] Present results as scores, confusion matrices, and ROC curves.
- [ ] Select the best-performing model.

### Deploying the model

- [ ] Use Flask to create an API.
- [ ] Serialize and deserialize the model using the pickle module.
- [ ] Build a client to test the model API service.

### Prepare a report

- [ ] Include an executive summary, overview of the solution, data exploration, feature selection, data modeling, and model building.


## Project References:

- Data Columns Explained:
  - X & Y: The coordinates in a specific projection system used for mapping and spatial analysis, representing longitude (X) and latitude (Y).
  - OBJECTID: The unique identifier for each record.
  - EVENT_UNIQUE_ID: The unique identifier for each reported event, probably assigned by the police department.
  - PRIMARY_OFFENCE: The main offence reported in the event, such as "THEFT UNDER" or "PROPERTY - FOUND".
  - OCC_DATE: The date and time when the offence occurred.
  - OCC_YEAR, OCC_MONTH, OCC_DOW (Day Of Week), OCC_DAY, OCC_DOY (Day Of Year), OCC_HOUR: detailed timing information about when the offence occurred, split into year, month, day, day of the week, day of the year, and hour of the day.
  - REPORT_DATE, REPORT_YEAR, REPORT_MONTH, REPORT_DOW, REPORT_DAY, REPORT_DOY, REPORT_HOUR: etailed timing information about when the offence was reported, split into year, month, day, day of the week, day of the year, and hour of the day.
  - DIVISION: The police division that recorded the event.
  - LOCATION_TYPE: The general type of the location where the event occurred, e.g., "Apartment" or "Commercial".
  - PREMISES_TYPE: The specific type of the premises, such as "House" or "Apartment".
  - BIKE_MAKE, BIKE_MODEL, BIKE_TYPE: The information about the bicycle, including the make, model, and type.
  - BIKE_SPEED: The number of speeds the bike has.
  - BIKE_COLOUR: The color of the bicycle.
  - BIKE_COST: The cost or value of the bicycle.
  - STATUS: Indicates whether the bicycle was stolen, recovered, etc.
  - HOOD_140 & NEIGHBOURHOOD_140 & HOOD_158 & NEIGHBOURHOOD_158: The numberic code 158 could be an identifier for data tracking purposes or a version number in a specific coding system for neighborhoods used by the Toronto Police
  - LONG_WGS84 & LAT_WGS84: The longitude and latitude of the event location in the WGS84 coordinate system (a standard used in cartography, geodesy, and navigation).