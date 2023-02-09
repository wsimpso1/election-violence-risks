# import libraries

import os
import pandas as pd
pd.set_option('display.max_columns',100)
from datetime import datetime, timedelta
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV



# load data

# data directory
data_dir = 'https://raw.githubusercontent.com/wsimpso1/election-violence-risks/main/data'
# dataframes
deco = pd.read_csv(data_dir+'/DECO_v.1.0.csv', parse_dates=['date_start','date_end'])
nelda = pd.read_csv(data_dir+'/NELDA.csv', encoding='latin-1')
nelda_look_up = pd.read_csv(data_dir+'/nelda_look_up.csv', encoding='latin-1')



# preprocessing

# datetime conversion
def custom_nelda_date_format(year, mmdd):
    '''
    parse year and month-day columns for NELDA data
    
    Parameters:
    ———————————
    year: int
    mmdd: int
        month (1-12) and day (01-31)
    
    Returns:
    ————————
    combined year-month-day date format
    '''
    try:
        date_string = f'{year}{mmdd}'
        return datetime.strptime(date_string, '%Y%m%d').date()
    # impute date for missing data 
    except:
        return datetime.strptime(f'{year}0101', '%Y%m%d').date()

# apply function        
nelda['date'] = pd.to_datetime([custom_nelda_date_format(year,date) for year, date in zip(nelda.year,nelda.mmdd)])



# clean NELDA data of historical elections for modeling
def process_nelda(nelda_data):
    '''
    data wrangling for NELDA data
    - filters relevant data to allow merging with DECO data
    - fill NA and reformat string features to numeric
    
    Parameters:
    ———————————
    nelda_data: pandas dataframe
        original NELDA data
        
    Returns:
    ————————
    nelda_data: pandas dataframe
        processed dataframe
    '''    
    # select only years covered by both datasets (1989-2017)
    nelda_data = nelda_data.loc[(nelda_data.year > 1988) & (nelda_data.year < 2018)]
    # select only countries that appear in both datasets
    deco_country_ids = list(deco.country_id.unique())
    nelda_data = nelda_data.loc[nelda_data.ccode.isin(deco_country_ids)] 
    
    # remove notes column
    nelda_data = nelda_data[[col for col in nelda_data.columns if not re.match(".+notes$", col)]]
    
    # exclude free text features of names and location
    free_txt_cols = ['nelda43', 'nelda44', 'nelda54']
    nelda_data = nelda_data[[col for col in nelda_data.columns if col not in free_txt_cols]]
    
    # fill NaN as 'N/A' for Nelda columns
    nelda_cols = [col[0] for col in [re.findall(r'nelda\d+$', col) for col in nelda_data.columns] if len(col)>=1]
    for col in nelda_cols:
        nelda_data[col] = nelda_data[col].fillna('n/a')
    
    # convert string features to numeric
    def str_to_num(string):
        if string == 'yes':
            return 2
        elif string == 'no':
            return 1
        elif string == 'n/a': 
            return 0
        else:  # string == 'unclear'
            return -1
        
    # apply string-to-numeric function    
    for col in nelda_cols:
        nelda_data[col] = [str_to_num(val) for val in nelda_data[col]]
    
    return nelda_data.reset_index(drop=True)

# apply cleaning function
nelda_clean = process_nelda(nelda)



# clean data of election violence (DECO) for modeling
def process_deco(deco_data):
    '''
    data wrangling for DECO data
    - filter columns
    - aggregate election violence data by country and date
    
    Parameters:
    ———————————
    deco_data: pandas dataframe
        original DECO data
        
    Returns:
    ————————
    deco_data: pandas dataframe
        processed and aggregated data
    '''
    # select relevant columns
    deco_data = deco_data[['country_id', 'best', 'date_end']]
    # rename columns
    deco_data = deco_data.rename(columns={'best':'num_fatalities', 'date_end':'date'})
    # sum number of fatalities by country and date
    deco_data = deco_data.groupby(by=['country_id', 'date']).sum()
    
    return deco_data.reset_index()

# apply cleaning function
deco_agg = process_deco(deco)



# Merge Preprocessed and Aggregated NELDA and DECO datasets
def fatalities_per_election(election_date, country_id):
    '''
    compute the total number of election related fatalities in 1 year leading up to election date
    
    Parameters:
    ———————————
    election_date: datetime object
        date of election
    country_id: int
        ISO country code
    
    Returns:
    –———————
    sum_election_fatalities: int
        aggregated number of election-related fatalities in x country 1 year leading up to election
    '''
    deco_agg_country = deco_agg.loc[deco_agg.country_id == country_id].copy()
    start_date = election_date - timedelta(days=365)
    deco_agg_country_1_year = deco_agg_country.loc[(deco_agg_country.date >= start_date) & 
                                                   (deco_agg_country.date <= election_date)]
    sum_election_fatalities = deco_agg_country_1_year.num_fatalities.sum()
    return sum_election_fatalities

# apply function
deco_election_fatalities = [fatalities_per_election(date, country) for date, country in zip(nelda_clean.date, nelda_clean.ccode)]

# create final combined dataframe
nelda_deco = nelda_clean.copy()
nelda_deco['election_fatalities'] = deco_election_fatalities



# Drop unecessary columns 
drop_cols = ['stateid','ccode', 'country', 'electionid', 'year', 'mmdd', 'types', 'notes', 'date']
nelda_deco = nelda_deco.drop(drop_cols, axis=1)

# reset index
nelda_deco.reset_index(inplace=True, drop=True)

# Remove Multicolinearity
# of the highly correlated features keep those that are most potentially informative
nelda_deco = nelda_deco.drop(columns=['nelda8','nelda21', 'nelda29',
                                                'nelda36', 'nelda37',
                                                'nelda40', 'nelda41'])




# Model Building

# convert target variable of fatalities to a categorical variable
def to_categorial(val):
    '''
    Discretizes numerical value of fatalities in a country-year
    
    Categories are defined according to a US Dept of Justice definition that an event
    with 4 or more fatalities constitutes mass murder
    https://www.ojp.gov/ncjrs/virtual-library/abstracts/serial-murder-multi-disciplinary-perspectives-investigators 
    
    Parameters:
    ———————————
    val: int
        number of fatalities
    
    Returns:
    ————————
    str: category of level of fatality
    '''
    if val == 0:
        return 'non-fatal'
    if 1 < val <= 3:
        return 'low fatality'
    else:
        return 'mass fatality'

# apply categorical conversion function
nelda_deco['election_fatalities'] = [to_categorial(row) for row in nelda_deco.election_fatalities]

# define target variable y (election fatalities) and features X (risk factors) 
y = nelda_deco.election_fatalities
X = nelda_deco.drop(['election_fatalities'], axis=1)

# Drop columns that contain information about election fatalities the model may cheat on
# - nelda 33 explicitly codes for the presence of fatalities
# - nelda 31 codes for the use of violence by the government against citizens
X = X.drop(columns=['nelda33', 'nelda31'])

# transform all categorical NELDA features to one-hot encoding
# Creates list of all column headers
all_columns = list(X)
# change datatype
X[all_columns] = X[all_columns].astype(str)
# one hot encoding
X_one = pd.get_dummies(X)



# Adjust class imblance to mitigate overfitting via under/oversampling

# undersample the non-fatal class
undersample = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_under, y_under = undersample.fit_resample(X_one, y)

# oversample the other classes to eliminate class imbalance 
oversample = RandomOverSampler(sampling_strategy='all', random_state=42)
X_over, y_over = oversample.fit_resample(X_under, y_under)



# split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_over, 
                                                    y_over, 
                                                    test_size=.2, 
                                                    random_state=42, 
                                                    stratify=y_over)



# Train Model and Search for Optimal Parameters

# define grid search and cross validation
parameter_grid = {'n_estimators':[50, 100, 200, 300, 500, 800, 1000],
              'max_depth':[3, 4, 5, 7], 
              'criterion': ['gini', 'entropy']}

# instantiate model 
rf_model = RandomForestClassifier(random_state=42)
rf_grid_cv = GridSearchCV(rf_model, parameter_grid, verbose=1, cv=5)

# train
rf_grid_cv.fit(X_train, y_train)

# best parameters
print('Best Parameters:', rf_grid_cv.best_params_)

# get predictions with best model
y_pred = rf_grid_cv.predict(X_test)

# training accuracy of best model
print('Train Accuracy:', round(rf_grid_cv.score(X_train, y_train)*100, 2), '%')

# testing accuracy of best model
print('Test Accuracy:', round(rf_grid_cv.score(X_test, y_test)*100, 2), '%')

# examine precision and recall from classification report
clf_rpt = classification_report(y_test, y_pred, target_names=rf_grid_cv.classes_)
print(clf_rpt)



# determine feature importances 
'''
Permutation Importance: 
A strategy to measure the decrease in model performance as the result of 
randomly shuffling one feature at a time. More important features in the model's final decision 
cause a larger drop in performance when shuffled.
'''

r = permutation_importance(rf_grid_cv, X_test, y_test,
                           n_repeats=5,
                           random_state=0)

perm_optimized = pd.DataFrame(columns=['AVG_Importance'], index=X_test.columns)
perm_optimized['AVG_Importance'] = r.importances_mean

perm_optimized = perm_optimized.sort_values('AVG_Importance', ascending=False)

# select features with importance above threshold
importance_threshold = 0.004
perm_optimized = perm_optimized.loc[perm_optimized.AVG_Importance > importance_threshold]



# get top most important risk factors to predicting historical election violence 
top_nelda_codes = [re.findall(r'nelda\d{1,2}',row) for row in perm_optimized.index]
top_nelda_codes = pd.DataFrame(np.concatenate(top_nelda_codes), columns=['nelda_feature'])

top_unique_nelda_codes = list(top_nelda_codes.nelda_feature.unique())

# look up text descriptions of top NELDA risk factors
top_nelda_code_descriptions = nelda_look_up.loc[nelda_look_up.nelda_code.isin(top_unique_nelda_codes)][['nelda_code', 'description_clean']]
top_nelda_code_descriptions = top_nelda_code_descriptions.rename(columns={'description_clean':'election_characteristic'})
top_nelda_code_descriptions.reset_index(inplace=True, drop=True)

# save list of characteristics of elections that correlate to election violence according to RF model
print(top_nelda_code_descriptions)
top_nelda_code_descriptions.to_csv('FINAL_OUTPUT_characteristics_of_electoral_violence.csv')
