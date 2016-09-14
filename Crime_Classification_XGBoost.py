
# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
#from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Global constants and variables
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'

train = pd.read_csv('../input/'+TRAIN_FILENAME, parse_dates=['Dates'], index_col=False)
test = pd.read_csv('../input/'+TEST_FILENAME, parse_dates=['Dates'], index_col=False)

train = train.drop(['Descript', 'Resolution', 'Address'], axis = 1)
test = test.drop(['Address'], axis = 1)

def feature_engineering(data):
    data['Day'] = data['Dates'].dt.day
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['DayOfWeek'] = data['Dates'].dt.dayofweek
    data['WeekOfYear'] = data['Dates'].dt.weekofyear
        
    data['IsWeekend'] = 0
    data.loc[data['DayOfWeek'] >4, 'IsWeekend'] = 1
    data['IsFriday'] = 0
    data.loc[data['DayOfWeek']==4, 'IsFriday'] = 1
    data['IsSaturday'] = 0
    data.loc[data['DayOfWeek']==4, 'IsSaturday'] = 1
    
    return data
    
train = feature_engineering(train)
test = feature_engineering(test)

from sklearn.preprocessing import LabelEncoder

dist_enc = LabelEncoder()
train['PdDistrict'] = dist_enc.fit_transform(train['PdDistrict'])

cat_enc = LabelEncoder()
cat_enc.fit(train['Category'])
train['CategoryEncoded'] = cat_enc.transform(train['Category'])

dist_enc = LabelEncoder()
test['PdDistrict']  = dist_enc.fit_transform(test['PdDistrict'])

x_cols = list(train.columns[2:15].values)
x_cols.remove('Minute')


clf = xgb.XGBClassifier(n_estimators=12, reg_alpha=0.05, max_depth=8)

clf.fit(train[x_cols], train['CategoryEncoded'])

test['predictions'] = clf.predict(test[x_cols])

# create dummy variables for each unique category
def dummy_cat(data):
    for new_col in data['Category'].unique():
        data[new_col]=(data['Category']== new_col).astype(int)
    return data
    
test['Category'] = cat_enc.inverse_transform(test['predictions'])
test = dummy_cat(test)

# Categories that do not get predicted need to be appended wiht the testing dataframe with
# value zero in all rows
unpredicted_cat = pd.Series(list(set(train['Category'].unique()) - (set(test['Category'].unique()))))
for new_col in unpredicted_cat:
    test[new_col] = 0


import time
PREDICTIONS_FILENAME_PREFIX = 'predictions_'
PREDICTIONS_FILENAME = PREDICTIONS_FILENAME_PREFIX + time.strftime('%Y%m%d-%H%M%S') + '.csv'

submission_cols = [test.columns[0]]+list(test.columns[17:])
print(submission_cols)

print(PREDICTIONS_FILENAME)
test[submission_cols].to_csv(PREDICTIONS_FILENAME, index = False)

