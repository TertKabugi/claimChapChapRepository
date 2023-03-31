import inline as inline
import matplotlib as matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pointbiserialr, stats
from imblearn.over_sampling import SMOTE, ADASYN

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

from xgboost import XGBClassifier
from xgboost import plot_importance

from mlens.ensemble import SuperLearner
from mlens.visualization import corrmat

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 500)

df = pd.read_csv('dataset.csv')

df.head()
df.tail()

missing_stats = []

for col in df.columns:
    missing_stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0],
                          df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))

stats_df = pd.DataFrame(missing_stats,
                        columns=['feature', 'unique_values', 'percent_missing', 'percent_biggest_cat', 'type'])
stats_df.sort_values('percent_missing', ascending=False)

df.describe()

# a row with no values have no value
df.drop('_c39', axis=1, inplace=True)

# extract out the year as
df['policy_bind_year'] = df['policy_bind_date'].str.extract('(\d{4})\-').astype('int32')

# extract the month out
# all accidents are from 2015 jan and feb, so year is not very important
df['incident_month'] = df['incident_date'].str.extract('\d{4}\-(\d{2})').astype('int32')

# replace "?" with undocumented
df['collision_type'] = df['collision_type'].replace("?", "undocumented")
df['police_report_available'] = df['police_report_available'].replace("?", "undocumented")
df['property_damage'] = df['property_damage'].replace("?", "undocumented")
df['auto_make'] = df['auto_make'].replace("Suburu", "Subaru")

# incident_severity can be seen as ordinal
# code them in order of severity
df['incident_severity'] = df['incident_severity'].map({"Trivial Damage": 0,
                                                       "Minor Damage": 1,
                                                       "Major Damage": 2,
                                                       "Total Loss": 3
                                                       }).astype("int32")
df.incident_severity.value_counts()

# umbrella limit is like an insruance topup that pays your liabilities in case you get sued
# cannot be zero
# this 0 is an error
df['umbrella_limit'].sort_values(ascending=True)

# edit to positve
df['umbrella_limit'].iloc[290] = 1000000

# check
print(df['umbrella_limit'].iloc[290])

# check
df['umbrella_limit'].sort_values(ascending=True)

all_var = list(df.columns)
len(all_var)

# continuous variables = ordinal, interval, ratio
cont_var = ['age', 'incident_hour_of_the_day',
            'number_of_vehicles_involved', 'total_claim_amount',
            'injury_claim', 'property_claim', 'vehicle_claim',
            'months_as_customer', 'policy_annual_premium', 'policy_deductable',
            'umbrella_limit', 'capital-gains', 'capital-loss',
            'auto_year', 'witnesses', 'bodily_injuries', 'policy_bind_year', 'incident_severity']

len(cont_var)

# ordinal var
ord_var = ['policy_deductable', 'witnesses', 'bodily_injuries', 'incident_severity']
len(ord_var)

# quan var = interval or ratio
quan_var = (list(set(cont_var) - set(ord_var)))
len(quan_var)

# norminal aka discrete var
nom_var = (list(set(all_var) - set(cont_var)))
len(nom_var)

# check for norminal data with vevry large number of categories
for col in nom_var:
    print("###", col, "###")
    print("  ")
    print(df[col].value_counts())
    print("---------------------------------------------")
    print("  ")

# for those that have not too many unique, we can plot them out
large_cat = []

for col in nom_var:
    if df[col].nunique() > 20:
        large_cat.append(col)
        print(col, df[col].nunique())
    else:
        pass

# incident location, insured zip, policy bind date, policy number too many unique to be meaningful

# get a data frame with var that have large num categories
# only auto model
large_dummy = pd.get_dummies(df[['auto_model', 'incident_date']], drop_first=True)

# put in the DV
large_dummy['fraud_reported'] = df['fraud_reported']

# numerical code the DV
large_dummy['fraud_reported'] = large_dummy['fraud_reported'].map({"Y": 1, "N": 0})

# unique to each case. wont be useful
large_dummy.head(10)


# Color negative numbers red, positive blue
def color(val):
    color = 'green' if val == 1 else 'red' if val < -0.3 else 'blue' if val > 0.3 else 'black'  # write like lambda
    return 'color: %s' % color


corr = large_dummy.corr()
corr.style.applymap(color)

# no correlation with make and fraud. drop var

# drop these variables from analysis
df.drop(large_cat, axis=1, inplace=True)

# redefine the norminal var
# remove dv from the list
nom_var.remove('fraud_reported')
nom_var = (list(set(nom_var) - set(large_cat)))
len(nom_var)

# check columns add up
print('total variable count:{} '.format(len(list(df.columns))))
print(list(df.columns))
print('continuous variables:{}'.format(len(cont_var)))
print(list(cont_var))
print('nominal variables:{}'.format(len(nom_var)))
print(list(nom_var))

# count each level of the Dv
df.fraud_reported.value_counts()

# EXPLORATORY
# count each level of the Dv
df.fraud_reported.value_counts()

# proportion of each level of DV
df.fraud_reported.value_counts(normalize=True)

# dist of dv
# plt.style.use('dark_background') for ppt
sns.set()
sns.countplot(x="fraud_reported", data=df, palette=['r', 'b']);


# variable correlation
# Color negative numbers red
def color(val):
    color = 'green' if val == 1 else 'red' if val < -0.3 else 'blue' if val > 0.3 else 'black'  # write like lambda
    return 'color: %s' % color


corr = df[cont_var].corr()
corr.style.applymap(color)

sns.set_style('white')

# heatmap from those with at least 0.3 magnitude in corr, includeing the DV
corr_list = ['age', 'months_as_customer', 'total_claim_amount',
             'injury_claim', 'property_claim', 'vehicle_claim',
             'incident_severity', 'fraud_reported']

corr_df = df[corr_list]
corr = round(corr_df.corr(), 2)

# Set the default matplotlib figure size to 7x7:
fix, ax = plt.subplots(figsize=(10, 10))

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True  # triu triangle upper

# Plot the heatmap with seaborn.
# Assign the matplotlib axis the function returns. This will let us resize the labels.
ax = sns.heatmap(corr, mask=mask, ax=ax, annot=True, cmap='OrRd')

# Resize the labels.
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10, ha='right', rotation=45)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10, va="center", rotation=0)

# If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.
plt.show()

# visualization
