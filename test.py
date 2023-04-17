import inline as inline
import matplotlib as matplotlib
import numpy as np
import pandas as pd
import scipy
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

from main import dum

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
corr = round(corr_df.corr(numeric_only=True), 2)

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

df['loss_by_claims'] = df['total_claim_amount'] - (df['policy_annual_premium'] * (2015 - df['policy_bind_year']))

# note this is not how much the insurance company earns because there are a lot of other cars with no indicents
sns.set_style("white")

print(sns.catplot(data=df, y="loss_by_claims", x="fraud_reported", hue='fraud_reported', kind='swarm',
                  palette=['red', 'silver']))

print(df[['capital-loss', 'capital-gains', 'loss_by_claims']])
print(df.groupby('fraud_reported')['loss_by_claims'].sum())
print(df.groupby('fraud_reported')['loss_by_claims'].mean())

print(df.groupby('fraud_reported')['loss_by_claims'].std())
print(df[['fraud_reported', 'loss_by_claims']].isnull().sum())

df_ttest = df[['fraud_reported', 'loss_by_claims']]
print(df_ttest.head())

print(stats.ttest_ind(df_ttest.loc[df_ttest['fraud_reported'] == 'Y', 'loss_by_claims'],
                      df_ttest.loc[df_ttest['fraud_reported'] == 'N', 'loss_by_claims']))

# Preprocessing
# DV numerical code
df['fraud_reported'] = df['fraud_reported'].map({"Y": 1, "N": 0})
print(df['fraud_reported'])

df['insured_sex'] = df['insured_sex'].map({"FEMALE": 0, "MALE": 1})
df['capital-loss'] = df['capital-loss'] * (-1)
df['capital-loss'].max()

# check that they are coded

df['pclaim_severity_int'] = df['property_claim'] * df['incident_severity']
df['vclaim_severity_int'] = df['vehicle_claim'] * df['incident_severity']
df['iclaim_severity_int'] = df['injury_claim'] * df['incident_severity']
df['tclaim_severity_int'] = df['total_claim_amount'] * df['incident_severity']

df['prem_claim_int'] = df['policy_annual_premium'] * df['total_claim_amount']
df['umlimit_tclaim_int'] = df['umbrella_limit'] * df['total_claim_amount']

# Dummy coding
rem = ['insured_sex', 'incident_month']
dum_list = [e for e in nom_var if e not in rem]
print(dum_list)
print(len(dum_list))

rem = ['insured_sex', 'incident_month']
dum_list = [e for e in nom_var if e not in rem]
print(len(dum_list))

dum.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)
df_dummied = pd.concat([dum, df], axis=1)
print(df_dummied.drop(nom_var, axis=1, inplace=True))
print(df_dummied.head())

print(df_dummied.isnull().sum().any())
print(df_dummied['umbrella_limit'].sort_values(ascending=True))

dd = df_dummied.describe()
print(dd.loc['min'])

x = df_dummied.drop('fraud_reported', axis=1)
y = df_dummied['fraud_reported']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)

# baseline accuracy = 75.2%
y_test.value_counts(normalize=True)

base_recall = 62 / 62
base_speci = 0 / 188
base_preci = 62 / 250
base_f1 = 2 * base_recall * base_preci / (base_recall + base_preci)

df_prob = pd.DataFrame()
df_prob['y'] = y_test
df_prob['pred'] = 1
auc_score = roc_auc_score(df_prob['y'], df_prob['pred'])

print("If we make a naive prediction that all claims are frauds, so that no frauds escape our watch, we will have an:")
print("")
print("Sensitivity:", base_recall)
print('Specificity:', base_speci)
print('Precision:', base_preci)
print('F1 score:', base_f1)
print('ROC AUC Score:', auc_score)


# modeling
# function to use for scoring

def scores(t, name):
    print(name, 'classification metric')
    print("CV scores:", round(t.best_score_, 3))
    print("train score:", round(t.score(x_train, y_train), 3))
    print("test score:", round(t.score(x_test, y_test), 3))

    # Evaluation metrics
    predictions = t.predict(x_test)

    TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()

    sensi = TP / (TP + FN)
    speci = TN / (TN + FP)
    preci = TP / (TP + FP)
    f1 = 2 * (preci * sensi) / (preci + sensi)

    print(f'Sensitivity: {round(sensi, 3)}')
    print(f'Specificity: {round(speci, 3)}')
    print(f'Precision: {round(preci, 3)}')
    print(f'F1: {round(f1, 3)}')

    pred_proba = [i[1] for i in t.predict_proba(x_test)]
    auc_score = roc_auc_score(y_test, pred_proba)
    print('ROC AUC Score:', round(auc_score, 3))

print("Logistic Regression")

# logistic regression
print(y_train.value_counts(normalize=True))

# ransearch log reg
# print("Logistic Regression")
# lr = LogisticRegression(max_iter=1000)
#
# lr_values = {'solver': ['liblinear'],
#              'penalty': ['l1', 'l2'],
#              'C': np.logspace(-5, 5, 50),
#              'class_weight': [{0: 0.246667, 1: 0.75333}, None]}
#
# rs_lr = RandomizedSearchCV(lr, lr_values, cv=10, n_jobs=-1, random_state=42)
# rs_lr.fit(x_train, y_train)
# print(rs_lr.best_params_)

print("KNN")
# knn = KNeighborsClassifier()
# ss = StandardScaler()
knn_pipe = Pipeline([
    ("ss", StandardScaler()),
    ("knn", KNeighborsClassifier(n_jobs=-1))])

knn_values = {'knn__n_neighbors': [3, 5, 7, 9, 11],
              'knn__weights': ['uniform', 'distance'],
              'knn__metric': ['minkowski', 'euclidean', 'manhattan'],
              'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'knn__leaf_size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
              }

rs_knn = RandomizedSearchCV(knn_pipe, knn_values, cv=10, n_jobs=-1, random_state=42)
rs_knn.fit(x_train, y_train)
print(rs_knn.best_params_)

print("Random Forest")
rf = RandomForestClassifier(n_jobs=-1)

rf_values = {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
             'min_samples_leaf': [1, 2, 3, 4, 5],
             'min_samples_split': [2, 5, 7, 9, 11],
             'max_features': ['sqrt'],
             'n_estimators': [150, 250, 350, 450, 500, 550, 600, 650],
             'class_weight': [{0: 0.246667, 1: 0.75333}, None]
             }

rs_rf = RandomizedSearchCV(rf, rf_values, cv=10, n_jobs=-1, random_state=42)
rs_rf.fit(x_train, y_train)
print(rs_rf.best_params_)

print(y_train.value_counts())

# print("XG Boost")
xg = XGBClassifier(booster='gbtree', n_jobs=-1)

xg_values = {'max_depth': [3, 4, 5, 6],
             'eta': [0.05, 0.1, 0.15, 0.3],
             'reg_lambda': [0.01, 0.05, 0.1, 0.5, 1],
             'reg_alpha': [0.01, 0.05, 0.1, 0.5, 1],
             'gamma': [0, 1, 2, 3],
             'n_estimators': [150, 250, 350, 450, 500, 550, 600, 650],
             'scale_pos_weight': [1, 3.054054054054054],
             }

rs_xg = RandomizedSearchCV(xg, xg_values, cv=10, n_jobs=-1, random_state=42)
rs_xg.fit(x_train, y_train)
print(rs_xg.best_params_)

# ada boost
print("ADA Boost")
ab = AdaBoostClassifier()
ab_values = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
             'learning_rate': [0.05, 0.1, 0.3, 0.5]
             }

rs_ab = RandomizedSearchCV(ab, ab_values, cv=10, n_jobs=-1, random_state=42)
rs_ab.fit(x_train, y_train)
print(rs_rf.best_params_)

classifiers = {'knn': rs_knn, 'Ranfor': rs_rf,
               'AdaBoost': rs_ab}

for key, value in classifiers.items():
    print(scores(value, key))
    print("__________________________")
    print(" ")
