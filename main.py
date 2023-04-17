import pandas as pd
import numpy as np
import rename as rename
import seaborn as sns
from matplotlib import pyplot as plt

# data retrieval
dataset = pd.read_csv('dataset.csv', header=0)
print(dataset.shape)
print(dataset.info())
print('------------------------------------------------------------------------')

# feature engineering
dataframe = dataset.drop(dataset.columns[[4, 9, 22, 23, 24, 39]], axis=1)
# extract out the year as
dataframe['policy_bind_year'] = dataframe['policy_bind_date'].str.extract('(\d{4})\-').astype('int32')
dataframe['incident_month'] = dataframe['incident_date'].str.extract('\d{4}\-(\d{2})').astype('int32')
print(dataframe.info())

dataframe['auto_make'] = dataframe['auto_make'].replace("Suburu", "Subaru")
dataframe['collision_type'] = dataframe['collision_type'].replace("?", "Undocumented")
dataframe['police_report_available'] = dataframe['police_report_available'].replace("?", "Undocumented")
dataframe['property_damage'] = dataframe['property_damage'].replace("?", "Undocumented")
print('------------------------------------------------------------------------')

# incident_severity transformation
dataframe['incident_severity'] = dataframe['incident_severity'].map({"Trivial Damage": 0,
                                                                     "Minor Damage": 1,
                                                                     "Major Damage": 2,
                                                                     "Total Loss": 3
                                                                     }).astype("int32")
print(dataframe.incident_severity.value_counts())
print(dataframe.describe())
print('------------------------------------------------------------------------')

# umbrella limit (insurance)
print(dataframe.umbrella_limit.sort_values(ascending=True))
print(dataframe.umbrella_limit.value_counts())

dataframe.umbrella_limit.iloc[290] = 1000000
print(dataframe.umbrella_limit.sort_values(ascending=True))
print(dataframe.umbrella_limit.iloc[290])
print('------------------------------------------------------------------------')

# variables
all_var = list(dataframe.columns)
print(all_var)
print('All Variables = ', len(all_var))

# continuous variables [ordinal, interval, ratio]
cont_var = ['months_as_customer', 'age', 'policy_bind_date', 'policy_deductible', 'policy_annual_premium',
            'umbrella_limit', 'capital_gains', 'capital_loss', 'incident_severity',
            'incident_hour_of_the_day', 'number_of_vehicles_involved', 'bodily_injuries',
            'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim',
            'vehicle_claim', 'auto_year']
print('continuous variables = ', len(cont_var))

# ordinal variables
ord_var = ['policy_deductible', 'witnesses', 'bodily_injuries', 'incident_severity']
print('ordinal variables = ', len(ord_var))

# interval & ratio variables
quan_var = (list(set(cont_var) - set(ord_var)))
print('Interval & Ratio variables = ', len(quan_var))

# nominal variables
nom_var = (list(set(all_var) - set(cont_var)))
print('Nominal Variables = ', len(nom_var))
print("---------------------------------------------")
print(" ")

# determining categories in nominal variable columns
for col in nom_var:
    print('###', col, '###')
    print(dataframe[col].value_counts())
    print("---------------------------------------------")

# finding columns with many categories
print(" ")
large_category = []
for col in nom_var:
    if dataframe[col].nunique() > 20:
        large_category.append(col)
        print(col, dataframe[col].nunique())
    else:
        pass
    # policy_number(1000) has many categories

# auto model dummies
large_dummy = pd.get_dummies(dataframe['auto_model'], drop_first=True)
large_dummy['fraud_reported'] = dataframe['fraud_reported']
large_dummy['fraud_reported'] = dataframe['fraud_reported'].map({'Y': 1, 'N': 0})
print(" ")
print(large_dummy.head())


# large dummy confusion matrix
def color(val):
    shade = 'green' if val == 1 else 'red' if val < -0.3 else 'blue' if val > 0.3 else 'black'
    return 'color: %s' % shade


corr = large_dummy.corr()
corr.style.applymap(color)
print(corr)

# remove variables from analysis
dataframe.drop(large_category, axis=1, inplace=True)

nom_var.remove('fraud_reported')
nom_var = (list(set(nom_var) - set(large_category)))
print(' ')
print('total variable count:{} '.format(len(list(dataframe.columns))))
print(list(dataframe.columns))
print('continuous variables:{}'.format(len(cont_var)))
print(list(cont_var))
print('nominal variables:{}'.format(len(nom_var)))
print(list(nom_var))

# EXPLORATORY
# count each level of the Dv
print(dataframe.fraud_reported.value_counts())

# proportion of each level of DV
print(dataframe.fraud_reported.value_counts(normalize=True))


# variable correlation
# Color negative numbers red
def color(val):
    color = 'green' if val == 1 else 'red' if val < -0.3 else 'blue' if val > 0.3 else 'black'  # write like lambda
    return 'color: %s' % color


corr = dataframe.corr(numeric_only=True)
corr.style.applymap(color)
print(corr)

sns.set_style('white')

# heatmap from those with at least 0.3 magnitude in corr, including the DV
corr_list = ['age', 'months_as_customer', 'total_claim_amount',
             'injury_claim', 'property_claim', 'vehicle_claim',
             'incident_severity', 'fraud_reported']

corr_dataframe = dataframe[corr_list]
corr = round(corr_dataframe.corr(numeric_only=True), 2)
print(corr)

# Set the default matplotlib figure size to 7x7:
fix, ax = plt.subplots(figsize=(10, 10))

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True  # true triangle upper

# Plot the heatmap with seaborn.
# Assign the matplotlib axis the function returns. This will let us resize the labels.
ax = sns.heatmap(corr, mask=mask, ax=ax, annot=True, cmap='OrRd')

# Resize the labels.
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10, ha='right', rotation=45)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10, va="center", rotation=0)

# If you put plt.show() at the bottom, it prevents those useless printouts from matplotlib.
plt.show()

# DV numerical code
dataframe['fraud_reported'] = dataframe['fraud_reported'].map({"Y": 1, "N": 0})
print(dataframe['fraud_reported'])

dataframe['insured_sex'] = dataframe['insured_sex'].map({"FEMALE": 0, "MALE": 1})

# new interactions
dataframe['pclaim_severity_int'] = dataframe['property_claim'] * dataframe['incident_severity']
dataframe['vclaim_severity_int'] = dataframe['vehicle_claim'] * dataframe['incident_severity']
dataframe['iclaim_severity_int'] = dataframe['injury_claim'] * dataframe['incident_severity']
dataframe['tclaim_severity_int'] = dataframe['total_claim_amount'] * dataframe['incident_severity']

dataframe['prem_claim_int'] = dataframe['policy_annual_premium'] * dataframe['total_claim_amount']
dataframe['umlimit_tclaim_int'] = dataframe['umbrella_limit'] * dataframe['total_claim_amount']

rem = ['insured_sex', 'incident_month']
dum_list = [e for e in nom_var if e not in rem]
print(dum_list)
print(len(dum_list))

dum = pd.get_dummies(dataframe[dum_list], drop_first=True)
print(dum.head())

