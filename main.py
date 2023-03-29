import pandas as pd
import numpy as np
import rename as rename

# data retrieval
dataset = pd.read_csv('dataset.csv', header=0)
print(dataset.shape)
print(dataset.info())
print('------------------------------------------------------------------------')

# feature engineering
dataframe = dataset.drop(dataset.columns[[4, 9, 17, 22, 23, 24, 39]], axis=1)
# extract out the year as
dataframe['policy_bind_year'] = dataframe['policy_bind_date'].str.extract('(\d{4})\-').astype('int32')
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

