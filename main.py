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
print(len(all_var))

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


