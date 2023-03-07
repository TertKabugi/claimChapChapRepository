import pandas as pd
import numpy as np
import rename as rename

# Data Retrieval
dataset = pd.read_csv('dataset.csv', header=0)
print(dataset.shape)
print(dataset.info())

# Feature Engineering
dataframe = dataset.drop(dataset.columns[[3, 4, 9, 17, 22, 23, 24, 39]], axis=1)
print(dataframe.info())

dataframe['auto_make'] = dataframe['auto_make'].replace("Suburu", "Subaru")
dataframe['collision_type'] = dataframe['collision_type'].replace("?", "Undocumented")
dataframe['police_report_available'] = dataframe['police_report_available'].replace("?", "Undocumented")
dataframe['property_damage'] = dataframe['property_damage'].replace("?", "Undocumented")

# incident_severity transformation
dataframe['incident_severity'] = dataframe['incident_severity'].map({"Trivial Damage": 0,
                                                                     "Minor Damage": 1,
                                                                     "Major Damage": 2,
                                                                     "Total Loss": 3
                                                                     }).astype("int32")
print(dataframe.incident_severity.value_counts())

