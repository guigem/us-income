
import pandas as pd

df_train = pd.read_csv("../original/data_train.csv")
df_test = pd.read_csv("../original/data_test.csv")

categorical_columns = ['workclass', 
                       'education', 
                       'marital-status', 
                       'occupation', 
                       'relationship',
                       'race',
                       'sex',
                       'native-country',
                       'income'
                      ]

# Concatenate the train and test dataframes to get all the possible values
df_combined = pd.concat([df_train, df_test], ignore_index=True)

#Remove leading and trailing whitespaces from every value
for col in df_combined.columns:
    df_combined[col] = df_combined[col].apply(lambda x: str(x).strip())

#Replace inconsistent target values
df_combined['income'].replace({'>50K.': '>50K', '<=50K.': '<=50K'}, inplace=True)

# Factorize categorical column values into their alphabetical index
mappings = {}
for col in categorical_columns:
    df_combined[col] = df_combined[col].astype('category')
    mappings[col] = dict(enumerate(df_combined[col].cat.categories))
    df_combined[col] = pd.factorize(df_combined[col])[0]

# Split the combined dataframe back into train and test set again
nb_rows_train = len(df_train)
df_train = df_combined.iloc[:nb_rows_train,: ]
df_test = df_combined.iloc[nb_rows_train:, : ]

df_train.to_csv("data_train.csv", index=None)
df_test.to_csv("data_test.csv", index=None)





