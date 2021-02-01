
import pandas as pd
import json 

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


with open("mapping.txt", "a") as file_handle:
    for col in categorical_columns:
        df_combined[col] = df_combined[col].astype('category')
        df_combined[col], uniques = pd.factorize(df_combined[col], sort=True)
        
        mapping_values = uniques.categories.tolist()
        mapping_codes = uniques.codes

        file_handle.write("\n\n" + col + " : \n \n")

        mappings = {k: str(v) for k,v in zip(mapping_values,mapping_codes)}
        file_handle.write(json.dumps(mappings))






# Split the combined dataframe back into train and test set again
nb_rows_train = len(df_train)
df_train = df_combined.iloc[:nb_rows_train,: ]
df_test = df_combined.iloc[nb_rows_train:, : ]

df_train.to_csv("data_train.csv", index=None)
df_test.to_csv("data_test.csv", index=None)





