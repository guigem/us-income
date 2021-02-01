# Cleaning of the dataset

The cleaned version of the dataset was cleaned using `clean.py` and is simply replacing the categorical column values by their index in alphabetical order. This was done using `pandas.df.factorize()`
The exact mapping can be found in `mapping.txt`.
The continuous columns were kept unchanged.
