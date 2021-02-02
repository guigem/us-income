
import os

import pandas as pd

from script.process import workflow

#Get absolute of the main.py file
basedir = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    
   #Getting path fior both training and testing data
   data_path_train = os.path.join(basedir, "Data/cleaned/data_train.csv")
   data_path_test = os.path.join(basedir, "Data/cleaned/data_test.csv")
   
   #Reading the files
   data_train = pd.read_csv(data_path_train)   
   data_test = pd.read_csv(data_path_test)   
   
   #Calling worflow class from process.py with the data and choice for upsampling and evaluation criteria
   score = workflow(data_train, data_test, 2, 0)
   
   score.process()
   



