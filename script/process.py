
from script.upsampling import upsampling
from script.model import model

import pandas as pd

class workflow:
    
    def __init__(self, data_train: pd.DataFrame, data_test: pd.DataFrame, resampling: int, model: int) -> None:
        '''
        
        This class represents the all workflow and process that is needed to classify
        our model:
            - 1st using or not a resampling technique
            - 2nd using an evaluation criteria or gridsearch to have a look 
            at the best hyperparameters


        Parameters
        ----------
        data_train : pd.DataFrame
            Data used to train our model.
        data_test : pd.DataFrame
            Data used to test our model .
        resampling : int
            Integer that indicates which resampling (or not) technique is asked:
                0: No resampling asked,
                1: Double the minority class (add a second time each observation),
                2: Using SMOTE technique to add new observations (still with the same majority)
                
        model : int
            Integer that indicates with evaluation method is asked:
                0: Holdout set that yields a classification report,
                1: Cross-validation that yields an accuracy.
                2: Gridsearch tha yields best parameters and best accuracy

        Returns
        -------
        None
            Does not return anything as outputs will just be printed in the console.

        '''
        self.data_train = data_train
        self.data_test = data_test
        self.resampling = resampling
        self.model = model
        
        #Calling upsampling class
        self.ups = upsampling(self.data_train, self.data_test)
        
        
    def process(self):
        '''
        
        This function does the flow described above.

        Returns
        -------
         Does not return anything as outputs will just be printed in the console.

        '''
        
        if self.resampling == 1:
            
            #Call function to double minority class
            self.data_train = self.ups.double_minority_class()
            
            #Get training and testing sets
            X_train, X_test, y_train, y_test = self.ups.split_train_test()
            
        elif self.resampling == 2:
            
            #Perform SMOTE algorithm
            X_train, X_test, y_train, y_test = self.ups.SMOTE()
            
        elif self.resampling == 3:
            
            X_train, X_test, y_train, y_test = self.ups.UnderSampler()
            
        else:
            
            #Get training and testing sets
            X_train, X_test, y_train, y_test = self.ups.split_train_test()
            
        #Choose the model wanted   
        model_choose = model(X_train, X_test, y_train, y_test)
        
        if self.model == 0:
            
            model_choose.random_forest_holdout()
            
        elif self.model == 1:
            
            model_choose.random_forest_cv()
            
        elif self.model == 2:
            
            model_choose.grid()
            
            
            
            
    
            
            
            
            
        