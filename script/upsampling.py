
import pandas as pd

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class upsampling:
    
    def __init__(self, data_train: pd.DataFrame, data_test: pd.DataFrame) -> None:
        '''
        Class takes the train and test dataset 

        Parameters
        ----------
        data_train : pd.DataFrame
            Data used to train our model.
        data_test : pd.DataFrame
            Data used to test our model .


        '''
        self.data_train = data_train
        self.data_test = data_test
        
    def double_minority_class(self) -> pd.DataFrame:
        '''
        Double the number of instances of the minority class

        Returns
        -------
        data_train : pd.DataFrame
            Returns the data-train with 2*the minority class.

        '''
        
        #Select only minority class
        data_train_def = self.data_train.loc[self.data_train["income"]==1]
        
        #Double it
        data_train = pd.concat([self.data_train, data_train_def])
        
        return data_train
    
    def split_train_test(self) -> pd.DataFrame:
        '''
        
        Divide the target from the independant variables for the two datasets

        Returns
        -------
        X_train : pd.DataFrame
            Dataset of independant variables for training.
        X_test : pd.DataFrame
            Dataset of independant variables for testing.
        y_train : pd.DataFrame
            Dataset of targets for training.
        y_test : pd.DataFrame
            Dataset of targets for testing.

        '''
        
        #Drop the target variable
        X_train = self.data_train.drop("income", axis=1)
        X_test = self.data_test.drop("income", axis=1)
        
        #Only select the target variable
        y_train = self.data_train["income"]
        y_test = self.data_test["income"]
        
        #Standardize both independant variables datasets
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def SMOTE(self) -> pd.DataFrame:
        '''
        Upsampling using SMOTE technique.

        Returns
        -------
        X_train : pd.DataFrame
            Dataset of independant variables for training.
        X_test : pd.DataFrame
            Dataset of independant variables for testing.
        y_train : pd.DataFrame
            Dataset of targets for training.
        y_test : pd.DataFrame
            Dataset of targets for testing.

        '''
        X_train, X_test, y_train, y_test = upsampling.split_train_test(self)
        
        #Minority class is 50% o majority class
        sm = SMOTE(random_state=42, sampling_strategy=0.5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    
    def UnderSampler(self) -> pd.DataFrame:
        '''
        Undersampling majority class as 2 times the minority class.

        Returns
        -------
        X_train : pd.DataFrame
            Dataset of independant variables for training.
        X_test : pd.DataFrame
            Dataset of independant variables for testing.
        y_train : pd.DataFrame
            Dataset of targets for training.
        y_test : pd.DataFrame
            Dataset of targets for testing.
       '''
        X_train, X_test, y_train, y_test = upsampling.split_train_test(self)
        
        under = RandomUnderSampler(sampling_strategy = 0.5, random_state = 42)
        X_train, y_train = under.fit_resample(X_train, y_train)
    
        return X_train, X_test, y_train, y_test
        
        