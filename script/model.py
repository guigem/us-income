
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

class model:
    
    def __init__(self, X_train, X_test, y_train, y_test):
        '''
        Initialise all the training and testing sets

        Parameters
        ----------
        X_train : pd.DataFrame
            Dataset of independant variables for training.
        X_test : pd.DataFrame
            Dataset of independant variables for testing.
        y_train : pd.DataFrame
            Dataset of targets for training.
        y_test : pd.DataFrame
            Dataset of targets for testing.


        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def random_forest_holdout(self):
        '''
        Create a random forest with the hyperparameters already tuned.
        The model is evaluated with a simple holdout set.

        Returns
        -------
        Does not return anything as outputs will just be printed in the console..

        '''
        
        #Create a Random forest classifier
        clf_forest=RandomForestClassifier(n_estimators= 500, max_depth= 30, random_state=42, max_features=0.3)

        clf_forest.fit(self.X_train, self.y_train)
        
        #Build prediction on testing set
        y_pred = clf_forest.predict(self.X_test)
        
        #Print the classification report
        print(classification_report(self.y_test,y_pred))
        
    def random_forest_cv(self):
        '''
        Create a random forest with the hyperparameters already tuned.
        The model is evaluated with a cross-validation with K=10.
        
        Returns
        -------
        Does not return anything as outputs will just be printed in the console.

        '''
        clf_forest=RandomForestClassifier(n_estimators= 500, max_depth= 30, random_state=42, max_features=0.3)
        
        #Build aggregated datasets for cv
        X_cv = pd.concat([self.X_train, self.X_test])
        y_cv = pd.concat([self.y_train, self.y_test])
        
        #Evaluation with cross validation 
        scores_randfor = cross_val_score(clf_forest, X_cv, y_cv, cv=10)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_randfor.mean(), scores_randfor.std()))
        
    
    def grid(self):
        '''
        
        Build a grid in order to test hyperparameters of random forest

        Returns
        -------
        Does not return anything as outputs will just be printed in the console.

        '''
        clf_forest=RandomForestClassifier(random_state=42)
        
        #Trying out 5 parameters
        params= {
            'n_estimators' : np.arange(300,600,50).tolist(),
            'max_depth' : np.arange(10,50,20).tolist(),
            'class_weight' : ["balanced", "balanced_subsample", None],
            'criterion' : ["gini", "entropy"],
            'max_features' : np.arange(0.1,1,0.1).tolist()
            
            }
        
        #Using accuracy as scoring method and using 10 folds
        gridsearch = GridSearchCV(estimator = clf_forest,
                                param_grid = params,
                                scoring = 'accuracy', 
                                cv = 10, # Use 5 folds
                                verbose = 1,
                                n_jobs = -1 #Use all but one CPU core
                                )
        
        result = gridsearch.fit(self.X_train, self.y_train)
        
        print("The best parameters are :", result.best_params_)
        print("The best accuracy is {:.2f}%:".format(result.best_score_ * 100))
        
        rf = result.best_estimator_
        score = rf.score(self.X_test, self.y_test)
        print("The generalization accuracy of the model is {:.2f}%".format(score * 100))
