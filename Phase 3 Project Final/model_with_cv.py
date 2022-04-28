from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

from sklearn.metrics import plot_roc_curve, roc_auc_score

from sklearn.feature_selection import RFE

from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
RANDOM_SEED = 8    # Set a random seed for reproducibility!

from sklearn.metrics import plot_confusion_matrix, recall_score,accuracy_score, precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier,VotingClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
import numpy as np
from numpy import nan
import pandas as pd
import copy

#scoring function (cross_val)
class ModelWithCV():
    '''Structure to save the model and more easily see its crossvalidation'''
    
    def __init__(self, model, model_name, X, y, cv_now=True):
        self.model = model
        self.name = model_name
        self.X = X
        self.y = y
        # For CV results
        self.cv_results = None
        self.cv_mean = None
        self.cv_median = None
        self.cv_std = None
        #
        if cv_now:
            self.cross_validate()
        
    def cross_validate(self, X=None, y=None, kfolds=10):
        '''
        Perform cross-validation and return results.
        
        Args: 
          X:
            Optional; Training data to perform CV on. Otherwise use X from object
          y:
            Optional; Training data to perform CV on. Otherwise use y from object
          kfolds:
            Optional; Number of folds for CV (default is 10)  
        '''
        
        cv_X = X if X else self.X
        cv_y = y if y else self.y

        self.cv_results = cross_val_score(self.model, cv_X, cv_y, cv=kfolds)
        self.cv_mean = np.mean(self.cv_results)
        self.cv_median = np.median(self.cv_results)
        self.cv_std = np.std(self.cv_results)

        
    def print_cv_summary(self):
        cv_summary = (
        f'''CV Results for `{self.name}` model:
            {self.cv_mean:.5f} ± {self.cv_std:.5f} accuracy
        ''')
        print(cv_summary)
        
        
     #scoring function (aoc-roc)
class ModelWithAOCROC(ModelWithCV):
    """Structure for scoring classfication models"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def plt_roc_curve(self,X=None,y=None):
        cv_X = X if X else self.X
        cv_y = y if y else self.y
        y_preds = self.model.predict(cv_X)
        plot_roc_curve(self.model,cv_X,cv_y)