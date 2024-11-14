import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from credit_risk_graph import Graphs


class InconsistentValues(Graphs):

    def __init__(self, file_path):
        # Inicialize the class Graphs and load the data
        super().__init__(file_path)

    def inconsistent_values(self):
        # Verifying inconsistent values
        print(self.credit_base.loc[self.credit_base["age"]<0])

        #Deleting the inconsistent values
        #credit_base2 = self.credit_base.drop(self.credit_base[self.credit_base["age"]<0].index)

        # Mean without inconsistent values
        mean = self.credit_base["age"][self.credit_base["age"]>0].mean()

        self.credit_base.loc[self.credit_base["age"]<0, "age"] = mean
        #print(self.credit_base.head(27))

    def missing_values(self):

        #print(self.credit_base.isnull().sum())
        print(self.credit_base.loc[pd.isnull(self.credit_base["age"])])

        self.credit_base["age"].fillna(self.credit_base["age"].mean(), inplace=True)

        print(self.credit_base.loc[pd.isnull(self.credit_base["age"])])

        # print(self.credit_base.loc(["clientid"]==29 | (self.credit_base["clientid"]==31), "clientid" | (self.credit_base["clientid"]==32)))

        print(self.credit_base.loc[self.credit_base["clientid"].isin([29,31,32])])

    # Division between predictors and classes
    def Division_predictors_class(self):

        print(type(self.credit_base))
        # Predictors
        x_credit = self.credit_base.iloc[:, 1:4].values
        print(type(x_credit))
        # Classes
        y_credit = self.credit_base.iloc[:, 4].values
        print(type(y_credit))










