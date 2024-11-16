import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from credit_risk_graph import Graphs
from sklearn.preprocessing import StandardScaler


class InconsistentValues(Graphs):

    def __init__(self, file_path):
        # Inicialize the class Graphs and load the data
        super().__init__(file_path)
        self.x_credit = None
        self.y_credit = None
        self.scaler_credit = None

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
    def division_predictors_class(self):

        print(type(self.credit_base))
        # Predictors
        self.x_credit = self.credit_base.iloc[:, 1:4].values
        print(type(self.x_credit))
        # Classes
        self.y_credit = self.credit_base.iloc[:, 4].values
        print(type(self.y_credit))

    def standard_scaler(self):

        # Printing the maximum and the minimum values
        print(self.x_credit[:, 0].min(), self.x_credit[:,1].min(), self.x_credit[:, 2].min())
        print(self.x_credit[:, 0].max(), self.x_credit[:, 1].max(), self.x_credit[:, 2].max())

        self.scaler_credit = StandardScaler()
        self.x_credit = self.scaler_credit.fit_transform(self.x_credit)

        # Printing the maximum and the minimum values
        print(self.x_credit[:, 0].min(), self.x_credit[:,1].min(), self.x_credit[:, 2].min())
        print(self.x_credit[:, 0].max(), self.x_credit[:, 1].max(), self.x_credit[:, 2].max())








