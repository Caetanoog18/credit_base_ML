import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from tensorflow.python.tpu.ops.gen_xla_ops import xla_sparse_core_adagrad_eager_fallback

from credit_risk_graph import Graphs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pre_processing.census_database import CensusDatabase


class InconsistentValues(Graphs):

    def __init__(self, file_path):
        # Inicialize the class Graphs and load the data
        super().__init__(file_path)
        self.x_credit = None
        self.y_credit = None
        self.scaler_credit = None
        self.x_credit_training = None
        self.x_credit_test = None
        self.y_credit_training = None
        self.y_credit_test = None

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

        self.credit_base["age"].fillna(self.credit_base["age"].mean())

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

    def training_credit_database(self):

        self.x_credit_training, self.x_credit_test, self.y_credit_training, self.y_credit_test = train_test_split(self.x_credit, self.y_credit, test_size=0.25, random_state=0)
        # print(self.x_credit_training.shape, self.x_credit_test.shape)
        # print(self.y_credit_training.shape, self.y_credit_test.shape)

    def save_variables(self):
        with open('/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/database/credit.pkl', mode='wb') as file:
            pickle.dump([self.x_credit_training, self.y_credit_training, self.x_credit_test, self.y_credit_test], file)

class Census(CensusDatabase):

    def __init__(self, file_path):
        self.x_census = None
        self.y_census = None
        self.label_encoder_test = None
        self.test = None
        self.x_census_training = None
        self.x_census_test = None
        self.y_census_training = None
        self.y_census_test = None
        super().__init__(file_path)

    def division_predictors_class_census(self):
        #print(self.census_data.columns)

        self.x_census = self.census_data.iloc[:, 0:14].values
        self.y_census = self.census_data.iloc[:, 14].values

        self.label_encoder_test = LabelEncoder()

        #print(self.x_census[:,1])

        self.test = self.label_encoder_test.fit_transform(self.x_census[:,1])

        #print(self.test)
        #print(self.x_census[0])

    def label_encoder(self):
        # Creating a variable to each category
        label_encoder_workclass = LabelEncoder()
        label_encoder_education = LabelEncoder()
        label_encoder_marital = LabelEncoder()
        label_encoder_occupation = LabelEncoder()
        label_encoder_relationship = LabelEncoder()
        label_encoder_race = LabelEncoder()
        label_encoder_sex = LabelEncoder()
        label_encoder_country = LabelEncoder()

        # Accessing x_census and applying the label encoder
        self.x_census[:, 1] = label_encoder_workclass.fit_transform(self.x_census[:,1])
        self.x_census[:, 3] = label_encoder_education.fit_transform(self.x_census[:,3])
        self.x_census[:, 5] = label_encoder_marital.fit_transform(self.x_census[:,5])
        self.x_census[:, 6] = label_encoder_occupation.fit_transform(self.x_census[:,6])
        self.x_census[:, 7] = label_encoder_relationship.fit_transform(self.x_census[:,7])
        self.x_census[:, 8] = label_encoder_race.fit_transform(self.x_census[:,8])
        self.x_census[:, 9] = label_encoder_sex.fit_transform(self.x_census[:,9])
        self.x_census[:, 13] = label_encoder_country.fit_transform(self.x_census[:,13])

        #print(self.x_census[0])
        #print(self.x_census)

    def one_hot_encoder(self):

        #print(len(np.unique(self.census_data["workclass"]))) # 9 different categories

        #print(len(np.unique(self.census_data["occupation"]))) # 15 different categories

        #print(self.x_census.shape)

        one_hot_encoder_census = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder="passthrough")

        self.x_census = one_hot_encoder_census.fit_transform(self.x_census).toarray()

        #print(self.x_census.shape)

    def scaling_values(self):

        scaler_census = StandardScaler()

        print(self.x_census)

        self.x_census = scaler_census.fit_transform(self.x_census)

        print(self.x_census[0])

    def training_census_database(self):

        self.x_census_training, self.x_census_test, self.y_census_training, self.y_census_test = train_test_split(self.x_census, self.y_census, test_size=0.15, random_state=0)
        # print(self.x_census_training.shape, self.y_census_training.shape)
        # print(self.x_census_test.shape, self.y_census_test.shape)

    def save_variables(self):

        with open('/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/database/census.pkl', mode='wb') as file:
            pickle.dump([self.x_census_training, self.y_census_training, self.x_census_test, self.y_census_test], file)
