import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from credit_risk_graph import Graphs
from sklearn.preprocessing import StandardScaler

class CensusDatabase:
    def __init__(self, file_path):
        self.file_path = file_path
        self.census_data = pd.read_csv(self.file_path)


    def statistics_census(self):
        print(self.census_data.describe())
        print(self.census_data.isnull().sum())