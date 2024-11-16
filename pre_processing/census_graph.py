import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pre_processing.census_database import *

class CensusGraph(CensusDatabase):

    def __init__(self, file_path):
        super().__init__(file_path)

    # Viewing the data
    def view_census_data(self, name):
        print(np.unique(self.census_data["income"], return_counts=True))
        sns.countplot(x=self.census_data["income"], hue=self.census_data["income"], palette=["blue", "orange"])
        plt.xlabel("Income")
        plt.ylabel("Count")
        plt.title("Income")
        plt.savefig(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/{name}.png")

    def historian(self):
        plt.figure()
        plt.hist(self.census_data["income"], bins=20, color="green")
        plt.xlabel("Income")
        plt.ylabel("Frequency")
        plt.title("Hist Income")
        plt.savefig(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/hist_income.png")

        plt.figure()
        plt.hist(self.census_data["education-num"], bins=20, color="red")
        plt.xlabel("Education")
        plt.ylabel("Frequency")
        plt.title("Hist Education")
        plt.savefig(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/hist_education.png")

        plt.figure()
        plt.hist(self.census_data["hour-per-week"], bins=20, color="blue")
        plt.xlabel("Hours")
        plt.ylabel("Frequency")
        plt.title("Hist Hours per week")
        plt.savefig(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/hist_hour_per_week.png")
        plt.show()

        graph = px.treemap(self.census_data, path=['workclass', 'age'])
        graph.write_html(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/treemap.html")
        graph.show()

        graph = px.treemap(self.census_data, path=['occupation', 'relationship', 'age'])
        graph.write_html(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/treemap_2.html")
        graph.show()

        graph = px.parallel_categories(self.census_data, dimensions=['workclass','occupation', 'income'])
        graph.write_html(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/parallel_categories.html")
        graph.show()



