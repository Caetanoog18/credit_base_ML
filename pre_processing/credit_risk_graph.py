import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


class Graphs:

    def __init__(self, file_path):
        self.file_path = file_path
        self.credit_base = pd.read_csv(file_path)

    # View database statistics
    def show_statistics(self):
        print(self.credit_base.describe())

    def show_first_and_last_data(self, head_count, tail_count):
        print("First data")
        print(self.credit_base.head(head_count))
        print("Last data")
        print(self.credit_base.tail(tail_count))

    def filter_highest_income(self):
        high_income = self.credit_base[self.credit_base["income"]>=69995.685578]
        print(high_income)

    def filter_lowest_loan(self):
        lowest_loan = self.credit_base[self.credit_base["loan"]<=1.377630]
        print(lowest_loan)

    def unique_data(self):
        print(np.unique(self.credit_base["default"], return_counts=True))

    def records_in_each_class(self, name):
        sns.countplot(x=self.credit_base["default"], hue=self.credit_base["default"], palette=["blue", "orange"])
        plt.xlabel("Default")
        plt.ylabel("Count")
        plt.title("Count of Default Values")
        plt.savefig(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/{name}.png")

    def historian_chart_age(self, name):
        plt.figure()
        plt.hist(x=self.credit_base["age"], bins=20, color="red")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.title("Distribuition of Age")
        plt.savefig(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/{name}.png")

    def historian_chart_income(self, name):
        plt.figure()
        plt.hist(x=self.credit_base["income"], bins=20, color="blue")
        plt.xlabel("Income")
        plt.ylabel("Frequency")
        plt.title("Distribuition of Income")
        plt.savefig(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/{name}.png")

    def historian_chart_loan(self, name):
        plt.figure()
        plt.hist(x=self.credit_base["loan"], bins=20, color="green")
        plt.xlabel("Loan")
        plt.ylabel("Frequency")
        plt.title("Distribuition of Loan")
        plt.savefig(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/{name}.png")

    def generate_dynamic_graph(self, name):
        # Generating a dynamic graph combining others parameters of database
        graph = px.scatter_matrix(self.credit_base, dimensions=["age", "income", "loan"], color="default")
        # Use xdg-open pre_processing/graphs/dynamic_graph to view this graph in an HTML page
        graph.write_html(f"/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/graphs/{name}.html")
        graph.show()




