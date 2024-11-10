import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Data Exploration
credit_base = pd.read_csv("/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/database/credit_data.csv")

# View the database
#print(credit_base)

# view the first 10 data points
#print(credit_base.head(10))

# View the last data in the database
#print(credit_base.tail(8))

# View database statistics
print(credit_base.describe())

# Applying a filter to view the client with the highest income
#print(credit_base[credit_base["income"]>= 69995.685578])

# Applying a filter to view the client with the lowest loan
#print(credit_base[credit_base["loan"]<=1.377630])

# Viewing of data - Count the uniques values
print(np.unique(credit_base["default"], return_counts=True))

# Count the number of records in each class and generate a chart
sns.countplot(x=credit_base["default"], hue=credit_base["default"], palette=["blue", "orange"])
plt.xlabel("Default")
plt.ylabel("Count")
plt.title("Count of Default Values")
plt.savefig("pre_processing/graphs/default_graph.png")

# Historian chart - Age
plt.figure()
plt.hist(x=credit_base["age"], bins=20, color="red")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribuition of Age")
plt.savefig("pre_processing/graphs/historian_age.png")

# Historian chart - Income
plt.figure()
plt.hist(x=credit_base["income"], bins=20, color="blue")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Distribuition of Income")
plt.savefig("pre_processing/graphs/historian_income.png")

# Historian chart - Loan
plt.figure()
plt.hist(x=credit_base["loan"], bins=20, color="green")
plt.xlabel("Loan")
plt.ylabel("Frequency")
plt.title("Distribuition of Loan")
plt.savefig("pre_processing/graphs/historian_loan.png")

# Generating a dynamic graph combining others parameters of database
graph = px.scatter_matrix(credit_base, dimensions=["age", "income", "loan"], color="default")
# Use xdg-open pre_processing/graphs/dynamic_graph to view this graph in a HTML page
graph.write_html("pre_processing/graphs/dynamic_graph")
graph.show()



