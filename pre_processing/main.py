from tensorflow.python.ops.summary_ops_v2 import graph

from pre_processing.handling_inconsistent_values import InconsistentValues
from pre_processing.credit_risk_graph import *


file_path = "/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/database/credit_data.csv"

graphs = Graphs(file_path)
graphs.generate_dynamic_graph("dynamic_1")
graphs.historian_chart_age("historian_age_1")

inconsistent_values = InconsistentValues(file_path)
inconsistent_values.inconsistent_values()

inconsistent_values.generate_dynamic_graph("dynamic_1")
inconsistent_values.historian_chart_age("historian_age_2")

inconsistent_values.missing_values()


inconsistent_values.Division_predictors_class()



