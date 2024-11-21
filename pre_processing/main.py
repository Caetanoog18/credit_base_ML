from tensorflow.python.ops.summary_ops_v2 import graph

from pre_processing.census_database import CensusDatabase
from pre_processing.census_graph import CensusGraph
from pre_processing.handling_inconsistent_values import InconsistentValues, Census
from pre_processing.credit_risk_graph import *
from pre_processing.census_database import *


file_path_credit = "/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/database/credit_data.csv"
file_path_census = "/home/gabriel-caetano/Desktop/Tensorflow/pre_processing/database/census.csv"

# graphs = Graphs(file_path_credit)
# graphs.generate_dynamic_graph("dynamic_1")
# graphs.historian_chart_age("historian_age_1")
#
# inconsistent_values = InconsistentValues(file_path_credit)
# inconsistent_values.inconsistent_values()
#
# inconsistent_values.generate_dynamic_graph("dynamic_1")
# inconsistent_values.historian_chart_age("historian_age_2")
#
# inconsistent_values.missing_values()
#
# inconsistent_values.division_predictors_class()

census_database = CensusDatabase(file_path_census)
census_database.statistics_census()

census_graph = CensusGraph(file_path_census)
census_graph.view_census_data("census_graph_income")
census_graph.historian()

census_graph = Census(file_path_census)
census_graph.division_predictors_class_census()
census_graph.label_encoder()
census_graph.one_hot_encoder()
census_graph.scaling_values()

