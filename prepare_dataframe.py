import numpy as np
from Units import *
import pandas as pd
import itertools

building_id = [-999]
concrete_comp_strength = [5, 12, 20, 28, 35]
steel_yield_strength = [370]
num_span = [3, 4]
num_storey_6storey = [3]
num_storey_8storey = [5]
span_length = [2.5, 3., 4.]
storey_height = [2.6, 2.8, 3.0]
soil_condition = [1130, 560, 270] # ZB, ZC, ZD
soft_story = ['Yes', 'No']
column_dimension_for_6storey = [0.25, 0.30, 0.35]
column_dimension_for_8storey = [0.30, 0.35, 0.40]
period = [-999]
period_class = [-999]

list_of_input_6storey = [building_id, concrete_comp_strength, steel_yield_strength, num_span, num_storey_6storey, span_length,
                         storey_height, soil_condition, soft_story, column_dimension_for_6storey, period, period_class]

list_of_input_8storey = [building_id, concrete_comp_strength, steel_yield_strength, num_span, num_storey_8storey, span_length,
                         storey_height, soil_condition, soft_story, column_dimension_for_8storey, period, period_class]

input_dataframe_6storey = pd.DataFrame(list(itertools.product(*list_of_input_6storey)),
                               columns=['building_id', 'concrete_comp_strength', 'steel_yield_strength', 'num_span', 'num_storey',
                                        'span_length', 'storey_height', 'soil_condition', 'soft_story',
                                        'column_dimension', 'period', 'period_class'])

input_dataframe_8storey = pd.DataFrame(list(itertools.product(*list_of_input_8storey)),
                               columns=['building_id', 'concrete_comp_strength', 'steel_yield_strength', 'num_span', 'num_storey',
                                        'span_length', 'storey_height', 'soil_condition', 'soft_story',
                                        'column_dimension', 'period', 'period_class'])

input_dataframe = pd.concat([input_dataframe_6storey, input_dataframe_8storey], ignore_index=True)
input_dataframe.to_hdf('data.h5', key='df', mode='w')



