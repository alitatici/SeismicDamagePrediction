import os
import openseespy.opensees as ops
from CreateModel import CreateModel
# import openseespy.postprocessing.ops_vis as opsv
import matplotlib.pyplot as plt
import opsvis as opsv
import math
import numpy as np
from Units import *
from GMProcess import *
import pandas as pd
import itertools

# HDF5 file that include uncompleted NLTH analyses
file = 'C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/03_Periods_GMSelection/Step_2/data_all_tables.h5'
all_data = pd.read_hdf(file, 'df')

failed_data = all_data[all_data['MIDR'] == -999]

## ## ## ## ##   ------------------------------------   ## ## ## ## ##
## ## ## ## ##   BELOW: prepared for old result file.   ## ## ## ## ##

# you need to delete soil condition insert and change column name of soil condition according to new data
# and also change 'index':'building_id' part to 'Building ID':'building_id'. etc. etc.
# ALL DEPENDS ON WHAT YOU INPUT AND WHAT YOU NEED.

soil_condition = [760] * 795
failed_data.insert(0, "soil_condition", soil_condition, True)
failed_data.reset_index(inplace=True)

failed_data.rename(columns = {'Number of Storey':'num_storey', 'Number of Span':'num_span', 'Span Length':'span_length',
                              'Storey Height':'storey_height', 'Column Width':'column_dimension',
                              'First Storey - Commercial Use':'soft_story', 'index':'building_id',
                              'Concrete Strength':'concrete_comp_strength', 'Steel Strength':'steel_yield_strength',
                              'First Mode Period':'period', 'Period Class':'period_class'}, inplace = True)

## ## ## ## ##   ABOVE: prepared for old result file.   ## ## ## ## ##
## ## ## ## ##   ------------------------------------   ## ## ## ## ##

with pd.HDFStore('C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/failed_data.h5') as store:
    store.put('df', failed_data, format='table')

print('stop here to see variables in debugging mode')