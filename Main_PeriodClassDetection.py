#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 04:56:58 2022

@author: onurulku
"""

import openseespy.opensees as ops
from CreateModel import CreateModel
# import openseespy.postprocessing.ops_vis as opsv
import matplotlib.pyplot as plt
import opsvis as opsv
import math
import numpy as np
from Units import *
from EarthquakeRecordProcess import *
import pandas as pd

data = pd.read_hdf('data.h5', 'df')
number_of_analysis = 0
for data_cluster in data.itertuples(index=False):
    # print(data_cluster.num_storey)
    ops.wipe()
    num_span = data_cluster.num_span
    num_storey = data_cluster.num_storey
    # span_length = [5.]
    span_length = data_cluster.span_length
    storey_height = data_cluster.storey_height
    # Column width is parallel to implemented force
    column_width = data_cluster.column_dimension * m
    column_depth = data_cluster.column_dimension * m
    # Beam width is parallel to implemented force
    beam_width = 0.25 * m
    beam_depth = 0.6 * m
    fc = data_cluster.concrete_comp_strength
    fy = data_cluster.steel_yield_strength * MPa
    cover = 4 * cm
    As_col_rat = 0.01
    As_beam_rat = 0.004
    live_load = 2 * kN / (m ** 2)
    dead_load = 2 * kN / (m ** 2)
    soft_story = data_cluster.soft_story
    num_modes = 10

    model = CreateModel(num_span, num_storey, span_length, storey_height, soft_story)
    model.create_nonlinear_model_precode(fc, fy, cover, As_col_rat, As_beam_rat, column_width,
                                         column_depth, beam_width, beam_depth, dead_load, live_load, soft_story)
    model.add_staticLoad(column_width, column_depth, beam_width, beam_depth, dead_load, live_load)
    model.static_analysis()
    model.mode_shape(1, num_modes)
    modalProps = ops.modalProperties('-return')
    T1 = modalProps['eigenPeriod'][0]
    data.loc[number_of_analysis, 'period'] = T1
    number_of_analysis = number_of_analysis + 1
    data.loc[number_of_analysis - 1, 'building_id'] = number_of_analysis
    print('Debug Stop Here')

T = data.loc[:, 'period']
period_intervals = np.linspace(np.min(T), np.max(T), num=11)  # Define the bin edges to create intervals
period_intervals[0] = math.floor(period_intervals[0]*100)/100 # Round 2 decimal the first value down and the last value up
period_intervals[-1] = math.ceil(period_intervals[-1]*100)/100
period_intervals_midpoints = [(period_intervals[i] + period_intervals[i+1])/2 for i in range(len(period_intervals)-1)]
period_intervals_midpoints = [round(x, 2) for x in period_intervals_midpoints]
print(period_intervals_midpoints)
period_class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]         # Define the class labels
data['period_class'] = pd.cut(data['period'], period_intervals, labels=period_class_labels)  # Use cut to create intervals based on the bin edges and assign class labels
with pd.HDFStore('data_results.h5') as store:
    store.put('df', data, format='table')

data_results = pd.read_hdf('data_results.h5', 'df')
data_results['period'][:].to_excel('toExcel_period.xlsx', index = False)
data_results['period_class'][:].to_excel('toExcel_periodClass.xlsx', index = False)
