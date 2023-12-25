#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 04:56:58 2022
@author: onurulku
"""
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

recorder_folder = 'C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/RecorderFolder/'

# Baker Records
SelectedRecords = 'C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/bakerMatlab/CS_Selection-master/Data'
BakerSelectedRecords = getAllFilesFromFolder(SelectedRecords)

num_span = 4 #data_cluster.num_span
num_storey = 6 #data_cluster.num_storey
span_length = 2.5 #data_cluster.span_length
storey_height = 2.6 #data_cluster.storey_height
column_width = 0.3 #data_cluster.column_dimension * m
column_depth = 0.3 #data_cluster.column_dimension * m
beam_width = 0.25 * m
beam_depth = 0.6 * m
fc = 25 #data_cluster.concrete_comp_strength
fy = 220 * MPa #data_cluster.steel_yield_strength * MPa
cover = 4 * cm
As_col_rat = 0.01
As_beam_rat = 0.004
live_load = 2 * kN / (m ** 2)
dead_load = 2 * kN / (m ** 2)
soft_story = ['No'] #data_cluster.soft_story
periodClass = '1'
num_modes = 10

for j in range(1, len(BakerSelectedRecords) + 1):
    if BakerSelectedRecords[j-1] == SelectedRecords + '/' + periodClass + '.dat':
        BakerOutput = BakerSelectedRecords[j-1]
        break

scaleFactor_list, recordURL_list = readBakerOutputTable(BakerOutput)

data_folder = 'C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/EQ_Record/Baker_Record/Selected CyberShake GMs'
for i in range(1, len(recordURL_list) + 1):
    record_number = i
    accfile = data_folder + '/' + recordURL_list[record_number-1]
    scaleFactor = float(scaleFactor_list[record_number-1])
    record_values, length_of_record, dt, factor = processBakerfile(accfile, scaleFactor)

    ops.wipe()

    model = CreateModel(num_span, num_storey, span_length, storey_height, soft_story)
    model.create_nonlinear_model_precode(fc, fy, cover, As_col_rat, As_beam_rat, column_width,
                                         column_depth, beam_width, beam_depth, dead_load, live_load, soft_story)
    model.add_staticLoad(column_width, column_depth, beam_width, beam_depth, dead_load, live_load)
    model.static_analysis()
    model.recorder_maxDisp(recorder_folder, record_number)
    model.add_dynamicLoad(record_values, dt, factor, num_modes)
    model.dynamic_analysis(length_of_record * dt + dt, dt)
ops.wipe()
MIDR = model.get_MIDR(recorder_folder, record_number)

print(MIDR)
print('Debug Stop Here')