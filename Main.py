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

import time
start = time.time()
print("Analyses and time started.")

dataset_main = pd.DataFrame([], columns=['Run ID', 'Building ID', 'Number of Storey', 'Number of Span', 'Span Length',
                                         'Storey Height', 'Column Width', 'Column Depth', 'Beam Width', 'Beam Depth',
                                         'Concrete Strength', 'Steel Strength', 'First Storey - Commercial Use',
                                         'Soil Condition', 'First Mode Period', 'Period Class', 'Earthquake URL', 'PGA',
                                         'PGV', 'Sa(T1)', 'Sd(T1)', 'CAV', 'MIDR'])

# Selected Records
SelectedRecords = 'C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/EQ_Record/Baker_Record/Selected_Record_File_Scale'
BakerSelectedRecords = getAllFilesFromFolder(SelectedRecords)
# Folder that keep maximum displacements of each building for temporarily.
recorder_folder = 'C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/RecorderFolder/'

# failed_data_file = 'C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/failed_data.h5'
# data = pd.read_hdf(failed_data_file, 'df')
data_file = 'C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/data_results.h5'
data = pd.read_hdf(data_file, 'df')

number_of_analysis = 0
for data_cluster in data.itertuples(index=False):
    building_number = data_cluster.building_id
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
    periodClass = data_cluster.period_class

    for j in range(1, len(BakerSelectedRecords) + 1):
        if BakerSelectedRecords[j - 1] == SelectedRecords + '/' + str(periodClass) + '.dat':
            BakerOutput = BakerSelectedRecords[j - 1]
            break

    scaleFactor_list, recordURL_list = readBakerOutputTable(BakerOutput)

    data_folder = 'C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/EQ_Record/Baker_Record/Selected CyberShake GMs'
    for i in range(1, len(recordURL_list) + 1):
        record_number = i
        accfile = data_folder + '/' + recordURL_list[record_number - 1]
        scaleFactor = float(scaleFactor_list[record_number - 1])
        record_values, length_of_record, dt, factor, PGA, PGV, Sa, Sd, CAV = processBakerfile(accfile, scaleFactor)

        ops.wipe()
        model = CreateModel(num_span, num_storey, span_length, storey_height, soft_story)
        model.create_nonlinear_model_precode(fc, fy, cover, As_col_rat, As_beam_rat, column_width,
                                             column_depth, beam_width, beam_depth, dead_load, live_load, soft_story)
        model.add_staticLoad(column_width, column_depth, beam_width, beam_depth, dead_load, live_load)
        model.static_analysis()
        # model.recorder_maxDisp(recorder_folder, record_number, building_number)
        model.add_dynamicLoad(record_values, dt, factor, num_modes)
        MIDR = model.dynamic_analysis(length_of_record * dt, dt)

        list_of_input = [number_of_analysis + 1, building_number, num_storey, num_span, span_length, storey_height, column_width,
                         column_depth, beam_width, beam_depth, fc, fy, soft_story, data_cluster.soil_condition,
                         data_cluster.period, periodClass, recordURL_list[record_number - 1], PGA, PGV, Sa, Sd, CAV]
        dataset_main.loc[len(dataset_main)] = list_of_input
        number_of_analysis = number_of_analysis + 1

    ops.wipe()
    MIDR = model.get_MIDR(recorder_folder, record_number, building_number)
    dataset_main.insert(len(dataset_main.columns), "MIDR", MIDR, True)
    # for i in range(1, len(recordURL_list) + 1):
    #     dataset_main.loc[number_of_analysis - len(recordURL_list) + i - 1, 'MIDR'] = MIDR[i - 1]

    if number_of_analysis % 100 == 0:
        end = time.time()
        time_spent = round((end - start), 2)
        print('\n' + str(number_of_analysis) + '. analysis is completed in ' + str(time_spent) + ' seconds.')

with pd.HDFStore('C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/data_all_tables_1xSteel.h5') as store:
    store.put('df', dataset_main, format='table')
dataset_main.to_excel('C:/Users/aliat/OneDrive - boun.edu.tr/Desktop/MasterThesis/Openseespy/OpenSees/toExcel_dataset_main_1xSteel.xlsx', index = False)
print('debug')
# TODO! Convergence hatasını düzelt. Try-Except
# Passed for now. Ask to Elif
# TODO! Parallel processing ile hızlandırılma yapılacak.
# TODO! PGA, PGV, Sa, Sd, CAV hesaplarını ekle.
# TODO! Newmark'tan gelen sonuçların karşılaştırılması.



