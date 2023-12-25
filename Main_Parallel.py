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
from joblib import Parallel, delayed
import os
from EarthquakeRecordProcess import *
import time
from SeismicIntensityMeasure import *

dirname = os.path.dirname(__file__)


def process_row(row):
    building_number = row.building_id
    print(str(building_number) + '. Building')
    ops.wipe()
    num_span = row.num_span
    num_storey = row.num_storey
    span_length = row.span_length
    storey_height = row.storey_height
    # Column width is parallel to implemented force
    column_width = row.column_dimension * m
    column_depth = row.column_dimension * m
    # Beam width is parallel to implemented force
    beam_width = 0.25 * m
    beam_depth = 0.6 * m
    fc = row.concrete_comp_strength
    fy = row.steel_yield_strength * MPa
    cover = 4 * cm
    As_col_rat = 0.01
    As_beam_rat = 0.004
    live_load = 2 * kN / (m ** 2)
    dead_load = 2 * kN / (m ** 2)
    soft_story = row.soft_story
    num_modes = 10
    periodClass = row.period_class
    Vs30 = row.soil_condition
    period = row.period

    selected_GMs = earthquake_data[(earthquake_data['Vs30_data'] == Vs30) & (earthquake_data['Tcond_data_class'] == periodClass)]
    record_number = 1
    for id, record_data in selected_GMs.iterrows():
        number_of_analysis = (building_number - 1) * len(selected_GMs.index) + record_number
        record_id = record_data.Record_ID
        ags = record_data.ground_acc
        dt = record_data.deltat
        factor = record_data.scale_factor
        length_of_record = len(ags)
        Sa, Sv, Sd = get_SpectralValues(period, 0.05, (np.array(ags) * factor).tolist(), dt)
        PGA = record_data.PGA
        PGV = record_data.PGV
        PGD = record_data.PGD
        CAV = record_data.CAV

        ops.wipe()
        model = CreateModel(num_span, num_storey, span_length, storey_height, soft_story)
        model.create_nonlinear_model_precode(fc, fy, cover, As_col_rat, As_beam_rat, column_width,
                                             column_depth, beam_width, beam_depth, dead_load, live_load, soft_story)
        model.add_staticLoad(column_width, column_depth, beam_width, beam_depth, dead_load, live_load)
        model.static_analysis()
        # model.recorder_maxDisp(recorder_folder, record_number, building_number)
        model.add_dynamicLoad(ags, dt, factor, num_modes)
        MIDR = model.dynamic_analysis(length_of_record * dt, dt)
        record_number += 1

        list_of_input = [number_of_analysis, row.building_id, num_storey, num_span, span_length, storey_height,
                         column_width,
                         column_depth, beam_width, beam_depth, fc, fy, soft_story, row.soil_condition,
                         row.period, periodClass, record_id, PGA, PGV, PGD, Sa, Sd, CAV, MIDR]
        dataset_main.loc[len(dataset_main)] = list_of_input
        # number_of_analysis = number_of_analysis + 1

    ops.wipe()
    # MIDR = model.get_MIDR(recorder_folder, record_number, building_number)
    # dataset_main.insert(len(dataset_main.columns), "MIDR", MIDR, True)
    # for i in range(1, len(recordURL_list) + 1):
    #     # dataset_main.loc[number_of_analysis - len(recordURL_list) + i - 1, 'MIDR'] = MIDR[i - 1]
    #     dataset_main.loc[(building_number - 1) * len(recordURL_list) + i - 1, 'MIDR'] = MIDR[i - 1]
    return dataset_main


start = time.time()
print("Analyses and time started.")

dataset_main = pd.DataFrame([], columns=['Run ID', 'Building ID', 'Number of Storey', 'Number of Span', 'Span Length',
                                         'Storey Height', 'Column Width', 'Column Depth', 'Beam Width', 'Beam Depth',
                                         'Concrete Strength', 'Steel Strength', 'First Storey - Commercial Use',
                                         'Soil Condition', 'First Mode Period', 'Period Class', 'Earthquake URL', 'PGA',
                                         'PGV', 'PGD', 'Sa(T1)', 'Sd(T1)', 'CAV', 'MIDR'])


# Selected Earthquakes
SelectedRecords = 'Data_Ali'
BakerSelectedRecords = getAllFilesFromFolder(SelectedRecords)
data_dict = {'Tcond_data_class': [], 'Vs30_data': [], 'no_database': [], 'file': []}
T_cond = [0.24, 0.33, 0.41, 0.5, 0.59, 0.67, 0.76, 0.85, 0.94, 1.02]
T_cond_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for files in BakerSelectedRecords:
    if not '._' in files:
        splitted_files = files.split('_')
        data_dict['Vs30_data'].append(int(splitted_files[6]))
        index = T_cond.index(float(splitted_files[4]))
        data_dict['Tcond_data_class'].append(T_cond_class[index])
        data_dict['file'].append(files)
        data_dict['no_database'].append(splitted_files[7].split('.')[0])

data_df = pd.DataFrame(data=data_dict)

earthquake_data = read_data_and_write_to_df(dirname, data_df)

data_file = os.path.join(dirname, 'data_results.h5')
data = pd.read_hdf(data_file, 'df')
data = data[1500:2540]

results = Parallel(n_jobs=1)(
    delayed(process_row)(row) for _, row in data.iterrows()
)

dataset_main_parallel = pd.concat(results)
end = time.time()
time_spent = round((end - start), 2)
print(time_spent)
dataset_main_parallel.to_hdf(os.path.join(dirname, 'dataset_all_1500_2540.h5'), key='df', mode='w', format='table')
# dataset_main_parallel = pd.read_hdf(os.path.join(dirname, 'data_all_tables_parallel.h5'), 'df')
dataset_main_parallel.to_excel(os.path.join(dirname, 'toExcel_dataset_1500_2540.xlsx'), index=False)
print('debug')




