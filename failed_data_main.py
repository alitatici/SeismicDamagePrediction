import os
import pandas as pd
from GMProcess import getAllFilesFromFolder
from EarthquakeRecordProcess import read_data_and_write_to_df
from joblib import Parallel, delayed
import openseespy.opensees as ops
from Units import *
from CreateModel import CreateModel
import time
dirname = os.path.dirname(__file__)

def process_failed_row(row):
    run_id = row['Run ID']
    building_number = row['Building ID']
    print(str(building_number) + '. Building')
    ops.wipe()
    num_span = row['Number of Span']
    num_storey = row['Number of Storey']
    span_length = row['Span Length']
    storey_height = row['Storey Height']
    # Column width is parallel to implemented force
    column_width = row['Column Width']
    column_depth = row['Column Depth']
    # Beam width is parallel to implemented force
    beam_width = row['Beam Width']
    beam_depth = row['Beam Depth']
    fc = row['Concrete Strength']
    fy = row['Steel Strength']
    cover = 4 * cm
    As_col_rat = 0.01
    As_beam_rat = 0.004
    live_load = 2 * kN / (m ** 2)
    dead_load = 2 * kN / (m ** 2)
    soft_story = row['First Storey - Commercial Use']
    num_modes = 10
    periodClass = row['Period Class']
    Vs30 = row['Soil Condition']
    period = row['First Mode Period']
    record_id = row['Earthquake URL']

    selected_GMs = earthquake_data[earthquake_data['Record_ID'] == record_id]
    ags = selected_GMs.ground_acc.values[0]
    dt = selected_GMs.deltat.values[0]
    factor = selected_GMs.scale_factor.values[0]
    length_of_record = len(ags)

    ops.wipe()
    model = CreateModel(num_span, num_storey, span_length, storey_height, soft_story)
    model.create_nonlinear_model_precode(fc, fy, cover, As_col_rat, As_beam_rat, column_width,
                                         column_depth, beam_width, beam_depth, dead_load, live_load, soft_story)
    model.add_staticLoad(column_width, column_depth, beam_width, beam_depth, dead_load, live_load)
    model.static_analysis()
    # model.recorder_maxDisp(recorder_folder, record_number, building_number)
    model.add_dynamicLoad(ags, dt, factor, num_modes)
    MIDR_max, NumberOfStep, totalStep, MIDR_last = model.get_max_MIDR_and_last_MIDR(length_of_record * dt, dt)
    # MIDR = model.dynamic_analysis(length_of_record * dt, dt)
    # model.animated_Node_Disp(length_of_record * dt, dt)

    list_of_input = [row['Run ID'], row['Building ID'], row['Number of Storey'], row['Number of Span'],
                     row['Span Length'], row['Storey Height'], row['Column Width'], row['Column Depth'],
                     beam_width, beam_depth, fc, fy, soft_story, Vs30, period, periodClass,
                     record_id, row.PGA, row.PGV, row.PGD, row['Sa(T1)'], row['Sd(T1)'], row.CAV, MIDR_max, MIDR_last,
                     NumberOfStep, totalStep]
    failed_dataset_main.loc[len(failed_dataset_main)] = list_of_input
    # number_of_analysis = number_of_analysis + 1

    ops.wipe()
    # MIDR = model.get_MIDR(recorder_folder, record_number, building_number)
    # dataset_main.insert(len(dataset_main.columns), "MIDR", MIDR, True)
    # for i in range(1, len(recordURL_list) + 1):
    #     # dataset_main.loc[number_of_analysis - len(recordURL_list) + i - 1, 'MIDR'] = MIDR[i - 1]
    #     dataset_main.loc[(building_number - 1) * len(recordURL_list) + i - 1, 'MIDR'] = MIDR[i - 1]
    return failed_dataset_main


start = time.time()
print("Analyses and time started.")

failed_dataset_main = pd.DataFrame([], columns=['Run ID', 'Building ID', 'Number of Storey', 'Number of Span',
                                                'Span Length', 'Storey Height', 'Column Width', 'Column Depth',
                                                'Beam Width', 'Beam Depth', 'Concrete Strength', 'Steel Strength',
                                                'First Storey - Commercial Use', 'Soil Condition', 'First Mode Period',
                                                'Period Class', 'Earthquake URL', 'PGA', 'PGV', 'PGD', 'Sa(T1)',
                                                'Sd(T1)', 'CAV', 'MIDR', 'LastStepMIDR', 'NumberOfStep', 'Total Step'])

# Selected Earthquakes
SelectedRecords = 'Data_Ali'
BakerSelectedRecords = getAllFilesFromFolder(SelectedRecords)
data_dict = {'Tcond_data_class': [], 'Vs30_data': [], 'no_database': [], 'file': []}
T_cond = [0.24, 0.33, 0.41, 0.5, 0.59, 0.67, 0.76, 0.85, 0.94, 1.02]
# T_cond = [0.4, 0.5, 0.59, 0.69, 0.78, 0.88, 0.97, 1.07, 1.16, 1.26]
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

# data_result_1 = pd.read_hdf(os.path.join(dirname, 'dataset_all_0_140.h5'), 'df')
# data_result_2 = pd.read_hdf(os.path.join(dirname, 'dataset_all_140_1100.h5'), 'df')
# data_result_3 = pd.read_hdf(os.path.join(dirname, 'dataset_all_1100_1500.h5'), 'df')
# data_result_4 = pd.read_hdf(os.path.join(dirname, 'dataset_all_1500_2540.h5'), 'df')
# data_result_5 = pd.read_hdf(os.path.join(dirname, 'dataset_all_2540_3240.h5'), 'df')
# data_result = pd.concat([data_result_1, data_result_2, data_result_3, data_result_4, data_result_5], axis=0)
# data_result = data_result.reset_index(drop=True)
# failed_data = data_result[data_result['MIDR'] == -999]

data_result_11 = pd.read_hdf(os.path.join(dirname, 'recalculatedMIDR_0_700.h5'), 'df')
data_result_21 = pd.read_hdf(os.path.join(dirname, 'recalculatedMIDR_700_2200.h5'), 'df')
data_result_31 = pd.read_hdf(os.path.join(dirname, 'recalculatedMIDR_2200_2540.h5'), 'df')
data_result_41 = pd.read_hdf(os.path.join(dirname, 'recalculatedMIDR_2540_2640.h5'), 'df')
data_result_51 = pd.read_hdf(os.path.join(dirname, 'recalculatedMIDR_2640_3240.h5'), 'df')
data_result1 = pd.concat([data_result_11, data_result_21, data_result_31, data_result_41, data_result_51], axis=0)
data_result1 = data_result1.reset_index(drop=True)
failed_data1 = data_result1[data_result1['MIDR'] == -999]

failed_data1 = failed_data1[:8]
# failed_data = pd.concat([data_result_0_7, failed_data], axis=0)
# failed_data = failed_data.reset_index(drop=True)
# failed_data = failed_data.iloc[1].to_frame().T

results = Parallel(n_jobs=8)(
    delayed(process_failed_row)(row) for _, row in failed_data1.iterrows()
)

failed_dataset_parallel = pd.concat(results)
end = time.time()
time_spent = round((end - start), 2)
print(time_spent)
failed_dataset_parallel.to_hdf(os.path.join(dirname, 'failedAnalyses_lastStep.h5'), key='df', mode='w', format='table')
# dataset_main_parallel = pd.read_hdf(os.path.join(dirname, 'data_all_tables_parallel.h5'), 'df')
failed_dataset_parallel.to_excel(os.path.join(dirname, 'failedAnalyses_lastStep.xlsx'), index=False)
print('debug')


print('Akıdeş')