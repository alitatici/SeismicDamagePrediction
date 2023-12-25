import matplotlib.pyplot as plt
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

def get_CumAbsVel_List(ground_acc, dt, factor):
    CAV = np.cumsum(np.abs(ground_acc) * dt)
    return CAV * factor

def get_table_for_thesis(dirname, df):
    earthquake_dataset = pd.DataFrame([],
                                columns=['Record_ID', 'Tcond_data_class', 'Vs30_data', 'scale_factor', 'file',
                                         'ground_acc', 'ground_vel', 'ground_disp', 'ground_CAV', 'deltat',
                                         'PGA', 'PGV', 'PGD'])
    record_id = 1
    for _, record_list in df.iterrows():
        if not '._' in record_list:
            file_name = record_list.file
            Vs30 = record_list.Vs30_data
            Tcond_data_class = record_list.Tcond_data_class
            scaleFactor_list_data, recordURL_list_data = readBakerOutputTable(os.path.join(dirname, file_name))
            for id, record in enumerate(recordURL_list_data):
                accfile = os.path.join(dirname, record)
                factor = float(scaleFactor_list_data[id])
                if '.asc' in accfile:
                    ags, dt = earthquakeRecord(accfile)
                else:
                    ags, dt = processBakerfile(accfile)

                time_series = np.arange(0, len(ags)*dt, dt)
                vgs = getVelRec(ags, dt).tolist()
                vgs_list = [item for sublist in vgs for item in sublist]
                dgs = getDispRec(vgs_list, dt).tolist()
                dgs_list = [item for sublist in dgs for item in sublist]
                PGA = get_PGA(ags, factor) / 9.81
                PGV = get_PGV(vgs, factor) * 100
                PGD = get_PGD(dgs, factor) * 100
                CAV = get_CumAbsVel_List(ags, dt, factor)
                # Sa, Sv, Sd = get_SpectralValues(record_list.Tcond, 0.05, (np.array(ags) * factor).tolist(), dt)


                list_of_variable = [record_id, Tcond_data_class, Vs30, factor, record, ags, vgs_list, dgs_list, CAV, dt, PGA, PGV, PGD]
                earthquake_dataset.loc[len(earthquake_dataset)] = list_of_variable
                record_id += 1

    return earthquake_dataset

# Selected Earthquakes
# SelectedRecords = 'Data_Onur'
SelectedRecords = 'Data_Ali'
BakerSelectedRecords = getAllFilesFromFolder(SelectedRecords)
data_dict = {'Tcond': [], 'Tcond_data_class': [], 'Vs30_data': [], 'no_database': [], 'file': []}
# T_cond = [0.4, 0.5, 0.59, 0.69, 0.78, 0.88, 0.97, 1.07, 1.16, 1.26]
T_cond = [0.24, 0.33, 0.41, 0.5, 0.59, 0.67, 0.76, 0.85, 0.94, 1.02]
T_cond_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for files in BakerSelectedRecords:
    if not '._' in files:
        splitted_files = files.split('_')
        data_dict['Vs30_data'].append(int(splitted_files[6]))
        index = T_cond.index(float(splitted_files[4]))
        data_dict['Tcond_data_class'].append(T_cond_class[index])
        data_dict['Tcond'].append(T_cond[index])
        data_dict['file'].append(files)
        data_dict['no_database'].append(splitted_files[7].split('.')[0])

data_df = pd.DataFrame(data=data_dict)

earthquake_data = get_table_for_thesis(dirname, data_df)

# Keep only the rows with unique values in column 'file'
duplicates = earthquake_data.duplicated('file', keep=False)
df_unique_record = earthquake_data[~duplicates | ~earthquake_data.duplicated('file', keep='first')]
df_unique_record = df_unique_record.reset_index(drop=True)

random_seed = 11     # nth record in df_unique_record

# Türkiye depremlerinden olsun.
# Vs30 - 1130
# Periodclass - 1
# Bundan figür al.
# istasyon ve depremi de yaz figüre.

ags_plot = np.array(df_unique_record['ground_acc'][random_seed])/9.81
vgs_plot = np.array(df_unique_record['ground_vel'][random_seed]) * 100
dgs_plot = np.array(df_unique_record['ground_disp'][random_seed]) * 100
CAV = np.array(df_unique_record['ground_CAV'][random_seed]) * 100
dt = df_unique_record['deltat'][random_seed]
time_series = np.arange(0, len(ags_plot)*dt, dt)

PGA = np.max(np.abs(ags_plot)) if np.abs(np.max(ags_plot)) > np.abs(np.max(-1 * ags_plot)) else -np.max(np.abs(ags_plot))
PGV = np.max(np.abs(vgs_plot)) if np.abs(np.max(vgs_plot)) > np.abs(np.max(-1 * vgs_plot)) else -np.max(np.abs(vgs_plot))
PGD = np.max(np.abs(dgs_plot)) if np.abs(np.max(dgs_plot)) > np.abs(np.max(-1 * dgs_plot)) else -np.max(np.abs(dgs_plot))

PGA_x_axis = np.argmax(np.abs(ags_plot)) * dt
PGV_x_axis = np.argmax(np.abs(vgs_plot)) * dt
PGD_x_axis = np.argmax(np.abs(dgs_plot)) * dt

plt.figure(figsize=(12, 12))
plt.subplot(411)
plt.plot(time_series, ags_plot, color='b', linewidth=1.0)
plt.plot([0, PGA_x_axis], [PGA, PGA], linestyle='--', color='k', linewidth=0.7)
plt.text(PGA_x_axis + 50*dt, PGA, f'PGA: {abs(PGA):.2f}', ha='left', va='center', fontsize=14)
plt.scatter(PGA_x_axis, PGA, marker='o', facecolor='none', edgecolors='r', s=30)
plt.ylabel('Acceleration (g)', fontsize=16)
plt.grid(alpha=0.5)
plt.xlim([0, time_series[-1]])
plt.ylim([-np.abs(PGA)-np.abs(PGA)/10, np.abs(PGA)+np.abs(PGA)/10])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Plot Velocity vs Time
plt.subplot(412)
plt.plot(time_series, vgs_plot, color='b', linewidth=1.0)
plt.plot([0, PGV_x_axis], [PGV, PGV], linestyle='--', color='k', linewidth=0.7)
plt.text(PGV_x_axis + 50*dt, PGV - PGV/20, f'PGV: {abs(PGV):.2f}', ha='left', va='center', fontsize=14)
plt.scatter(PGV_x_axis, PGV, marker='o', facecolor='none', edgecolors='r', s=30)
plt.ylabel('Velocity (cm/s)', fontsize=16)
plt.grid(alpha=0.5)
plt.xlim([0, time_series[-1]])
plt.ylim([-np.ceil(np.abs(PGV)/10)*10-np.abs(PGV)/100, np.ceil(np.abs(PGV)/10)*10+np.abs(PGV)/100])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Plot Displacement vs Time
plt.subplot(413)
plt.plot(time_series, dgs_plot, color='b', linewidth=1.0)
plt.plot([0, PGD_x_axis], [PGD, PGD], linestyle='--', color='k', linewidth=0.7)
plt.text(PGD_x_axis + 50*dt, PGD, f'PGD: {abs(PGD):.2f}', ha='left', va='center', fontsize=14)
plt.scatter(PGD_x_axis, PGD, marker='o', facecolor='none', edgecolors='r', s=30)
plt.ylabel('Displacement (cm)', fontsize=16)
plt.grid(alpha=0.5)
plt.xlim([0, time_series[-1]])
plt.ylim([-np.ceil(np.abs(PGD)/10)*10-np.abs(PGD)/10, np.ceil(np.abs(PGD)/10)*10+np.abs(PGD)/10])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Plot Cumulative Absolute Velocity vs Time
plt.subplot(414)
plt.plot(time_series, CAV, color='b', linewidth=1.0)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Cumulative Absolute\nVelocity (cm/s)', fontsize=16)
plt.grid(alpha=0.5)
plt.xlim([0, time_series[-1]])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
# plt.show()

dirname = os.path.dirname(__file__)
# data_file = os.path.join(dirname, 'data_results.h5')
# data = pd.read_hdf(data_file, 'df')

plt.savefig(os.path.join(dirname, (f'eqRecordFigure_seed_{random_seed:.0f}.png')))

# earthquake_data
# columns_to_drop = ['ground_acc', 'ground_vel', 'ground_disp', 'ground_CAV', 'deltat']
# earthquake_data_csv = earthquake_data.drop(columns=columns_to_drop)
#
# earthquake_data_csv.to_csv(os.path.join(dirname, (f'earthquake_dataset_{random_seed:.0f}.csv')), index=False)
print("debug")