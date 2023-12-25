# -*- coding: utf-8 -*-

import numpy as np
from Units import *
import os
from NumericalEvaluationDynamicResponse import *
from GMProcess import *
from SeismicIntensityMeasure import *


def earthquakeRecord(filename, no_skiprows=64):
    dt, units = get_record_attr(filename)

    ag_txt = np.loadtxt(filename, skiprows=no_skiprows)
    if units == "cm/s^2":
        groundacc = ag_txt[:] * cm / sec ** 2
    elif units == 'm/s^2':
        groundacc = ag_txt[:] * m / sec ** 2
    else:
        raise ValueError('record unit error')

    ags = groundacc.flatten("C")
    ags = ags.tolist()
    length_of_record = len(ags)

    return ags, dt


# def geomean(filenames):
#
#     acc_geomean_of_all_records = []
#     length_list_of_all_records = []
#     dt_list_of_all_records = []
#     for i in range(len(filenames)):
#         ags_E, length_of_record, dt = earthquakeRecord(filenames[i][0])
#         # ags_N, _, _ = earthquakeRecord(filenames[i][1])
#         # acc_geomean = (np.array(ags_E) * np.array(ags_N)) ** 0.5
#
#         length_list_of_all_records.append(length_of_record)
#         dt_list_of_all_records.append(dt)
#         acc_geomean_of_all_records.append(ags_E)
#
#     return acc_geomean_of_all_records, length_list_of_all_records, dt_list_of_all_records

def get_record_attr(filename):
    interval = []
    unit_val = []
    with open(filename, "r", encoding="utf-8") as file:
        test_value = "INTERVAL_S: "
        file.seek(file.read().find(test_value) + len(test_value))
        interval = file.readline()
        interval = interval[:-1]
        interval = [''.join(interval)]
        dt = interval[0]
        dt = float(dt)
    with open(filename, "r", encoding="utf-8") as file:
        unit_value = "UNITS: "
        file.seek(file.read().find(unit_value) + len(unit_value))
        unit_val = file.readline()
        unit_val = unit_val[:-1]
        unit_val = [''.join(unit_val)]
        unit = unit_val[0]

    return dt, unit


def getFiles(data_folder):
    os.chdir(data_folder)
    folder_list = []
    file_list = []
    for i in os.listdir():
        if i.startswith("._") == False:
            folder_list.append(i)
    for j in folder_list:
        fullpath = os.path.join(data_folder, j)
        os.chdir(fullpath)
        record_folder_list = []
        for k in os.listdir():
            if k.startswith("._") == False and k.endswith("U.asc") == False:
                fullpath = os.path.join(data_folder, j, k)
                record_folder_list.append(fullpath)
                # record_folder_list.append(j+ "/" +k)
        file_list.append(sorted(record_folder_list))

    # for l in range(0, len(file_list)):
    #     with open(data_folder + "/" + file_list[l], "r", encoding="utf-8") as file:
    #         test_value="VS30_M/S: "
    #         file.seek(file.read().find(test_value)+len(test_value))
    #         Vs30_value=file.readline()
    #         Vs30_value=Vs30_value[:-1]
    #         Vs30_value=[''.join(Vs30_value)]
    #         Vs30_value=Vs30_value[0]
    #         if Vs30_value=="None":
    #             os.remove(data_folder + "/" + file_list[l])
    #             file_list.remove(data_folder + "/" + file_list[l])
    return file_list


# def getPGV(filename):
#     # Vs30_value = []
#     with open(filename, "r", encoding="utf-8") as file:
#         test_value = "VS30_M/S: "
#         file.seek(file.read().find(test_value) + len(test_value))
#         Vs30_value = file.readline()
#         Vs30_value = Vs30_value[:-1]
#         Vs30_value = [''.join(Vs30_value)]
#         Vs30_value = Vs30_value[0]
#         Vs30_value = float(Vs30_value)
#
#     return Vs30_value

def getstationid(filename):
    Station_id = []
    with open(filename, "r", encoding="utf-8") as file:
        test_value = "STATION_CODE: "
        file.seek(file.read().find(test_value) + len(test_value))
        Station_id = file.readline()
        Station_id = Station_id[:-1]
        Station_id = [''.join(Station_id)]
        Station_id = Station_id[0]
        Station_id = Station_id

    return Station_id


def getVs30(filename):
    # Vs30_value = []
    with open(filename, "r", encoding="utf-8") as file:
        test_value = "VS30_M/S: "
        file.seek(file.read().find(test_value) + len(test_value))
        Vs30_value = file.readline()
        Vs30_value = Vs30_value[:-1]
        Vs30_value = [''.join(Vs30_value)]
        Vs30_value = Vs30_value[0]
        try:
            Vs30_value = float(Vs30_value)
        except:
            Vs30_value = None

    return Vs30_value


def getUnit(filename):
    # Vs30_value = []
    with open(filename, "r", encoding="utf-8") as file:
        test_value = "UNITS: "
        file.seek(file.read().find(test_value) + len(test_value))
        unit = file.readline()
        unit = unit[:-1]
        unit = [''.join(unit)]
        unit = unit[0]
        # unit = float(unit)

    return unit


def getEpiDistance(filename):
    # Vs30_value = []
    with open(filename, "r", encoding="utf-8") as file:
        test_value = "EPICENTRAL_DISTANCE_KM: "
        file.seek(file.read().find(test_value) + len(test_value))
        dist = file.readline()
        dist = dist[:-1]
        dist = [''.join(dist)]
        dist = dist[0]
        dist = float(dist)

    return dist


def findInterval(filename):
    interval = []
    with open(filename, "r", encoding="utf-8") as file:
        test_value = "INTERVAL_S: "

        file.seek(file.read().find(test_value) + len(test_value))
        interval = file.readline()
        interval = interval[:-1]
        interval = [''.join(interval)]
        dt = interval[0]
        dt = float(dt)

    return dt


def getVelRec(ground_acc, dt):
    t_amount = len(ground_acc)
    vgs = np.zeros((t_amount, 1))
    for i in range(1, t_amount):
        vgs[i] = vgs[i - 1] + (((ground_acc[i - 1] + ground_acc[i]) / 2) * dt)

    return vgs


def getDispRec(ground_vel, dt):
    t_amount = len(ground_vel)
    dgs = np.zeros((t_amount, 1))
    for i in range(1, t_amount):
        dgs[i] = dgs[i - 1] + (((ground_vel[i - 1] + ground_vel[i]) / 2) * dt)

    return dgs


def get_SpectralValues(T, ksi, ags, dt):
    dyn_res_E = NewmarkMethod(T, ksi, ags, dt)
    omega = 2 * np.pi / T
    [_, _, disp] = dyn_res_E.results()
    [SA, SV, SD] = dyn_res_E.PseudoSpectralValues(disp, omega)

    return SA / g, SV / g, SD / g


def read_data_and_write_to_df(dirname, df):
    earthquake_dataset = pd.DataFrame([],
                                columns=['Record_ID', 'Tcond_data_class', 'Vs30_data', 'scale_factor', 'file', 'ground_acc', 'deltat',
                                         'PGA', 'PGV', 'PGD', 'CAV'])
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
                PGA = get_PGA(ags, factor) / 9.81
                PGV = get_PGV(vgs, factor) * 100
                PGD = get_PGD(dgs, factor) * 100
                CAV = get_CumAbsVel(ags, dt, factor)

                # import matplotlib.pyplot as plt
                # plt.subplot(3, 1, 1)
                # plt.plot(time_series, np.array(ags) * factor)
                # plt.grid()
                # plt.subplot(3, 1, 2)
                # plt.plot(time_series, np.array(vgs) * factor)
                # plt.grid()
                # plt.subplot(3, 1, 3)
                # plt.plot(time_series, np.array(dgs) * factor)
                # plt.grid()


                list_of_variable = [record_id, Tcond_data_class, Vs30, factor, record, ags, dt, PGA, PGV, PGD, CAV]
                earthquake_dataset.loc[len(earthquake_dataset)] = list_of_variable
                record_id += 1

    return earthquake_dataset

