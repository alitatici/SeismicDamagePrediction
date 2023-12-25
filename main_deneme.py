import pandas as pd

from EarthquakeRecordProcess import *
from NumericalEvaluationDynamicResponse import *
from SeismicIntensityMeasure import *
import numpy as np
import Units
import matplotlib.pyplot as plt
import scipy.io
import os
from EarthquakeRecordProcess import *
import time
from NumericalEvaluationDynamicResponse import NewmarkMethod

CyberShake_meta_data = scipy.io.loadmat('/Volumes/Elements/Dersler/Master Courses/Thesis/Ground Motion Selection/CS_Selection-master/Databases/CyberShake_meta_data.mat')

# Set the directory path
folders_path = r'/Volumes/Elements/Dersler/Master Courses/Thesis/Ground Motion Selection/CS_Selection-master/EQsinTurkey_NearFault'
Vs30_1 = []
# List all the files in the directory
# Loop through all subdirectories of the parent folder
def spactral_acc_and_rotxx(filename1, filename2, CyberShake_meta_data):
        [ags_E, dt_E] = earthquakeRecord(filename1)
        [ags_N, dt_N] = earthquakeRecord(filename2)
        RotD50Sa = []
        RotD100Sa = []
        T_list = CyberShake_meta_data['Periods']
        Sa_E = []
        Sa_N = []
        for T in T_list[0]:
            omega = 2 * np.pi / T
            ksi = 0.05
            t_E = np.arange(0, len(ags_E) * dt_E, dt_E)
            t_N = np.arange(0, len(ags_N) * dt_N, dt_N)
            dyn_res_E = NewmarkMethod(T, ksi, ags_E, dt_E)
            dyn_res_N = NewmarkMethod(T, ksi, ags_N, dt_N)
            [_, _, disp1] = dyn_res_E.results()
            [_, _, disp2] = dyn_res_N.results()
            Rot_Disp = get_ROTDpp(disp1, disp2)

            Rot_Acc = np.dot(Rot_Disp, (omega ** 2) / g)

            RotD50Sa.append(np.median(Rot_Acc))
            RotD100Sa.append(np.max(Rot_Acc))
            [Sa_t_E, _, _] = dyn_res_E.PseudoSpectralValues(disp1, omega)
            [Sa_t_N, _, _] = dyn_res_N.PseudoSpectralValues(disp2, omega)
            Sa_E.append(Sa_t_E / g)
            Sa_N.append(Sa_t_N / g)

        return Sa_E, Sa_N, RotD50Sa, RotD100Sa


# for root, dirs, files in os.walk(folders_path):
#     for dir_name in dirs:
#         if "Earthquake" in dir_name:
#             if "Izmir" in dir_name:
#                 Mw = 6.6
#             elif "Elazig" in dir_name:
#                 Mw = 6.8
#             elif "Duzce" in dir_name:
#                 Mw = 7.1
#             elif "Golcuk" in dir_name:
#                 Mw = 7.6
#             elif "Golyaka" in dir_name:
#                 Mw = 6.0
#             elif "Nurdagi" in dir_name:
#                 Mw = 6.6
#             elif "Van" in dir_name:
#                 Mw = 7.0
#             folder_full_path = os.path.join(root, dir_name)
#             for roots, dirs_eq, files_eq in os.walk(folder_full_path):
#                 for file_eq_data in files_eq:
#                     if not file_eq_data.startswith("."):
#                         df = pd.read_excel(roots + "/" +  file_eq_data)
#                         df["Code"] = df["Code"].astype("str")
#                 for dirs_eq_data in dirs_eq:
#                     if "AllData" in dirs_eq_data:
#                         folder_full_path_eq_data = os.path.join(roots, dirs_eq_data)
#                         for root_stations, dirs_stations, files_stations in os.walk(folder_full_path_eq_data):
#                             for dirs_station in dirs_stations:
#                                 start = time.time()
#                                 folder_full_path_eq_station = os.path.join(root_stations, dirs_station)
#                                 if not dirs_station.startswith("."):
#                                     # List all the files in the folder
#                                     file_list = os.listdir(folder_full_path_eq_station)
#                                     file_list_new = [element for element in file_list if not element.startswith(".")]
#                                     file_list = [element for element in file_list_new if not element.endswith("U.asc")]
#                                     eq_record_1st_direction = os.path.join(root_stations, dirs_station, file_list[0])
#                                     eq_record_2nd_direction = os.path.join(root_stations, dirs_station, file_list[1])
#                                     Vs30 = getVs30(eq_record_1st_direction)
#                                     epi_dist = getEpiDistance(eq_record_1st_direction)
#                                     station_id = getstationid(eq_record_1st_direction)
#                                     Rjb = df["Rjb"][df["Code"] == station_id]
#                                     if Vs30 is None:
#                                         continue
#                                     print(folder_full_path_eq_station)
#                                     CyberShake_meta_data["Filename_1"] = np.append(CyberShake_meta_data["Filename_1"],
#                                                                                    np.array(file_list[0]).reshape(1, 1),
#                                                                                    axis=0)
#                                     CyberShake_meta_data["Filename_2"] = np.append(CyberShake_meta_data["Filename_2"],
#                                                                                    np.array(file_list[1]).reshape(1, 1),
#                                                                                    axis=0)
#                                     CyberShake_meta_data["dirLocation"] = np.append(CyberShake_meta_data["dirLocation"],
#                                                                                     np.array(os.path.join(root_stations,
#                                                                                   dirs_station)).reshape(1, 1), axis=0)
#                                     CyberShake_meta_data["closest_D"] = np.append(CyberShake_meta_data["closest_D"],
#                                                                                   np.array(Rjb).reshape(1, 1),
#                                                                                   axis=0)
#                                     CyberShake_meta_data["Source_Name"] = np.append(CyberShake_meta_data["Source_Name"],
#                                                                                     np.array(dir_name).reshape(1, 1),
#                                                                                     axis=0)
#                                     CyberShake_meta_data["soil_Vs30"] = np.append(CyberShake_meta_data["soil_Vs30"],
#                                                                                   np.array(Vs30).reshape(1, 1), axis=0)
#                                     CyberShake_meta_data["Site_Name"] = np.append(CyberShake_meta_data["Site_Name"],
#                                                                                   np.array(station_id).reshape(1, 1),
#                                                                                   axis=0)
#                                     CyberShake_meta_data["magnitude"] = np.append(CyberShake_meta_data["magnitude"],
#                                                                                   np.array(Mw).reshape(1, 1), axis=0)
#
#                                     Sa_E, Sa_N, RotD50Sa, RotD100Sa = spactral_acc_and_rotxx(eq_record_1st_direction,
#                                                                                              eq_record_2nd_direction,
#                                                                                              CyberShake_meta_data)
#
#                                     CyberShake_meta_data["Sa_1"] = np.append(CyberShake_meta_data["Sa_1"],
#                                                                               np.array(Sa_E).reshape(1, np.size(
#                                                                                   CyberShake_meta_data["Periods"], 1)),
#                                                                              axis=0)
#
#                                     CyberShake_meta_data["Sa_2"] = np.append(CyberShake_meta_data["Sa_2"],
#                                                                              np.array(Sa_N).reshape(1, np.size(
#                                                                                  CyberShake_meta_data["Periods"], 1)),
#                                                                              axis=0)
#
#                                     CyberShake_meta_data["Sa_RotD50"] = np.append(CyberShake_meta_data["Sa_RotD50"],
#                                                                              np.array(RotD50Sa).reshape(1, np.size(
#                                                                                  CyberShake_meta_data["Periods"], 1)),
#                                                                              axis=0)
#
#                                     CyberShake_meta_data["Sa_RotD100"] = np.append(CyberShake_meta_data["Sa_RotD100"],
#                                                                              np.array(RotD100Sa).reshape(1, np.size(
#                                                                                  CyberShake_meta_data["Periods"], 1)),
#                                                                              axis=0)
#                                 end = time.time()
#                                 print(end-start)
#
#
# scipy.io.savemat("Turkish_EQ_meta_data_near_fault.mat", CyberShake_meta_data)


# plt.plot(CyberShake_meta_data["Periods"][0], RotD50Sa)
# plt.xlabel('Period (sec)', fontsize=30, fontweight='bold')
# plt.ylabel('RotD50Sa (g)', fontsize=30, fontweight='bold')
# plt.title('RotD50Sa (g) - T (sn)', fontsize=40, fontweight='bold')
# plt.tick_params(labelsize=25)
# plt.grid(True)
# plt.legend(fontsize=30)
#
# plt.plot(CyberShake_meta_data["Periods"][0], Sa_E)
# plt.xlabel('Period (sec)', fontsize=30, fontweight='bold')
# plt.ylabel('Sa_E (g)', fontsize=30, fontweight='bold')
# plt.title('Sa_E (g) - T (sn)', fontsize=40, fontweight='bold')
# plt.tick_params(labelsize=25)
# plt.grid(True)
# plt.legend(fontsize=30)
#
# plt.plot(CyberShake_meta_data["Periods"][0], Sa_N)
# plt.xlabel('Period (sec)', fontsize=30, fontweight='bold')
# plt.ylabel('Sa_N (g)', fontsize=30, fontweight='bold')
# plt.title('Sa_N (g) - T (sn)', fontsize=40, fontweight='bold')
# plt.tick_params(labelsize=25)
# plt.grid(True)
# plt.legend(fontsize=30)

# import scipy.io
# mat = scipy.io.loadmat('/Volumes/Elements/Dersler/Master Courses/Thesis/Ground Motion Selection/CS_Selection-master/Databases/CyberShake_meta_data.mat')


for root, dirs, files in os.walk(folders_path):
    for dir_name in dirs:
        if "Earthquake" in dir_name:
            if "Elazig" in dir_name:
                Mw = 6.8
            elif "Duzce" in dir_name:
                Mw = 7.1
            elif "Golcuk" in dir_name:
                Mw = 7.6
            elif "Golyaka" in dir_name:
                Mw = 6.0
            elif "Nurdagi" in dir_name:
                Mw = 6.6
            elif "Elbistan" in dir_name:
                Mw = 7.6
            elif "Pazarcik" in dir_name:
                Mw = 7.7
            folder_full_path = os.path.join(root, dir_name)
            for roots, dirs_eq, files_eq in os.walk(folder_full_path):
                if roots == folder_full_path:
                    for file_eq_data in files_eq:
                        if not file_eq_data.startswith("."):
                            if "Elbistan" in dir_name or "Pazarcik" in dir_name:
                                df = pd.read_excel(roots + "/" + file_eq_data, header=0, converters=
                                {'Code': str, 'Rjb': int})
                            else:
                                df = pd.read_excel(roots + "/" + file_eq_data, header=0, converters=
                                {'Code': str, 'Longitude': int, 'Latitude': int, 'PGA_NS': int, 'PGA_EW': int,
                                 'PGV_UD': int, 'Rjb': int, 'Rrup': int, 'Repi': int, 'Rhyp': int})
                    for dirs_eq_data in dirs_eq:
                        if "AllData" in dirs_eq_data:
                            folder_full_path_eq_data = os.path.join(roots, dirs_eq_data)
                            for root_stations, dirs_stations, files_stations in os.walk(folder_full_path_eq_data):
                                for dirs_station in dirs_stations:
                                    start = time.time()
                                    folder_full_path_eq_station = os.path.join(root_stations, dirs_station)
                                    if not dirs_station.startswith("."):
                                        # List all the files in the folder
                                        file_list = os.listdir(folder_full_path_eq_station)
                                        file_list_new = [element for element in file_list if not element.startswith(".")]
                                        file_list = [element for element in file_list_new if not element.endswith("U.asc")]
                                        eq_record_1st_direction = os.path.join(root_stations, dirs_station, file_list[0])
                                        eq_record_2nd_direction = os.path.join(root_stations, dirs_station, file_list[1])
                                        Vs30 = getVs30(eq_record_1st_direction)
                                        epi_dist = getEpiDistance(eq_record_1st_direction)
                                        station_id = getstationid(eq_record_1st_direction)
                                        Rjb = df["Rjb"][df["Code"] == station_id]
                                        if Vs30 is None:
                                            continue
                                        print(folder_full_path_eq_station)
                                        CyberShake_meta_data["Filename_1"] = np.append(CyberShake_meta_data["Filename_1"],
                                                                                       np.array(file_list[0]).reshape(1, 1),
                                                                                       axis=0)
                                        CyberShake_meta_data["Filename_2"] = np.append(CyberShake_meta_data["Filename_2"],
                                                                                       np.array(file_list[1]).reshape(1, 1),
                                                                                       axis=0)
                                        CyberShake_meta_data["dirLocation"] = np.append(CyberShake_meta_data["dirLocation"],
                                                                                        np.array(os.path.join(root_stations,
                                                                                      dirs_station)).reshape(1, 1), axis=0)
                                        CyberShake_meta_data["closest_D"] = np.append(CyberShake_meta_data["closest_D"],
                                                                                      np.array(Rjb).reshape(1, 1),
                                                                                      axis=0)
                                        CyberShake_meta_data["Source_Name"] = np.append(CyberShake_meta_data["Source_Name"],
                                                                                        np.array(dir_name).reshape(1, 1),
                                                                                        axis=0)
                                        CyberShake_meta_data["soil_Vs30"] = np.append(CyberShake_meta_data["soil_Vs30"],
                                                                                      np.array(Vs30).reshape(1, 1), axis=0)
                                        CyberShake_meta_data["Site_Name"] = np.append(CyberShake_meta_data["Site_Name"],
                                                                                      np.array(station_id).reshape(1, 1),
                                                                                      axis=0)
                                        CyberShake_meta_data["magnitude"] = np.append(CyberShake_meta_data["magnitude"],
                                                                                      np.array(Mw).reshape(1, 1), axis=0)

                                        Sa_E, Sa_N, RotD50Sa, RotD100Sa = spactral_acc_and_rotxx(eq_record_1st_direction,
                                                                                                 eq_record_2nd_direction,
                                                                                                 CyberShake_meta_data)

                                        CyberShake_meta_data["Sa_1"] = np.append(CyberShake_meta_data["Sa_1"],
                                                                                  np.array(Sa_E).reshape(1, np.size(
                                                                                      CyberShake_meta_data["Periods"], 1)),
                                                                                 axis=0)

                                        CyberShake_meta_data["Sa_2"] = np.append(CyberShake_meta_data["Sa_2"],
                                                                                 np.array(Sa_N).reshape(1, np.size(
                                                                                     CyberShake_meta_data["Periods"], 1)),
                                                                                 axis=0)

                                        CyberShake_meta_data["Sa_RotD50"] = np.append(CyberShake_meta_data["Sa_RotD50"],
                                                                                 np.array(RotD50Sa).reshape(1, np.size(
                                                                                     CyberShake_meta_data["Periods"], 1)),
                                                                                 axis=0)

                                        CyberShake_meta_data["Sa_RotD100"] = np.append(CyberShake_meta_data["Sa_RotD100"],
                                                                                 np.array(RotD100Sa).reshape(1, np.size(
                                                                                     CyberShake_meta_data["Periods"], 1)),
                                                                                 axis=0)
                                    end = time.time()
                                    print(end-start)


from copy import copy
CyberShake_meta_data1 = CyberShake_meta_data.copy()
CyberShake_meta_data1['closest_D'] = np.delete(CyberShake_meta_data1['closest_D'], range(0, 320))
CyberShake_meta_data1['Filename_1'] = np.delete(CyberShake_meta_data1['Filename_1'], range(0, 320))
CyberShake_meta_data1['Filename_2'] = np.delete(CyberShake_meta_data1['Filename_2'], range(0, 320))
CyberShake_meta_data1['dirLocation'] = np.delete(CyberShake_meta_data1['dirLocation'], range(0, 320))
CyberShake_meta_data1['Source_Name'] = np.delete(CyberShake_meta_data1['Source_Name'], range(0, 320))
CyberShake_meta_data1['soil_Vs30'] = np.delete(CyberShake_meta_data1['soil_Vs30'], range(0, 320))
CyberShake_meta_data1['Site_Name'] = np.delete(CyberShake_meta_data1['Site_Name'], range(0, 320))
CyberShake_meta_data1['magnitude'] = np.delete(CyberShake_meta_data1['magnitude'], range(0, 320))
CyberShake_meta_data1['Sa_1'] = np.delete(CyberShake_meta_data1['Sa_1'], list(range(0, 320)), 0)
CyberShake_meta_data1['Sa_2'] = np.delete(CyberShake_meta_data1['Sa_2'], list(range(0, 320)), 0)
CyberShake_meta_data1['Sa_RotD50'] = np.delete(CyberShake_meta_data1['Sa_RotD50'], list(range(0, 320)), 0)
CyberShake_meta_data1['Sa_RotD100'] = np.delete(CyberShake_meta_data1['Sa_RotD100'], list(range(0, 320)), 0)
CyberShake_meta_data1['closest_D'] = CyberShake_meta_data1['closest_D'].reshape(-1,1)
CyberShake_meta_data1['Filename_1'] = CyberShake_meta_data1['Filename_1'].reshape(-1,1)
CyberShake_meta_data1['Filename_2'] = CyberShake_meta_data1['Filename_2'].reshape(-1,1)
CyberShake_meta_data1['dirLocation'] = CyberShake_meta_data1['dirLocation'].reshape(-1,1)
CyberShake_meta_data1['Source_Name'] = CyberShake_meta_data1['Source_Name'].reshape(-1,1)
CyberShake_meta_data1['soil_Vs30'] = CyberShake_meta_data1['soil_Vs30'].reshape(-1,1)
CyberShake_meta_data1['Site_Name'] = CyberShake_meta_data1['Site_Name'].reshape(-1,1)
CyberShake_meta_data1['magnitude'] = CyberShake_meta_data1['magnitude'].reshape(-1,1)
CyberShake_meta_data1['Sa_1'] = CyberShake_meta_data1['Sa_1'].reshape(-1,111)
CyberShake_meta_data1['Sa_2'] = CyberShake_meta_data1['Sa_2'].reshape(-1,111)
CyberShake_meta_data1['Sa_RotD50'] = CyberShake_meta_data1['Sa_RotD50'].reshape(-1,111)
CyberShake_meta_data1['Sa_RotD100'] = CyberShake_meta_data1['Sa_RotD100'].reshape(-1,111)
scipy.io.savemat("Turkish_EQ_meta_data_near_fault.mat", CyberShake_meta_data1)
print('Tabi Efenim')
