# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from Units import *
import os

def processBakerfile(filename, no_skiprows=2):
    with open(filename, 'r') as f:
        second_line = f.readlines()[1]
    stepNumber_dt = second_line.split(" ")
    dt = float(stepNumber_dt[1])
    # dt = 0.025 # Cybershake verileri için analizi hızlandırmak için doğrudan bu satır kullanılabilir.

    ag_txt = np.loadtxt(filename, skiprows=no_skiprows)
    groundacc = ag_txt[:] * g

    ags = groundacc.flatten("C")
    ags = ags.tolist()
    length_of_record = len(ags)

    # PGA, PGV, Sa, Sd, CAV = 1, 1, 1, 1, 1

    return ags, dt


def readBakerOutputTable(filename):
    # Reads Baker ground motion code's output file.
    # Gets scale factor and record URL.
    flag = 0
    scaleFactor_list = []
    recordURL_list = []
    with open(filename, 'r') as f:
        for line in f:
            if 'Record' in line:
                flag = 1
            if flag == 1:
                recordLine = line.split(" ")
                scaleFactor_list.append(recordLine[6])
                if recordLine[14].endswith('.acc'):
                    recordURL_list.append(" ".join(recordLine[12:15]))
                else:
                    if 'Record' in line:
                        recordURL_list.append(recordLine[12])
                    else:
                        recordLine_new = recordLine[17].split('/')[2:]
                        recordURL_list.append(os.path.join(*recordLine_new))
            else:
                continue
    scaleFactor_list.pop(0)
    recordURL_list.pop(0)
    return scaleFactor_list, recordURL_list

def getAllFilesFromFolder(data_folder):
    os.chdir(data_folder)
    folder_list = []
    file_list = []
    for i in os.listdir():
        folder_list.append(i)
    for j in folder_list:
        file_name = data_folder + '/' + j
        file_list.append(file_name)
    return file_list