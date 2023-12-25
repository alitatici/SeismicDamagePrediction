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
    # span_length = (np.array(span_length) * m ).tolist()
    span_length = data_cluster.span_length
    storey_height = data_cluster.storey_height
    # storey_height= (np.array(storey_height) * m).tolist()
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

    # from MaterialCalc import *
    #
    # load = Loads(span_length, storey_height, beam_width, beam_depth, column_width, column_depth, num_span, num_storey)
    #
    # aa = load.calc_beam_seismic_weight()

    # accfile = '/Volumes/Elements/Dersler/Master Courses/Thesis/OpenSees/BM68elc.acc'
    # desc, npts, dt, time, inp_acc = processNGAfile(accfile)

    # with open('BM68elc.acc') as f:
    #     lines = f.readlines()
    #     acc_data = []
    #     for line in lines:
    #         line_str = line.split()
    #         for l in line_str:
    #             a = float(l)
    #             acc_data.append(a)
    #         inp_acc = list(np.asarray(acc_data))
    # foldername = r"/Volumes/Elements/Dersler/Master Courses/Thesis/OpenSees/EQ_Record"
    # filenames = getFiles(foldername)
    #
    # acc_geomean_of_all_records, length_list_of_all_records, dt_list_of_all_records = geomean(filenames)
    # acc_geomean = acc_geomean_of_all_records[0]
    # dt = dt_list_of_all_records[0]
    # length_of_record = length_list_of_all_records[0]

    # factor = 1
    num_modes = 10

    model = CreateModel(num_span, num_storey, span_length, storey_height, soft_story)
    model.create_nonlinear_model_precode(fc, fy, cover, As_col_rat, As_beam_rat, column_width,
                                         column_depth, beam_width, beam_depth, dead_load, live_load, soft_story)

    # model.create_elastic_model(fc, column_width, column_depth, beam_width, beam_depth, dead_load, live_load, soft_story)

    model.add_staticLoad(column_width, column_depth, beam_width, beam_depth, dead_load, live_load)
    model.static_analysis()

    # opsv.plot_model()

    # power = model.add_dynamicLoad(acc_geomean, dt, factor, num_modes)
    # model.dynamic_analysis(length_of_record * dt + dt, dt)

    # IDR, MIDR = model.get_MIDR()

    # 3. plot N, V, M forces diagrams

    ## OUTPUTS

    # r1 = ops.nodeReaction(2)
    # print("r1 = ", r1)

    ## PLOTS
    ## Plot structure
    # opsv.plot_model()
    ## Plot N, V, M forces diagrams
    # opsv.plot_loads_2d()
    # plt.show()
    ## Plot N, V, M forces diagrams
    # sfacN, sfacV, sfacM = 2.e-2, 2.e-2, 2.e-2
    # opsv.section_force_diagram_2d('N', sfacN)
    # plt.title('Axial force distribution')
    # opsv.section_force_diagram_2d('T', sfacV)
    # plt.title('Shear force distribution')
    # opsv.section_force_diagram_2d('M', sfacM)
    # plt.title('Bending moment distribution')
    model.mode_shape(1, num_modes)

    # ops.modalProperties('-print', '-file', 'ModalReport.txt', '-unorm')

    modalProps = ops.modalProperties('-return')
    T1 = modalProps['eigenPeriod'][0]
    # data['period'][number_of_analysis] = T1
    data.loc[number_of_analysis, 'period'] = T1
    number_of_analysis = number_of_analysis + 1
    print('Debug Stop Here')

T = data.loc[:, 'period']
period_intervals = np.linspace(np.min(T), np.max(T), num=11)  # Define the bin edges to create intervals
# Round 2 decimal the first value down and the last value up
period_intervals[0] = math.floor(period_intervals[0]*100)/100
period_intervals[-1] = math.ceil(period_intervals[-1]*100)/100
period_intervals_midpoints = [(period_intervals[i] + period_intervals[i+1])/2 for i in range(len(period_intervals)-1)]
period_intervals_midpoints = [round(x, 2) for x in period_intervals_midpoints]
print(period_intervals_midpoints)
period_class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]         # Define the class labels
data['period_class'] = pd.cut(data['period'], period_intervals, labels=period_class_labels)  # Use cut to create intervals based on the bin edges and assign class labels
# data.to_hdf('data_results.h5', key='df', mode='w')
with pd.HDFStore('YEDEK/data_results.h5') as store:
    store.put('df', data, format='table')

data_results = pd.read_hdf('YEDEK/data_results.h5', 'df')
data_results['period'][:].to_excel('toExcel_period.xlsx', index = False)
data_results['period_class'][:].to_excel('toExcel_periodClass.xlsx', index = False)

# TODO! %80 Yapıldı. - Record'u okuması için script yazılacak.
    # Geomean yapılacak mı? -Hayır iki yön ayrı ayrı kullanılacak.

# TODO! Yapı stoğu için yapı özellikleri belirlenecek.
    # Number of span kaç adet kullanılacak.
# TODO! SAP2000 verifikasyonu yap.

# TODO! Newmark'tan gelen sonuçların karşılaştırılması.
# TODO! Ground Motion selection
# TODO! Her bir girdi dataframe satırı olarak ifade edilecek olup tek for yazılarak tüm olası girdiler analiz edilecek.
# Kolon boyutları ve ivme girdileri eklenecek.
# TODO! Parallel processing ile hızlandırılma yapılacak.

## Sunum notları
# "Shear-flexure effect in tall buildings" gibi loss hesabına etki eden
# parametreler hakkında başka çalışmalar nelerdir?
# - Bal 2008 eski kalıyor.
# - Concrete Compressive Strength 5 az gibi. Bu dağılıma DEZIM datasından bak.
# - DEZIM yerine IBB veya IMM?
# - ML methodları neden seçtiğini, literatür'de hangi metodların kullanıldığını açıkla.