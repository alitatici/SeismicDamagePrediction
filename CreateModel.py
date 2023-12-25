#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 21:09:46 2022

@author: onurulku
"""

import openseespy.opensees as ops
import math
from Units import *
import opsvis as opsv
import matplotlib.pyplot as plt
from MaterialCalc import *
import numpy as np
import os


class CreateModel:
    def __init__(self, num_span, num_storey, span_length, storey_height, soft_story):
        self.num_span = num_span
        self.num_storey = num_storey
        self.span_length = span_length
        self.storey_height = self.__calc_story_height(storey_height, soft_story)

    def __calc_story_height(self, storey_height, soft_story):
        if soft_story == 'Yes':
            difference_height = 3.5 - storey_height
            story_height = storey_height * np.ones((1, self.num_storey))
            story_height[0, 0] = story_height[0, 0] + round(difference_height, 2)
            story_height = story_height.tolist()[0]
        else:
            story_height = np.array(storey_height) * np.ones((1, self.num_storey)).tolist()[0]

        return story_height

    def create_2D_model(self):
        ndm = 2
        ndf = int(ndm * (ndm + 1) / 2)

        ops.model('Basic', '-ndm', ndm, '-ndf', ndf)

        # =============================================================================
        # # Model Command
        # # '-ndm' (int) ---> number of dimensions (1,2,3)
        # # '-ndf' (int) ---> number of degree of freedom [ndf = ndm * (ndm + 1) / 2]
        # =============================================================================

        if isinstance(self.span_length, float) == 1:
            x_coord = (np.multiply(np.arange(0, self.num_span + 1), self.span_length)).tolist()
        else:
            x_coord = np.insert(np.cumsum((np.multiply(np.ones([1, self.num_span]), self.span_length))), 0, 0).tolist()

        if len(self.storey_height) == 1:
            z_coord = (np.multiply(np.arange(0, self.num_storey + 1), self.storey_height)).tolist()
        else:
            z_coord = np.insert(np.cumsum((np.multiply(np.ones([1, len(self.storey_height)]), self.storey_height))), 0,
                                0).tolist()
            z_coord = [round(elem, 2) for elem in z_coord]

        number_of_nodes = len(x_coord) * len(z_coord)
        coordinates = np.array(np.meshgrid(x_coord, z_coord)).T.reshape(-1, 2)
        coordinates = coordinates[np.lexsort((coordinates[:, 0], coordinates[:, 1])), :]
        column_nodes = list(zip(range(1, number_of_nodes - self.num_span),
                                range(self.num_span + 1 + 1, number_of_nodes + 1)))

        beam_nodes = list(zip(range(self.num_span + 1 + 1, number_of_nodes),
                              range(self.num_span + 1 + 1 + 1, number_of_nodes + 1)))

        index_del = 1
        for k in (np.arange(1, self.num_storey) * (self.num_span + 1)).tolist():
            beam_nodes.pop(k - index_del)
            index_del += 1

        # print('Model is built.')

        return coordinates, column_nodes, beam_nodes

    def create_elastic_model(self, fc, column_width, column_depth, beam_width,
                             beam_depth, dead_load, live_load, soft_story):

        coordinates, column_nodes, beam_nodes = self.create_2D_model()

        # Column Properties
        # column_width = 0.3
        # Column width is parallel to implemented force
        # column_depth = 0.3
        # column length is parpendicular to implemented force
        column_area = column_width * column_depth
        column_inertia = (column_width * column_depth ** 3) / 12
        loads = Loads(self.span_length, self.storey_height, beam_width, beam_depth, column_width, column_depth,
                      self.num_span, self.num_storey)
        mass_of_columns = loads.calc_column_seismic_weight() / g

        # Beam Properties
        # beam_width = 0.2                                                                                               # Beam width is parallel to implemented force
        # beam_depth = 0.3                                                                                               # Beam length is parallel to implemented force
        beam_area = beam_width * beam_depth
        beam_inertia = (beam_width * beam_depth ** 3) / 12
        weight_of_beams = loads.calc_beam_seismic_weight(dead_load, live_load)
        # mass_of_beams = np.reshape(np.multiply(np.divide(weight_of_beams, g), np.ones((self.num_storey, 1))), -1)
        mass_of_beams = np.divide(weight_of_beams, g)

        # Material Properties
        E = (3250 * fc ** 0.5 + 14000) * MPa

        # Create Nodes
        for node_tag, node in enumerate(coordinates):
            ops.node(node_tag + 1, *node.tolist())
            if math.isclose(node[1], 0.0):
                ops.fix(node_tag + 1, *[1, 1, 1])

        # Determine nodes of elements
        coordTransf = "Linear"  # Linear, PDelta, Corotational
        ops.geomTransf(coordTransf, 1, *[])
        ops.geomTransf(coordTransf, 2, *[])
        # ops.geomTransf(coordTransf, 1, '-jntOffset', 0, -beam_depth / 2)
        # ops.geomTransf(coordTransf, 2, '-jntOffset', column_depth / 2, -column_depth / 2)


        # Timoshenko Beam used for verification with SAP2000.
        # Only for elastic model with the properties of 2D frame given below:
        # column_width = 0.3 * m
        # column_depth = 0.3 * m
        # beam_width = 0.2 * m
        # beam_depth = 0.4 * m
        # Shear area for column
        Avy_c = 0.075
        # Avy_b = 0.0667

        # column_width = 0.25 * m
        # column_depth = 0.25 * m
        # beam_width = 0.25 * m
        # beam_depth = 0.60 * m
        # Avy_c = 0.0521
        # Shear area for beam
        Avy_b = 0.125

        poisson_ratio = 0.2
        G_mod = E / (2 * (1 + poisson_ratio))

        ele_tag = 1
        for ele_c in column_nodes:
            # ops.element('elasticBeamColumn', ele_tag, *ele_c, column_area, E, column_inertia, 1, '-mass',
            #             mass_of_columns)
            ops.element('ElasticTimoshenkoBeam', ele_tag, *ele_c, E, G_mod, column_area, column_inertia, Avy_c,
                    1, '-mass', mass_of_columns)
            ele_tag += 1

        for key, ele_b in enumerate(beam_nodes):
            # ops.element('elasticBeamColumn', ele_tag, *ele_b, beam_area, E, beam_inertia, 2, '-mass', mass_of_beams[key])
            ops.element('ElasticTimoshenkoBeam', ele_tag, *ele_b, E, G_mod, beam_area, beam_inertia, Avy_b,
                    2, '-mass', mass_of_beams[key])
            ele_tag += 1

    def create_nonlinear_model_precode(self, fc, fy, cover, As_col_rat, As_beam_rat, column_width,
                                       column_depth, beam_width, beam_depth, dead_load, live_load, soft_story):

        # coordinates, column_nodes, beam_nodes, story_height = self.create_2D_model(soft_story)
        coordinates, column_nodes, beam_nodes = self.create_2D_model()

        loads = Loads(self.span_length, self.storey_height, beam_width, beam_depth,
                      column_width, column_depth, self.num_span, self.num_storey)
        # Column Properties
        mass_of_columns = loads.calc_column_seismic_weight() / g

        # Beam Properties
        weight_of_beams = loads.calc_beam_seismic_weight(dead_load, live_load)
        mass_of_beams = np.divide(weight_of_beams, g)

        # Create Nodes
        for node_tag, node in enumerate(coordinates):
            ops.node(node_tag + 1, *node.tolist())
            if math.isclose(node[1], 0.0):
                ops.fix(node_tag + 1, *[1, 1, 1])

        # Concrete
        epsc = 0.002        # floating point values defining concrete strain at maximum strength - TBDY2018, unconfined
        # epsc = 0.02
        epscu = 0.005       # floating point values defining concrete strain at crushing strength - TBDY2018, unconfined
        # epscu = 0.05       # floating point values defining concrete strain at crushing strength - to test - to overcome convergence
        Ec = 5000 * (fc ** 0.5) * MPa       # floating point values defining initial stiffness
        concTag = 1
        ops.uniaxialMaterial('Concrete04', concTag, -fc * MPa, -epsc, -epscu, Ec)

        # Reinforcing Steel
        E = 2 * 10 ** 5 * MPa
        steelTag = 2
        ops.uniaxialMaterial('Steel01', steelTag, fy, E, 0.034)

        # Determine nodes of elements
        coordTransf = 'Linear'
        ColTransfTag = 1
        BeamTransfTag = 2
        ops.geomTransf('PDelta', ColTransfTag, *[])
        ops.geomTransf(coordTransf, BeamTransfTag, *[])

        # Number of integration points along length of element
        num_int = 5

        Nfcore, Nfcover, Nfs = 10, 10, 4

        # Column Section
        As_col = column_depth * column_width * As_col_rat
        Atop_col, Abot_col, Aside_col = As_col / 2, As_col / 2, 0
        ColSecTag = 1
        ops.section('RCSection2d', ColSecTag, concTag, concTag, steelTag, column_depth, column_width, cover, Atop_col,
                    Abot_col, Aside_col, Nfcore, Nfcover, Nfs)

        # Beam Section
        As_beam = beam_depth * beam_width * As_beam_rat
        Atop_beam, Abot_beam, Aside_beam = As_beam / 2, As_beam / 2, 0
        BeamSecTag = 2
        ops.section('RCSection2d', BeamSecTag, concTag, concTag, steelTag, beam_depth, beam_depth, cover, Atop_beam,
                    Abot_beam, Aside_beam, Nfcore, Nfcover, Nfs)

        ele_tag = 1
        for ele_c in column_nodes:
            ops.element('nonlinearBeamColumn', ele_tag, *ele_c, num_int, ColSecTag, ColTransfTag, '-mass',
                        mass_of_columns, '-lMass')
            ele_tag += 1

        for key, ele_b in enumerate(beam_nodes):
            ops.element('nonlinearBeamColumn', ele_tag, *ele_b, num_int, BeamSecTag, BeamTransfTag, '-mass',
                        mass_of_beams[key], '-lMass')
            ele_tag += 1

        # print('Nonlinear model is created.')

    def add_staticLoad(self, column_width, column_depth, beam_width, beam_depth, dead_loads, live_loads):

        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)

        loads = Loads(self.span_length, self.storey_height, beam_width, beam_depth, column_width, column_depth,
                      self.num_span, self.num_storey)
        # Column Properties
        weight_of_columns = loads.calc_column_seismic_weight()

        # Beam Properties
        weight_of_beams_exc_ll = loads.calc_beam_seismic_weight(dead_loads, live_loads)
        live_load_retained = 0.7 * loads.calc_ll_weight(live_loads)
        total_load = weight_of_beams_exc_ll + np.tile(np.array(live_load_retained), self.num_storey)

        # if len(self.span_length) == 1:
        #     span_lengths = np.repeat(np.repeat(np.array(self.span_length), self.num_span), self.num_storey)
        # else:
        #     span_lengths = np.repeat(np.array(self.span_length), self.num_storey)

        # beam_load = np.multiply(np.ones((self.num_storey, 1)), span_lengths * (live_load + dead_load)).reshape(
        #     -1).tolist()

        # beam_load = np.reshape(np.multiply(np.array(total_load), np.ones((self.num_storey, 1))), -1)

        number_of_columns = (self.num_span + 1) * self.num_storey

        element_tags = ops.getEleTags()
        beam_tags = range(number_of_columns + 1, len(element_tags) + 1)
        column_tags = range(1, number_of_columns + 1)

        for key, i in enumerate(beam_tags):
            ops.eleLoad('-ele', i, '-type', '-beamUniform', -total_load[key])
            # ops.eleLoad('-ele', i, '-type', '-beamUniform', -mass_of_beams[key])

        for key, j in enumerate(column_tags):
            ops.eleLoad('-ele', j, '-type', '-beamUniform', 0., -weight_of_columns)

        # print('Static loads are loaded.')

    def static_analysis(self):

        # ------------------------------
        # Start of analysis generation
        # ------------------------------

        # Create the system of equation, a sparse solver with partial pivoting (large models: 'UmfPack')
        ops.system('BandGeneral')

        # Create the constraint handler, the transformation method
        try:
            ops.constraints('Plain')
        except:
            ops.constraints('Transformation')

        # Create the DOF numberer, the reverse Cuthill-McKee algorithm
        ops.numberer('RCM')

        # Create the convergence test, the norm of the residual with a tolerance of
        # 1e-7 and a max number of iterations of 10
        ops.test('NormDispIncr', 1.0e-7, 10, 5)

        # Create the solution algorithm, a Newton-Raphson algorithm: updates tangent stiffness at every iteration
        ops.algorithm('Newton')

        # Create the integration scheme, the LoadControl scheme using steps of 0.1
        ops.integrator('LoadControl', 0.1)

        # Create the analysis object
        ops.analysis('Static')

        # perform the gravity load analysis, requires 10 steps to reach the load level
        ops.analyze(10)

        ops.record()

        ops.reactions()

        ops.wipeAnalysis()

        # print('Static analysis is done.')

    def add_dynamicLoad(self, values, dt, factor, num_modes):
        if isinstance(factor, int):
            factor = float(factor)
        elif isinstance(factor, float):
            pass
        else:
            raise ValueError('factor must be float or number')

        ops.loadConst('-time', 0.0)   # hold gravity constant and restart time

        # Rayleigh Damping
        eigen = ops.eigen('-fullGenLapack', num_modes)
        power = math.pow(eigen[0], 0.5)  # Frequency of the first mode
        damping_ratio = 0.05

        power_array_for_all_modes = [math.pow(eigen[i - 1], 0.5) for i in range(1, len(eigen) + 1)]

        # modes that chosen to calculate mass and stiffness proportional damping ratios
        mode_i = 0
        mode_j = 3

        a0 = damping_ratio * 2 * power_array_for_all_modes[mode_i] * power_array_for_all_modes[mode_j] / (power_array_for_all_modes[mode_i] + power_array_for_all_modes[mode_j])
        a1 = damping_ratio * 2 / (power_array_for_all_modes[mode_i] + power_array_for_all_modes[mode_j])
        #
        # ## Plot for Rayleigh Damping
        # max_frequency = int(round(power_array_for_all_modes[-1])*10)
        # frequencies = [i / 10 for i in range(1,max_frequency,1)]
        #
        # ksi0_list = [a0 / (frequencies[i-1] * 2) for i in range(1, len(frequencies) + 1)]
        # ksi1_list = [a1 * frequencies[i-1] / 2 for i in range(1, len(frequencies) + 1)]
        # ksi_total_list = [ksi0_list[i-1] + ksi1_list[i-1] for i in range(1, len(frequencies) + 1)]
        #
        # plt.rc('font', family='Times New Roman', size=12)
        # plt.plot(frequencies, ksi0_list, color='green', label='Mass Proportional Damping (a0)')
        # plt.plot(frequencies, ksi1_list, color='blue', label='Stiffness Proportional Damping (a1)')
        # plt.plot(frequencies, ksi_total_list, color='red', label='Rayleigh Damping')
        # xmax_plot = power_array_for_all_modes[mode_j] / power_array_for_all_modes[-1]
        # plt.axhline(y=0.05, xmin=0, xmax=xmax_plot, color='m', linestyle='--')
        # plt.axvline(x=power_array_for_all_modes[mode_i], ymax=0.5, color='m', linestyle='--')
        # plt.axvline(x=power_array_for_all_modes[mode_j], ymax=0.5, color='m', linestyle='--')
        # plt.xlabel('Natural Frequency (1/sec)')
        # plt.ylabel('Modal Damping Ratio')
        # plt.title('Variation of modal damping ratio with natural frequencies \n'
        #           'a0={:.4f} 1/sec, a1={:.4f} sec'.format(a0, a1))
        # plt.legend(loc='best')
        # plt.xlim(0, power_array_for_all_modes[-1])
        # plt.ylim(0, .1)
        # plt.show()

        ops.rayleigh(a0, 0.0, 0.0, a1)

        ops.timeSeries('Path', 2, '-dt', dt, '-values', *values, '-factor', factor)
        # factor(float): A factor to multiply load factors by. (optional) - Openseespy definition
        # factor can be scale factor or a factor for changing unit, for instance g to meter/s2
        ops.pattern('UniformExcitation', 2, 1, '-accel', 2)
        # dir: "1" corresponds to translation along the global X axis

        # print('Time series of ground motion record is applied to the model.')
        #
        # return power

    def mode_shape(self, modeNo, num_modes):
        eigen = ops.eigen('-fullGenLapack', num_modes)
        power = math.pow(eigen[modeNo - 1], 0.5)  # Frequency of the first mode
        f_modeNo = power / (2 * np.pi)

        fmt_defo = {'color': 'blue', 'linestyle': 'solid', 'linewidth': 3.0,
                    'marker': '', 'markersize': 6}

        # anim = opsv.anim_mode(modeNo, fmt_defo=fmt_defo,
        #                       xlim=[-5, 10], ylim=[-2, 10], fig_wi_he=(30., 22.))
        # plt.title(f'Mode {modeNo}, f_{modeNo}: {f_modeNo:.3f} Hz')

        # plt.show()
        return eigen

    def periods_of_structures(self, num_modes):
        eigen = ops.eigen('-fullGenLapack', num_modes)
        power = np.asarray(np.power(eigen, 0.5))
        T = 2*np.pi / power
        return T

    def dynamic_analysis(self, total_time, dt):

        first_node = self.num_span + 1
        last_node = ((self.num_span + 1) * (self.num_storey + 1)) + 1
        story_nodes = list(range(first_node, last_node, self.num_span + 1))

        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        # Convergence Test (try and except must be written here)
        ops.test('NormDispIncr', 1e-6, 500)
        ops.algorithm('Newton')
        # Newmark beta integrator
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')
        controlTime = ops.getTime()  # Update the control time
        cIndex = 0
        ok = 0
        step = 0
        analysis_failed = False
        total_number_of_steps = int((total_time / dt) + 1)
        nodeDisplacements = []
        while cIndex == 0 and round(controlTime, 3) <= total_time and ok == 0:  ## ONEMLI: 500 = GRAVITY ANALYSIS NUMBER OF STEPS
            # controlTime = ops.getTime()  # Update the control time
            ok = ops.analyze(1, dt) # Run a step of the analysis
            if ok == 0:
                nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                controlTime = ops.getTime()  # Update the control time
                step = step + 1
            else:
                # print('Regular Newton failed at time t=%.2f' % controlTime)
                # print("Trying Newton with Initial Tangent ..")
                ops.algorithm('Newton', '-initial')
                ok = ops.analyze(1, dt)  # Run a step of the analysis
                if ok == 0:
                    nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                    controlTime = ops.getTime()  # Update the control time
                    step = step + 1
                else:
                    # print('"Newton -initial" failed at time t=%.2f' % controlTime)
                    # print("Trying Broyden ..")
                    ops.algorithm('Broyden', 20)
                    ok = ops.analyze(1, dt)  # Run a step of the analysis
                    if ok == 0:
                        nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                        controlTime = ops.getTime()  # Update the control time
                        step = step + 1
                    else:
                        # print("Trying NewtonwithLineSearch..")
                        ops.algorithm('NewtonLineSearch', 0.8)
                        ok = ops.analyze(1, dt)  # Run a step of the analysis
                        if ok == 0:
                            nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                            controlTime = ops.getTime()  # Update the control time
                            step = step + 1
                        else:
                            # print("reducing time step.. ")
                            ops.test('NormDispIncr', 1.0e-6, 500, 1)
                            ok = ops.analyze(1, dt / 10)  # Run a step of the analysis
                            if ok == 0:
                                nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                                controlTime = ops.getTime()  # Update the control time
                                step = step + 1
                            else:
                                # print("reducing tolerance..")
                                ops.test('NormDispIncr', 1.0e-5, 500, 1)
                                ok = ops.analyze(1, dt)  # Run a step of the analysis
                                if ok == 0:
                                    nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                                    controlTime = ops.getTime()  # Update the control time
                                    step = step + 1
                                    print("~~~ Everything looks good! Doing Great!")
                                else:
                                    analysis_failed = True
                                    print("Sorry for you, keep trying, unsuccessful trial.")

        # ops.integrator('Newmark', 0.5, 0.25)
        # ops.analysis('Transient')
        # ops.analyze(int(total_time / dt), dt)
        # ops.record()
        # ops.reactions()
        ops.wipeAnalysis()
        # print('Dynamic analysis is successfully performed.')

        MIDR = []
        if analysis_failed == False:
        # if len(nodeDisplacements) == total_number_of_steps:
            inter_story_drifts = [(u[1:] - u[0:-1]) for u in np.array(nodeDisplacements)]
            MIDR_each_step_of_record = [np.max(abs(np.divide(o[:], self.storey_height))) for o in inter_story_drifts]
            MIDR.append(np.max(abs(np.array(MIDR_each_step_of_record))))
        else:
            MIDR.append(-999)
        return MIDR[0]

    def recorder_maxDisp(self, recorder_folder, record_number, building_number):

        first_node = self.num_span + 1
        last_node = ((self.num_span + 1) * (self.num_storey + 1)) + 1
        story_nodes = list(range(first_node, last_node, self.num_span + 1))
        filename = os.path.join(recorder_folder, 'NodeDisplacement_building' + str(building_number) + '_record_' + str(record_number) + '.txt')
        ops.recorder('Node', '-file', filename, '-time', '-node', *story_nodes, '-dof', 1, 'disp')

    def get_MIDR(self, recorder_folder, record_number, building_number):
        MIDR = []
        for record_number_i in range(1, record_number + 1):
            filename = os.path.join(recorder_folder, 'NodeDisplacement_building' + str(building_number) + '_record_' + str(record_number_i) +'.txt')
            with open(filename, 'r') as f:
                displacements = f.readlines()[:]
            if len(displacements) == 12002:
                story_disp = [x.split(" ") for x in displacements]
                story_disp_last = [y[-1].split('\n') for y in story_disp]
                [z.pop(-1) for z in story_disp]
                [z.pop(0) for z in story_disp]
                [story_disp[e].append(v[0]) for e, v in enumerate(story_disp_last)]
                story_disp_array = np.array(story_disp)
                story_disp_float = [q.astype(np.float) for q in story_disp_array]
                inter_story_disp_diff = [(u[1:] - u[0:-1]) for u in story_disp_float]
                MIDR_each_step_of_record = [np.max(np.divide(o, self.storey_height)) for o in inter_story_disp_diff]
                MIDR.append(np.max(abs(np.array(MIDR_each_step_of_record))))
            else:
                MIDR.append(-999)
        return MIDR

    def get_max_MIDR_and_last_MIDR(self, total_time, dt):

        first_node = self.num_span + 1
        last_node = ((self.num_span + 1) * (self.num_storey + 1)) + 1
        story_nodes = list(range(first_node, last_node, self.num_span + 1))

        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        # Convergence Test (try and except must be written here)
        ops.test('NormDispIncr', 1e-6, 500)
        ops.algorithm('Newton')
        # Newmark beta integrator
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')
        controlTime = ops.getTime()  # Update the control time
        cIndex = 0
        ok = 0
        step = 0
        total_number_of_steps = int((total_time / dt) + 1)
        nodeDisplacements = []
        while cIndex == 0 and round(controlTime,
                                    3) <= total_time and ok == 0:  ## ONEMLI: 500 = GRAVITY ANALYSIS NUMBER OF STEPS
            # controlTime = ops.getTime()  # Update the control time
            ok = ops.analyze(1, dt)  # Run a step of the analysis
            if ok == 0:
                nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                controlTime = ops.getTime()  # Update the control time
                step = step + 1
            else:
                # print('Regular Newton failed at time t=%.2f' % controlTime)
                # print("Trying Newton with Initial Tangent ..")
                ops.algorithm('Newton', '-initial')
                ok = ops.analyze(1, dt)  # Run a step of the analysis
                if ok == 0:
                    nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                    controlTime = ops.getTime()  # Update the control time
                    step = step + 1
                else:
                    # print('"Newton -initial" failed at time t=%.2f' % controlTime)
                    # print("Trying Broyden ..")
                    ops.algorithm('Broyden', 20)
                    ok = ops.analyze(1, dt)  # Run a step of the analysis
                    if ok == 0:
                        nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                        controlTime = ops.getTime()  # Update the control time
                        step = step + 1
                    else:
                        # print("Trying NewtonwithLineSearch..")
                        ops.algorithm('NewtonLineSearch', 0.8)
                        ok = ops.analyze(1, dt)  # Run a step of the analysis
                        if ok == 0:
                            nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                            controlTime = ops.getTime()  # Update the control time
                            step = step + 1
                        else:
                            # print("reducing time step.. ")
                            ops.test('NormDispIncr', 1.0e-6, 500, 1)
                            ok = ops.analyze(1, dt / 10)  # Run a step of the analysis
                            if ok == 0:
                                nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                                controlTime = ops.getTime()  # Update the control time
                                step = step + 1
                            else:
                                # print("reducing tolerance..")
                                # ops.test('NormDispIncr', 1.0e-5, 500, 1)
                                ops.test('NormDispIncr', 1.0e-1, 1000, 1)
                                ok = ops.analyze(1, dt)  # Run a step of the analysis
                                if ok == 0:
                                    nodeDisplacements.append([ops.nodeDisp(i, 1) for i in story_nodes])
                                    controlTime = ops.getTime()  # Update the control time
                                    step = step + 1
                                    print("~~~ Everything looks good! Doing Great!")
                                else:
                                    print("Sorry for you, keep trying, unsuccessful trial.")

        # ops.integrator('Newmark', 0.5, 0.25)
        # ops.analysis('Transient')
        # ops.analyze(int(total_time / dt), dt)
        # ops.record()
        # ops.reactions()
        ops.wipeAnalysis()
        # print('Dynamic analysis is successfully performed.')

        inter_story_drifts = [(u[1:] - u[0:-1]) for u in np.array(nodeDisplacements)]
        MIDR_each_step_of_record = [np.max(abs(np.divide(o[:], self.storey_height))) for o in inter_story_drifts]
        MIDR_max = (np.max(abs(np.array(MIDR_each_step_of_record))))
        MIDR_last = MIDR_each_step_of_record[-1]

        return MIDR_max, step, total_number_of_steps, MIDR_last

    def animated_Node_Disp(self, total_time, dt):
        input_parameters = (20.8, 300., 8.)

        pf, sfac_a, tkt = input_parameters
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        # Convergence Test (try and except must be written here)
        ops.test('NormDispIncr', 1e-6, 500)
        ops.algorithm('Newton')
        # Newmark beta integrator
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')

        el_tags = ops.getEleTags()
        n_steps = int((total_time / dt))
        nels = len(el_tags)

        Eds = np.zeros((n_steps, nels, 6))
        timeV = np.zeros(n_steps)

        # transient analysis loop and collecting the data
        for step in range(n_steps):
            ops.analyze(1, dt)
            timeV[step] = ops.getTime()
            # collect disp for element nodes
            for el_i, ele_tag in enumerate(el_tags):
                nd1, nd2 = ops.eleNodes(ele_tag)
                Eds[step, el_i, :] = [ops.nodeDisp(nd1)[0],
                                      ops.nodeDisp(nd1)[1],
                                      ops.nodeDisp(nd1)[2],
                                      ops.nodeDisp(nd2)[0],
                                      ops.nodeDisp(nd2)[1],
                                      ops.nodeDisp(nd2)[2]]

        # 1. animate the deformed shape
        anim = opsv.anim_defo(Eds, timeV, sfac_a, interpFlag=1)

        # MIDR = []
        # for record_number_i in range(1, record_number + 1):
        #     filename = recorder_folder + 'EnvelopeNodeDisplacement_' + str(record_number_i) + '.txt'
        #     with open(filename, 'r') as f:
        #         max_displacement = f.readlines()[1]
        #     story_disp = max_displacement.split(" ")
        #     story_disp_last = story_disp[-1].split('\n')
        #     story_disp.pop(-1)
        #     story_disp.append(story_disp_last[0])
        #     story_disp_float = [float(x) for x in story_disp]
        #     story_disp_float = np.array(story_disp_float)
        #     inter_story_disp_diff = story_disp_float[1:] - story_disp_float[0:-1]
        #     IDR = np.divide(inter_story_disp_diff, self.storey_height)
        #     MIDR.append(np.max(IDR))
        # return MIDR
