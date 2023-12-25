       
        ###############################################################################################################################################################################
        start_gravity = timer()
        
        def do_gravity(nSteps = 500, Oflag = 0):
            
            #  analysis parameters
            # ------------------------------------------------------------------------
            # Wipe any previous analysis object
            ops.wipeAnalysis()
        
            # Constraints handler: determines how the constraint equations are enforced in the analysis -- how it handles the boundary conditions/imposed displacements
            ops.constraints('Transformation')
            # ops.constraints('Plain')
            # ops.constraints('Lagrange') #need to be used when rigid diaph used??
            # ops.constraints('Penalty',1e3,1e3)
            
            # DOF_Numberer -- determines the mapping between equation numbers and degrees-of-freedom
            ops.numberer('RCM')
            
            # SystemOfEqn/Solver -- within the solution algorithm, it specifies how to store and solve the system of equations in the analysis
            ops.system('BandGeneral')
            # ops.system('FullGeneral')
            
            nSteps = 10
            
            # Convergence Test -- determines when convergence has been achieved.
            tol = 1.0e-5  #  the tolerance (default) default:1e-8
            iterMax = 1000  #  the max bumber of iterations (default) default=50
            pFlag = 0     # Optional print flag (default is 0). Valid options: 0-5
            nType = 2     # optional type of norm (default is 2). Valid options: 0-2
            ops.test('NormDispIncr', tol, iterMax, pFlag, nType)
            # ops.test('EnergyIncr', tol, iterMax, pFlag, nType)
        
            # SolutionAlgorithm -- determines the sequence of steps taken to solve the non-linear equation at the current time step
            ops.algorithm('Newton', '-initial') 
            # ops.algorithm('Newton') 
            
            # Integrator -- determines the predictive step for time t+dt
            dLambda = 1/nSteps # the load factor increment
            ops.integrator('LoadControl', dLambda)
            
            # AnalysisType -- defines what type of analysis is to be performed ('Static', 'Transient' etc.)
            ops.analysis('Static')
            
            # Recorders
            ops.recorder('Node', '-file', "basenodesreactions.txt",'-time', '-closeOnWrite', '-node', *base_node_tags, '-dof',3, 'reaction')
            ops.recorder('Node', '-file', "topnodedisp.txt",'-time', '-closeOnWrite', '-node', Rd_masternodes[-1], '-dof',1, 'disp')
            
            # Perform the analysis
            # ------------------------------------------------------------------------
            ops.analyze(nSteps)
            
            ops.record()
            
            # Maintain constant gravity loads and re time to zero
            # ------------------------------------------------------------------------
            ops.loadConst('-time', 0.0)	
            
            # Save element forces and nodal displacements, base reactions
            # ------------------------------------------------------------------------
            ops.reactions() # Must call this command before using nodeReaction() command.
            
            BaseReactions = 0
            for node in ops.getNodeTags():
                forces = np.array(ops.nodeReaction(node))
                BaseReactions += forces
           
            eleForces = {}
            for ele in ops.getEleTags():
                forces = ops.eleForce(ele)
                eleForces[ele] = np.array(forces)
            
            nodalDisps = {}
            for node in ops.getNodeTags():
                disps = ops.nodeDisp(node)
                nodalDisps[node] = np.array(disps)
                
            NodeForces = {}
            for node in ops.getNodeTags():
                NForces = ops.nodeReaction(node)
                NodeForces[node] = np.array(NForces)
            
            # ------------------------------------------------------------------------
         
            # IMPORTANT: Make sure to issue a wipe() command to close all the recorders. Not issuing a wipe() command
            # ... can cause errors in the plot_deformedshape() command.
            
            # ops.wipe()
            
            return eleForces, nodalDisps, BaseReactions, NodeForces
        
        # Perform static analysis under gravity loads
        # ------------------------------------------------------------------------
        eleForces, nodalDisps, BaseReactions, NodeForces = do_gravity()
        print("Gravity analysis completed.")
        end_gravity=timer()
        duration_gravityanalysis=round(end_gravity-start_gravity,2)
        print('Time for gravity analysis:', duration_gravityanalysis,'s') 		
        
        ####################################
        ### Now plot mode shape 2 with scale factor of 300 and the deformed shape using the recorded output data
        
        # opsplt.plot_modeshape(1, 20, Model="3DGenericFrame")
        # opsplt.plot_deformedshape(Model="3DGenericFrame", LoadCase="Gravity")
        		
        
        ##############################################################################################################################################################################
        print('Dynamic Analysis has started.')
        
        
        start_NTHA = timer()
        #  analysis tings
        # ------------------------------------------------------------------------
        # Wipe any previous analysis object
        ops.wipeAnalysis() 
        # loadConst -time 0.0
        ops.loadConst('-time', 0.0)	
               
        ################################################################################
        # DAMPING COEFICIENTS
        ################################################################################ 
        # ------------ define & apply damping
        # RAYLEIGH damping parameters, Where to put M/K-prop damping, switches (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/1099.htm)
        # D=$alphaM*M + $betaKcurr*Kcurrent + $betaKcomm*KlastCommit + $beatKinit*$Kinitial
        xDamp=0.05					# damping ratio
        MpropSwitch=1.0
        KcurrSwitch=0.0
        KcommSwitch=1.0
        KinitSwitch=0.0
        nEigenI=1		# mode 1
        nEigenJ=4		# mode 4
        lambdas = ops.eigen('-fullGenLapack', nEigenJ) # eigenvalue analysis (for 4 modes)  #EIGENVALUES
        lambdaI=lambdas[nEigenI-1] 		# eigenvalue mode i
        lambdaJ=lambdas[nEigenJ-1] 		# eigenvalue mode j
        omegaI=lambdaI**0.5
        omegaJ=lambdaJ**0.5
        alphaM=MpropSwitch*xDamp*(2*omegaI*omegaJ)/(omegaI+omegaJ)	# M-prop. damping; D = alphaM*M
        betaKcurr=KcurrSwitch*2.*xDamp/(omegaI+omegaJ)         		# current-K;      +beatKcurr*KCurrent
        betaKcomm=KcommSwitch*2.*xDamp/(omegaI+omegaJ)   				# last-committed K;   +betaKcomm*KlastCommitt
        betaKinit=KinitSwitch*2.*xDamp/(omegaI+omegaJ)       			# initial-K;     +beatKinit*Kini
        
    # for gm_no in range(2): 
        ops.wipeAnalysis() 
        ################################################################################
        # GROUND MOTION INPUT
        ################################################################################ 
        # Dt=0.01/2*sec	# time-step dt for lateral analysis
        # Dt=0.005*sec	# time-step dt for lateral analysis
        Dt=Dt_DD2[gm_no]*sec	# time-step dt for lateral analysis
        print(f'Dt={Dt} sec')
        # Tmax=18.0*sec	# maximum duration of ground-motion analysis
        # Tmax=40.0*sec	# maximum duration of ground-motion analysis
        Tmax=Tmax_DD2[gm_no]*sec	# maximum duration of ground-motion analysis
        print(f'Tmax={Tmax} sec')
    
        gtoms2=9.81 #Scaling of the GM (multiplies the original file accelerations)
        sf=scalefactor[gm_no]
        _lambda=round(gtoms2*sf,2)
        
        # Uniform EXCITATION: acceleration input
        IDloadTag = 400			# load tag
        GMdirection = 1
        
        print(f'Current ground motion file:{gmID_DD2[gm_no]}')    
    
        ops.timeSeries('Path', gm_no+3, '-dt', Dt, '-filePath', DD2records[gm_no], '-factor', _lambda)
        print(f'timeSeries(Path, {gm_no+3}, -dt, {Dt}, -filePath, {DD2records[gm_no]}, -factor, {_lambda})')
                
        ops.pattern('UniformExcitation', gm_no+IDloadTag, GMdirection, '-accel', gm_no+3) # up Analysis Parameters ---------------------------------------------
        print(f'ops.pattern(UniformExcitation, {gm_no+IDloadTag}, {GMdirection}, -accel, {gm_no+3}')# up Analysis Parameters ---------------------------------------------
        # CONSTRAINTS handler -- Determines how the constraint equations are enforced in the analysis (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/617.htm)
        #          Plain Constraints -- Removes constrained degrees of freedom from the system of equations
        #          Lagrange Multipliers -- Uses the method of Lagrange multipliers to enforce constraints
        #          Penalty Method -- Uses penalty numbers to enforce constraints
        #          Transformation Method -- Performs a condensation of constrained degrees of freedom
        # Constraints handler: determines how the constraint equations are enforced in the analysis -- how it handles the boundary conditions/imposed displacements
        ops.constraints('Transformation')
        
        # DOF NUMBERER (number the degrees of freedom in the domain): (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/366.htm)
        #   determines the mapping between equation numbers and degrees-of-freedom
        #          Plain -- Uses the numbering provided by the user
        #          RCM -- Renumbers the DOF to minimize the matrix band-width using the Reverse Cuthill-McKee algorithm
        # DOF_Numberer -- determines the mapping between equation numbers and degrees-of-freedom
        ops.numberer('RCM')
        
        # SYSTEM (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/371.htm)
        #   Linear Equation Solvers (how to store and solve the system of equations in the analysis)
        #   -- provide the solution of the linear system of equations Ku = P. Each solver is tailored to a specific matrix topology.
        #          ProfileSPD -- Direct profile solver for symmetric positive definite matrices
        #          BandGeneral -- Direct solver for banded unsymmetric matrices
        #          BandSPD -- Direct solver for banded symmetric positive definite matrices
        #          SparseGeneral -- Direct solver for unsymmetric sparse matrices (-piv option)
        #          SparseSPD -- Direct solver for symmetric sparse matrices
        #          UmfPack -- Direct UmfPack solver for unsymmetric matrices
        # SystemOfEqn/Solver -- within the solution algorithm, it specifies how to store and solve the system of equations in the analysis
        ops.system('BandGeneral')
        
        # TEST: # convergence test to
        # Convergence TEST (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/360.htm)
        #   -- Accept the current state of the domain as being on the converged solution path
        #   -- determine if convergence has been achieved at the end of an iteration step
        #          NormUnbalance -- Specifies a tolerance on the norm of the unbalanced load at the current iteration
        #          NormDispIncr -- Specifies a tolerance on the norm of the displacement increments at the current iteration
        #          EnergyIncr-- Specifies a tolerance on the inner product of the unbalanced load and displacement increments at the current iteration
        #          RelativeNormUnbalance --
        #          RelativeNormDispIncr --
        #          RelativeEnergyIncr --
        
        # Convergence Test -- determines when convergence has been achieved.
        # tol = 1.0e-8  # the tolerance (default)
        tol = 1.0e-6  # the tolerance (default)
        # iterMax = 50  # the max bumber of iterations (default)
        # iterMax = 200  # the max bumber of iterations (default)
        iterMax = 500  # the max bumber of iterations (default)
        pFlag = 0     # Optional print flag (default is 0). Valid options: 0-5
        nType = 2     # optional type of norm (default is 2). Valid options: 0-2
        ops.test('NormDispIncr', tol, iterMax, pFlag, nType)
        # ops.test('EnergyIncr', tol, iterMax, pFlag, nType)
        
        # SolutionAlgorithm -- determines the sequence of steps taken to solve the non-linear equation at the current time step
        ops.algorithm('Newton')
        # ops.algorithm('Newton', '-initial')
        # ops.algorithm('NewtonLineSearch')
        
        # Solution ALGORITHM: -- Iterate from the last time step to the current (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/682.htm)
        #          Linear -- Uses the solution at the first iteration and continues
        #          Newton -- Uses the tangent at the current iteration to iterate to convergence
        #          ModifiedNewton -- Uses the tangent at the first iteration to iterate to convergence
        #          NewtonLineSearch --
        #          KrylovNewton --
        #          BFGS --
        #          Broyden --
        # # SolutionAlgorithm -- determines the sequence of steps taken to solve the non-linear equation at the current time step
        # ops.algorithm("Newton")
        
        # Static INTEGRATOR: -- determine the next time step for an analysis  (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/689.htm)
        #          LoadControl -- Specifies the incremental load factor to be applied to the loads in the domain
        #          DisplacementControl -- Specifies the incremental displacement at a specified DOF in the domain
        #          Minimum Unbalanced Displacement Norm -- Specifies the incremental load factor such that the residual displacement norm in minimized
        #          Arc Length -- Specifies the incremental arc-length of the load-displacement path
        # Transient INTEGRATOR: -- determine the next time step for an analysis including inertial effects
        #          Newmark -- The two parameter time-stepping method developed by Newmark
        #          HHT -- The three parameter Hilbert-Hughes-Taylor time-stepping method
        #          Central Difference -- Approximates velocity and acceleration by centered finite differences of displacement
        # Integrator -- determines the predictive step for time t+dt
        gamma = 0.5   # Newmark gamma coefficient (also HHT)
        beta = 0.25   # Newmark beta coefficient
        ops.integrator('Newmark', gamma, beta)
        
        # ANALYSIS  -- defines what type of analysis is to be performed (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/324.htm)
        #          Static Analysis -- solves the KU=R problem, without the mass or damping matrices.
        #          Transient Analysis -- solves the time-dependent analysis. The time step in this type of analysis is constant. The time step in the output is also constant.
        #          variableTransient Analysis -- performs the same analysis type as the Transient Analysis object. The time step, however, is variable. This method is used when
        #                 there are convergence problems with the Transient Analysis object at a peak or when the time step is too small. The time step in the output is also variable.
        # AnalysisType -- defines what type of analysis is to be performed ('Static', 'Transient' etc.)
        ops.analysis('Transient')
        ops.rayleigh(alphaM,betaKcurr,betaKinit,betaKcomm)				# RAYLEIGH damping      
        
        # Initialize some parameters
        # ------------------------------------------------------------------------
        cIndex = 0         # Initially define the control index (-1 for non-converged, 0 for stable, 1 for global collapse)
        controlTime = 0.0  # Start the controlTime
        step = 0
        ok = 0             # the convergence to 0 (initially converged)
        mflr = 0           # the initial pier collapse location
        h = []             # storey heights
        mdrft = []         # the interstorey drift values
        Tol=1.0e-6
        
        # ### RECORDERS
        # BaseReactions = 0
        # for node in ops.getNodeTags():
        #     forces = np.array(ops.nodeReaction(node))
        #     BaseReactions += forces
           
        # eleForces = {}
        # for ele in ops.getEleTags():
        #     forces = ops.eleForce(ele)
        #     eleForces[ele] = np.array(forces)
        
        # nodalDisps = {}
        # for node in ops.getNodeTags():
        #     disps = ops.nodeDisp(node)
        #     nodalDisps[node] = np.array(disps)
            
        # NodeForces = {}
        # for node in ops.getNodeTags():
        #     NForces = ops.nodeReaction(node)
        #     NodeForces[node] = np.array(NForces)
        ### RECORDERS
        
       
        
        # Perform the analysis
        # ------------------------------------------------------------------------
        print("NTHA started.")
        
        # ### RECORDERS
        # BaseReactions = 0
        # for node in ops.getNodeTags():
        #     forces = np.array(ops.nodeReaction(node))
        #     BaseReactions += forces
           
        # eleForces = {}
        # for ele in ops.getEleTags():
        #     forces = ops.eleForce(ele)
        #     eleForces[ele] = np.array(forces)
        
        # nodalDisps = {}
        # for node in ops.getNodeTags():
        #     disps = ops.nodeDisp(node)
        #     nodalDisps[node] = np.array(disps)
            
        # NodeForces = {}
        # for node in ops.getNodeTags():
        #     NForces = ops.nodeReaction(node)
        #     NodeForces[node] = np.array(NForces)
        
        controlTime = ops.getTime()  # Update the control time
        
        while cIndex == 0 and controlTime <= Tmax and ok == 0: ## ONEMLI: 500 = GRAVITY ANALYSIS NUMBER OF STEPS
            step=step+1
            if step%100==0:
                print("This is step:",step)
            controlTime = ops.getTime()  # Update the control time
            ok = ops.analyze(1, Dt)      # Run a step of the analysis
            if ok!=0:
                print('Regular Newton failed at time t=%.2f' % controlTime)
                print("Trying Newton with Initial Tangent ..")
                ops.algorithm('Newton','-initial')
                ok = ops.analyze(1, Dt)      # Run a step of the analysis
                if ok!=0:
                    controlTime = ops.getTime()  # Update the control time
                    print('"Newton -initial" failed at time t=%.2f' % controlTime)
                    print("Trying Broyden ..")
                    ops.algorithm('Broyden',20)
                    ok = ops.analyze(1, Dt)      # Run a step of the analysis
                    if ok!=0:
                        print("Trying NewtonwithLineSearch..")
                        ops.algorithm('NewtonLineSearch',0.8)
                        ok = ops.analyze(1, Dt)      # Run a step of the analysis
                        if ok!=0:
                            print("reducing time step.. ")
                            ops.test('NormDispIncr',1.0e-8,50,1)
                            ok = ops.analyze(1, Dt/10)      # Run a step of the analysis
                            if ok!=0:
                                print("reducing tolerance..")
                                ops.test('NormDispIncr',1.0e-6,50,1)
                                ok = ops.analyze(1, Dt)      # Run a step of the analysis
                                if ok==0:
                                    print("~~~ Everything looks good! Doing Great!")
                                else:
                                    print("Sorry for you, keep trying, unsuccessful trial.")
                                    
        if ok==0:                                         
            ### RECORDERS
            BaseReactions = 0
            for node in ops.getNodeTags():
                forces = np.array(ops.nodeReaction(node))
                BaseReactions += forces
               
            eleForces = {}
            for ele in ops.getEleTags():
                forces = ops.eleForce(ele)
                eleForces[ele] = np.array(forces)
            
            nodalDisps = {}
            for node in ops.getNodeTags():
                disps = ops.nodeDisp(node)
                nodalDisps[node] = np.array(disps)
                
            NodeForces = {}
            for node in ops.getNodeTags():
                NForces = ops.nodeReaction(node)
                NodeForces[node] = np.array(NForces)
    # '''
    # Filter nodal displacements results dictionary by rigid diaphragm list
    # '''        
        # if ok==0:        
            Rdnodedisps = { Rdkeys: nodalDisps[Rdkeys] for Rdkeys  in Rd_masternodes }           
            Story_Disps=[]
            Story_Drifts=[]
            IntersDriftRatio=[]
            Story_Disps.append(0)
            IntersDriftRatio.append(0)
            
            Story_Heights=[]
            for i in range(len(elev)-1):
                h=round(elev[i+1]-elev[i],2)
                Story_Heights.append(h)
                    
            for i in Rd_masternodes:
                storydisp=Rdnodedisps[i][GMdirection-1] #x yonundeki deplasman alindi. 
                Story_Disps.append(storydisp)
            
                # drift=Rdnodedisps[Rdnode][GMdirection-1]-Rdnodedisps[Rdnode+1][GMdirection-1]
            for k in range(len(Story_Disps)-1):
                story_drift=Story_Disps[k+1]-Story_Disps[k]
                Story_Drifts.append(story_drift)
                interstoryDriftRatio=story_drift/Story_Heights[k]
                IntersDriftRatio.append(interstoryDriftRatio)
                
                
            IntersDriftRatio=[abs(i) for i in IntersDriftRatio]
            max_IntersDriftRatio=max(IntersDriftRatio)
            story_number=list(range(len(elev)))
    
            cur_Result={'MIDR':[max_IntersDriftRatio],
                            'TopDisp':[Story_Disps[-1]],
                            'IDRlist':IntersDriftRatio,
                            'story_number':story_number,}
            analysis_results[f'ModelNo{model_dictname}_Results'][f'GM ID.{gmID_DD2[gm_no]}']=cur_Result
            
    
        # Print the final status of the analysis
        # ------------------------------------------------------------------------
        # if cIndex == -1:
            # Analysis = "Analysis is FAILED to converge at %.3f of %.3f" % (controlTime, Tmax)
        if ok == 0:
            # text = ["\nInterstorey drift: %.4f% at floor %d" % (IntersDriftRatio[i],i) for i in story_number]
            Analysis = ''.join([f'ModelNo{model_dictname} GM ID.{gmID_DD2[gm_no]} Analysis is SUCCESSFULLY completed'])
        # if cIndex == 1:
            # Analysis = "Analysis is STOPPED, peak interstorey drift ratio, %d%%, is exceeded, global COLLAPSE is observed" % Dc
        print('------------------------------------------------------------------------')
        print(Analysis)
        
        #### DETERMINE THE DURATION OF NTHA ####
        print("\nNTHA completed.")
        end_NTHA=timer()
        duration_NTHA=round(end_NTHA-start_NTHA,2)
        print('Time for NTHA:', duration_NTHA,'s') 
        print(f'Ground Motion No.{gmID_DD2[gm_no]} completed. ')		

import json
with open(f'genericframes_story{num_story}_DD2X.json', 'w') as fp:
    json.dump(analysis_results, fp, indent=2)
    
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    