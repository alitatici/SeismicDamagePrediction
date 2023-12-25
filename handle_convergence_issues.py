
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
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    