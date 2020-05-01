# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:45:21 2020
Analyzes the kinematics of real fish data from VR experiments
Includes code for:
    - getting speed and orientation from position (getSpeed, getOri)
    - smoothing and finding jumps/anomalies in data (cleanData)
    - identifying burst-glide information and getting fits for each burst (getBGCycle)
    - getting the distribution of turn angles around each burst (getTurns)
    - finding periods of activity vs inactivity over the experiment (getSwimTime)

@author: katic
"""

# make it possible to set threshold for pos/vel/time gaps as inputs?
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from scipy.optimize import curve_fit

class Invalid(Exception):
    pass


def getSpeed(pos, time = None, xyOnly = True, lag = 1): 
    '''Gets fish speed from x,y, and optionally z positions; lag = # of timesteps back for calculations'''
      
    pos = checkInputs(pos) #pos should be array with x, y, and (optionally) z as columns
    dPos = np.subtract(pos[lag:,:],pos[:-lag,:])
    speed = np.linalg.norm(dPos, axis = 1)
               
    if type(time) == np.ndarray:
        dt = time[lag:] - time[:-lag] #make sure this is actually time in s, not just framenumber
    else:
        dt = 0.01
        print('Time data not provided, using timestep = 0.01s')
            
    speed = np.divide(speed, dt) #speed in m/s, I think xxxx check this
    return speed #in m/s
            

def getOri(pos, lag = 1): 
    '''
    Gets orientation from x and y positions; 
    not currently doing z, since change in depth isn't necessarily directly associated with change in fish pitch
    '''
    
    pos = checkInputs(pos)
    dPos = np.subtract(pos[lag:,:],pos[:-lag,:])
    
    if np.linalg.norm(dPos[0,:]) == 0:
        firstNon0 = np.where(dPos !=0)[0][0]
        print("Warning: Initial fish speed = 0. Starting orientation estimate will be inaccurate until frame" , firstNon0+lag)
    
    #xxx add fix here for when dpos = 0
    ind = np.where(dPos == 0)
    for i in ind:
        dPos[i] = dPos[i-1]  #not the most elegant, but doing in loop so that each 0 speed value is replaced by the first nonzero speed value that precedes it
    
    ori = np.arctan2(dPos[:,1],dPos[:,0])

    return ori #calculated only in xy; since they don't necessarily orient up or down when changing depth, unsure if adding z component makes sense


#used in getOri, getSpeed
def checkInputs(x, xyOnly = False):
    '''
    Checks whether position inputs have the right format and dimensions
    Removes Z data from 3d position input if XYonly is True
    '''

    #check if xyz/pos are arrays
    if type(x) != np.ndarray:
        raise Invalid('Position must be a numpy array with columns for X,Y, and optionally Z')
        #convert to array if possible?
    
    #check # of columns    
    if xyOnly == True:
        x = x[:,:2]
    elif x.shape[1] > 3 or x.shape[1] < 2:
        raise Invalid('Check dimensions. Position array should have 2 or 3 columns.')

    return x
    

#is NONE doesn't work well for arrays the way I'm trying, need to fix XXXX
def cleanData(pos = None, speed = None, ori = None, time = None, doSmooth = False, nbinsToAve=3, interpSpdGaps = True):
    '''
    Find gaps/jumps in position, speed, orientation, and/or time across the course of the trial
    Smooth if desired
    Return smoothed variables (if doSmooth), ndarray with locations of jumps, and list of columns in the returned array
    Todo: - maybe change output to pd.DataFrame 
          - this could be refactored a bit, smoothing and assignment to output array are reused bits
    '''
   #maybe do: build fn based on this that splits data into multiple trajectories at disjuncts/jump points
    
    if type(speed) != np.ndarray and type(pos) != np.ndarray and type(ori) != np.ndarray and type(time) != np.ndarray: #not supr elegant, but works
        raise Invalid('Data must be np.ndarray to clean/smooth')
        
    colNames = []
    
    if doSmooth:  #running average for now, consider more complex smoothing later if relevant
        ibin = math.floor(nbinsToAve/2)

    if type(pos) == np.ndarray:        
        pos = checkInputs(pos) #pos should be array with x, y, and (optionally) z as columns        
        if doSmooth:  #running average for now, consider more complex smoothing later if relevant
            smoothPos = np.copy(pos[ibin:-ibin,:])
            for i in range(ibin,smoothPos.shape[0]-ibin): #there is for sure a more pythonic way to do this, but for now this works
                smoothPos[i,:] = np.mean(pos[i-ibin:i+ibin,:],axis = 0)
                #print(smoothPos[i,:])
            colNames.append('xSmooth')
            colNames.append('ySmooth')
            if pos.shape[1] == 3:
                colNames.append('zSmooth')
            pos = smoothPos  # truncate by ibin at beginning and end to remove unsmoothed data
            valsOut = pos.T
                       
        dPos = np.subtract(pos[1:,:],pos[:-1,:])
        posStep = np.linalg.norm(dPos, axis = 1)
        threshold = 0.003 #about 3x the standard shift at the max point in one burst, can change if needed - xxx make settable parameter?
        #will smoothing mess up this thresholding? I oon't think so, though may make lower threhsold desirable
        posGaps = posStep > threshold
        posGaps = np.hstack((False, posGaps)) #no jump at t = 0
        colNames.append('posGaps')
        
        if 'valsOut' in locals():
            try:
                valsOut = np.vstack((valsOut,posGaps))
            except: 
                raise Invalid('Input data should have same number of datapoints')
        else:
                valsOut = np.copy(posGaps)
                     
    if type(speed) == np.ndarray:    
        if interpSpdGaps:
            #Find, Label & Interpolate loopbio step issues - not a great fix, still pretty jumpy
            zeros = speed == 0.0 
            zeroInd = np.where(speed == 0.)[0]
            consec = np.isin(zeroInd-1,zeroInd)
            for i in range(zeroInd.shape[0]):
                qq = 1
                ind = zeroInd[i]
                aux = True
                if consec[i]:
                    continue
                else:
                    while aux == True:
                        qq += 1
                        if ind+qq-1 == speed.size:
                            break
                        if speed[ind + qq-1] != 0.:
                            aux = False
                    stepinterval = 1.0/qq
                    step = np.multiply(speed[qq],stepinterval)
                    speed[ind:ind+qq] += np.multiply(step,np.arange(1,qq+1))  
            colNames.append('speedZeros')
            if doSmooth:
                zeros = zeros[ibin:-ibin]
            if 'valsOut' in locals(): #using this a lot, maybe make little function
                try:
                    valsOut = np.vstack((valsOut,zeros))
                except: 
                    raise Invalid('Input data should have same number of datapoints')
            else:
                valsOut = np.copy(zeros)
        
        if doSmooth:  
            smoothSpeed = np.copy(speed[ibin:-ibin])
            for i in range(ibin,smoothSpeed.shape[0]-ibin): #there is for sure a more pythonic way to do this, but for now this works
                smoothSpeed[i] = np.mean(speed[i-ibin:i+ibin])
            colNames.append('speedSmooth')
            speed = smoothSpeed
            if 'valsOut' in locals():
                try:
                    valsOut = np.vstack((valsOut,speed))
                except: 
                    raise Invalid('Input data should have same number of datapoints')
            else:
                valsOut = np.copy(speed)
                      
        #either before or after smooth: find speed "jumps" and mark/remove?
        threshold = 0.45 #abt 3x typical max burst speed
        speedGaps = speed > threshold
        colNames.append('speedJumps')
        if 'valsOut' in locals(): #using this a lot, maybe make little function
            try:
                valsOut = np.vstack((valsOut,speedGaps))
            except: 
                raise Invalid('Input data should have same number of datapoints')
        else:
            valsOut = np.copy(speedGaps)
     
    if type(ori) == np.ndarray:
        if doSmooth:
            smoothOri = np.copy(ori[ibin:-ibin])
            for i in range(ibin,smoothOri.shape[0]-ibin): #there is for sure a more pythonic way to do this, but for now this works
                if np.max(ori[i-ibin:i+ibin]) - np.min(ori[i-ibin:i+ibin]) > np.pi:
                    aux = np.copy(ori[i-ibin:i+ibin])
                    aux[np.where(aux<0.)] += 2.*np.pi
                    smoothOri[i] = np.mean(aux)
                    if smoothOri[i] > np.pi:
                        smoothOri[i] = smoothOri[i] - 2.*np.pi
                else:
                    smoothOri[i] = np.mean(ori[i-ibin:i+ibin])
            colNames.append('oriSmooth')      
            ori = smoothOri
            if 'valsOut' in locals():
                try:
                    valsOut = np.vstack((valsOut,ori))
                except: 
                    raise Invalid('Input data should have same number of datapoints')
            else:
                valsOut = np.copy(ori)
                
        threshold = np.pi*0.6 # look at actual distribution in data to see if this is an appropriate threshold
        oriStep = np.subtract(ori[1:],ori[:-1])
        oriStep[np.where(oriStep>np.pi)] = 2.*np.pi - oriStep[np.where(oriStep>np.pi)]
        oriJumps = oriStep > threshold
        oriJumps = np.hstack((False,oriJumps)) #no jump at t = 0
        colNames.append('oriJumps')
        if 'valsOut' in locals():
            try:
                valsOut = np.vstack((valsOut,oriJumps))
            except: 
                raise Invalid('Input data should have same number of datapoints')
        else:
            valsOut = np.copy(oriJumps)
        
    if type(time) == np.ndarray:
        if doSmooth: #note: time is not smoothed, but can be truncated to match other smoothed data
            time = time[ibin:-ibin]
        dt = time[1:] - time[:-1]
        colNames.append('Time')
        if 'valsOut' in locals():
            try:
                valsOut = np.vstack((valsOut,time))
            except: 
                raise Invalid('Input data should have same number of datapoints')
        else:
            valsOut = np.copy(time)
                
        if np.absolute(np.median(dt) - 0.01) > 0.001:
            print('Warning: median time step is different from expected')
        timeGap = dt > 0.011 #define time break if time step >10% greater than expected for VR 
        colNames.append('timeGaps')
        timeGap = np.hstack((False,timeGap))
        if 'valsOut' in locals():
            try:
                valsOut = np.vstack((valsOut,timeGap))
            except: 
                raise Invalid('Input data should have same number of datapoints')
        else:
            valsOut = np.copy(timeGap)           
    # currently smoothing is just running average, consider kernel smoothing
    # consider kalman filter? prb no
    
    valsOut = valsOut.T
    cleanedVals = pd.DataFrame(valsOut, columns = colNames)

    return cleanedVals #change to pd.dataframe

#functions for fits in getBGcycle
def exponential(x, a, c, d):
    return a*np.exp(-c*x)+d

def linear(x, m, b):
    return m*x + b

def quadratic(x, m, a, b):
    return m*x*x + a*x + b


def getBGcycle(speed, time = None, nbins= 8, threshold = 0.0, burstHtthresh = 0.0, twinsize = 1000, stepsize = 100):   #analyze BG cycle
    '''
    Identifies burst-glide cycle peaks and valleys, gets fits for each burst, and finds a sliding burstrate across time
    vars: speed = input of speed at each timepoint across whole experiment; time = time at each step (defined as 0:0.01:speed.length if not inputed)
          nbins = time steps before and after peak (or valley) for which speed must be incr/decr on average (to avoid mini-peaks)
          threshold = minimum peak speed,     burstHtthresh = minimum peak-valley difference
          twinsize and stepsize are nbins to average and slidingstepsize for calculating sliding burstrate
    Consider making burstrate into separate fn time and bursts dataframe as inputs    
    TO DO: add functionality for dealing with NaNs in speed'''
  
    #baddata = np.isnan(speed) | np.isnan(time)
    peak = np.zeros(speed.shape[0],dtype=np.int8)     
    valley = np.zeros(speed.shape[0],dtype=np.int8)
    falsevalley = np.zeros(speed.shape[0],dtype=np.int8)
    falsepeak = np.zeros(speed.shape[0],dtype = np.int8)
    
    #can make more efficient, but for now "done is better than perfect
    for i in range(nbins,speed.shape[0]-nbins):
        #if nan in range i-nbins:i+nbins, no peak, no valley, continue
        past = speed[i] - speed[i-nbins]
        future = speed[i+nbins] - speed[i]
        locMax = np.max(speed[i-nbins:i+nbins+1])
        locMin = np.min(speed[i-nbins:i+nbins+1])
        if past > 0 and future < 0 and speed[i] == locMax and speed[i] > threshold: #add threshold here if desired
            peak[i] = 1
        elif past < 0 and future > 0 and speed[i] == locMin and speed[i] > 0.0:
            valley[i] = 1
            #xxx: sometimes valley is a bit too early, due to noisiness at low speeds; alt strategy could be to pick first min before each peak and call that valley (assumes rise time is less noisy, may be viable)
    
    #if there are multiple valleys between peaks, only keep last valley
    peaktime = np.where(peak)[0]
    valtime = np.where(valley)[0]
    
    for i in range(peaktime.shape[0]-1):
        aux = []
        nvals = np.multiply(valtime > peaktime[i], valtime < peaktime[i+1])        
        if np.sum(nvals) > 1:
            aux = np.where(nvals)[0]
            falsevalley[valtime[aux[:-1]]] = 1
            valley[valtime[aux[:-1]]] = 0
            
    #if there are multiple peaks between valleys, only keep first peak
    for i in range(valtime.shape[0]-1):
        aux = []
        nvals = np.multiply(peaktime > valtime[i], peaktime < valtime[i+1])        
        if np.sum(nvals) > 1:
            aux = np.where(nvals)[0]
            falsepeak[peaktime[aux[1:]]] = 1
            peak[peaktime[aux[1:]]] = 0
    npeaks = np.sum(peak)  
    
    #Burst = valley followed by peak; therefore ignore peak if not preceded by valley, and ignore valley if not followed by peak    
    if np.min(np.where(valley)[0]) > np.min(np.where(peak)[0]):
        npeaks -= 1
        peak[np.min(np.where(peak)[0])] = 0
    if np.max(np.where(valley)[0]) > np.max(np.where(peak)[0]):
        valley[np.max(np.where(valley)[0])] = 0
        
    bursts = pd.DataFrame(np.zeros([npeaks, 5]),columns = ['n','valleyTime','peakTime','minSp','peakSp'])
    bursts['n'] = np.arange(npeaks)
    bursts['valleyTime'] = time[np.where(valley)[0]]
    bursts['peakTime']= time[np.where(peak)[0]]
    bursts['minSp'] = speed[np.where(valley)[0]]
    bursts['peakSp'] = speed[np.where(peak)[0]]    
    #maybe later: decide if false peaks/valleys worth doign something with; not currently worth it xxx
  
    #check whether these assignments work correctly 
    bursts['burstHt'] = bursts['peakSp'] - bursts['minSp']  
    bursts['IBI'] = bursts.loc[:,'peakTime']                
    bursts.loc[1:,'IBI'] =  bursts.loc[1:,'IBI'].values - bursts.loc[:npeaks-2,'IBI'].values
    bursts.loc[0,'IBI'] = -1.0 #first IBI unknown, set negative for easy filter - maybe test if you can just set empty?
    bursts = bursts[bursts.burstHt > burstHtthresh] #rmv bursts that are too small or spuriously negative - careful with this, check that it's not rmv too much
    bursts.reset_index(drop = True,inplace = True) #reset index to remove gaps
    
    linfit = np.empty([bursts.shape[0],2])
    expfit = np.empty([bursts.shape[0],3])
    quadrisefit = np.empty([bursts.shape[0],3])
    
    xlin_all = np.array([])
    ylin_all = np.array([])
    xexp_all = np.array([])
    yexp_all = np.array([])
    failed_expfit = np.zeros(bursts.shape[0],dtype = np.int8)
    
    # fit 2 part fn to each BG: linear rise, expDecay fall 
    #(change incr to quadr or sigmoid, as is not great fits) XXX
    # maybe also change decr? so more likely to not have fit pblms?
    # also find overall fit for all data pooled (and alignedat peak)
    for i in range(20):#range(bursts.shape[0]):
        tmin =  bursts.loc[i,'valleyTime']
        tpeak = bursts.loc[i,'peakTime']
        if i != bursts.shape[0]-1:
            tend = bursts.loc[i+1,'valleyTime']
        else:
            tend = time[-1] #for last peak, no next valley, so set last timepoint as end time
            
        x_lin = time[np.where(time == tmin)[0][0]:np.where(time==tpeak)[0][0]] 
        x_exp = time[np.where(time == tpeak)[0][0]:np.where(time==tend)[0][0]] 
        y_lin = speed[np.where(time == tmin)[0][0]:np.where(time==tpeak)[0][0]]
        y_exp = speed[np.where(time == tpeak)[0][0]:np.where(time==tend)[0][0]] 
        #for exp decay, should I also limit to certain num of steps? XXX
        
        #set starting t to zero. should I fully normalize and then undo after?? might make fit easier XXX
        x_lin = x_lin - tpeak
        x_exp = x_exp - tpeak
        
#        xlin_all = np.hstack((xlin_all,x_lin))
#        xexp_all = np.hstack((xexp_all,x_exp))
#        ylin_all = np.hstack((ylin_all,y_lin))
#        yexp_all = np.hstack((yexp_all,y_exp))
    
        popt_linear, pcov_linear = scipy.optimize.curve_fit(linear, x_lin, y_lin, p0=[.1, 0])
        popt_quadrise,pcov_quadrise = scipy.optimize.curve_fit(quadratic,x_lin,y_lin,p0=[.1,.1,0])
    
        try:
            popt_exp, pcov_exp = curve_fit(exponential, x_exp, y_exp, p0=(1, 10, 1)) 
        except(TypeError):
            pass
            print("peak ",i," has too few timepoints")
            failed_expfit = 1
            popt_exp[:] = np.nan
            pcov_exp[:] = np.nan
        except: #if not too few timepoints but exponential does not converge, fit w quadratic instead
            popt_exp, pcov_exp = curve_fit(quadratic, x_exp, y_exp, p0=(1, .1, 1)) #quadratic where exponential fails, keep track of index
            failed_expfit = 1
    
        linfit[i,:] = popt_linear
        expfit[i,:] = popt_exp
        quadrisefit[i,:] = popt_quadrise
        
#    #consider rmving outliers first
#    popt_linpooled, pcov_linpooled = scipy.optimize.curve_fit(linear, xlin_all, ylin_all, p0=[0.01, 0])
#    popt_quadpooled, pcov_quadpooled = scipy.optimize.curve_fit(quadratic,xlin_all,ylin_all, p0 = [0.1,0.1,0])
#    popt_exppooled, pcov_exppooled = curve_fit(exponential, xexp_all, yexp_all, p0=(1, .01, 1)) #prb need to tweak starting pt
#    #looks weird, this pooling may not work
    
    # find BG frequency over session, & over time in sliding window 
    burstrate = pd.DataFrame(np.zeros([int((time.shape[0]-twinsize)/stepsize),3]), columns = ['burstrate','tstart','tend'])
    for i in range(burstrate.shape[0]):
        timerange = (time[i*stepsize],time[i*stepsize+twinsize])
        burstrate.tstart[i] = timerange[0]
        burstrate.tend[i] = timerange[1]
        nbursts = sum((bursts['peakTime'] >= timerange[0]) & (bursts['peakTime']<timerange[1]))
        burstrate.burstrate[i] = nbursts/(timerange[1]-timerange[0])
 
    #add linfit and expfit to bursts dataframe
    bursts['linfit_slope'] = linfit[:,0]
    bursts['linfit_int'] = linfit[:,1]
    bursts['quadrise_A'] = quadrisefit[:,0]
    bursts['quadrise_B'] = quadrisefit[:,1]
    bursts['expfit_A'] = expfit[:,0]
    bursts['expfit_C'] = expfit[:,1]
    bursts['expfit_D'] = expfit[:,2]
    return bursts, burstrate

''' for other analysis of BG cycle:
    #plots: IBI distr, pkHt distr; corr pkHt:IBI?, typical exp fit with variance? overlay of bursts? 
    # more plots: example trajectories (raw), fitted trajectory over example(s), fitted trajectory with variance, BGfreq/time (sliding window), I.B.I vs. minV
    #ind = np.arange(2000,4000)
    #plt.plot(valley[ind])
    #plt.plot(peak[ind])
    
    #compare pooled fit to indiv fits (median, IQR)
    
    #corrs to check:    --- prb also separate fn
    #exp and linear fit - distr of parameters for each burst
    #burst height and IBI
    #do corrs w scipy.stats.pearsonr() or .spearmanr() - prb better choice than pandas df.corr, can get pval
    ''' 

def getTurns(ori, BGmat, time, nsteps = 40): 
    ''' 
    Finds when in the burst-glide cycle changes in orientation occur, and what the pattern/distribution of those changes is
    Inputs: np.ndarray ori of orientations, pd.Dataframe bursts with burst info, np.ndarray time 
    Output: ndarray of orientations around each burst, subtracting first datapoint
    May try to do detection of burst signatures in dOri - there is a standard pattern, but noise in data and possibly other swim bhv makes it messy
    '''
    
    dOri = ori[1:]-ori[:-1] #should this look across >1 time step? maybe make input parameter
    dOri[dOri > np.pi] -= 2.0*np.pi
    dOri[dOri < -np.pi] += 2.0*np.pi
    #dOri not currently used, but I may want to detect peaks in it XXX
    
    #find spikes in dOri - twin of interest
    #in twin of interest, find points of interest: e.g. start of initial anti-deviation; 0-crossing; peak dOri; 
    #compare dOri spikes to burst time
    
    #align ori around bursts, see where changes are/what average dOri trajectory is 
    oriAligned = np.empty([BGmat.shape[0],nsteps*2])
    #get ori aligned to each peak in bursts
    for row in BGmat.index:
        indAlign = np.where(time == BGmat.loc[row,"peakTime"])[0][0]
        if indAlign < nsteps:
            oriRow = ori[0:indAlign+nsteps].copy()
            if np.max(oriRow)-np.min(oriRow) > np.pi: #adjust for cases where ori circles around from -pi <--> +pi,
                oriRow[oriRow < 0] += 2*np.pi #may not want this for oriAligned, not sure; definitely want it for dOri; can prb do after loop
            oriRow -= oriRow[0]
            oriAligned[row,-oriRow.size:] = oriRow
        elif indAlign + nsteps > ori.shape[0]:
            oriRow = ori[indAlign-nsteps:].copy()
            if np.max(oriRow)-np.min(oriRow) > np.pi: #adjust for cases where ori circles around from -pi <--> +pi,
                oriRow[oriRow < 0] += 2*np.pi #may not want this for oriAligned, not sure; definitely want it for dOri; can prb do after loop
            oriRow -= oriRow[0]
            oriAligned[row,0:oriRow.size] = oriRow
        else:
            oriRow = ori[indAlign-nsteps:indAlign+nsteps].copy() #- ori[indAlign-nsteps]
#            oriRow = oriRow % (np.pi*2.0) - np.pi
            if np.max(oriRow)-np.min(oriRow) > np.pi: #adjust for cases where ori circles around from -pi <--> +pi,
                oriRow[oriRow < 0] += 2*np.pi #may not want this for oriAligned, not sure; definitely want it for dOri; can prb do after loop
            oriRow -= oriRow[0]
            oriAligned[row,:] = oriRow
            
    indFlip = oriAligned[:,nsteps+1] < 0 
    oriRect = oriAligned.copy()
    oriRect[indFlip,:] = -oriAligned[indFlip,:]  #maybe flip oriAligned so that things w overall neg dOri are rectified to pos (symmetrize direction of turn)
    
    #plt.plot(np.mean(oriRect, axis = 0))     

    # find when within burst cycle turning occurs
    # ccg of dOri and dSpeed?
    # distr of dOri? (overall and/or at each burst transition - maybe filter for @ burst and not)
    # corr between dOri and IBI? dOri and dSpeed?
    return oriAligned #orientation aligned with each burst
    # do I also want to get info abt that? e.g. turn angle change, max deflections, timing?
    # I want that but it's hard bc data is noisy and not always template-like


def getSwimTime(speed = None, time = None, BGmat = pd.DataFrame({'A' : []}), tThresh = 3.0, spThresh = 0.02):
    '''if speed < speedThresh for t > timeThreshold, define fish as "still"/"inactive"
    can be calc'd w speed and time or with bursts dataframe'''
    
    if type(speed) == np.ndarray and type(time) == np.ndarray:
    #do with speed and time if available
        active = np.where(speed > spThresh)[0]
        t_active = time[active]
        gaps = t_active[1:]-t_active[:-1]
        longgapInd = np.where(gaps>tThresh)[0]
        gapstart = t_active[longgapInd]
        gaptime = gaps[longgapInd]
        stillwin = np.vstack([gapstart,(gapstart+gaptime),gaptime]).T #array w column of start and stoptimes for non-swimming periods
        #xxx make stillwin pandas dataframe?
        #note: the starttime technichally is the last twin when fish is >thresh, consider shifting to make first twin when fish is still
    elif BGmat.empty:
        raise Invalid('Inputs must include either np.ndarrays "speed" and "time" or pd.DataFrame "bursts"')
    else:
        BGmat.drop(np.where(BGmat["peakSp"] > spThresh)[0],inplace = True) #filter bursts df to remove bursts w peaksp < spthresh
        BGmat.reset_index(drop = True,inplace = True)
        BGmat.loc[1:,'IBI'] = BGmat.loc[1:,'peakTime'].values-BGmat.loc[:BGmat.shape[0]-2,'peakTime'].values    #recalc IBI on preserved bursts    
        gaptime = BGmat.loc[(BGmat["IBI"]>tThresh),"IBI"].values
        gapstart = BGmat.loc[BGmat.IBI>tThresh,"peakTime"].values
        stillwin = np.vstack([gapstart,(gapstart+gaptime),gaptime]).T
    
    inact = pd.DataFrame(stillwin, columns = ['lastSwimTime','nextSwimTime','inactTime'])    
   # totalInact = np.sum(gaptime)
    return inact


def runAll(pos, time = None, speed = None, ori = None, xyOnly = True ):
    '''
    get speed & ori from position (if needed), then identify data gaps (cleanData),
    identify bursts, find dOri at each burst, and find when fish was inactive vs active
    ''' 
 #maybe later: add something to get pos xyz and time from loopbio csv output
 
    if type(time) != np.ndarray:
        time = np.arange(0.0,(pos.shape[0])/100.0,.01)
    if type(speed) != np.ndarray:
        speed = getSpeed(pos, time = time)
    if type(ori) != np.ndarray:
        ori = getOri(pos)
    
    pos = pos[1:,:]
    time = time[1:]
    dataGaps = cleanData(pos = pos, time = time, speed = speed, ori = ori)       
    bursts, burstrate = getBGcycle(speed = speed, time = time)
    whenInact = getSwimTime(speed = speed, time = time, BGmat = bursts.copy())
    oriAlignedBurst = getTurns(ori = ori, time = time, BGmat = bursts.copy())
    
    if xyOnly == True:
        kinMat = pd.DataFrame(np.vstack([pos[:,0],pos[:,1],speed,ori,time]).T,columns = ['posX','posY','speed','orientation','time' ])
    else:
        kinMat = pd.DataFrame(np.vstack([pos[:,0],pos[:,1],pos[:,2],speed,ori,time]).T,columns = ['posX','posY','speed','orientation','time' ])
           
    return kinMat, bursts, burstrate, whenInact, oriAlignedBurst, dataGaps