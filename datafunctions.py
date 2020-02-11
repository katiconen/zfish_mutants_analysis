import numpy as np
import scipy.signal as signal
import scipy
import pandas as pd
import pickle

numfish=6

########################################################################################################################
# import and processing functions
########################################################################################################################

def getcat(treatment,datadir,trialsel='all'):
    [trial_speeds,trial_trajectories,trial_headings,trial_theta,
                    trial_smoothspeeds,trial_smoothtrajectories,trial_smoothheadings,
                    trial_ellipses,trial_arena,trial_sex,
                    datafiles,trial_trackingerrors] = pickle.load(open(datadir+treatment+'-alltrials.pkl','rb'))
    
    numtrials = len(datafiles)
    numfish = trial_speeds[0].shape[1]

    # transpose theta, because the other quantities are stored that way:
    trial_theta = [t.T for t in trial_theta]

#     print(treatment,',', numtrials, 'trials')

    ## Calculate distance and rotated coordinates, or import if already calculated

    ddfilename = datadir+treatment+'-dcoords+dist-heading.pkl'
    trial_dcoords,trial_dist = pickle.load(open(ddfilename,'rb'))
        
    # select which trial
    # concatenate!  this introduces a few points of noise, where they concatenate, but should be min
    trial_id = [np.tile(n,len(trial_trajectories[n])) for n in range(numtrials)]
    
    if trialsel == 'all':
        alldist = np.concatenate(trial_dist)
        alldcoords_rotated = np.concatenate(trial_dcoords)
        smoothspeeds = np.concatenate(trial_smoothspeeds)
        theta = np.concatenate(trial_theta)
        heading = np.concatenate(trial_smoothheadings)
        smoothtrajectories = np.concatenate(trial_smoothtrajectories)        
        trajectories = np.concatenate(trial_trajectories)
        trialids = np.concatenate(trial_id)
        trackingerrors = np.concatenate(trial_trackingerrors)
        
    else:
#         trialsel = trialsel.astype(int)
        alldist = trial_dist[trialsel]
        alldcoords_rotated = trial_dcoords[trialsel]
        smoothspeeds = trial_smoothspeeds[trialsel]    
        theta = trial_theta[trialsel]
        heading = trial_smoothheadings[trialsel]
        smoothtrajectories = trial_smoothtrajectories[trialsel]
        trajectories = trial_trajectories[trialsel]
        trialids = trial_id[trialsel]
        trackingerrors = trial_trackingerrors[trialsel]
    return alldist, alldcoords_rotated, smoothtrajectories, smoothspeeds, theta, heading, trajectories, trialids, trackingerrors


def getmidr(traj,thr):
    coords = np.reshape(traj,(-1,2))
    minx = np.nanquantile(coords[:,0],thr)
    maxx = np.nanquantile(coords[:,0],1-thr)
    miny = np.nanquantile(coords[:,1],thr)
    maxy = np.nanquantile(coords[:,1],1-thr)
    mid = np.array([(maxx+minx)/2,(maxy+miny)/2])
    r = np.sqrt((coords[:,0]-mid[0])**2 + (coords[:,1]-mid[1])**2)
    return mid, np.nanquantile(r,1-thr)

def getboundaries(trajectories,trialids,numtrials,boundaryquantilethreshold = 0.001, wrongboundarythreshold = 23):
  # boundaryquntilethresohld is for getting quantile distributions of points and using them to calculate the boundary
  # I looked at WT data to get wrongboundarythreshold this... it looks reasonable.  This is correct for cases where the fish just stay in a single "corner" the whole time, and so the min/max are clearly wrong, so for these use median values instead
    
    trial_arena_r = np.zeros((numtrials))
    trial_arena_mid = np.zeros((numtrials,2))
    #  Get the boundary coordinates for each one
    for trialnum in range(numtrials):
        sel = (trialids==trialnum)
        # get an estimate of the arena boundary by only considering points where the tracking is "sure"
        trial_arena_mid[trialnum], trial_arena_r[trialnum] = getmidr(trajectories[sel],boundaryquantilethreshold)

    # Correct the boundary coordinates and set to the median, for ones that are far away from the median.  Do this ONLY if the inferred arena is smaller than the median.  Or if the center point is far off
    medx = np.median(trial_arena_mid[:,0])
    medy = np.median(trial_arena_mid[:,1])
    medr = np.median(trial_arena_r)
    for trialnum in range(numtrials):
        centerdifference = np.sqrt((medx-trial_arena_mid[trialnum,0])**2 + (medy-trial_arena_mid[trialnum,1])**2)
        radiusdifference = np.abs(medr-trial_arena_r[trialnum])
        if (centerdifference>wrongboundarythreshold) | (radiusdifference>wrongboundarythreshold):
            trial_arena_mid[trialnum] = [medx,medy]
            trial_arena_r[trialnum] = medr

    return trial_arena_mid, trial_arena_r

########################################################################################################################
# Basic functions
########################################################################################################################
def fixanglerange(angles): # Puts all angles into a range of [-Pi,Pi] 
    return np.arctan2(np.sin(angles),np.cos(angles))


def smooth_vector(x, window_length=15):
    """Smooth a vector with a Savgol filter."""
    degree = 3
    
    # These parameters work best for our dataset, might need some finetuning.
    x = signal.savgol_filter(x, window_length=window_length, polyorder=degree, mode='constant', deriv=0)
    x_deriv = signal.savgol_filter(x, window_length=window_length, polyorder=degree, mode='constant', delta=0.01, deriv=1)

    return x, x_deriv

def removenan(x):
    return x[np.logical_not(np.isnan(x))]

# yes, it is actually this complicated to flatten a list into a numpy array
def flattenr(L):
    for item in L:
        try:
            yield from flattenr(item)
        except TypeError:
            yield item
def flatten(L):
    return np.array(list(flattenr(L)))

# this is useful!
def initarray(lengths):
    if len(lengths)==1:
        return [[] for _ in range(lengths[0])]
    else:
        return [initarray(lengths[1:]) for _ in range(lengths[0])]


def movingaverage(data,N):
    # Pandas syntax is ridiculous to me, but this indeed works.  see https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/30141358#30141358
    return pd.Series(data).rolling(window=N).mean().values
def movingmedian(data,N):
    # Pandas syntax is ridiculous to me, but this indeed works.  see https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/30141358#30141358
#     return pd.Series(data).rolling(window=N).median().iloc[N-1:].values
    return pd.Series(data).rolling(window=N,center=True).median().values
def mmed_all(smoothspeeds,N,skipcalc=5,threshold=27.129):  # 27.129=all_speedquantiles20[1]
    skipcalc = 1 if N==1 else skipcalc        
    allmed = np.array([movingmedian(smoothspeeds[::skipcalc,i],int(N/skipcalc))for i in range(6)]).T
    return allmed, np.mean(allmed<threshold)



########################################################################################################################
# Changes in heading
########################################################################################################################

def get_headingchange(data='heading-or-theta',shift=60):
    return fixanglerange(data[shift:]-data[:-shift])

def get_vectorheadingchange(data='trajectories',data2='heading',shift=60):
    dtraj = data[shift:]-data[:-shift]
    vangle = np.arctan2(dtraj[:,:,1],dtraj[:,:,0])
    return fixanglerange(vangle-data2[0:-shift])

