import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy
import datafunctions as dfunc

thetanumbins = 32
theta_edge = np.linspace(-np.pi,np.pi,thetanumbins+1) 
dtheta = 2*np.pi/(thetanumbins)

numfish=6

## note:  the decomposition functions were in v3.  look there if want to bring them back

    
### for getting and showing error fraction    
def geterrorfrac(model,params,inputs,outputs,outputsraw):
    prediction = np.array(model.model_prediction(torch.from_numpy(inputs.astype(int)),torch.from_numpy(params)))
    prediction_discrete = np.sign(prediction)
    fraction_correct = np.mean(prediction_discrete==np.sign(outputs))    
    
    osel = (np.abs(outputsraw)>20/180*np.pi) & (np.abs(outputsraw)<160/180*np.pi)
    fraction_correct_largeturns = np.mean(prediction_discrete[osel]==np.sign(outputs[osel]))
    return [fraction_correct,fraction_correct_largeturns]

def geterror_all(model,params,inputs,outputs,outputsraw):
    pt = torch.from_numpy(params)
    prediction = model.model_prediction(torch.from_numpy(inputs.astype(int)),pt)
    prediction_discrete = np.sign(np.array(prediction))
    fraction_correct = np.mean(prediction_discrete==np.sign(outputs))    
    osel = (np.abs(outputsraw)>20/180*np.pi) & (np.abs(outputsraw)<160/180*np.pi)
    fraction_correct_largeturns = np.mean(prediction_discrete[osel]==np.sign(outputs[osel]))
    
    ot = torch.from_numpy(outputs)
    ot = ot.type(torch.FloatTensor)
    loss = model.error_norm(prediction,ot,pt)
    loss_zeronorm = model.error_norm(prediction,ot,pt,lambda_mult=0)
    
    return np.array([loss.detach().numpy(), loss_zeronorm.detach().numpy(), fraction_correct, fraction_correct_largeturns,])

def printerrorfrac(model,params,inputs,outputs,outputsraw):
    fraction_correct,fraction_correct_largeturns = geterrorfrac(model,params,inputs,outputs,outputsraw)
    print('Fraction all turn sign correct',np.round(fraction_correct,3),'.  Turns 20<th<160 deg:', np.round(fraction_correct_largeturns,3))    
    

#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################
 ### Training and processing


def defaultviewfn(params):
    return

def trainmodel(inputs,outputs,model,viewstep=50,lr=1e-2,nopt=300,viewfn=defaultviewfn,lossthreshold=1e-4,inputinitialparams=[],holdconst=[]):

    # Define the model
    # input
    if len(inputinitialparams)==0:
        w_initial = model.initialparams()
    else:
        w_initial = inputinitialparams.copy()
    w_initial_tensor = torch.FloatTensor(w_initial)
        
    # holdconst is a selector mask, for which parameters to use when fitting/updating
    if len(holdconst)==0:
        model_prediction_function = model.model_prediction
        error_function = model.error_norm
    else:
        holdconst_indices = np.where(holdconst)[0]  # need to do this, because Torch doesn't support boolean indexing like numpy
        def mpf(x,w):
            wcopy = w.clone()
            wcopy[holdconst_indices] = w_initial_tensor[holdconst_indices]
            return model.model_prediction(x,wcopy)
        model_prediction_function = mpf
        
        #  assume that if any are held are constant, it includes the angle values
        # in addition to that, there is the option set here to hold F functions constant.  This needs to be passed into the error function, so that the normalization of it is not counted
        Fsel = np.logical_not(np.array([np.any(a) for a in model.parseparams(holdconst)][2:8])).astype(int)
        def mef(pred,data,w):
            return model.error_norm(pred,data,w,lambda_angle=0,Fsel=Fsel)
        error_function = mef
            
    

    x = Variable(torch.from_numpy(inputs.astype(int)),requires_grad=False)  
    w = Variable(w_initial_tensor,requires_grad=True)
    y = torch.from_numpy(outputs)
    y = y.type(torch.FloatTensor)

    # error
    loss_history = []

    # optimizer
    opt = torch.optim.Adam([w], lr=lr)

    for step in range(nopt):
        opt.zero_grad()
        preds = model_prediction_function(x,w)
        loss = error_function(preds, y, w)
        loss_history.append(loss.data)
        loss.backward()
        opt.step()
        if np.mod(step,viewstep)==0:
            print('Step ',step,'/',nopt,'\tLoss:',loss_history[-1])
            viewfn(w.detach().numpy())
        # see if the change in loss has been below the threshold for consecutive evaluations
        percentdiff = np.diff(loss_history)/loss_history[1:]
        if np.all(percentdiff[-50:]<lossthreshold) & (len(percentdiff)>50):
            break
    plt.plot(loss_history[-100:])
    plt.title("number of steps: "+str(step))

    params = w.detach().numpy()
    return params, np.array(loss_history)





###### FUNCTIONS TO MAKE INPUT-OUTPUT STRUCTURE
def getbins(inputdata, binedges,freeze_threshold=-1,speed_threshold=-1):
    smoothspeeds,heading,boundarydist,groupmembership,alldist,alldcoords_rotated,allbcoords_rotated,medianavgspeeds,_ = inputdata
    speed_edge, nspeed_edge, ndist_edge, bbin_edge, theta_edge = binedges

    rnumbins = len(ndist_edge)-1 # number of neighbor distance bins
    thetanumbins = len(theta_edge)-1
    brnumbins=len(bbin_edge)-1  # number of boundary bins
    numsteps = smoothspeeds.shape[0]
    numfish = smoothspeeds.shape[1]
    
    if freeze_threshold==-1:  # by default, make these the right edge of the first bin
        freeze_threshold = speed_edge[1]
    if speed_threshold==-1:
        speed_treshold = speed_edge[1]

    ## speed bins
    speedbins = np.digitize(smoothspeeds,speed_edge)
    speedbins[speedbins==len(speed_edge)]=len(speed_edge)-1  # should not happen unless it is for 'equality'.. then it should be only once per treatment

    nspeedbins = np.digitize(smoothspeeds,nspeed_edge)
    nspeedbins[nspeedbins==len(nspeed_edge)]=len(nspeed_edge)-1  # should not happen unless it is for 'equality'.. then it should be only once per treatment    

    ## Neighbor positions and binning
    sel_offdiag = np.logical_not(np.diag(np.ones(numfish).astype(int)))

    rbins = np.digitize(np.reshape(alldist[:,sel_offdiag],(-1,numfish,numfish-1)),ndist_edge)
    rbins[rbins==rnumbins+1]=rnumbins  # should never happen, but keep in for the equality case
    dcrsel = np.reshape(alldcoords_rotated[:,sel_offdiag],(-1,numfish,numfish-1,2))
    phibins = np.digitize(np.arctan2(dcrsel[:,:,:,1],dcrsel[:,:,:,0]),theta_edge)
    del dcrsel
    phibins[phibins==thetanumbins+1] = thetanumbins  # correct bin numbers equal to exactly +Pi

    # this returns the rank of each one, and to do this, need two argsorts:  see:  https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy
    nnums = np.argsort(np.argsort(np.reshape(alldist[:,np.logical_not(np.diag(np.ones(numfish).astype(int)))],(-1,numfish,numfish-1)),axis=2),axis=2)

    relativeheading = np.array([heading - np.tile(heading[:,i],(numfish,1)).T for i in range(numfish)]).swapaxes(0,1)
    relativeheading = np.reshape(relativeheading[:,sel_offdiag],(-1,numfish,numfish-1))
    headingbins = np.digitize(np.arctan2(np.sin(relativeheading),np.cos(relativeheading)),theta_edge)
    headingbins[headingbins==thetanumbins+1] = thetanumbins  # correct bin numbers equal to exactly +Pi
    del relativeheading

    # boundary bins
    bbins = np.digitize(boundarydist,bbin_edge)    
    bbins[bbins==brnumbins+1]=brnumbins  # this should never happen anymore, but keep this in for the 'equality' case
    bphibins = np.digitize(np.arctan2(allbcoords_rotated[:,:,1],allbcoords_rotated[:,:,0]),theta_edge)
    bphibins[bphibins==thetanumbins+1] = thetanumbins  # correct bin numbers equal to exactly +Pi


    groupmembershiptile = np.array([(np.tile(groupmembership[:,i],(numfish,1)).T == groupmembership).T for i in range(numfish)]).T
    groupmembershiptile = np.reshape(groupmembershiptile[:,sel_offdiag],(-1,numfish,numfish-1)).astype(int)       

    # speed and freezing selection integers
    speedselbins = np.zeros(smoothspeeds.shape)
    speedselbins[smoothspeeds>speed_threshold] = 1
    speedselbins[medianavgspeeds>freeze_threshold] = 2
    speedselbins[(smoothspeeds>speed_threshold) & (medianavgspeeds>freeze_threshold)] = 3        
    
    ibins = [bbins,bphibins,speedbins,nspeedbins,rbins,phibins,headingbins,nnums,groupmembershiptile,speedselbins]
    return ibins


def flipthetabins(bins):
    halfnum = np.round((thetanumbins/2)).astype(int)
    return -(bins-halfnum) + halfnum

def doubledata(data):
    return np.concatenate((data,data))

def double_with_flip(data):
    return np.concatenate((data,flipthetabins(data)))

def adddim(data):
    return np.expand_dims(data, axis=-1)


ind_tnum = 0
ind_trial = 1
ind_bdist = 2
ind_bphi = 3
ind_si = 4
ind_sj = np.arange(numfish-1)+5
ind_rj = np.arange(numfish-1)+ind_sj[-1]+1
ind_phij = np.arange(numfish-1)+ind_rj[-1]+1
ind_thetaj = np.arange(numfish-1)+ind_phij[-1]+1
ind_nnum = np.arange(numfish-1)+ind_thetaj[-1]+1
ind_groupmem = np.arange(numfish-1)+ind_nnum[-1]+1  # this is group membership - which ones are in the current group as the focal fish
ind_speedseli = ind_groupmem[-1]+1

def getinputoutputs(bindata,sel,tnum,trialids,dskip=1): 
    
    [bbins,bphibins,speedbins,nspeedbins,rbins,phibins,headingbins,nnums,groupmembershiptile,speedselbins], headingchanges = bindata
    numbinsteps = speedbins.shape[0]
    bsel = sel[0:numbinsteps]  #all of the bindata should have the same number of steps, but the selector can be based directly on data, so it could be longer
    # check if any of the bins are negative, then don't use them (shouldn't happen, but could could if the first bin does not start at zero)
    bsel = bsel & (np.sum(rbins==0,axis=2)==0) & (np.sum(phibins==0,axis=2)==0) & (np.sum(headingbins==0,axis=2)==0) & (bbins>0) & (bphibins>0) & (speedbins>0) & (nspeedbins>0)
    
    tnumtile = np.tile(tnum,(numbinsteps,numfish))
    trialidtile = np.tile(trialids[0:numbinsteps],(numfish,1)).T
    sel_offdiag = np.logical_not(np.diag(np.ones(numfish).astype(int)))
    neighborspeedbins = np.reshape(np.reshape(np.tile(nspeedbins,numfish),(-1,numfish,numfish))[:,sel_offdiag],(-1,numfish,numfish-1))  # this should use nspeedbins, which can differ from speedbins, if different edges are used
    inputs = np.concatenate((
        adddim(doubledata(tnumtile[bsel][::dskip])),  # [0] treatment number
        adddim(doubledata(trialidtile[bsel][::dskip])),  # [1] trial number        
        adddim(doubledata(bbins[bsel][::dskip])-1),  # [2] boundary:  dist
        adddim(double_with_flip(bphibins[bsel][::dskip])-1),  # [3] boundary:  angle
        adddim(doubledata(speedbins[bsel][::dskip])-1), # [4] speed of focal fish
        doubledata(neighborspeedbins[bsel][::dskip]-1),  # [5-10] speed of neighboring fish
        doubledata(rbins[bsel][::dskip]) - 1, # [10-15] distance
        double_with_flip(phibins[bsel][::dskip]) - 1, # [15-20] phi  (relative position)
        double_with_flip(headingbins[bsel][::dskip]) - 1, # [20-25] theta (relative orientation)
        doubledata(nnums[bsel][::dskip]), # [25-30] neighbor number
        doubledata(groupmembershiptile[bsel][::dskip]),  # [31-25] Group membership
        adddim(doubledata(speedselbins[bsel][::dskip])) # number for selectively taking out 'slow', 'frozen', 'slow+frozen'
        ),axis=1) 
    def outputfn(h,fn):
        hc = fn(h[bsel[0:len(h)]][::dskip])
        return np.concatenate((hc,-hc))
    outputs = [outputfn(h,np.sign) for h in headingchanges]
    outputsraw = [outputfn(h,np.array) for h in headingchanges]
    return inputs, outputs, outputsraw

def getnorm(tw):
    return (torch.sum(tw*tw,dim=-1)*dtheta-1)**2    

def getabsnorm(tw):
	return (torch.sum(torch.abs(tw),dim=-1)-tw.shape[-1])**2

def gettoponorm(tw):
    return (torch.sum(torch.abs(tw),dim=-1)-1)**2    




########################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################
 ### Models

class model_4_FiFjFkHH:	
    def __init__(self,sibins,sjbins,rnumbins,turnclassifier=True,bfit=True,topofit=False,fiti=True,fitj=True,fitk=True,fitH2=True,grouponly=False):        
        self.turnclassifier=turnclassifier  
        self.numparams = (sibins*rnumbins + thetanumbins) + (sibins + sjbins + rnumbins + thetanumbins*2)*2 + 2 + (numfish-1)
        # selecting which parameters are which
        self.psel_bsr = np.arange(0,sibins*rnumbins)
        self.psel_bphi = sibins*rnumbins + np.arange(0,thetanumbins)
        self.psel_Fatt_i = (sibins*rnumbins + thetanumbins) + np.arange(0,sibins)
        self.psel_Fatt_j = (sibins*rnumbins + thetanumbins) + sibins + np.arange(0,sjbins)
        self.psel_Fatt_k = (sibins*rnumbins + thetanumbins) + sibins + sjbins + np.arange(0,rnumbins)
        self.psel_Fali_i = (sibins*rnumbins + thetanumbins) + sibins + sjbins + rnumbins + np.arange(0,sibins)        
        self.psel_Fali_j = (sibins*rnumbins + thetanumbins) + sibins*2 + sjbins + rnumbins + np.arange(0,sjbins)        
        self.psel_Fali_k = (sibins*rnumbins + thetanumbins) + sibins*2 + sjbins*2 + rnumbins + np.arange(0,rnumbins)        
        self.psel_Hatt_phi = self.psel_Fali_k[-1]+1 + np.arange(0,thetanumbins)
        self.psel_Hatt_theta = self.psel_Hatt_phi[-1]+1 + np.arange(0,thetanumbins)
        self.psel_Hali_phi = self.psel_Hatt_theta[-1]+1 + np.arange(0,thetanumbins)
        self.psel_Hali_theta = self.psel_Hali_phi[-1]+1 + np.arange(0,thetanumbins)
        self.psel_topo = self.psel_Hali_theta[-1]+1 + np.arange(numfish-1)            
        self.psel_attract = self.numparams-2
        self.psel_align = self.numparams-1

        self.bfit=bfit
        self.topofit=topofit
        self.fiti=fiti
        self.fitj=fitj
        self.fitk=fitk
        self.fitH2=fitH2
        self.sibins=sibins
        self.sjbins=sjbins   
        self.rnumbins=rnumbins
        self.grouponly=grouponly


    def parseparams(self,w):
        bsr = w[self.psel_bsr]
        bsr = np.reshape(bsr,(self.rnumbins,-1)).T
        bphi = w[self.psel_bphi]
        Fatt_i, Fatt_j, Fatt_k = w[self.psel_Fatt_i], w[self.psel_Fatt_j], w[self.psel_Fatt_k]
        Fali_i, Fali_j, Fali_k = w[self.psel_Fali_i], w[self.psel_Fali_j], w[self.psel_Fali_k]
        Hatt_phi, Hatt_theta = w[self.psel_Hatt_phi], w[self.psel_Hatt_theta]
        Hali_phi, Hali_theta = w[self.psel_Hali_phi], w[self.psel_Hali_theta]
        attract, align = w[self.psel_attract], w[self.psel_align]
        topo = w[self.psel_topo]
        return bsr, bphi, Fatt_i, Fatt_j, Fatt_k, Fali_i, Fali_j, Fali_k, Hatt_phi, Hatt_theta, Hali_phi, Hali_theta, attract, align, topo
    
    def initialparams(self,vb=0.5,vn=1): 
        thetavals = theta_edge[0:-1] + (theta_edge[1]-theta_edge[0])/2
        theta_initial = np.sin(thetavals)/np.sqrt(np.pi)
        w_initial = np.ones(self.numparams)
        w_initial[self.psel_bsr] = vb
        w_initial[self.psel_bphi] = -theta_initial
        w_initial[self.psel_Fatt_i] = 1
        w_initial[self.psel_Fali_i] = 1
        w_initial[self.psel_Fatt_j] = 1
        w_initial[self.psel_Fali_j] = 1        
        w_initial[self.psel_Fatt_k] = 1
        w_initial[self.psel_Fali_k] = 1        
        w_initial[self.psel_Hatt_phi] = theta_initial
        w_initial[self.psel_Hali_theta] = theta_initial
        w_initial[self.psel_Hatt_theta] = np.sqrt(1/(2*np.pi))
        w_initial[self.psel_Hali_phi] = np.sqrt(1/(2*np.pi))
        w_initial[self.psel_attract], w_initial[self.psel_align] = 1, 1
        w_initial[self.psel_topo] = 1/(numfish-1)
        return w_initial
    
    def model_prediction(self,x,w):
        # boundary 
        if self.bfit:
	        b1 = w[self.psel_bsr][x[:,ind_bdist]]  #b s-r
	        b2 = w[self.psel_bphi][x[:,ind_bphi]]  #bphi
	        boundaryprediction = b1*b2
        else:
            boundaryprediction = 0
        # neighbors
        if self.fiti:
            Fi = (w[self.psel_Fatt_i][x[:,ind_si]]).unsqueeze(1)  # speed
            # Gi = (w[self.psel_Fali_i][x[:,ind_si]]).unsqueeze(1)  # speed        
            Gi = 1
        else:
            Fi, Gi = 1, 1
        if self.fitj:
	        Fj = w[self.psel_Fatt_j][x[:,ind_sj]]
	        Gj = w[self.psel_Fali_j][x[:,ind_sj]]
        else:
            Fj, Gj = 1, 1
        if self.fitk:
	        Fk = w[self.psel_Fatt_k][x[:,ind_rj]]
	        Gk = w[self.psel_Fali_k][x[:,ind_rj]]
        else:
            Fk, Gk = 1, 1
        
        p = w[self.psel_Hatt_phi][x[:,ind_phij]]  # attract-phi
        t = w[self.psel_Hali_theta][x[:,ind_thetaj]]  # align-theta
        if self.fitH2:
            pp = w[self.psel_Hatt_theta][x[:,ind_thetaj]]  # attract-theta
            tt = w[self.psel_Hali_phi][x[:,ind_phij]]  # align-phi
        else:
            pp, tt = np.sqrt(1/(2*np.pi)), np.sqrt(1/(2*np.pi))
        attract, align = w[self.psel_attract], w[self.psel_align]
        if self.topofit:
        	neighborweight = w[self.psel_topo][x[:,ind_nnum]]  # topo
        elif self.grouponly:
            neighborsum = (torch.sum(x[:,ind_groupmem],1)).type(torch.FloatTensor)
            neighborsum[neighborsum==0] = 1e6
            neighborweight = ((x[:,ind_groupmem]).type(torch.FloatTensor))/(neighborsum.unsqueeze(1))
        else:
        	neighborweight = 1/(numfish-1) 
        neighborprediction = torch.sum(neighborweight*(attract*Fi*Fj*Fk*p*pp + Gi*Gj*Gk*t*tt), dim=1)
        return boundaryprediction + neighborprediction 
    
    def error_norm(self,pred,data,w,lambda_mult=0.01,lambda_angle=0.01,Fsel=[1,1,1,0,1,1]):
        if self.turnclassifier:
            error = torch.mean(torch.log(1+torch.exp(-w[self.psel_align]*pred*data)))  # soft-margin classification loss - see https://pytorch.org/docs/stable/nn.html
        else:
            diff = (pred-data)
            error = torch.mean(diff*diff)
        tw1 = w[self.psel_bphi]
        tw2 = w[self.psel_Hatt_phi]
        tw3 = w[self.psel_Hatt_theta]
        tw4 = w[self.psel_Hali_phi]
        tw5 = w[self.psel_Hali_theta]
        Fatt_i, Fatt_j, Fatt_k = w[self.psel_Fatt_i], w[self.psel_Fatt_j], w[self.psel_Fatt_k]
        Fali_i, Fali_j, Fali_k = w[self.psel_Fali_i], w[self.psel_Fali_j], w[self.psel_Fali_k]        
        anglefnmult = lambda_angle * ( getnorm(tw1) + getnorm(tw2) + getnorm(tw3) + getnorm(tw4) + getnorm(tw5) )
        Ffnmult = lambda_mult * (Fsel[0]*getabsnorm(Fatt_i) + Fsel[1]*getabsnorm(Fatt_j) + Fsel[2]*getabsnorm(Fatt_k) + Fsel[3]*getabsnorm(Fali_i) + Fsel[4]*getabsnorm(Fali_j) + Fsel[5]*getabsnorm(Fali_k) )
        nw  = w[self.psel_topo]
        topomult = lambda_mult * (torch.sum(torch.abs(nw))-1)**2        
        return error + anglefnmult + Ffnmult + topomult
