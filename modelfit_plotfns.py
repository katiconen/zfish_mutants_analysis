import numpy as np
import matplotlib.pyplot as plt
import headingchange_models_v6 as hm
import pickle
import seaborn as sns
snscolors=sns.color_palette()
snscolors=np.tile(np.array(snscolors),(4,1))
import os
home = os.path.expanduser("~")
resultdir = '../savedresults/'
# immport combined quantiles
outfile = resultdir+'combinedquantiles.pkl'
[all_speedquantiles10, all_speedquantiles20, all_ndistquantiles, all_bdistquantiles] = pickle.load(open(outfile,'rb'))

numfish=6
pxpercm = 4.02361434 * 10  # from tracker

# used by other functions
def plot_thetaparams(a,thetaparams,ylim=0.8,showdots=True,label=''):
    if showdots:
        style = '-o'
    else:
        style = '-'
    a.plot(hm.theta_edge[0:-1]+hm.dtheta/2,thetaparams,style,label=label)
    a.set_xlim([-np.pi,np.pi])
    a.set_ylim([-0.8,0.8])
    l=1
    for div in [1,2,2/3]:
        a.plot([np.pi/div,np.pi/div],[-l,l],'k--',alpha=0.2)
        a.plot([-np.pi/div,-np.pi/div],[-l,l],'k--',alpha=0.2)
    a.plot([0,0],[-l,l],'k--',alpha=0.2)    


# hierarchical model plotting
def plotTfit(plotfn,model,params,numtrials,alphatrial=0.2,sizetrial=0.7,color='k',xspeed=all_speedquantiles20,xdist=all_ndistquantiles):
    whichone=0
    ax = plotfn(model,params[whichone*model.numparams:(whichone+1)*model.numparams],sizescale=1,alphaval=1,color=color,xspeed=xspeed,xdist=xdist)    

    for whichone in range(1,numtrials+1):
        plotfn(model,params[whichone*model.numparams:(whichone+1)*model.numparams],sizescale=sizetrial,alphaval=alphatrial, ax=ax,color=color,xspeed=xspeed,xdist=xdist)    
    return ax



# fits
def plotfit(model,params,xspeed=all_speedquantiles20,xdist=all_ndistquantiles,ax='',alphaval=1,sizescale=1,color='k'):
    bsr, bphi, sphi_i, sphi_j, sphi_k, stheta_i, stheta_j, stheta_k, phi_att, phi_ali, head_att, head_ali, attract, align, topo = model.parseparams(params)    

    x_speed_i = xspeed[:-1:2] if model.sibins==10 else xspeed[:-1]
    x_speed_j = xspeed[:-1:2] if model.sjbins==10 else xspeed[:-1]    
    x_dist_j = xdist[:-1:2] if model.rnumbins==10 else xdist[:-1]   
        
    
    f,ax = plt.subplots(1,6)
    f.set_size_inches(25,4)
    a=ax[0]
    a.plot(x_speed_i,sphi_i,'-o',label='$F_{attract}^{(1)}(s_i)$')
    a.plot(x_speed_i,stheta_i,'-o',label='$F_{align}^{(1)}(s_i)$')
    a.set_title('F(Focal speed)',fontsize=14)
    a.set_xlabel('Focal fish speed',fontsize=14)    
    a=ax[1]
    a.plot(x_speed_j,sphi_j,'-o',label='$F_{attract}^{(2)}(s_j)$')
    a.plot(x_speed_j,stheta_j,'-o',label='$F_{align}^{(2)}(s_j)$')
    a.set_title('F(Neighbor speed)',fontsize=14)
    a.set_xlabel('Neighbor speed',fontsize=14)    
    a=ax[2]
    a.plot(x_dist_j,sphi_k,'-o',label='$F_{attract}^{(3)}(r_j)$')
    a.plot(x_dist_j,stheta_k,'-o',label='$F_{align}^{(3)}(r_j)$')
    a.set_title('F(Neighbor distance)',fontsize=14)
    a.set_xlabel('Neighbor distance',fontsize=14)    
    [a.legend() for a in ax[0:3]]
    [a.set_ylim([0,2.5]) for a in ax[0:3]]

    a=ax[3]
    plot_thetaparams(a,phi_att,label='$H_{attract}(\phi)$')
    plot_thetaparams(a,phi_ali,label='$H_{align}^{(2)}(\phi)$')
    a.set_title('Relative angular position',fontsize=14)
    a.set_xlabel('Relative angular position ($\phi$)',fontsize=14)    
    
    a=ax[4]
    plot_thetaparams(a,head_att,label='$H_{attract}^{(2)}(\\theta)$')    
    plot_thetaparams(a,head_ali,label='$H_{align}(\\theta)$')
    a.set_title('Relative orientation',fontsize=14)    
    a.set_xlabel('Relative orientation ($\\theta$)',fontsize=14)        

    a = ax[5]
    a.bar(np.arange(numfish-1),topo)
    a.plot([-1,5],np.mean(topo)*np.ones(2),'k--',label='Uniform weights value')
    a.set_xlim([-0.5,4.5])
    a.set_ylabel('Weight')
    a.set_xlabel('Neighbor number')
    a.set_title('Neighbor weights')
    
    [a.legend(fontsize=14) for a in ax]
    return ax


# fits
def plotfit_simple(model,params,ax='',alphaval=1,sizescale=1,color='k',xspeed=all_speedquantiles20,xdist=all_ndistquantiles):
    bsr, bphi, sphi_i, sphi_j, sphi_k, stheta_i, stheta_j, stheta_k, phi_att, phi_ali, head_att, head_ali, attract, align, topo = model.parseparams(params)    

    x_speed_i = xspeed[:-1:2] if model.sibins==10 else xspeed[:-1]
    x_speed_j = xspeed[:-1:2] if model.sjbins==10 else xspeed[:-1]    
    x_dist_j = xdist[:-1:2] if model.rnumbins==10 else xdist[:-1]   
    
    makenewax = (len(ax)==0)
    if makenewax:
        f,ax = plt.subplots(1,4,gridspec_kw={'width_ratios': [0.2, 1, 1, 1]})
        f.set_size_inches(12,3)
    
    a=ax[0]
    a.scatter([1 + (0 if alphaval>=0.95 else 0.1*(np.random.rand()-0.5))],[attract], color='k', alpha=alphaval, s=30*sizescale**2)
    # a.set_ylim(bottom=0)
    a.set_xticks([1])
    a.set_xticklabels(['Magnitude\nratio'],fontsize=14)
    a.set_title('$\\alpha$',fontsize=14)    

    lw = 1.5  # base size for line width
    s = 20 # base size for points
    z = 10 if (alphaval>=0.95) else 1
    showdots = (alphaval==1)
    
    
    a=ax[1]
    x,y = x_speed_i,sphi_i/stheta_i
    a.plot(x,y,c='k', alpha=alphaval, linewidth=sizescale*lw, zorder=z)
    a.scatter(x,y,c='k', alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None
    a.set_title('$F(s_i)$',fontsize=16)
    a.set_xlabel('Focal fish speed (cm/sec)',fontsize=14)    

    a=ax[2]
    x,y = x_dist_j,sphi_k
    a.plot(x,y,label='$G_{att}(r_j)$', c='k', alpha=alphaval, linewidth=sizescale*lw, zorder=z)
    a.scatter(x,y,c=['k'], alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None
    x,y = x_dist_j,stheta_k
    a.plot(x,y,label='$G_{ali}(r_j)$', c='k',linestyle='--', alpha=alphaval, linewidth=sizescale*lw, zorder=z)
    a.scatter(x,y,c=['k'], alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None   
    a.set_title('$G_*(r_j)$',fontsize=16)
    a.set_xlabel('Neighbor distance (cm)',fontsize=14)    

    a=ax[3]
    x,y = hm.theta_edge[0:-1]+hm.dtheta/2, phi_att
    a.plot(x,y,label='$H_{att}(\phi)$', c='k', alpha=alphaval, linewidth=sizescale*lw, zorder=z)
#     a.scatter(x,y,c=[snscolors[0]], alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None   
    y = head_ali
    a.plot(x,y,label='$H_{ali}(\\theta)$', c='k',linestyle='--', alpha=alphaval, linewidth=sizescale*lw, zorder=z)
#     a.scatter(x,y,c=[snscolors[1]], alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None
    a.set_title('$H_*(\cdot)$',fontsize=16)
    a.set_xlabel('$\phi$ or $\\theta$ (radians)',fontsize=14)        

    if makenewax:
        ax[0].set_ylim([0,5])    
        ax[1].set_ylim([0,2.5])
        ax[2].set_ylim([-1,3])    
        ax[3].set_ylim([-0.9,0.9])
        ax[3].axhline(0,c='k',linewidth=1)
        [a.axhline(0,c='k',linewidth=1) for a in ax]
        [a.axhline(1,c='k',linestyle='--',linewidth=1) for a in ax]
        [a.legend(fontsize=12) for a in ax[2:4]]
        # add extra things for the angular plot
        a=ax[3]
        a.set_xlim([-np.pi,np.pi])
        l=1
        for div in [1,2,2/3]:
            a.plot([np.pi/div,np.pi/div],[-l,l],'k--',alpha=0.2)
            a.plot([-np.pi/div,-np.pi/div],[-l,l],'k--',alpha=0.2)
        a.plot([0,0],[-l,l],'k--',alpha=0.2)           
    return ax



# fits
def plotfit_noangle(model,params,ax='',alphaval=1,sizescale=1,color='k',xspeed=all_speedquantiles20,xdist=all_ndistquantiles):
    bsr, bphi, sphi_i, sphi_j, sphi_k, stheta_i, stheta_j, stheta_k, phi_att, phi_ali, head_att, head_ali, attract, align, topo = model.parseparams(params)    

    x_speed_i = xspeed[:-1:2] if model.sibins==10 else xspeed[:-1]
    x_speed_j = xspeed[:-1:2] if model.sjbins==10 else xspeed[:-1]    
    x_dist_j = xdist[:-1:2] if model.rnumbins==10 else xdist[:-1]    
    
    makenewax = (len(ax)==0)
    if makenewax:
        f,ax = plt.subplots(1,4,gridspec_kw={'width_ratios': [0.2, 1, 1, 1]})
        f.set_size_inches(12,3)
    
    a=ax[0]
    a.scatter([1 + (0 if alphaval>=0.95 else 0.1*(np.random.rand()-0.5))],[attract], color=color, alpha=alphaval, s=30*sizescale**2)
    # a.set_ylim(bottom=0)
    a.set_xticks([1])
    a.set_xticklabels(['Magnitude\nratio'],fontsize=12)
    a.set_title('$\\alpha$',fontsize=14)    

    lw = 1.5  # base size for line width
    s = 20 # base size for points
    z = 10 if (alphaval>=0.95) else 1
    showdots = (alphaval==1)  

    
    a=ax[1]
    x,y = x_speed_i,sphi_i/stheta_i
    a.plot(x,y,label='$F_{att}(s_i)/F_{ali}(s_i)$',c=color, alpha=alphaval, linewidth=sizescale*lw, zorder=z)
    a.scatter(x,y,c=color, alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None
    a.set_title('$F(s_i)$',fontsize=14)
    a.set_xlabel('Focal fish speed',fontsize=14)    

    a=ax[2]
    x,y = x_dist_j,sphi_k
    a.plot(x,y,label='$G_{attract}(r_j)$', c=color, alpha=alphaval, linewidth=sizescale*lw, zorder=z)
    a.scatter(x,y,c=[color], alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None
    a.set_title('$G_A(r_j)$',fontsize=14)
    a.set_xlabel('Neighbor distance',fontsize=14)    
    
    a=ax[3]
    x,y = x_dist_j,stheta_k
    a.plot(x,y,label='$G_{align}(r_j)$', c=color, alpha=alphaval, linewidth=sizescale*lw, zorder=z)
    a.scatter(x,y,c=[color], alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None   
    a.set_title('$G_O(r_j)$',fontsize=14)
    a.set_xlabel('Neighbor distance',fontsize=14)      

    if makenewax:
        ax[0].set_ylim([0,3])    
        ax[1].set_ylim([0,2.5])
        ax[2].set_ylim([-0.5,3])    
        ax[3].set_ylim([-0.5,3])            
        [a.axhline(1,c='k',linewidth=1) for a in ax]
    return ax

def plotfit_justdist(model,params,ax='',alphaval=1,sizescale=1,color='k',xspeed=all_speedquantiles20,xdist=all_ndistquantiles):
    bsr, bphi, sphi_i, sphi_j, sphi_k, stheta_i, stheta_j, stheta_k, phi_att, phi_ali, head_att, head_ali, attract, align, topo = model.parseparams(params)    

    x_speed_i = (xspeed[:-1:2] if model.sibins==10 else xspeed[:-1]) / pxpercm
    x_speed_j = (xspeed[:-1:2] if model.sjbins==10 else xspeed[:-1]) / pxpercm
    x_dist_j = (xdist[:-1:2] if model.rnumbins==10 else xdist[:-1]) / pxpercm   
    
    makenewax = (len(ax)==0)
    if makenewax:
        f,ax = plt.subplots(1,3,gridspec_kw={'width_ratios': [0.2, 1, 1]},sharey=False)
        f.set_size_inches(8.5,3)
        set_share_axes(ax[1:], sharex=True,sharey=True)    
        plt.subplots_adjust(wspace=0.2)
    
    a=ax[0]
    a.scatter([1 + (0 if alphaval>=0.95 else 0.1*(np.random.rand()-0.5))],[attract], color=color, alpha=alphaval, s=30*sizescale**2)
    # a.set_ylim(bottom=0)
    a.set_xticks([1])
    a.set_title('$\\alpha$',fontsize=16)    

    lw = 1.5  # base size for line width
    s = 20 # base size for points
    z = 10 if (alphaval>=0.95) else 1
    showdots = (alphaval==1)  

    a=ax[1]
    x,y = x_dist_j,sphi_k
    a.plot(x,y, c=color, alpha=alphaval, linewidth=sizescale*lw, zorder=z)
    a.scatter(x,y,c=[color], alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None
    a.set_title('$G_{att}(r_j)$',fontsize=16)
    a.set_xlabel('Neighbor distance (cm)',fontsize=14)    
    
    a=ax[2]
    x,y = x_dist_j,stheta_k
    a.plot(x,y, c=color, alpha=alphaval, linewidth=sizescale*lw, zorder=z)
    a.scatter(x,y,c=[color], alpha=alphaval, s=sizescale*s, zorder=z) if showdots else None   
    a.set_title('$G_{ali}(r_j)$',fontsize=16)
    a.set_xlabel('Neighbor distance (cm)',fontsize=14)      

    if makenewax:
        [a.set_ylim([-1,3.5]) for a in ax]
        [a.axhline(0,c='k',linewidth=1) for a in ax]
        [a.axhline(1,c='k',linestyle='--',linewidth=1) for a in ax]
        [a.tick_params(labelsize=12) for a in ax]
        ax[0].set_xticklabels(['Magnitude\nratio'],fontsize=14)   
        plt.setp(ax[2].get_yticklabels(), visible=False)
        [a.set_xlim([x_dist_j[0],x_dist_j[-1]]) for a in ax[1:]]
    return ax

def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)