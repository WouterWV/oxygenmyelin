"""Script to compare different methods to extract profiles:
calc flux
AG, April 9, 2013
AG, April 25, 2013
AG, Jan 11, 2016"""

import numpy as np

from outreading import read_F_D_edges, read_Drad
import matplotlib.pyplot as plt

##### UNITS #####
# F -- in kBT
# D -- in angstrom**2/ps
# edges -- in angstrom

# TODO
# rate*lagtime
# lagtime in ps
# rate in 1/dt
# so do *dt or /dt somewhere 


##### READ #####

def read_distfile(filename):
    D = []
    f = file(filename)
    for line in f:
        if not line.startswith("#"):
            words = line.split()
            D.append([float(words[0]),float(words[1]),float(words[2]),])
    f.close()
    D = np.array(D)
    print("shape(D)",D.shape)
    return D

def read_realfile(filename):
    F = []
    D = []
    f = file(filename)
    for line in f:
        if not line.startswith("#"):
            words = line.split()
            F.append(float(words[0]))
            D.append(float(words[1]))
    f.close()
    F = np.array(F)
    D = np.array(D)
    print("shape(D)",D.shape)
    return F-min(F),D

def set_filenames(system,lt,doradial=True,letter="",dofit=False):

# TODO letter

    if dofit:
        filename = "/home/an/diffusion/rescale-with-t0_membranes/fit.%s.select.dat"%(system)
        print("filename is ",filename)
    else:
        if system in ["popc","mito"]:
            if doradial:
                #filename = "../anal_rv_%s/%s/rad.ncos10-6.lmax50.run2/out.nbins100.%i.pbc.dat" %(system,system,lt)
                #filename = "../anal_rv_%s/%s/rad.ncos10-6-10.lmax50.run4/out.nbins100.%i.pbc.dat10" %(system,system,lt)
                filename = "../anal_rv_%s/%s/%s/rad.ncosf10d6.lmax50.run4/out.nbins100.%i.pbc.dat" %(system,system,letter,lt)
         # TODO
                filename = "../anal_rv_%s/%s/rad.ncosf10d6.lmax50.run4/out.nbins100.%i.pbc.dat" %(system,system,lt)
            else:
                filename = "../anal_rv_%s/%s/%s/out.ncosf10d6.run2/out.nbins100.%i.pbc.dat" %(system,system,letter,lt)
        elif system == "HexdWat":
            if doradial:
                #filename = "../anal_rv_%s/%s/rad.ncos10-6.lmax50.run2/out.nbins100.%i.pbc.dat" %("hexdwat",system,lt)
                filename = "../anal_rv_hexdwat2/%s/%s/rad.ncosf10d6.lmax50.run4/out.nbins100.%i.pbc.dat" %(system,letter,lt)
            else:
                filename = "../anal_rv_hexdwat/%s/%s/out.ncos10.run3/out.nbins100.%i.pbc.dat" %(system,letter,lt)
        else:
            raise ValueError("system  %s  is not recognized" %system)
    return filename


##########

def extend_profile(F,D,Drad,edges,mult):
    """Extend the profile by repeating it n=mult times"""
    F1 = np.array(F.tolist()*mult)
    D1 = np.array(D.tolist()*mult)
    Drad1 = np.array(Drad.tolist()*mult)
    dz = (edges[-1]-edges[0])/(len(edges)-1.)  # angstrom
    nK = mult*len(F)
    edges1 = np.arange(nK+1)*dz   # no longer symmetry about z=0
    return F1,D1,Drad1,edges1

def extend_profile_water(F,D,Drad,edges,extrawat):
    """Extend the profile by adding n=extrawat bins with properties of starting/ending bin"""
    # bin[0] and bin[-1] are usually the water layer, hence the name
    extraF    = [(F[-1]+F[0])*0.5 for i in range(extrawat)]
    extraD    = [D[-1] for i in range(extrawat)]
    extraDrad = [Drad[-1] for i in range(extrawat)]
    F1 = np.array(F.tolist()+extraF)
    D1 = np.array(D.tolist()+extraD)
    Drad1 = np.array(Drad.tolist()+extraDrad)
    dz = (edges[-1]-edges[0])/(len(edges)-1.)  # angstrom
    edges1 = np.arange(len(F)+1)*dz   # no longer symmetry about z=0
    #WOUTER: shouldn't it be F1??
    #dz = np.arange(len(F1)+1)*dz
    return F1,D1,Drad1,edges1

def extend_profile_water_LR(F,D,Drad,edges,extrawat):
    """Extend the profile by adding n=extrawat bins left and right of the membrane"""
    extraF = [(F[-1]+F[0])*0.5 for i in range(extrawat)]
    extraD = [D[-1] for i in range(extrawat)]
    extraDrad = [Drad[-1] for i in range(extrawat)]
    F1 = np.array(extraF+F.tolist()+extraF)
    D1 = np.array(extraD+D.tolist()+extraD)
    Drad1 = np.array(extraDrad+Drad.tolist()+extraDrad)
    dz = (edges[-1]-edges[0])/(len(edges)-1.)
    edges1 = np.arange(len(F1)+1)*dz
    return F1,D1,Drad1,edges1

def divide_bins_profile(F,D,Drad,edges,divide):
    """Adapt the bins without changing the profile"""
    nbins = len(F)
    nbins1 = len(F)*divide
    F1    = np.resize(F,(divide,len(F))).T.ravel()
    D1    = np.resize(D,(divide,len(D))).T.ravel()
    Drad1 = np.resize(Drad,(divide,len(Drad))).T.ravel()
    dz = (edges[-1]-edges[0])/(len(edges)-1.)  # angstrom
    dz1 = dz/float(divide)
    edges1 = np.arange(edges[0],edges[-1]+1.e-9,dz1)
    return F1,D1,Drad1,edges1


#################################
##### PLOT #####
#################################

#############

# maybe this fits in "plot propagator" ???? # TODO # used nowhere???
def use_rate_matrix_mfpt(F,D,dx,dt,figname="userate.png"):
    rate = calc_rate_matrix_mfpt_left(F,D,dx,dt,)
    #print "check quality",np.sum(rate,0)
    #plot_propagator_bin12(propagator,startbin,factor,edges,redges,figname,lagtime)
    initprob = np.zeros(len(rate),float)
    initprob[0]=1.
    plt.figure()
    import scipy
    lagtimes = [0.01,0.1,1.,10.,100.,1000.,10000.]
    for lagtime in lagtimes:
        prop = np.dot(scipy.linalg.expm2(rate*lagtime),initprob)
        plt.plot(prop)
    plt.legend(lagtimes)
    plt.title("prop at various lagtimes (in ps)")
    plt.xlabel("bins")
    plt.ylabel("propagator")
    plt.savefig(figname)

def plot_propagator_bin12(propagator,startbin,factor,edges,redges,figname,lagtime):
    """plot propagator"""
    plt.figure()
    print("starting at:",0.5*(edges[startbin]+edges[startbin+1]))
#    plt.contourf(redges,edges[:-1],np.sum(prop,2).transpose())
    CS = plt.contourf(redges,edges[:-1],propagator[:,:,startbin].transpose()*factor,) #levels=levels,extend='both')
    plt.xlabel("radial distance r [A]")
    plt.ylabel("z [A]")
    plt.xlim(0,30)
    plt.title("distribution after %i ps when started at z=%4.2f" %(lagtime,0.5*(edges[startbin]+edges[startbin+1])))

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('probability*%i'%factor)
    ## Add the contour line levels to the colorbar
    ##cbar.add_lines(CS2)

    plt.savefig(figname)
    plt.close()
    print("fig written...", figname)

def plot_propagator_bin12_ax(ax,propagator,startbin,factor,edges,redges,lagtime,levels=None):
    """plot propagator"""
    print("starting at:",0.5*(edges[startbin]+edges[startbin+1]))
    CS = ax.contourf(redges,edges[:-1],propagator[:,:,startbin].transpose()*factor,extend='both',levels=levels)
    plt.xlabel("radial distance r [A]")
    plt.ylabel("z [A]")
    plt.xlim(0,70)
    plt.title("%i ps" %lagtime)
    #ax.set_title("distribution after %i ps when started at z=%4.2f" %(lagtime,0.5*(edges[startbin]+edges[startbin+1])))
    #ax.text(1.,0.85,'%i ps'%lagtime,verticalalignment='bottom',transform=ax.transAxes,
    ax.text(0.75,0.85,'%i ps'%lagtime,verticalalignment='bottom',transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='blue', pad=10.0),
            fontsize=18)

    return CS

    # Make a colorbar for the ContourSet returned by the contourf call.
    #cbar = plt.colorbar(CS)
    #cbar.ax.set_ylabel('probability*%i'%factor)
    ## Add the contour line levels to the colorbar
    ##cbar.add_lines(CS2)

    #plt.savefig(figname)
    #plt.close()
    #print "fig written...", figname

