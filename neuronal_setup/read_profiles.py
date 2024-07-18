"""Script to read profiles"""


import numpy as np
import mcdiff
from mcdiff.outreading import read_F_D_edges, read_Drad
from analyzeprofiles import *
from mcdiff.permeability.perm import *
from mcdiff.permeability.deff import *

#================
# INPUT
#================
#system = "popc"
#system = "mito"
#system = "HexdWat"
import sys
system = sys.argv[1]

#+++++++++++++++++++
lt = 10.
#lt = 20.   # !!!!!!!!!!!!!!!!!!!!!!
dofit = True    #!!!!!!!!!!!!!!!!!!!! choose! !!
#+++++++++++++++++++

# standard
doradial = True
letter = None

filename = set_filenames(system,lt,doradial=doradial,letter=letter,dofit=dofit)

F,D,edges = read_F_D_edges(filename)
if doradial:
    Drad,Edges = read_Drad(filename)
#================
nbins = len(F)
zpbc = edges[-1]-edges[0]
dz = (edges[-1]-edges[0])/(len(edges)-1.)  # angstrom
dt = 1.
dtc = 1.  # ps
lmax = 50
dr = 0.5  # angstrom
nrad = 100
redges = np.arange(0.,nrad)*dr
#================


