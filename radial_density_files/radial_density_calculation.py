# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:33:58 2021

@author: jonat
"""

import pandas as pd
import numpy as np
import skimage.io
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt

ch = 640
pos = 32
min_params = 'lf30.0_wf0.0_sf0.0_dr0'

pnts_fname = "decoded/points_ch_{ch}_pos_{pos}_".format(ch=ch, pos=pos)+min_params+".csv"

ps = pd.read_csv(pnts_fname)

nd = ps.loc[ps.decoded==0]
off_target = ps.loc[ps.decoded > 591]
dcd = ps.loc[np.logical_and(ps.decoded < 592, ps.decoded != 0)]


rs = np.arange(0,5,0.04)

all_ps_tree = KDTree(ps.loc[:,['x','y']])

nd_ps_tree = KDTree(nd.loc[:,['x','y']])
offtgt_tree = KDTree(off_target.loc[:,['x','y']])
ontgt_tree = KDTree(dcd.loc[:, ['x','y']])


all_ps_nbs = all_ps_tree.count_neighbors(all_ps_tree, rs)
nd_ps_nbs = nd_ps_tree.count_neighbors(nd_ps_tree, rs)
offtgt_nbs = offtgt_tree.count_neighbors(offtgt_tree, rs)
ontgt_nbs = ontgt_tree.count_neighbors(ontgt_tree, rs)

nself_ps = all_ps_nbs - len(ps)
nself_nd_ps = nd_ps_nbs - len(nd) 
nself_ontgt = ontgt_nbs - len(dcd)
nself_offtgt = offtgt_nbs - len(off_target)


def radial_density(rs, vec):
    return (vec[1:] - vec[:-1])/(2*np.pi*rs[1:]*(rs[1])-rs[0])

plt.plot(rs[1:51], radial_density(rs, nself_ps)[:50]/len(ps), label="all")
plt.plot(rs[1:51], radial_density(rs, nself_nd_ps)[:50]/len(nd), label="not decoded")
plt.plot(rs[1:51], radial_density(rs, nself_ontgt)[:50]/len(dcd), label="on target")
plt.plot(rs[1:51], radial_density(rs, nself_offtgt)[:50]/len(off_target), label="off target")
plt.xlabel("Radial distance from Dot Center ( Pixels)")
plt.ylabel("Average Dot density Across All Hybs (Dots/Pixel)")
plt.legend()
plt.show()
