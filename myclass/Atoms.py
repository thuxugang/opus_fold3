# -*- coding: utf-8 -*-
"""
Created on Sat May 30 07:14:18 2015

@author: XuGang
"""

from myclass import Residues

class Atom:
    def __init__(self, atomid, name1, resname, resid, position, chainid="A", occ=1.0, bfactor=100.0, name2=""):
        self.atomid = atomid
        self.name1 = name1
        self.resname = Residues.singleResname(resname)
        self.resid = resid
        self.position = position
        self.chainid = chainid
        self.occ = occ
        self.bfactor = bfactor
        
        if name2 == "":
            self.name2 = name1[0]
        else:
            self.name2 = name2
            
        if self.name1 in ['N','CA','C','O','CB']:
            self.ismainchain = True
        else:
            self.ismainchain = False 
