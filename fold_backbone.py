# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:11:13 2016

@author: XuGang
"""

import os
from scripts import script_fold_backbone
from potential import Rama
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':

    lists = []
    f = open('../list_cameo_hard61')
    for i in f.readlines():
        lists.append(i.strip())
    f.close()    
    
    model = 0
    
    if model == 0:
        # backbone folding process using the results from OPUS-TASS2 as initial phi, psi, omega
        # require predicted .tass2 file / backbone trrosetta-like constrains file
        from_pdb = False
        from_tass = True
    elif model == 1:
        # backbone folding process using the random values as initial phi, psi, omega
        # require backbone trrosetta-like constrains file
        from_pdb = False
        from_tass = False
    elif model == 2:
        # backbone folding process from the real values calculated from input PDB as initial phi, psi, omega
        # require backbone .pdb file / backbone trrosetta-like constrains file
        from_pdb = True
        from_tass = False
    
    multi_iters = []
    for filename in lists:

        output_path = "./predictions/" + filename + "_fold3.pdb"
        
        fasta_path = os.path.join("../cons/native", filename + ".fasta")
        rama_cons = Rama.readRama("./lib/ramachandran.txt")

        params = {}
        params["rama_cons"] = rama_cons
        params["fasta_path"] = fasta_path
        params["output_path"] = output_path  
        
        # backbone folding params
        mctrr_cons_path = os.path.join("../cons/true", filename + ".labels2.npz") # backbone trrosetta-like constrains file
        # mctrr_cons_path = os.path.join("../cons/pred", filename + ".contact.npz") # backbone trrosetta-like constrains file
        params["mctrr_cons_path"] = mctrr_cons_path        
      
        params["from_pdb"] = from_pdb
        params["from_tass"] = from_tass
        
        if params["from_pdb"]: 
            params["pdb_path"] = os.path.join("../cons/native", filename + ".pdb")
            print ("Read:", params["pdb_path"])
            
        if params["from_tass"]: 
            params["tass_path"] = os.path.join("../cons/pred", filename + ".tass2")
            print ("Read:", params["tass_path"])
            
        multi_iters.append(params)
    
    pool = multiprocessing.Pool(30)
    pool.map(script_fold_backbone.run_script, multi_iters)
    pool.close()
    pool.join()  





        

