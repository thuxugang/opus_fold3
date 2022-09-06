# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:11:13 2016

@author: XuGang
"""

import os
from scripts import script_fold_sidechain
from potential import Rama
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':

    lists = []
    f = open('./examples/list_fold')
    for i in f.readlines():
        lists.append(i.strip())
    f.close()    
    
    model = 0
    
    if model == 0:
        # side chains folding process with fixed backbone, and the results from OPUS-RontaNN2 (or others) as initial x1x2x3x4
        # require backbone .pdb file / predicted .rotann2 file / side chains trrosetta-like constrains file
        from_pdb = True
        fixed_backbone = True
        from_rotann = True  
    elif model == 1:
        # side chains folding process with fixed backbone, and the random values as initial x1x2x3x4
        # require backbone .pdb file / side chains trrosetta-like constrains file
        from_pdb = True
        fixed_backbone = True
        from_rotann = False          
    elif model == 2:
        # side chains folding process with flexible backbone at last optimaziton round, and the random values as initial x1x2x3x4
        # require backbone .pdb file / backbone & side chains trrosetta-like constrains file
        from_pdb = True
        fixed_backbone = False
        from_rotann = False        
    elif model == 3:
        # conjugate backbone and side chains folding process using the random values as initial phi, psi, omega, x1x2x3x4
        # require backbone & side chains trrosetta-like constrains file
        from_pdb = False
        fixed_backbone = False
        from_rotann = False        
    
    multi_iters = []
    for filename in lists:
        
        output_path = "./predictions2/" + filename + "_fold3.pdb"
        
        fasta_path = os.path.join("./examples/fold", filename + ".fasta")
        rama_cons = Rama.readRama("./lib/ramachandran.txt")
        
        params = {}
        params["rama_cons"] = rama_cons
        params["fasta_path"] = fasta_path
        params["output_path"] = output_path  

        # side chains folding params
        scx1trr_cons_path = os.path.join("./examples/fold", filename + ".x1_labels.npz") # side chains trrosetta-like constrains file
        params["scx1trr_cons_path"] = scx1trr_cons_path  
        
        scx2trr_cons_path = None
        scx3trr_cons_path = None
        scx4trr_cons_path = None
        
        # ===optional for optimize x2, x3, x4===
        scx2trr_cons_path = os.path.join("./examples/fold", filename + ".x2_labels.npz") # side chains trrosetta-like constrains file
        params["scx2trr_cons_path"] = scx2trr_cons_path  

        scx3trr_cons_path = os.path.join("./examples/fold", filename + ".x3_labels.npz") # side chains trrosetta-like constrains file
        params["scx3trr_cons_path"] = scx3trr_cons_path  

        scx4trr_cons_path = os.path.join("./examples/fold", filename + ".x4_labels.npz") # side chains trrosetta-like constrains file
        params["scx4trr_cons_path"] = scx4trr_cons_path  
        # ===optional for optimize x2, x3, x4===
        
        params["from_pdb"] = from_pdb
        params["fixed_backbone"] = fixed_backbone
        params["from_rotann"] = from_rotann
        
        if params["from_pdb"]: 
            params["pdb_path"] = os.path.join("./examples/fold", filename + ".pdb")
            print ("Read:", params["pdb_path"])
             
        if params["from_rotann"]: 
            params["rotann_path"] = os.path.join("./examples/fold", filename + ".rotann2")
            print ("Read:", params["rotann_path"])

        if not params["fixed_backbone"]: 
            params["mctrr_cons_path"] = os.path.join("./examples/fold", filename + ".labels.npz")
            print ("Read:", params["mctrr_cons_path"])
            
        multi_iters.append(params)
    
    pool = multiprocessing.Pool(30)
    pool.map(script_fold_sidechain.run_script, multi_iters)
    pool.close()
    pool.join()  

