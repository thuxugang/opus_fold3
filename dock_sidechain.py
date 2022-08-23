# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:11:13 2016

@author: XuGang
"""

import os
from scripts import script_dock_sidechain
from potential import Rama
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    
    """
    pdb7b26 B,A pdb7b26 C
    protein A, chain_A, protein B, chain B
    
    for simplicity, protein A [len_a]and protein B [len_b] are combined sequentially to
    construct one backbone trrosetta-like constrains file [len_a+len_b, len_a+len_b, 100]
    """
    
    lists = []
    f = open('../list_dock_cameo75')
    for i in f.readlines():
        lists.append(i.strip())
    f.close()    


    # For oligomer targets, fold_sidechain.py can be used for 
    # their side chains modeling process with fixed backbone,
    # just combine them and consider it as a single protein
    model = 2
    if model == 0:
        # two proteins side chains docking process with flexible backbone at last optimaziton round, and the results from OPUS-RontaNN2 (or others) as initial x1x2x3x4
        # require protein_A & protein_B .pdb file / predicted .rotann2 file / backbone & side chains trrosetta-like constrains file (between two proteins & inside each protein)
        from_pdb = True
        from_rotann = True
    elif model == 1:
        # two proteins side chains docking process with flexible backbone at last optimaziton round, and the random values as initial x1x2x3x4
        # require protein_A & protein_B .pdb file / backbone & side chains trrosetta-like constrains file (between two proteins & inside each protein)
        from_pdb = True
        from_rotann = False
    elif model == 2:
        # two proteins backbone docking process with flexible and randomly initialized backbone, and the random values as initial x1x2x3x4
        # require backbone & side chains trrosetta-like constrains file (between two proteins && inside each protein)
        # Note: two chains only
        from_pdb = False
        from_rotann = False
       
    multi_iters = []
    for content in lists:
        
        pdb_name_a, chains_a, pdb_name_b, chains_b = content.split()
        
        chains_a = chains_a.split(',')
        chains_b = chains_b.split(',')
        
        print (pdb_name_a, chains_a, pdb_name_b, chains_b)

        output_path = "./predictions/" + pdb_name_a + "_fold3.pdb"

        fasta_a_path = os.path.join("../cons/native_dock2", pdb_name_a + "_a.fasta")
        fasta_b_path = os.path.join("../cons/native_dock2", pdb_name_b + "_b.fasta")
        rama_cons = Rama.readRama("./lib/ramachandran.txt")

        params = {}
        params["fasta_a_path"] = fasta_a_path
        params["fasta_b_path"] = fasta_b_path
        params["chains_a"] = chains_a
        params["chains_b"] = chains_b
        
        params["rama_cons"] = rama_cons
        params["output_path"] = output_path  

        # backbone docking params
        # here, pdb_a and pdb_b use a combined cons file for demonstration
        params["mctrr_cons_path"] = os.path.join("../cons/true_dock2", pdb_name_a + ".labels2.npz")
        print ("Read:", params["mctrr_cons_path"])
            
        # side chains docking params
        # here, pdb_a and pdb_b use a combined cons file for demonstration
        scx1trr_cons_path = os.path.join("../cons/true_dock_sc2", pdb_name_a + ".x1_labels2.npz") # side chains trrosetta-like constrains file
        params["scx1trr_cons_path"] = scx1trr_cons_path  
        
        scx2trr_cons_path = None
        scx3trr_cons_path = None
        scx4trr_cons_path = None
        
        # ===optional for optimize x2, x3, x4===
        scx2trr_cons_path = os.path.join("../cons/true_dock_sc2", pdb_name_a + ".x2_labels2.npz") # side chains trrosetta-like constrains file
        params["scx2trr_cons_path"] = scx2trr_cons_path  

        scx3trr_cons_path = os.path.join("../cons/true_dock_sc2", pdb_name_a + ".x3_labels2.npz") # side chains trrosetta-like constrains file
        params["scx3trr_cons_path"] = scx3trr_cons_path  

        scx4trr_cons_path = os.path.join("../cons/true_dock_sc2", pdb_name_a + ".x4_labels2.npz") # side chains trrosetta-like constrains file
        params["scx4trr_cons_path"] = scx4trr_cons_path  
        # ===optional for optimize x2, x3, x4===

        params["from_pdb"] = from_pdb
        params["from_rotann"] = from_rotann
        
        if params["from_pdb"]: 
            params["pdb_a_path"] = os.path.join("../cons/native_dock2", pdb_name_a + ".pdb")
            params["pdb_b_path"] = os.path.join("../cons/native_dock2", pdb_name_b + ".pdb")
            print ("Read:", params["pdb_a_path"], params["pdb_b_path"])
             
        if params["from_rotann"]: 
            # here, pdb_a and pdb_b use a combined cons file for demonstration
            params["rotann_path"] = os.path.join("../cons/native_dock2", pdb_name_a + ".dihedrals")
            print ("Read:", params["rotann_path"])

        multi_iters.append(params)
    
    pool = multiprocessing.Pool(30)
    pool.map(script_dock_sidechain.run_script, multi_iters)
    pool.close()
    pool.join() 

