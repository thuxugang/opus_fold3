# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:11:13 2016

@author: XuGang
"""

import os
from scripts import script_dock_backbone
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

    model = 0
    
    if model == 0:
        # two proteins backbone docking process with fixed backbone
        # require protein_A & protein_B .pdb file / backbone trrosetta-like constrains file (between two proteins)
        from_pdb = True
        fixed_backbone = True
    elif model == 1:
        # two proteins backbone docking process with flexible backbone at last optimaziton round
        # require protein_A & protein_B .pdb file / backbone trrosetta-like constrains file (between two proteins)
        from_pdb = True
        fixed_backbone = False
    elif model == 2:
        # two proteins backbone docking process with flexible and randomly initialized backbone
        # require backbone trrosetta-like constrains file (between two proteins && inside each protein)
        # Note: two chains only (one for each participant)
        from_pdb = False
        fixed_backbone = False
        
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
        mctrr_cons_path = os.path.join("../cons/true_dock2", pdb_name_a + ".labels2.npz") # backbone trrosetta-like constrains file
        params["mctrr_cons_path"] = mctrr_cons_path  

        params["from_pdb"] = from_pdb
        params["fixed_backbone"] = fixed_backbone
        
        if params["from_pdb"]:
            pdb_a_path = os.path.join("../cons/native_dock2", pdb_name_a + ".pdb")
            params["pdb_a_path"] = pdb_a_path
            
            pdb_b_path = os.path.join("../cons/native_dock2", pdb_name_b + ".pdb")
            params["pdb_b_path"] = pdb_b_path
            
            print ("Read:", params["pdb_a_path"], params["pdb_b_path"])

        multi_iters.append(params)
    
    pool = multiprocessing.Pool(30)
    pool.map(script_dock_backbone.run_script, multi_iters)
    pool.close()
    pool.join()  




        

