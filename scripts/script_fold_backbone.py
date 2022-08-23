# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:11:13 2016

@author: XuGang
"""

import os
from myclass import Residues, Myio
from buildprotein import RebuildStructure
from potential import TrrPotential, Rama, Potentials
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def _run(atoms_matrix_len, params, residuesData, geosData, 
         init_torsions, rama_cons, epochs, lr, resgap, save=False):
    
    print ("LR: " + str(lr) + "; Epochs: " + str(epochs) + "; ResGap: " + str(resgap))

    if save:
        best_potential = 1e6
        best_atoms_matrix = None

    mctrr_cons = TrrPotential.readTrrCons(params["mctrr_cons_path"], resgap=resgap)
    MCTrr_matrix = TrrPotential.init_MCTrr_matrix(residuesData, mctrr_cons)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=epochs,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            for torsions in init_torsions:
                if torsions > 180:
                    torsions.assign_sub(360)
                elif torsions < -180:
                    torsions.assign_add(360) 
            atoms_matrix = RebuildStructure.rebuild_main_chain(init_torsions, geosData, residuesData)
            assert atoms_matrix_len == len(atoms_matrix)
            
            loss = Potentials.get_potentials(MCTrr_matrix, atoms_matrix, init_torsions, rama_cons)

        gradients = tape.gradient(loss, init_torsions, unconnected_gradients="zero")
        optimizer.apply_gradients(zip(gradients, init_torsions))

        if save and epoch >= 100:
            if loss.numpy() < best_potential:
                best_atoms_matrix = atoms_matrix
                best_potential = loss.numpy()
    
    if save:
        return best_atoms_matrix
    else:
        return init_torsions

def run_script(multi_iter):
    
    params = multi_iter
    
    if not os.path.exists(params["output_path"]):


        # ############################## initialization ##############################            
        
        name, fasta = Myio.readFasta(params["fasta_path"])
        seq_len = len(fasta)
        
        rama_cons = Rama.getRamaCons(params["rama_cons"], fasta)
        
        if params["from_tass"]:
            init_torsions = Myio.readTASS(params["tass_path"], phi_idx=3, psi_idx=4) 
        else:
            init_torsions = Myio.randomTorsions(fasta)
            
        if params["from_pdb"]:
            print ("Modeling with pdb: " + params["pdb_path"])
            residuesData = Residues.getResidueDataFromPDB(params["pdb_path"]) 
            geosData = RebuildStructure.getGeosData(residuesData, params["from_pdb"])
            init_torsions = RebuildStructure.initTorsionFromGeo(geosData)
            
        else:
            print ("Modeling: " + name)
            print (len(fasta), fasta)
            
            residuesData = Residues.getResidueDataFromSequence(fasta) 
            geosData = RebuildStructure.getGeosData(residuesData)

        assert len(init_torsions) == 3*len(residuesData) == 3*seq_len
        init_torsions = np.array(init_torsions, dtype=np.float32)  
        init_torsions = [tf.Variable(i) for i in init_torsions]

        atoms_matrix_len = seq_len*Residues.NUM_MAX_ATOMS
        residuesData = RebuildStructure.save_index_in_matrix(residuesData, atoms_matrix_len)
        
        # ############################## initialization ##############################            

        # ############################## optimaztion ##############################            
        if params["from_pdb"] or params["from_tass"]:
            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=200, lr=0.5, resgap=6)
            
            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=200, lr=0.5, resgap=12)
            
            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=200, lr=0.5, resgap=24)

            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=600, lr=0.5, resgap=None)

            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=600, lr=0.25, resgap=None)

            best_atoms_matrix = _run(atoms_matrix_len, params, residuesData, geosData, 
                                     init_torsions, rama_cons, epochs=600, lr=0.1, resgap=None, save=True)
        else:
            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=200, lr=0.5, resgap=2)
            
            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=200, lr=0.5, resgap=3)
    
            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=200, lr=0.5, resgap=5)
    
            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=200, lr=0.5, resgap=9)
    
            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=200, lr=0.5, resgap=16)
            
            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=200, lr=0.5, resgap=24)

            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=600, lr=0.5, resgap=None)

            init_torsions = _run(atoms_matrix_len, params, residuesData, geosData, 
                                init_torsions, rama_cons, epochs=600, lr=0.25, resgap=None)
            
            best_atoms_matrix = _run(atoms_matrix_len, params, residuesData, geosData, 
                                     init_torsions, rama_cons, epochs=600, lr=0.1, resgap=None, save=True)       

        # ############################## optimaztion ##############################            
        
        assert len(best_atoms_matrix) == atoms_matrix_len
        Myio.outputPDB(residuesData, best_atoms_matrix, params["output_path"])
                

        

