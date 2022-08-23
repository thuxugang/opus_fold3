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

def _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b, params,
         residuesData, geosData, rama_cons, quat_transl_variables, 
         fixed_backbone, torsion_variables, epochs, lr, resgap, save=False):
    
    print ("LR: " + str(lr) + "; Epochs: " + str(epochs) + "; ResGap: " + str(resgap))

    if save:
        best_potential = 1e6
        best_atoms_matrix = None

    residuesData = RebuildStructure.save_index_in_matrix(residuesData, Residues.NUM_MAX_ATOMS*(len_a+len_b))

    mctrr_cons = TrrPotential.readTrrCons(params["mctrr_cons_path"], resgap=resgap)
    MCTrr_matrix = TrrPotential.init_MCTrr_matrix(residuesData, mctrr_cons, len_a=len_a)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=epochs,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            if not fixed_backbone:
                for torsions in torsion_variables:
                    if torsions > 180:
                        torsions.assign_sub(360)
                    elif torsions < -180:
                        torsions.assign_add(360) 
            
            if not fixed_backbone:
                residuesData_a = RebuildStructure.save_index_in_matrix(residuesData[:len_a], Residues.NUM_MAX_ATOMS*len_a)
                atoms_matrix_init_a = RebuildStructure.rebuild_main_chain(torsion_variables[:len_a*3], geosData[:len_a], residuesData_a)

                residuesData_b = RebuildStructure.save_index_in_matrix(residuesData[len_a:], Residues.NUM_MAX_ATOMS*len_b)
                atoms_matrix_init_b = RebuildStructure.rebuild_main_chain(torsion_variables[len_a*3:], geosData[len_a:], residuesData_b)
                
                assert len(atoms_matrix_init_a) == Residues.NUM_MAX_ATOMS*len_a
                assert len(atoms_matrix_init_b) == Residues.NUM_MAX_ATOMS*len_b
                
                atoms_matrix_init_a = tf.cast(atoms_matrix_init_a, tf.float32)
                atoms_matrix_init_b = tf.cast(atoms_matrix_init_b, tf.float32)
                
            atoms_matrix_b = RebuildStructure.transform_with_quat_transl(quat_transl_variables, atoms_matrix_init_b)
            assert atoms_matrix_init_b.shape[0] == len(atoms_matrix_b)
            
            atoms_matrix = tf.concat([atoms_matrix_init_a, atoms_matrix_b], axis=0)
            
            loss = Potentials.get_dock_potentials(MCTrr_matrix, atoms_matrix, torsion_variables, rama_cons, fixed_backbone)
                
        if fixed_backbone:
            variables = quat_transl_variables
        else:
            variables = quat_transl_variables + torsion_variables
        
        gradients = tape.gradient(loss, variables, unconnected_gradients="zero")
        optimizer.apply_gradients(zip(gradients, variables))

        if save and epoch >= 100:
            if loss.numpy() < best_potential:
                best_atoms_matrix = atoms_matrix
                best_potential = loss.numpy()
    
    if save:
        return best_atoms_matrix
    else:
        return quat_transl_variables, torsion_variables

def run_script(multi_iter):
    
    params = multi_iter
    
    if not os.path.exists(params["output_path"]):

        # ############################## initialization ##############################            
        
        name, fasta_a = Myio.readFasta(params["fasta_a_path"])
        name, fasta_b = Myio.readFasta(params["fasta_b_path"])
        fasta = fasta_a + fasta_b
        
        seq_len = len(fasta)
        len_a = len(fasta_a)
        len_b = len(fasta_b)
        
        atoms_matrix_len = seq_len*Residues.NUM_MAX_ATOMS
        atoms_matrix_len_a = len_a*Residues.NUM_MAX_ATOMS
        atoms_matrix_len_b = len_b*Residues.NUM_MAX_ATOMS
        
        print (name, len(fasta_a), len(fasta_b), len(fasta))
        
        if params["from_pdb"]:
            residuesData_a = Residues.getResidueDataFromPDB(params["pdb_a_path"], chains=params["chains_a"]) 
            geosData_a = RebuildStructure.getGeosData(residuesData_a, params["from_pdb"])
            init_torsions_a = RebuildStructure.initTorsionFromGeo(geosData_a)
            residuesData_a = RebuildStructure.save_index_in_matrix(residuesData_a, atoms_matrix_len_a)
            atoms_matrix_init_a = RebuildStructure.rebuild_main_chain(init_torsions_a, geosData_a, residuesData_a)
            
            residuesData_b = Residues.getResidueDataFromPDB(params["pdb_b_path"], chains=params["chains_b"]) 
            geosData_b = RebuildStructure.getGeosData(residuesData_b, params["from_pdb"])
            init_torsions_b = RebuildStructure.initTorsionFromGeo(geosData_b)
            residuesData_b = RebuildStructure.save_index_in_matrix(residuesData_b, atoms_matrix_len_b)
            atoms_matrix_init_b = RebuildStructure.rebuild_main_chain(init_torsions_b, geosData_b, residuesData_b)
        else:
            residuesData_a = Residues.getResidueDataFromSequence(fasta_a, chainid=params["chains_a"][0]) 
            geosData_a = RebuildStructure.getGeosData(residuesData_a)        
            init_torsions_a = Myio.randomTorsions(fasta_a)
            residuesData_a = RebuildStructure.save_index_in_matrix(residuesData_a, atoms_matrix_len_a)
            atoms_matrix_init_a = RebuildStructure.rebuild_main_chain(init_torsions_a, geosData_a, residuesData_a)
            
            residuesData_b = Residues.getResidueDataFromSequence(fasta_b, chainid=params["chains_b"][0]) 
            geosData_b = RebuildStructure.getGeosData(residuesData_b)        
            init_torsions_b = Myio.randomTorsions(fasta_b)
            residuesData_b = RebuildStructure.save_index_in_matrix(residuesData_b, atoms_matrix_len_b)
            atoms_matrix_init_b = RebuildStructure.rebuild_main_chain(init_torsions_b, geosData_b, residuesData_b)            
            
        residuesData = residuesData_a + residuesData_b
        geosData = geosData_a + geosData_b
        init_torsions = init_torsions_a + init_torsions_b
        atoms_matrix_init = atoms_matrix_init_a + atoms_matrix_init_b

        assert seq_len == len(fasta) == len(residuesData) == len(geosData) == len(init_torsions)/3
        
        atoms_matrix_init_a = np.array(atoms_matrix_init_a, dtype=np.float32)
        atoms_matrix_init_b = np.array(atoms_matrix_init_b, dtype=np.float32)
        atoms_matrix_init = np.array(atoms_matrix_init, dtype=np.float32)
        init_torsions = np.array(init_torsions, dtype=np.float32)

        rama_cons = Rama.getRamaCons(params["rama_cons"], fasta)
        
        quat_transl_variables = np.array([1,0,0,0,1,0,0,0,0]).astype(np.float32)
        quat_transl_variables = [tf.Variable(i) for i in quat_transl_variables]
        
        torsion_variables = [tf.Variable(i) for i in init_torsions]

        # ############################## initialization ##############################            

        # ############################## optimaztion ##############################  
        
        if params["from_pdb"]:
            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=True, torsion_variables=torsion_variables,
                                                            epochs=300, lr=0.5, resgap=None)
            
            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=True, torsion_variables=torsion_variables, 
                                                            epochs=300, lr=0.25, resgap=None)
    
            best_atoms_matrix = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                 params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                 fixed_backbone=params["fixed_backbone"], torsion_variables=torsion_variables,
                                 epochs=600, lr=0.1, resgap=None, save=True)
        else:
            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=False, torsion_variables=torsion_variables,
                                                            epochs=200, lr=0.5, resgap=2)

            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=False, torsion_variables=torsion_variables,
                                                            epochs=200, lr=0.5, resgap=3)
            
            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=False, torsion_variables=torsion_variables,
                                                            epochs=200, lr=0.5, resgap=5)

            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=False, torsion_variables=torsion_variables,
                                                            epochs=200, lr=0.5, resgap=9)

            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=False, torsion_variables=torsion_variables,
                                                            epochs=200, lr=0.5, resgap=16)
            
            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=False, torsion_variables=torsion_variables,
                                                            epochs=200, lr=0.5, resgap=24)

            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=False, torsion_variables=torsion_variables,
                                                            epochs=600, lr=0.5, resgap=None)

            quat_transl_variables, torsion_variables = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                                            params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                                            fixed_backbone=False, torsion_variables=torsion_variables,
                                                            epochs=600, lr=0.25, resgap=None)
            
            best_atoms_matrix = _run(atoms_matrix_init_a, len_a, atoms_matrix_init_b, len_b,
                                 params, residuesData, geosData, rama_cons, quat_transl_variables, 
                                 fixed_backbone=False, torsion_variables=torsion_variables,
                                 epochs=600, lr=0.1, resgap=None, save=True)            

        # ############################## optimaztion ##############################  
        
        residuesData = RebuildStructure.save_index_in_matrix(residuesData, atoms_matrix_len)
        Myio.outputPDB(residuesData, best_atoms_matrix, params["output_path"])

