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

def _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
         init_torsions, init_rotamers, rama_cons, epochs, lr, resgap, 
         fixed_backbone=True, save=False):

    print ("LR: " + str(lr) + "; Epochs: " + str(epochs) + "; ResGap: " + str(resgap))

    if save:
        best_potential = 1e6
        best_atoms_matrix = None
    
    SCTrr_matrixs = []
    if params["scx1trr_cons_path"]:
        scx1trr_cons = TrrPotential.readTrrCons(params["scx1trr_cons_path"], resgap=resgap)
        SCX1Trr_matrix = TrrPotential.init_SCTrr_matrix(residuesData, scx1trr_cons, dihedral_index=0)
        SCTrr_matrixs.append(SCX1Trr_matrix)
    if params["scx2trr_cons_path"]:
        scx2trr_cons = TrrPotential.readTrrCons(params["scx2trr_cons_path"], resgap=resgap)
        SCX2Trr_matrix = TrrPotential.init_SCTrr_matrix(residuesData, scx2trr_cons, dihedral_index=1)
        SCTrr_matrixs.append(SCX2Trr_matrix)

    if params["scx3trr_cons_path"]:
        scx3trr_cons = TrrPotential.readTrrCons(params["scx3trr_cons_path"], resgap=resgap)
        SCX3Trr_matrix = TrrPotential.init_SCTrr_matrix(residuesData, scx3trr_cons, dihedral_index=2)
        SCTrr_matrixs.append(SCX3Trr_matrix)

    if params["scx4trr_cons_path"]:
        scx4trr_cons = TrrPotential.readTrrCons(params["scx4trr_cons_path"], resgap=resgap)
        SCX4Trr_matrix = TrrPotential.init_SCTrr_matrix(residuesData, scx4trr_cons, dihedral_index=3)
        SCTrr_matrixs.append(SCX4Trr_matrix)

    if not fixed_backbone:
        mctrr_cons = TrrPotential.readTrrCons(params["mctrr_cons_path"], resgap=resgap)
        MCTrr_matrix = TrrPotential.init_MCTrr_matrix(residuesData, mctrr_cons)
        
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=epochs,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    atoms_matrix = atoms_matrix_init
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            if not fixed_backbone:
                for torsions in init_torsions:
                    if torsions > 180:
                        torsions.assign_sub(360)
                    elif torsions < -180:
                        torsions.assign_add(360)
                    
            for torsions in init_rotamers:
                if torsions > 180:
                    torsions.assign_sub(360)
                elif torsions < -180:
                    torsions.assign_add(360)
            
            if not fixed_backbone:
                atoms_matrix = RebuildStructure.rebuild_main_chain(init_torsions, geosData, residuesData)
                
            atoms_matrix = RebuildStructure.rebuild_side_chain_parallel(init_rotamers, geosData, residuesData, atoms_matrix)
            assert len(atoms_matrix_init) == atoms_matrix.shape[0]
            
            loss = 0
            if not fixed_backbone:
                loss += Potentials.get_potentials(MCTrr_matrix, atoms_matrix, init_torsions, rama_cons)

            loss += Potentials.get_scpotentials(SCTrr_matrixs, atoms_matrix)

        if fixed_backbone:
            variables = init_rotamers
        else:
            variables = init_rotamers + init_torsions
            
        gradients = tape.gradient(loss, variables, unconnected_gradients="zero")
        optimizer.apply_gradients(zip(gradients, variables))

        if save and epoch >= 100:
            if loss.numpy() < best_potential:
                best_atoms_matrix = atoms_matrix
                best_potential = loss.numpy()
    
    if save:
        return best_atoms_matrix
    else:
        return init_torsions, init_rotamers

def run_script(multi_iter):
    
    params = multi_iter
    
    if not os.path.exists(params["output_path"]):

        # ############################## initialization ##############################            
        name, fasta = Myio.readFasta(params["fasta_path"])
        seq_len = len(fasta)
        
        rama_cons = Rama.getRamaCons(params["rama_cons"], fasta)
        
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
            init_torsions = Myio.randomTorsions(fasta)

        assert len(init_torsions) == 3*len(residuesData) == 3*seq_len
        
        init_torsions = np.array(init_torsions, dtype=np.float32)  
        init_torsions = [tf.Variable(i) for i in init_torsions]

        num_atoms = sum([i.num_side_chain_atoms for i in residuesData]) + 5*len(residuesData) # contain Gly CB
        num_atoms_real = sum([i.num_atoms for i in residuesData]) # exclude Gly CB

        atoms_matrix_len = seq_len*Residues.NUM_MAX_ATOMS
        residuesData = RebuildStructure.save_index_in_matrix(residuesData, atoms_matrix_len)
        atoms_matrix_init = RebuildStructure.rebuild_main_chain(init_torsions, geosData, residuesData)
        
        if params["from_rotann"]:
            init_rotamers, _ = Myio.readRotaNN(params["rotann_path"], start_idx=1, end_idx=5)
        else:
            num_rotamers = sum([i.num_dihedrals for i in residuesData])
            init_rotamers = [np.random.randint(-180, 180) for _ in range(num_rotamers)]
            
        assert len(init_rotamers) == sum([i.num_dihedrals for i in residuesData])
        init_rotamers = np.array(init_rotamers, dtype=np.float32)    
        init_rotamers = [tf.Variable(i) for i in init_rotamers]
        
        atoms_matrix_init, atoms_matrix_name = RebuildStructure.rebuild_side_chain(init_rotamers, geosData, residuesData, atoms_matrix_init)
        
        # ############################## initialization ##############################            

        # ############################## optimaztion ##############################            
        if params["from_pdb"]:
            if params["from_rotann"]:
                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=6)
                
                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=12)
                
                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=24)

                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.5, resgap=None)

                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.25, resgap=None)
                
                best_atoms_matrix = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                         init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.1, resgap=None, 
                                         fixed_backbone=params["fixed_backbone"], save=True)
            else:
                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=2)

                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=3)
                
                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=5)
        
                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=9)
        
                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=16)
                
                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=24)

                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.5, resgap=None)

                init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                    init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.25, resgap=None)
                
                best_atoms_matrix = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                          init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.1, resgap=None, 
                                          fixed_backbone=params["fixed_backbone"], save=True)                  
        else:
            init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=2,
                                                fixed_backbone=False)

            init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=3,
                                                fixed_backbone=False)
            
            init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=5,
                                                fixed_backbone=False)
    
            init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=9,
                                                fixed_backbone=False)
    
            init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=16,
                                                fixed_backbone=False)
            
            init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                init_torsions, init_rotamers, rama_cons, epochs=200, lr=0.5, resgap=24,
                                                fixed_backbone=False)

            init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.5, resgap=None,
                                                fixed_backbone=False)

            init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                                init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.25, resgap=None,
                                                fixed_backbone=False)
            
            best_atoms_matrix = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData, 
                                     init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.1, resgap=None, 
                                     fixed_backbone=False, save=True)  

        # ############################## optimaztion ##############################            

        Myio.outputPDBALL(residuesData, best_atoms_matrix, atoms_matrix_name, params["output_path"])
         
