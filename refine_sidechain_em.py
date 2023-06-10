# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:11:13 2016

@author: XuGang
"""

import os, re
from myclass import Residues, Myio
from buildprotein import RebuildStructure
from potential import Rama, Potentials
import time
import tensorflow as tf
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

APIX = 1.5
def _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData,
         init_torsions, init_rotamers, rama_cons, epochs, lr, rotamers_cons, iteration, find_shift, shift, save=False):

    print ("LR: " + str(lr) + "; Epochs: " + str(epochs) + "; find_shift: " + str(find_shift))

    exp_map, original_apix = Potentials.read_map(params["emd_path"], apix=APIX)

    if save:
        best_potential = 1e6
        best_atoms_matrix = None

    atoms_mask = RebuildStructure.make_atoms_mask(residuesData)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=epochs,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    atoms_matrix = atoms_matrix_init
    for epoch in range(epochs):
        iteration += 1
        with tf.GradientTape() as tape:
            start = time.time()

            for torsions in init_rotamers:
                if torsions > 180:
                    torsions.assign_sub(360)
                elif torsions < -180:
                    torsions.assign_add(360)

            atoms_matrix = RebuildStructure.rebuild_side_chain_parallel(init_rotamers, geosData, residuesData, atoms_matrix)
            assert len(atoms_matrix_init) == atoms_matrix.shape[0]

            loss = 0
            if find_shift:
                shift = Potentials.get_shift(atoms_matrix, atoms_mask, exp_map=exp_map, apix=APIX, original_apix=original_apix)
                find_shift = False
            loss = Potentials.get_EMpotentials(atoms_matrix, atoms_mask, exp_map=exp_map, apix=APIX, original_apix=original_apix, shift=shift)
            print ("Epoch:", iteration, loss.numpy(), shift)

        variables = init_rotamers

        gradients = tape.gradient(loss, variables, unconnected_gradients="zero")
        optimizer.apply_gradients(zip(gradients, variables))

        print (time.time() - start)

        if save and epoch >= 100:
            if loss.numpy() < best_potential:
                best_atoms_matrix = atoms_matrix
                best_potential = loss.numpy()

    if save:
        return iteration, shift, best_atoms_matrix
    else:
        return iteration, shift, init_torsions, init_rotamers

def run_script(multi_iter):

    params = multi_iter
    
    iteration = -1
    shift = 0.5
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
            init_rotamers, _ = Myio.readRotaNN(params["rotann_path_real"], start_idx=2, end_idx=6)
        else:
            print ("Modeling: " + name)
            print (len(fasta), fasta)

            residuesData = Residues.getResidueDataFromSequence(fasta)
            geosData = RebuildStructure.getGeosData(residuesData)
            init_torsions = Myio.randomTorsions(fasta)

        assert len(init_torsions) == 3*len(residuesData) == 3*seq_len

        init_torsions = np.array(init_torsions, dtype=np.float32)
        init_torsions = [tf.Variable(i) for i in init_torsions]

        atoms_matrix_len = seq_len*Residues.NUM_MAX_ATOMS
        residuesData = RebuildStructure.save_index_in_matrix(residuesData, atoms_matrix_len)
        atoms_matrix_init = RebuildStructure.rebuild_main_chain(init_torsions, geosData, residuesData, set_first=True)

        if params["from_rotann"]:
            init_rotamers_pred, _ = Myio.readRotaNN(params["rotann_path"], start_idx=2, end_idx=6)
            assert len(init_rotamers) == len(init_rotamers_pred)
            for i in range(len(init_rotamers)):
                if init_rotamers[i] == 181:
                    init_rotamers[i] = init_rotamers_pred[i]
        
        assert len(init_rotamers) == sum([i.num_dihedrals for i in residuesData])
        init_rotamers = np.array(init_rotamers, dtype=np.float32)
        rotamers_cons = np.array(init_rotamers, dtype=np.float32)
        init_rotamers = [tf.Variable(i) for i in init_rotamers]

        atoms_matrix_init, atoms_matrix_name = RebuildStructure.rebuild_side_chain(init_rotamers, geosData, residuesData, atoms_matrix_init)

        # ############################## initialization ##############################

        # ############################## optimaztion ##############################
        iteration, shift, init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData,
                                                              init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.5, 
                                                              rotamers_cons=rotamers_cons, iteration=iteration, find_shift=True, shift=shift)
        
        iteration, shift, init_torsions, init_rotamers = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData,
                                                              init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.25, 
                                                              rotamers_cons=rotamers_cons, iteration=iteration, find_shift=False, shift=shift)

        iteration, shift, best_atoms_matrix = _run(atoms_matrix_init, atoms_matrix_name, params, residuesData, geosData,
                                                   init_torsions, init_rotamers, rama_cons, epochs=600, lr=0.1, 
                                                   rotamers_cons=rotamers_cons, iteration=iteration, find_shift=False, shift=shift, save=True)
        
        # ############################## optimaztion ##############################
        Myio.outputPDBALL(residuesData, best_atoms_matrix, atoms_matrix_name, params["output_path"])

if __name__ == '__main__':

    filenames = ["pdb8eqj"]
    
    emb_names = ["emd_28538.mrc"]

    # side chains folding process with fixed backbone, and the results from PDB (or others) as initial x1x2x3x4
    # require backbone .pdb file / real dihedral .dihedral file / pred dihedral .mut file / em density .mrc file
    from_pdb = True
    from_rotann = True
    use_emd = True
    
    multi_iters = []
    for filename, emb_name in zip(filenames, emb_names):
        
        output_path = "./predictions/" + filename + "_fold3.pdb"

        fasta_path = os.path.join("./examples/em_refine", filename + ".fasta")
        rama_cons = Rama.readRama("./lib/ramachandran.txt")

        params = {}
        params["rama_cons"] = rama_cons
        params["fasta_path"] = fasta_path
        params["output_path"] = output_path

        params["from_pdb"] = from_pdb
        params["from_rotann"] = from_rotann
        params["use_emd"] = use_emd

        if params["from_pdb"]:
            params["pdb_path"] = os.path.join("./examples/em_refine", filename + ".pdb")
            params["rotann_path_real"] = os.path.join("./examples/em_refine", filename + ".dihedrals")
            print ("Read:", params["pdb_path"])
            print ("Read:", params["rotann_path_real"])

        if params["from_rotann"]:
            params["rotann_path"] = os.path.join("./examples/em_refine", filename + ".mut")
            print ("Read:", params["rotann_path"])

        if params["use_emd"]:
            params["emd_path"] = os.path.join("./examples/em_refine", emb_name)
            print ("Read:", params["emd_path"])

        multi_iters.append(params)

    for multi_iter in multi_iters:
        run_script(multi_iter)

