# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:15:25 2016

@author: XuGang
"""
import numpy as np
import tensorflow as tf
from buildprotein import Geometry
from buildprotein import PeptideBuilder
from myclass import Residues

def getParams(residue_before, residue, residue_after, geo):

    if(residue_before == None):
        geo.phi = -60
        geo.omega = 180
    else:
        geo.phi = PeptideBuilder.get_np_dihedral(residue_before.atoms["C"].position, 
                                                 residue.atoms["N"].position, 
                                                 residue.atoms["CA"].position, 
                                                 residue.atoms["C"].position)

        geo.omega = PeptideBuilder.get_np_dihedral(residue_before.atoms["CA"].position, 
                                                   residue_before.atoms["C"].position, 
                                                   residue.atoms["N"].position, 
                                                   residue.atoms["CA"].position)
        
        geo.peptide_bond = PeptideBuilder.get_np_distance(residue_before.atoms["C"].position, residue.atoms["N"].position)
        
        geo.CA_C_N_angle = PeptideBuilder.get_np_angle(residue_before.atoms["CA"].position, 
                                                       residue_before.atoms["C"].position, 
                                                       residue.atoms["N"].position)

        geo.C_N_CA_angle = PeptideBuilder.get_np_angle(residue_before.atoms["C"].position, 
                                                       residue.atoms["N"].position, 
                                                       residue.atoms["CA"].position)
        
    geo.CA_N_length = PeptideBuilder.get_np_distance(residue.atoms["N"].position, residue.atoms["CA"].position)
    
    geo.CA_C_length = PeptideBuilder.get_np_distance(residue.atoms["CA"].position, residue.atoms["C"].position)

    geo.N_CA_C_angle = PeptideBuilder.get_np_angle(residue.atoms["N"].position, 
                                                   residue.atoms["CA"].position,
                                                   residue.atoms["C"].position) 

    if(residue_after == None):
        geo.psi_im1 = 60
    else:
        geo.psi_im1 = PeptideBuilder.get_np_dihedral(residue.atoms["N"].position, 
                                                     residue.atoms["CA"].position, 
                                                     residue.atoms["C"].position, 
                                                     residue_after.atoms["N"].position)
    
    geo.C_O_length = PeptideBuilder.get_np_distance(residue.atoms["C"].position, residue.atoms["O"].position)
    
    geo.CA_C_O_angle = PeptideBuilder.get_np_angle(residue.atoms["CA"].position, 
                                                   residue.atoms["C"].position,
                                                   residue.atoms["O"].position) 
    
    geo.N_CA_C_O_diangle = PeptideBuilder.get_np_dihedral(residue.atoms["N"].position, 
                                                          residue.atoms["CA"].position, 
                                                          residue.atoms["C"].position, 
                                                          residue.atoms["O"].position)
    
    if geo.residue_name != 'G':
        try:
            geo.CA_CB_length = PeptideBuilder.get_np_distance(residue.atoms["CA"].position, residue.atoms["CB"].position)
    
            geo.C_CA_CB_angle = PeptideBuilder.get_np_angle(residue.atoms["C"].position,
                                                            residue.atoms["CA"].position,
                                                            residue.atoms["CB"].position)  
            
            geo.N_C_CA_CB_diangle = PeptideBuilder.get_np_dihedral(residue.atoms["C"].position, 
                                                                   residue.atoms["N"].position, 
                                                                   residue.atoms["CA"].position, 
                                                                   residue.atoms["CB"].position)
        except:
            print (residue.chainid, residue.resid, residue.resname, "no CB atom...")

    return geo

def getGeoFromPDB(geosData, residuesData):

    for seq, (geo, residue) in enumerate(zip(geosData, residuesData)):
        
        if(seq == 0):
            geosData[seq] = getParams(None, residuesData[seq], residuesData[seq+1], geo)
        elif(seq == len(residuesData)-1):
            geosData[seq] = getParams(residuesData[seq-1], residuesData[seq], None, geo)
        else:
            geosData[seq] = getParams(residuesData[seq-1], residuesData[seq],residuesData[seq+1], geo)
        
    return geosData 

def getGeosData(residuesData, from_pdb=False):
    
    geosData = []
    for residue in residuesData:
        geo = Geometry.geometry(residue.resname)
        geosData.append(geo)

    if from_pdb:
        assert len(geosData) == len(residuesData)
        geosData = getGeoFromPDB(geosData, residuesData)     
        
    return geosData   

def initTorsionFromGeo(geosData):
    
    init_torsions = []
    
    for geo in geosData:
        init_torsions.extend([geo.phi, geo.psi_im1, geo.omega])

    return init_torsions

def rebuild_main_chain(torsions, geosData, residuesData, set_first=False):

    atoms_matrix = []
    count = 0
    assert len(residuesData) == len(geosData)
    length = len(residuesData)
    for idx in range(length):
        
        if idx == 0:
            atoms_matrix.extend(PeptideBuilder.get_mainchain(idx, None, None, residuesData[idx], geosData[idx], set_first=set_first))
                        
        else:
            # phi, psi, omega, psi_
            torsion = [torsions[count], torsions[count-2], torsions[count+2], torsions[count+1]]
            atoms_matrix.extend(PeptideBuilder.get_mainchain(idx, torsion, 
                 atoms_matrix[residuesData[idx-1].main_chain_atoms_matrixid["N"]:residuesData[idx-1].main_chain_atoms_matrixid["C"]+1], residuesData[idx], geosData[idx]))
        count += 3
    
        for _ in range(residuesData[idx].num_pad_main_atoms):
            atoms_matrix.extend([np.zeros(3).astype(np.float32)])
            
    assert count == len(torsions)
    
    return atoms_matrix

def rebuild_side_chain(rotamers, geosData, residuesData, atoms_matrix_old):
    
    atoms_matrix = []
    count = 0
    atoms_matrix_name = []
    assert len(residuesData) == len(geosData)
    length = len(residuesData)
    for idx, (residue, geo) in enumerate(zip(residuesData, geosData)):
        
        atoms_matrix.append(atoms_matrix_old[residue.main_chain_atoms_matrixid["N"]])
        atoms_matrix.append(atoms_matrix_old[residue.main_chain_atoms_matrixid["CA"]])
        atoms_matrix.append(atoms_matrix_old[residue.main_chain_atoms_matrixid["C"]])
        atoms_matrix.append(atoms_matrix_old[residue.main_chain_atoms_matrixid["O"]])
        atoms_matrix.append(atoms_matrix_old[residue.main_chain_atoms_matrixid["CB"]])
        
        atoms_matrix_name.append(["N", "CA", "C", "O", "CB"])
        
        if not residue.resname in ['G', 'A']:
            num_rotamers = residue.num_dihedrals
            
            rotamer = rotamers[count: count+num_rotamers]
            
            side_atoms, side_atoms_names = PeptideBuilder.get_sidechain(rotamer, residue, geo, atoms_matrix)
            
            atoms_matrix.extend(side_atoms)
            atoms_matrix_name[-1].extend(side_atoms_names)
            
            count += num_rotamers
 
        for _ in range(residue.num_pad_side_atoms):
            atoms_matrix.extend([np.zeros(3).astype(np.float32)])
            
    assert count == len(rotamers)
    assert len(atoms_matrix) == len(atoms_matrix_old)
    assert length == len(atoms_matrix_name)
    
    return atoms_matrix, atoms_matrix_name

def rebuild_side_chain_parallel(rotamers, geosData, residuesData, atoms_matrix_old):
    
    length = len(residuesData)
    atoms_matrix_old = tf.reshape(atoms_matrix_old, (length, Residues.NUM_MAX_ATOMS, 3))
    atoms_matrix = atoms_matrix_old[:, :5, :]
    
    # exclude N, CA, C, O, CB
    # side9_counters -- counters of three required atoms (Residues.NUM_MAX_SCATOMS*length, 3)
    # side9_params -- dist, angle, dihedral (Residues.NUM_MAX_SCATOMS*length, 3)
    side9_counters, side9_params = get_scfeatures(residuesData, geosData, rotamers)

    side9_counters = tf.reshape(side9_counters, (length, Residues.NUM_MAX_SCATOMS, 3))
    side9_params = tf.reshape(side9_params, (length, Residues.NUM_MAX_SCATOMS, 3))
    
    for i in range(Residues.NUM_MAX_SCATOMS):
        
        atoms_matrix_old = tf.reshape(atoms_matrix_old, (length*Residues.NUM_MAX_ATOMS, 3))
        
        ind = side9_counters[:,i,:]
        atoms = tf.stack([tf.gather(atoms_matrix_old, ind[i]) for i in range(length)]) # (L, 3, 3) 1. atoms
        params = side9_params[:,i,:] # (L, 3) 1. L, ang, di
        
        new_atom = PeptideBuilder.calculateCoordinates_batch(atoms[:,0,:], atoms[:,1,:], atoms[:,2,:],
                                                             params[:,0], params[:,1], params[:,2])
        new_atom = tf.expand_dims(new_atom, 1) # (L, 1, 3)
        
        atoms_matrix = tf.concat([atoms_matrix, new_atom], 1)            
        
        atoms_matrix_old = tf.reshape(atoms_matrix_old, (length, Residues.NUM_MAX_ATOMS, 3))
        
        atoms_matrix_old = tf.concat([atoms_matrix_old[:,:5+i,:], new_atom, atoms_matrix_old[:,6+i:,:]], 1)
        assert atoms_matrix_old.shape == (length, Residues.NUM_MAX_ATOMS, 3) 
    
    atoms_matrix = tf.reshape(atoms_matrix, (length*Residues.NUM_MAX_ATOMS, 3))
    assert atoms_matrix.shape == (length*Residues.NUM_MAX_ATOMS, 3)     
    
    return atoms_matrix

def make_atoms_mask(residuesData):
    
    length = len(residuesData)
    
    atoms_mask = []
    for residue in residuesData:
        
        for _ in range(residue.num_atoms):
            atoms_mask.append(1)
        if residue.resname == "G":
            num_pad = residue.num_pad_side_atoms + 1
        else:
            num_pad = residue.num_pad_side_atoms
        for _ in range(num_pad):
            atoms_mask.append(0)
            
    atoms_mask = tf.reshape(atoms_mask, (length*Residues.NUM_MAX_ATOMS,))
    assert atoms_mask.shape == (length*Residues.NUM_MAX_ATOMS,)     
    
    return atoms_mask

def sixd_to_rot(x, use_bias=False):
    # x (6,)
    if use_bias:
        bias1 = tf.convert_to_tensor([1., 0., 0.])
        bias2 = tf.convert_to_tensor([0., 1., 0.])
    else:
        bias1 = tf.convert_to_tensor([0., 0., 0.])
        bias2 = tf.convert_to_tensor([0., 0., 0.])

    c1 = tf.nn.l2_normalize(x[:3] + bias1, axis=-1)
    c3 = tf.linalg.cross(c1, x[3:] + bias2)
    c3 = tf.nn.l2_normalize(c3, axis=-1)
    c2 = tf.linalg.cross(c3, c1)
    x = tf.stack([c1, c2, c3], axis=-1)
    return x # (3, 3)


def transform_with_quat_transl(quat_transl, atoms_matrix):
    
    rots = sixd_to_rot(quat_transl[:6]) # (3, 3)
    transl = quat_transl[6:] # (3, )
    
    pred_backbone = tf.einsum("ij,kj->ik", rots, atoms_matrix) # (3, L) 0.coor 1.atoms
    transl_expand = tf.expand_dims(transl, axis=1) # (3, 1)
    pred_backbone += transl_expand # (3, L)
    
    pred_backbone = tf.transpose(pred_backbone, [1, 0]) # (L, 3)
    
    return pred_backbone

def extractmc(atomsData):
    
    # main_chain_only
    atomsData_main_chain_only = []
    for atom in atomsData:
        if atom.ismainchain:
            atomsData_main_chain_only.append(atom)
            
    return atomsData_main_chain_only

def save_index_in_matrix(residuesData, atoms_matrix_len):
    
    counter = 0
    for idx, residue in enumerate(residuesData):
        
        residue.main_chain_atoms_matrixid["N"] = counter
        counter += 1
        residue.main_chain_atoms_matrixid["CA"] = counter
        counter += 1
        residue.main_chain_atoms_matrixid["C"] = counter
        counter += 1
        residue.main_chain_atoms_matrixid["O"] = counter
        counter += 1
        residue.main_chain_atoms_matrixid["CB"] = counter
        counter += 1      
        
        if residue.resname == "S":
            residue.side_chain_atoms_matrixid["OG"] = counter
            counter += 1   
        elif residue.resname == "C":
            residue.side_chain_atoms_matrixid["SG"] = counter
            counter += 1             
        elif residue.resname == "V":
            residue.side_chain_atoms_matrixid["CG1"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CG2"] = counter
            counter += 1  
        elif residue.resname == "I":
            residue.side_chain_atoms_matrixid["CG1"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CG2"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CD1"] = counter
            counter += 1 
        elif residue.resname == "L":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CD1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CD2"] = counter
            counter += 1 
        elif residue.resname == "T":
            residue.side_chain_atoms_matrixid["OG1"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CG2"] = counter
            counter += 1 
        elif residue.resname == "R":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CD"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["NE"] = counter
            counter += 1  
            residue.side_chain_atoms_matrixid["CZ"] = counter
            counter += 1
            residue.side_chain_atoms_matrixid["NH1"] = counter
            counter += 1
            residue.side_chain_atoms_matrixid["NH2"] = counter
            counter += 1
        elif residue.resname == "K":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CD"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CE"] = counter
            counter += 1  
            residue.side_chain_atoms_matrixid["NZ"] = counter
            counter += 1
        elif residue.resname == "D":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["OD1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["OD2"] = counter
            counter += 1  
        elif residue.resname == "N":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["OD1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["ND2"] = counter
            counter += 1
        elif residue.resname == "E":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CD"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["OE1"] = counter
            counter += 1            
            residue.side_chain_atoms_matrixid["OE2"] = counter
            counter += 1  
        elif residue.resname == "Q":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CD"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["OE1"] = counter
            counter += 1            
            residue.side_chain_atoms_matrixid["NE2"] = counter
            counter += 1 
        elif residue.resname == "M":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["SD"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CE"] = counter
            counter += 1            
        elif residue.resname == "H":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["ND1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CD2"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CE1"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["NE2"] = counter
            counter += 1 
        elif residue.resname == "P":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CD"] = counter
            counter += 1 
        elif residue.resname == "F":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CD1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CD2"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CE1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CE2"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CZ"] = counter
            counter += 1 
        elif residue.resname == "Y":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CD1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CD2"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CE1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CE2"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CZ"] = counter
            counter += 1             
            residue.side_chain_atoms_matrixid["OH"] = counter
            counter += 1    
        elif residue.resname == "W":
            residue.side_chain_atoms_matrixid["CG"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CD1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CD2"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["NE1"] = counter
            counter += 1 
            residue.side_chain_atoms_matrixid["CE2"] = counter
            counter += 1   
            residue.side_chain_atoms_matrixid["CE3"] = counter
            counter += 1             
            residue.side_chain_atoms_matrixid["CZ2"] = counter
            counter += 1    
            residue.side_chain_atoms_matrixid["CZ3"] = counter
            counter += 1  
            residue.side_chain_atoms_matrixid["CH2"] = counter
            counter += 1  
    
        counter += residue.num_pad_side_atoms
        
    assert counter == atoms_matrix_len

    return residuesData

def get_side9(idx, residue, geo, rotamer):
    
    side9_counter = [] # (9, 3)
    side9_param = [] # (9, 3)
    
    if (residue.resname == "V"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG1_length, geo.CA_CB_CG1_angle, rotamer[0]])

        side9_counter.append([residue.side_chain_atoms_matrixid["CG1"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG2_length, geo.CG1_CB_CG2_angle, geo.CG1_CA_CB_CG2_diangle])
        
    elif (residue.resname == "I"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG1_length, geo.CA_CB_CG1_angle, rotamer[0]])
        
        side9_counter.append([residue.side_chain_atoms_matrixid["CG1"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG2_length, geo.CG1_CB_CG2_angle, geo.CG1_CA_CB_CG2_diangle])

        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG1"]])
        side9_param.append([geo.CG1_CD1_length, geo.CB_CG1_CD1_angle, rotamer[1]])
        
    elif (residue.resname == "L"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD1_length, geo.CB_CG_CD1_angle, rotamer[1]])

        side9_counter.append([residue.side_chain_atoms_matrixid["CD1"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD2_length, geo.CD1_CG_CD2_angle, geo.CD1_CB_CG_CD2_diangle])
        
    elif (residue.resname == "S"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_OG_length, geo.CA_CB_OG_angle, rotamer[0]])
        
    elif (residue.resname == "T"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_OG1_length, geo.CA_CB_OG1_angle, rotamer[0]])

        side9_counter.append([residue.side_chain_atoms_matrixid["OG1"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG2_length, geo.OG1_CB_CG2_angle, geo.OG1_CA_CB_CG2_diangle])
          
    elif (residue.resname == "D"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_OD1_length, geo.CB_CG_OD1_angle, rotamer[1]])

        side9_counter.append([residue.side_chain_atoms_matrixid["OD1"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_OD2_length, geo.CB_CG_OD2_angle, geo.OD1_CB_CG_OD2_diangle])
        
    elif (residue.resname == "N"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_OD1_length, geo.CB_CG_OD1_angle, rotamer[1]])

        side9_counter.append([residue.side_chain_atoms_matrixid["OD1"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_ND2_length, geo.CB_CG_ND2_angle, geo.OD1_CB_CG_ND2_diangle])
        
    elif (residue.resname == "E"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD_length, geo.CB_CG_CD_angle, rotamer[1]])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD"]])
        side9_param.append([geo.CD_OE1_length, geo.CG_CD_OE1_angle, rotamer[2]])
        
        side9_counter.append([residue.side_chain_atoms_matrixid["OE1"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD"]])
        side9_param.append([geo.CD_OE2_length, geo.CG_CD_OE2_angle, geo.OE1_CG_CD_OE2_diangle])
        
    elif (residue.resname == "Q"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD_length, geo.CB_CG_CD_angle, rotamer[1]])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD"]])
        side9_param.append([geo.CD_OE1_length, geo.CG_CD_OE1_angle, rotamer[2]])
        
        side9_counter.append([residue.side_chain_atoms_matrixid["OE1"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD"]])
        side9_param.append([geo.CD_NE2_length, geo.CG_CD_NE2_angle, geo.OE1_CG_CD_NE2_diangle])
        
    elif (residue.resname == "K"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD_length, geo.CB_CG_CD_angle, rotamer[1]])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD"]])
        side9_param.append([geo.CD_CE_length, geo.CG_CD_CE_angle, rotamer[2]])
        
        side9_counter.append([residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD"], residue.side_chain_atoms_matrixid["CE"]])
        side9_param.append([geo.CE_NZ_length, geo.CD_CE_NZ_angle, rotamer[3]])
        
    elif (residue.resname == "R"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD_length, geo.CB_CG_CD_angle, rotamer[1]])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD"]])
        side9_param.append([geo.CD_NE_length, geo.CG_CD_NE_angle, rotamer[2]])
        
        side9_counter.append([residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD"], residue.side_chain_atoms_matrixid["NE"]])
        side9_param.append([geo.NE_CZ_length, geo.CD_NE_CZ_angle, rotamer[3]])

        side9_counter.append([residue.side_chain_atoms_matrixid["CD"], residue.side_chain_atoms_matrixid["NE"], residue.side_chain_atoms_matrixid["CZ"]])
        side9_param.append([geo.CZ_NH1_length, geo.NE_CZ_NH1_angle, geo.CD_NE_CZ_NH1_diangle])

        side9_counter.append([residue.side_chain_atoms_matrixid["NH1"], residue.side_chain_atoms_matrixid["NE"], residue.side_chain_atoms_matrixid["CZ"]])
        side9_param.append([geo.NH1_NH2_length, geo.NE_CZ_NH2_angle, geo.NH1_NE_CZ_NH2_diangle])
        
    elif (residue.resname == "C"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_SG_length, geo.CA_CB_SG_angle, rotamer[0]])
    
    elif (residue.resname == "M"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_SD_length, geo.CB_CG_SD_angle, rotamer[1]])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["SD"]])
        side9_param.append([geo.SD_CE_length, geo.CG_SD_CE_angle, rotamer[2]])
        
    elif (residue.resname == "F"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD1_length, geo.CB_CG_CD1_angle, rotamer[1]])

        side9_counter.append([residue.side_chain_atoms_matrixid["CD1"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD2_length, geo.CB_CG_CD2_angle, geo.CD1_CB_CG_CD2_diangle])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD1"]])
        side9_param.append([geo.CD1_CE1_length, geo.CG_CD1_CE1_angle, geo.CB_CG_CD1_CE1_diangle])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD2"]])
        side9_param.append([geo.CD2_CE2_length, geo.CG_CD2_CE2_angle, geo.CB_CG_CD2_CE2_diangle])

        side9_counter.append([residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD1"], residue.side_chain_atoms_matrixid["CE1"]])
        side9_param.append([geo.CE1_CZ_length, geo.CD1_CE1_CZ_angle, geo.CG_CD1_CE1_CZ_diangle])
        
    elif (residue.resname == "Y"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD1_length, geo.CB_CG_CD1_angle, rotamer[1]])

        side9_counter.append([residue.side_chain_atoms_matrixid["CD1"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD2_length, geo.CB_CG_CD2_angle, geo.CD1_CB_CG_CD2_diangle])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD1"]])
        side9_param.append([geo.CD1_CE1_length, geo.CG_CD1_CE1_angle, geo.CB_CG_CD1_CE1_diangle])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD2"]])
        side9_param.append([geo.CD2_CE2_length, geo.CG_CD2_CE2_angle, geo.CB_CG_CD2_CE2_diangle])

        side9_counter.append([residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD1"], residue.side_chain_atoms_matrixid["CE1"]])
        side9_param.append([geo.CE1_CZ_length, geo.CD1_CE1_CZ_angle, geo.CG_CD1_CE1_CZ_diangle])

        side9_counter.append([residue.side_chain_atoms_matrixid["CD1"], residue.side_chain_atoms_matrixid["CE1"], residue.side_chain_atoms_matrixid["CZ"]])
        side9_param.append([geo.CZ_OH_length, geo.CE1_CZ_OH_angle, geo.CD1_CE1_CZ_OH_diangle])
        
    elif (residue.resname == "W"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD1_length, geo.CB_CG_CD1_angle, rotamer[1]])

        side9_counter.append([residue.side_chain_atoms_matrixid["CD1"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD2_length, geo.CB_CG_CD2_angle, geo.CD1_CB_CG_CD2_diangle])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD1"]])
        side9_param.append([geo.CD1_NE1_length, geo.CG_CD1_NE1_angle, geo.CB_CG_CD1_NE1_diangle])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD2"]])
        side9_param.append([geo.CD2_CE2_length, geo.CG_CD2_CE2_angle, geo.CB_CG_CD2_CE2_diangle])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD2"]])
        side9_param.append([geo.CD2_CE3_length, geo.CG_CD2_CE3_angle, geo.CB_CG_CD2_CE3_diangle])
        
        side9_counter.append([residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD2"], residue.side_chain_atoms_matrixid["CE2"]])
        side9_param.append([geo.CE2_CZ2_length, geo.CD2_CE2_CZ2_angle, geo.CG_CD2_CE2_CZ2_diangle])

        side9_counter.append([residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD2"], residue.side_chain_atoms_matrixid["CE3"]])
        side9_param.append([geo.CE3_CZ3_length, geo.CD2_CE3_CZ3_angle, geo.CG_CD2_CE3_CZ3_diangle])
        
        side9_counter.append([residue.side_chain_atoms_matrixid["CD2"], residue.side_chain_atoms_matrixid["CE2"], residue.side_chain_atoms_matrixid["CZ2"]])
        side9_param.append([geo.CZ2_CH2_length, geo.CE2_CZ2_CH2_angle, geo.CD2_CE2_CZ2_CH2_diangle])
        
    elif (residue.resname == "H"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_ND1_length, geo.CB_CG_ND1_angle, rotamer[1]])

        side9_counter.append([residue.side_chain_atoms_matrixid["ND1"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD2_length, geo.CB_CG_CD2_angle, geo.ND1_CB_CG_CD2_diangle])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["ND1"]])
        side9_param.append([geo.ND1_CE1_length, geo.CG_ND1_CE1_angle, geo.CB_CG_ND1_CE1_diangle])

        side9_counter.append([residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"], residue.side_chain_atoms_matrixid["CD2"]])
        side9_param.append([geo.CD2_NE2_length, geo.CG_CD2_NE2_angle, geo.CB_CG_CD2_NE2_diangle])
        
    elif (residue.resname == "P"):
        side9_counter.append([residue.main_chain_atoms_matrixid["N"], residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"]])
        side9_param.append([geo.CB_CG_length, geo.CA_CB_CG_angle, rotamer[0]])
        
        side9_counter.append([residue.main_chain_atoms_matrixid["CA"], residue.main_chain_atoms_matrixid["CB"], residue.side_chain_atoms_matrixid["CG"]])
        side9_param.append([geo.CG_CD_length, geo.CB_CG_CD_angle, rotamer[1]])
    
    assert len(side9_counter) == len(side9_param) == residue.num_side_chain_atoms
    
    for _ in range(residue.num_pad_side_atoms):
        # useless info
        side9_counter.append([0, 1, 2])
        side9_param.append([1.0, 90.0, 90.0])

    assert len(side9_counter) == len(side9_param) == Residues.NUM_MAX_SCATOMS
        
    return side9_counter, side9_param

def get_scfeatures(residuesData, geosData, rotamers):
    
    length = len(residuesData)
    
    assert len(geosData) == len(residuesData)
    
    # exclude N, CA, C, O, CB
    side9_counters = [] # counters of three required atoms
    side9_params = [] # dist, angle, dihedral
    count = 0
    for idx in range(length):
        
        num_rotamers = residuesData[idx].num_dihedrals
        
        rotamer = rotamers[count: count+num_rotamers]
        
        side9_counter, side9_param = get_side9(idx, residuesData[idx], geosData[idx], rotamer)
            
        side9_counters.extend(side9_counter)
        side9_params.extend(side9_param)
        
        count += num_rotamers   
        
    assert count == len(rotamers)
    assert Residues.NUM_MAX_SCATOMS*length == len(side9_counters) == len(side9_params)
        
    return side9_counters, side9_params
    
