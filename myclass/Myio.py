# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:32:13 2015

@author: XuGang
"""

from myclass import Atoms, Residues
import numpy as np    
import random

#pick phi/psi randomly from:
#-140  153 180 0.135 B
# -72  145 180 0.155 B
#-122  117 180 0.073 B
# -82  -14 180 0.122 A
# -61  -41 180 0.497 A
#  57   39 180 0.018 L
def random_dihedral():
    phi=0
    psi=0
    r=random.random()
    if(r<=0.135):
        phi=-140
        psi=153
    elif(r>0.135 and r<=0.29):
        phi=-72
        psi=145
    elif(r>0.29 and r<=0.363):
        phi=-122
        psi=117
    elif(r>0.363 and r<=0.485):
        phi=-82
        psi=-14
    elif(r>0.485 and r<=0.982):
        phi=-61
        psi=-41
    else:
        phi=57
        psi=39
    return phi, psi

def readTASS(path, phi_idx, psi_idx):
    
    init_torsions = []
    with open(path,'r') as r:
        for i in r.readlines():
            if i.strip().split()[0][0] == '#':
                continue
            else:
                context = i.strip().split()
                assert len(context) == 17
                phi = float(context[phi_idx])
                psi = float(context[psi_idx])
                omega = 180.0
                init_torsions.extend([phi, psi, omega])
                
    return init_torsions

def readRotaNN(rota_path, start_idx, end_idx):
    # 182: no specific dihedral
    
    init_rotamers = []
    x1_index = []
    f = open(rota_path)
    for i in f.readlines():
        if i[0] == '#' or i.strip() == "":
            continue
        x1_4 = i.strip().split()[start_idx:end_idx]
        assert len(x1_4) == 4
        x1_index.append(len(init_rotamers))
        init_rotamers.extend([float(j) for j in x1_4 if float(j) != 182])
    f.close() 
    
    return init_rotamers, x1_index

def randomTorsions(fasta):
    
    init_torsions = []
    for _ in fasta:
        phi, psi = random_dihedral()
        omega = 180.0
        init_torsions.extend([phi, psi, omega])
                
    return init_torsions

def readFasta(path):
    
    with open(path,'r') as r:
        results = [i.strip() for i in r.readlines()]
    return results[0][1:], results[1]

def readPDB(filename, chains=None):
    f = open(filename,'r')
    atomsDatas = []
    for line in f.readlines():   
        line = line.strip()
        if (line == "" or line[:3] == "TER"):
            break
        else:
            if (line[:4] == 'ATOM' or line[:6] == 'HETATM'):
                atomid = line[6:11].strip()
                name1 = line[11:16].strip()
                resname = line[16:20].strip()
                chainid = line[21].strip()
                
                #B confomation        
                if(len(resname) == 4 and resname[0] != "A"):
                    continue
                
                resid = line[22:27].strip()

                x = line[30:38].strip()
                y = line[38:46].strip()
                z = line[46:54].strip()
                
                if chains == None or chainid in chains:
                    if(name1[0] in ["N","O","C","S"]):
                        position = np.array([float(x), float(y), float(z)], dtype=np.float32)
                        atom = Atoms.Atom(atomid, name1, resname, resid, position, chainid)
                        atomsDatas.append(atom)
    f.close()
    return atomsDatas

def outputRotaNN(rotamers, residuesData, rota_path):
    
    count = 0
    f = open(rota_path, "w")
    for idx, residue in enumerate(residuesData):
        
        num_rotamers = residue.num_dihedrals
        rotamer = rotamers[count: count+num_rotamers]
        if  num_rotamers == 0:
            f.write(str(idx+1) + " " + str(182) + " " + str(182) + " " + str(182) + " " + str(182) + "\n")
        elif num_rotamers == 1:
            f.write(str(idx+1) + " " + str(rotamer[0].numpy()) + " " + str(182) + " " + str(182) + " " + str(182) + "\n")
        elif  num_rotamers == 2:
            f.write(str(idx+1) + " " + str(rotamer[0].numpy()) + " " + str(rotamer[1].numpy()) + " " + str(182) + " " + str(182) + "\n")
        elif  num_rotamers == 3:
            f.write(str(idx+1) + " " + str(rotamer[0].numpy()) + " " + str(rotamer[1].numpy()) + " " + str(rotamer[2].numpy()) + " " + str(182) + "\n")
        elif  num_rotamers == 4:
            f.write(str(idx+1) + " " + str(rotamer[0].numpy()) + " " + str(rotamer[1].numpy()) + " " + str(rotamer[2].numpy()) + " " + str(rotamer[3].numpy()) + "\n")
                                                                
        count += num_rotamers

    assert count == len(rotamers)
    
    f.close() 
        
def outputPDB(residuesData, atoms_matrix, pdb_path):
    
    atom_id = 1
    counter = 0
    f = open(pdb_path, 'w')
    for residue in residuesData:
        for idx, name1 in enumerate(["N", "CA", "C", "O", "CB"]):
            if residue.resname == "G" and name1 == "CB": 
                counter += 1
                continue
            atom_id2 = atom_id + idx
            string = 'ATOM  '
            id_len = len(list(str(atom_id2)))
            string = string + " "*(5-id_len) + str(atom_id2)
            string = string + " "*2
            name1_len = len(list(name1))
            string = string + name1 + " "*(3-name1_len)
            resname = Residues.triResname(residue.resname)
            resname_len = len(list(resname))
            string = string + " "*(4-resname_len) + resname
            string = string + " "*2
            resid = str(residue.resid)
            resid_len = len(list(resid))
            string = string + " "*(4-resid_len) + str(resid)
            string = string + " "*4
            x = format(atoms_matrix[counter][0],".3f")
            x_len = len(list(x))
            string = string + " "*(8-x_len) + x
            y = format(atoms_matrix[counter][1],".3f")
            y_len = len(list(y))
            string = string + " "*(8-y_len) + y
            z = format(atoms_matrix[counter][2],".3f")        
            z_len = len(list(z))
            string = string + " "*(8-z_len) + z  
            
            string_list = list(string)
            string_list[21] = residue.chainid
            string = "".join(string_list)
            
            f.write(string)
            f.write("\n")
            
            counter += 1
        
        counter += residue.num_pad_main_atoms
        atom_id += residue.num_atoms
        
    assert len(atoms_matrix) == counter
    f.close()

def outputPDBALL(residuesData, atoms_matrix, atoms_matrix_name, pdb_path):
    
    if type(atoms_matrix) != list:
        atoms_matrix = atoms_matrix.numpy()
        
    atom_id = 1
    counter = 0
    f = open(pdb_path, 'w')
    assert len(residuesData) == len(atoms_matrix_name)
    for residue, atom_names in zip(residuesData, atoms_matrix_name):
        for idx, name1 in enumerate(atom_names):
            if residue.resname == "G" and name1 == "CB": 
                counter += 1
                continue
            atom_id2 = atom_id + idx
            string = 'ATOM  '
            id_len = len(list(str(atom_id2)))
            string = string + " "*(5-id_len) + str(atom_id2)
            string = string + " "*2
            name1_len = len(list(name1))
            string = string + name1 + " "*(3-name1_len)
            resname = Residues.triResname(residue.resname)
            resname_len = len(list(resname))
            string = string + " "*(4-resname_len) + resname
            string = string + " "*2
            resid = str(residue.resid)
            resid_len = len(list(resid))
            string = string + " "*(4-resid_len) + str(resid)
            string = string + " "*4
            x = format(atoms_matrix[counter][0],".3f")
            x_len = len(list(x))
            string = string + " "*(8-x_len) + x
            y = format(atoms_matrix[counter][1],".3f")
            y_len = len(list(y))
            string = string + " "*(8-y_len) + y
            z = format(atoms_matrix[counter][2],".3f")        
            z_len = len(list(z))
            string = string + " "*(8-z_len) + z  
            
            string_list = list(string)
            string_list[21] = residue.chainid
            string = "".join(string_list)
            
            f.write(string)
            f.write("\n")
            
            counter += 1
        
        counter += residue.num_pad_side_atoms
        atom_id += residue.num_atoms
        
    assert len(atoms_matrix) == counter
    f.close()