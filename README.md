# OPUS-Fold3

OPUS-Fold3 is ***a gradient-based protein all-atom folding and docking framework***, which is able to accurately generate 3D protein structures in compliance with specified constraint, as long as it can be expressed as a function of positions of heavy atoms. In addition, ***experimental cryo-EM density map*** can be included as a differentiable constraint and be integrated with other constraints such as those derived from structure prediction methods to jointly guide the optimization process, thus making a brige between the ***reconstruction of cryo-EM density map*** and ***protein structure prediction techniques***. In summary, OPUS-Fold3 can be flexibly utilized to generate a protein 3D structure following multiple sources of constraints, which is crucial for protein structure refinement and protein design. Moreover, developed using Python and TensorFlow 2.4, OPUS-Fold3 is user-friendly for any source-code level modifications and can be seamlessly combined with other deep learning models, thus facilitating collaboration between the ***biology*** and ***AI*** communities.

## Performance of OPUS-Fold3
Some protein folding and docking trajectories are presented as movies [here](https://github.com/thuxugang/opus_fold3/tree/main/examples/movies).

## How to introduce your constraint into OPUS-Fold3
If a constraint can be expressed as ***a function of heavy atoms' positions***, it can be included in OPUS-Fold3. Here, we use the introduction of Lennard Jones potential in backbone folding as an example.

### Gernal workflow

In each epoch, we use `RebuildStructure.rebuild_main_chain` to generate the positions of every main-chain atoms (`atoms_matrix`) from three trainable variable phi, psi and omega (protein torsion angles) of each residue. Note that, the index of each atom corresponds to its relatived residue is generated by `RebuildStructure.save_index_in_matrix` at the begining, which will save the index of each atom in its corresponding residue (main-chain atoms in `residue.main_chain_atoms_matrixid` and side-chain atoms in `residue.side_chain_atoms_matrixid`). 

### Before optimaztion

We need to save the index of atom a (`LJ_matrix_atoma`), index of atom b (`LJ_matrix_atomb`) and the params (`LJ_matrix_aij`, `LJ_matrix_eij` and `LJ_matrix_lambda`) used for calculating LJ potential in `LJ_matrixs`.
```
LJ_matrixs = init_LJ_matrix(residuesData)

def accelerate_ae():
    LJ_parms = Atoms.LJ_parms
    LJ_ac = {}
    for type1 in LJ_parms:
        for type2 in LJ_parms:
            if type2 < type1:
                continue
            aij = None
            if type1==9 or type1==10 or type1==11: #Special cases for some atoms
                if type2==11 or type2==12:
                    aij=3.1
                elif type2==13 or type2==14:
                    aij=2.9
                elif type2==15 or type2==16:
                    aij=2.9
            elif type1==12:
                if type2==15 or type2==16:
                    aij=2.9
            elif type1==13:
                if type2==15 or type2==16:
                    aij=2.8
            elif type1==15 or type1==16:
                if type2==16:
                    aij=2.8
            elif type1==17:
                if type2==17:
                    aij=2
            elif type1==8:
                if type2==17:
                    aij=3
                    
            if aij == None:
                aij = (LJ_parms[type1][0]+LJ_parms[type2][0])
                
            eij = np.sqrt(LJ_parms[type1][1]*LJ_parms[type2][1])
            
            LJ_ac[str(type1)+","+str(type2)] = [aij, eij]
            
    return LJ_ac

LJ_ac = accelerate_ae()

def init_LJ_matrix(residuesData):
    
    LJ_matrix_atoma = []
    LJ_matrix_atomb = []
    LJ_matrix_aij = []
    LJ_matrix_eij = []
    LJ_matrix_lambda = []
    for idx_a, residue_a in enumerate(residuesData):
        for residue_b in residuesData[idx_a+2:]:
            for a_atom in residue_a.main_chain_atoms_matrixid:
                for b_atom in residue_b.main_chain_atoms_matrixid:
                    if residue_a.atoms_lj_type[a_atom] == -1 or residue_b.atoms_lj_type[b_atom] == -1: continue

                    LJ_matrix_atoma.append(residue_a.main_chain_atoms_matrixid[a_atom])
                    LJ_matrix_atomb.append(residue_b.main_chain_atoms_matrixid[b_atom])
                    
                    if residue_a.atoms_lj_type[a_atom] > residue_b.atoms_lj_type[b_atom]:
                        aij, eij = LJ_ac[str(residue_b.atoms_lj_type[b_atom])+","+str(residue_a.atoms_lj_type[a_atom])]
                    else:
                        aij, eij = LJ_ac[str(residue_a.atoms_lj_type[a_atom])+","+str(residue_b.atoms_lj_type[b_atom])]
                    
                    LJ_matrix_aij.append(aij)
                    LJ_matrix_eij.append(eij)
    
                    if residue_a.atoms_lj_type[a_atom] == 6 and residue_b.atoms_lj_type[b_atom] == 6: 
                        lambd = 1
                    else:
                        lambd = 1.6
                    
                    LJ_matrix_lambda.append(lambd)
                    
    return [np.array(LJ_matrix_atoma, dtype=np.int32), np.array(LJ_matrix_atomb, dtype=np.int32), \
           np.array(LJ_matrix_aij, dtype=np.float32), np.array(LJ_matrix_eij, dtype=np.float32), \
           np.array(LJ_matrix_lambda, dtype=np.float32)]
```

### During the optimaztion

In each epoch, after generating the positions of new main-chain atoms (`RebuildStructure.rebuild_main_chain`), we can add `cal_LJPotential' in `Potentials.get_potentials`. In `cal_LJPotential', we use `a_pos = tf.gather(atoms_matrix, LJ_matrix_atoma)` and ` b_pos = tf.gather(atoms_matrix, LJ_matrix_atomb)` to get the atoms' postitions in `atoms_matrix` and do the calculation accordingly.
```
def cal_LJPotential(LJ_matrixs, atoms_matrix):
    
    LJ_matrix_atoma, LJ_matrix_atomb, LJ_matrix_aij, LJ_matrix_eij, LJ_matrix_lambda = LJ_matrixs
    
    atoms_matrix = tf.cast(atoms_matrix, tf.float32)
    
    lj_potential = 0
    
    a_pos = tf.gather(atoms_matrix, LJ_matrix_atoma)
    b_pos = tf.gather(atoms_matrix, LJ_matrix_atomb)
    
    dij = tf.sqrt(tf.reduce_sum(tf.square(a_pos - b_pos), -1))
    dij_star = dij/LJ_matrix_aij
    

    case1 = tf.squeeze(tf.where(dij_star<0.015), -1)
    case2 = tf.squeeze(tf.where((0.015<=dij_star) & (dij_star<1)), -1)
    case3 = tf.squeeze(tf.where((1<=dij_star) & (dij_star<1.9)), -1)

    dij_star = tf.where(dij_star<0.015, 10, dij_star)
    
    lj_potential += tf.reduce_sum(tf.gather(dij_star, case1))
    lj_potential += tf.reduce_sum(10.0*(tf.gather(dij_star, case2)-1.0)/(0.015-1.0))
    lj_potential += tf.reduce_sum(4.0*tf.gather(LJ_matrix_eij, case3)*
                          (tf.math.pow(tf.gather(dij_star, case3), -12)-tf.math.pow(tf.gather(dij_star, case3), -6)))
    
    return lj_potential / (case1.shape[0] + case2.shape[0] + case3.shape[0] + 1e-6)
```
