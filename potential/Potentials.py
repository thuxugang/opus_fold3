# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:46:08 2016

@author: XuGang
"""

from potential import TrrPotential, Rama

def get_potentials(Trr_matrix, atoms_matrix, init_torsions, rama_cons):
    
    potentials = 0
    
    dist_potential, omega_potential, theta_potential, phi_potential = \
        TrrPotential.cal_TrrPotential(Trr_matrix, atoms_matrix)
    potentials += (10*dist_potential + 8*omega_potential + 8*theta_potential + 8*phi_potential)
    
    rama_potential, o_potential = Rama.cal_RamaPotential(init_torsions, rama_cons)
    potentials += (0.1*rama_potential + 0.05*o_potential)

    return potentials

def get_dock_potentials(Trr_matrix, atoms_matrix, init_torsions, rama_cons, fixed_backbone):
    
    potentials = 0
    
    dist_potential, omega_potential, theta_potential, phi_potential = \
        TrrPotential.cal_TrrPotential(Trr_matrix, atoms_matrix)
    potentials += (10*dist_potential + 8*omega_potential + 8*theta_potential + 8*phi_potential)
    
    if not fixed_backbone:
        rama_potential, o_potential = Rama.cal_RamaPotential(init_torsions, rama_cons)
        potentials += (0.1*rama_potential + 0.05*o_potential)
    
    return potentials


def get_scpotentials(SCTrr_matrixs, atoms_matrix):
    
    potentials = 0
    
    for SCTrr_matrix in SCTrr_matrixs:
        dist_potential, omega_potential, theta_potential, phi_potential = \
            TrrPotential.cal_TrrPotential(SCTrr_matrix, atoms_matrix)
        potentials += (10*dist_potential + 8*omega_potential + 8*theta_potential + 8*phi_potential)
    
    potentials /= len(SCTrr_matrixs)
   
    return potentials

def get_dock_scpotentials(SCTrr_matrixs, atoms_matrix):
    
    potentials = 0
    
    for SCTrr_matrix in SCTrr_matrixs:
        dist_potential, omega_potential, theta_potential, phi_potential = \
            TrrPotential.cal_TrrPotential(SCTrr_matrix, atoms_matrix)
        potentials += (10*dist_potential + 8*omega_potential + 8*theta_potential + 8*phi_potential)
    
    potentials /= len(SCTrr_matrixs)
    
    return potentials