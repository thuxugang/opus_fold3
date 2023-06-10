# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:46:08 2016

@author: XuGang
"""

import tensorflow as tf
import numpy as np
import neurite as ne

from potential import TrrPotential, Rama, mrc

atomdefs={'H':(1.0,1.00794),'HO':(1.0,1.00794),'C':(6.0,12.0107),'A':(7.0,14.00674),'N':(7.0,14.00674),'O':(8.0,15.9994),'P':(15.0,30.973761),'K':(19.0,39.0983),
            'S':(16.0,32.066),'W':(18.0,1.00794*2.0+15.9994),'AU':(79.0,196.96655) }

def read_map(filename, apix=1.5):

    exp_map, header = mrc.parse_mrc(filename, False)
    exp_map = tf.convert_to_tensor(exp_map)
    # resample to target apix
    original_apix = header.get_apix()
    origin = header.get_origin()
    if not (origin == np.zeros(3)).all():
        print ("The origin of density map should be [0,0,0], current is", origin)
        assert origin == np.zeros(3)

    ori_shape = tf.shape(exp_map)
    tgt_shape = ori_shape.numpy() * original_apix / apix
    tgt_shape = tgt_shape.astype(np.int32)
    exp_map = resample_map(exp_map, tgt_shape)

    return exp_map, original_apix

def crop_map(mrc, box):
    cropped_map = mrc[box[2][0]:box[2][1], box[1][0]:box[1][1], box[0][0]:box[0][1]]
    return cropped_map

def resample_map(mrc, tgt_shape):
    # resample map from vol_shape to target shape
    vol_shape = mrc.shape
    nb_dims = len(vol_shape)
    mrc = tf.expand_dims(mrc, axis=-1)
    mesh = ne.utils.volshape_to_meshgrid(tgt_shape, indexing='ij')

    ratios = [vol_shape[d]/tgt_shape[d] for d in range(nb_dims)]
    loc = [tf.cast(mesh[d], 'float32')*ratios[d] for d in range(nb_dims)]
    mrc = ne.utils.interpn(mrc, loc, interp_method='linear', fill_value=None)
    mrc = tf.squeeze(mrc, axis=-1)
    
    return mrc

def real_space_correlation(exp_map, ref_map):
    corr = tf.reduce_sum(exp_map*ref_map)
    corr /= (tf.norm(exp_map) * tf.norm(ref_map))
    return -corr

def get_shift(coords, mask, exp_map=None, apix=1., filename="", original_apix=1.):

    max_corr = 0
    best_shift = -1
    for s in range(-10, 11):
        shift = s / 10.0
        corr = get_EMpotentials(coords, mask, exp_map, apix, filename, original_apix, shift=shift)
        if corr < max_corr:
            max_corr = corr
            best_shift = shift
            print ("best_shift:", best_shift)
    return best_shift

def get_EMpotentials(coords, mask, exp_map=None, apix=1., filename="", original_apix=1., shift=0.5):

    indices = tf.squeeze(tf.where(tf.math.equal(mask, 1)), 1)
    coords = tf.gather(coords, indices)

    amax = np.max(coords, axis=-2) # (x, y ,z)
    amin = np.min(coords, axis=-2)

    # center coords
    atoms = coords
    min_coords = tf.cast(tf.reduce_min(tf.round(amin/apix)), tf.int32)
    max_coords = tf.cast(tf.reduce_max(tf.round(amax/apix)), tf.int32)
    interval = max_coords - min_coords
    if interval % 2 == 0:
        min_coords -= 5
        max_coords += 5
    else:
        min_coords -= 5
        max_coords += 6
    outbox = np.array([max_coords - min_coords]*3)

    exp_map = exp_map[min_coords:max_coords, min_coords:max_coords, min_coords:max_coords]

    #shift by min of coords
    atoms = atoms/apix - tf.cast(min_coords, tf.float32) - shift #0.5 #+ original_apix/apix
    atoms = tf.reverse(atoms, [1])

    out_map = tf.zeros(outbox, dtype=tf.float32)
    Bi = 15.15
    cvar = 1.5 * (np.pi * apix) ** 2 - Bi

    atoms = tf.convert_to_tensor(atoms)
    atoms_int = tf.cast(tf.round(atoms), tf.int32)
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-3, 4):
                atoms_cur = atoms_int + tf.convert_to_tensor([h, k, l])
                atoms_cur_float = tf.cast(atoms_cur, dtype=tf.float32)
                d = tf.reduce_sum(tf.pow(atoms - atoms_cur_float, 2), axis=-1)
                out_map = tf.tensor_scatter_nd_add(out_map, atoms_cur, tf.exp(-d/1.5))

    # we then sharpening the map
    y = tf.range(outbox[0], dtype=tf.float32)/outbox[0] - 0.5
    Z, Y, X = tf.meshgrid(y, y, y)

    radius = tf.stack((X, Y, Z), axis=-1)
    radius = tf.reduce_sum(tf.pow(radius, 2), axis=-1)
    gaussian_kernel = tf.exp(radius*cvar / apix ** 2) * np.sqrt(apix ** 2 / (1.5*Bi)) ** 3

    out_map = tf.complex(out_map, tf.zeros_like(out_map))
    out_map_fft = tf.signal.fft3d(out_map)

    # shift map
    gaussian_kernel = tf.signal.fftshift(gaussian_kernel, axes=(0, 1, 2))
    gaussian_kernel = tf.complex(gaussian_kernel, tf.zeros_like(gaussian_kernel))

    # divide by the gaussian
    out_map_fft *= gaussian_kernel

    out_map = tf.signal.ifft3d(out_map_fft)
    out_map = tf.math.real(out_map)

    out_map_true = exp_map
    return real_space_correlation(out_map_true, out_map)

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
