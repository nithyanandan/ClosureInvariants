import pytest
import numpy as NP
from .. import graphUtils as GU
from .. import vectorInvariants as VI

def test_pol_corrs_list_on_loops_shape(pol_correlations_array, antenna_pairs, loops):
    result = VI.corrs_list_on_loops(pol_correlations_array, antenna_pairs, loops)
    assert len(result) == len(loops)
    for corr_loop in result:
        assert len(corr_loop) == len(loops[0])

def test_pol_corrs_list_on_loops_type(pol_correlations_array, antenna_pairs, loops):
    result = VI.corrs_list_on_loops(pol_correlations_array, antenna_pairs, loops)
    for corr_loop in result:
        for corr in corr_loop:
            assert isinstance(corr, NP.ndarray)

def test_pol_corrs_list_on_loops_values(pol_correlations_array, antenna_pairs, loops):
    result = VI.corrs_list_on_loops(pol_correlations_array, antenna_pairs, loops)
    
    for loop, corr_loop in zip(loops, result):
        for i in range(len(loop)):
            conj = False
            antpair_in_loop = (loop[i], loop[(i + 1) % len(loop)])
            if antpair_in_loop in antenna_pairs:
                pair_index = antenna_pairs.index(antpair_in_loop)
            elif tuple(reversed(antpair_in_loop)) in antenna_pairs:
                pair_index = antenna_pairs.index(tuple(reversed(antpair_in_loop)))
                conj = True
            else:
                assert False, "Loop antenna pair {0} not found in input antenna pairs".format(antpair_in_loop)
            inpcorr = pol_correlations_array[pair_index]
            if conj:
                inpcorr = inpcorr.T.conj()
            assert NP.allclose(corr_loop[i], inpcorr)

def test_pol_advariant(pol_corrs_list1, polaxes, pol_advariant1):
    # Call the function
    result = VI.advariant(pol_corrs_list1, pol_axes=polaxes)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    

    # Check if the result has the expected shape
    assert result.shape == pol_advariant1.shape

    # Check if the values are as expected
    assert NP.allclose(result, pol_advariant1)

def test_pol_advariants_multiple_loops(pol_xc_lol, pol_advariant_loops):
    result = VI.advariants_multiple_loops(pol_xc_lol)

    # Check if the result has the expected shape
    assert result.shape == pol_advariant_loops.shape

    # Check if the values are as expected
    assert NP.allclose(result, pol_advariant_loops)

def test_pol_advariants_multiple_loops_old(pol_corrs_lol, pol_advariants_on_list):
    # Call the function
    result = VI.advariants_multiple_loops(pol_corrs_lol)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    
    
    # Check if the result has the expected shape
    assert result.shape == pol_advariants_on_list.shape

    # Check if the values are as expected
    assert NP.allclose(result, pol_advariants_on_list)

def test_vectors_from_advariants(pol_advariant_loops, polaxes):
    indims = pol_advariant_loops.ndim
    polaxes = NP.array(polaxes)
    polaxes = (polaxes + indims) % indims 
    npol = len(polaxes)

    pauli_mat0 = NP.identity(2, dtype=complex).reshape(tuple(NP.ones(indims-2, dtype=int))+(npol,npol))
    pauli_mat1 = NP.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex).reshape(tuple(NP.ones(indims-2, dtype=int))+(npol,npol))
    pauli_mat2 = NP.asarray([[0.0, -1j], [1j, 0.0]], dtype=complex).reshape(tuple(NP.ones(indims-2, dtype=int))+(npol,npol))
    pauli_mat3 = NP.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex).reshape(tuple(NP.ones(indims-2, dtype=int))+(npol,npol))

    z0 = 0.5 * NP.trace(pol_advariant_loops@pauli_mat0, axis1=-2, axis2=-1)
    z1 = 0.5 * NP.trace(pol_advariant_loops@pauli_mat1, axis1=-2, axis2=-1)
    z2 = 0.5 * NP.trace(pol_advariant_loops@pauli_mat2, axis1=-2, axis2=-1)
    z3 = 0.5 * NP.trace(pol_advariant_loops@pauli_mat3, axis1=-2, axis2=-1)

    z4v = NP.concatenate([z0[...,NP.newaxis], z1[...,NP.newaxis], z2[...,NP.newaxis], z3[...,NP.newaxis]], axis=-1) # shape=(...,4)

    pkg_result = VI.vector_from_advariant(pol_advariant_loops)
    
    # Check if the result has the expected shape
    assert pkg_result.shape == z4v.shape

    # Check if the values are as expected
    assert NP.allclose(pkg_result, z4v)

def test_vectors_from_advariants_old(pol_advariants_on_list, vectors_from_pol_advariants):
    result = VI.vector_from_advariant(pol_advariants_on_list)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    
    
    # Check if the result has the expected shape
    assert result.shape == vectors_from_pol_advariants.shape

    # Check if the values are as expected
    assert NP.allclose(result, vectors_from_pol_advariants)

def test_minkowski_dot_products(vectors_from_pol_advariants, minkowski_dot_products):
    z2 = vectors_from_pol_advariants[:,:2,:]
    z1 = vectors_from_pol_advariants[:,[2],:]
    mdp22 = VI.minkowski_dot(z2)
    mdp21 = VI.minkowski_dot(z2, z1)
    mdp22_expected, mdp21_expected = minkowski_dot_products
    assert mdp22.shape == mdp22_expected.shape
    assert mdp21.shape == mdp21_expected.shape
    assert NP.allclose(mdp22, mdp22_expected)
    assert NP.allclose(mdp21, mdp21_expected)

def test_complete_minkowski_dots(vectors_from_pol_advariants, complete_minkowski_dots):
    result = VI.complete_minkowski_dots(vectors_from_pol_advariants)
    assert result.size == complete_minkowski_dots.size
    assert result.size == (4*5)//2 + 2*(vectors_from_pol_advariants.shape[-2]-2)*4
    assert NP.allclose(result, complete_minkowski_dots)    

def test_remove_scaling_factor_minkoski_dots(complete_minkowski_dots, minkoski_dots_scaling_factor_removed):
    result = VI.remove_scaling_factor_minkoski_dots(complete_minkowski_dots)
    assert NP.allclose(result, minkoski_dots_scaling_factor_removed)
    assert NP.allclose(NP.sum(result**2), 1.0)

def test_invariance(pol_xc, pol_complex_gains, example_ids, baseid_ind, polaxes):
    bl_axis = -3
    element_axis = -3
    element_pairs = [(example_ids[i], example_ids[j]) for i in range(len(example_ids)) for j in range(i + 1, len(example_ids))]
    triads_indep = GU.generate_independent_triads(example_ids, baseid_ind)
    pol_xc_lol = VI.corrs_list_on_loops(pol_xc, element_pairs, triads_indep, bl_axis=bl_axis, pol_axes=polaxes)
    advars_in = VI.advariants_multiple_loops(pol_xc_lol)
    z4v_in = VI.vector_from_advariant(advars_in)
    mdp_in = VI.complete_minkowski_dots(z4v_in)
    ci_in = VI.remove_scaling_factor_minkoski_dots(mdp_in)

    prefactor_gains = NP.take(pol_complex_gains, NP.array(element_pairs)[:,0], axis=element_axis) # A collection of g_a
    postfactor_gains = NP.take(pol_complex_gains, NP.array(element_pairs)[:,1], axis=element_axis) # A collection of g_b
    pol_xc_mod = VI.corrupt_visibilities(pol_xc, prefactor_gains, postfactor_gains, pol_axes=polaxes)
    pol_xc_lol_mod = VI.corrs_list_on_loops(pol_xc_mod, element_pairs, triads_indep, bl_axis=bl_axis, pol_axes=polaxes)
    advars_out = VI.advariants_multiple_loops(pol_xc_lol_mod)
    z4v_out = VI.vector_from_advariant(advars_out)
    mdp_out = VI.complete_minkowski_dots(z4v_out)
    ci_out = VI.remove_scaling_factor_minkoski_dots(mdp_out)

    assert NP.allclose(ci_in, ci_out)

    scale_factor = NP.abs(NP.linalg.det(pol_complex_gains))**2

    assert NP.allclose(scale_factor[...,[baseid_ind]], mdp_out / mdp_in)

def test_invariance_old(pol_corrs_list1, pol_corrs_list2, pol_corrs_list3, pol_complex_gains):
    corrs_in = [pol_corrs_list1, pol_corrs_list2, pol_corrs_list3]
    advars_in = VI.advariants_multiple_loops(corrs_in)
    z4v_in = VI.vector_from_advariant(advars_in)
    mdp_in = VI.complete_minkowski_dots(z4v_in)
    ci_in = VI.remove_scaling_factor_minkoski_dots(mdp_in)

    irun = 0
    preinds = NP.concatenate([NP.zeros((3,1), dtype=int), 1+NP.arange(6, dtype=int).reshape(3,2)], axis=-1)
    postinds = NP.roll(preinds, -1, axis=-1)
    corrs_out = [VI.corrupt_visibilities(NP.array(corrs_in[loopi]), pol_complex_gains[irun,preinds[loopi]], pol_complex_gains[irun,postinds[loopi]]) for loopi in range(len(corrs_in))]
    advars_out = VI.advariants_multiple_loops(corrs_out)
    z4v_out = VI.vector_from_advariant(advars_out)
    mdp_out = VI.complete_minkowski_dots(z4v_out)
    ci_out = VI.remove_scaling_factor_minkoski_dots(mdp_out)

    assert NP.allclose(ci_in, ci_out)

