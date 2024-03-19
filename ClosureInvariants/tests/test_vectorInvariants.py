import pytest
import numpy as NP
from .. import vectorInvariants as VI

def test_corrs_list_on_loops_shape(correlations_array, antenna_pairs, loops):
    result = VI.corrs_list_on_loops(correlations_array, antenna_pairs, loops)
    assert len(result) == len(loops)
    for corr_loop in result:
        assert len(corr_loop) == len(loops[0])

def test_corrs_list_on_loops_type(correlations_array, antenna_pairs, loops):
    result = VI.corrs_list_on_loops(correlations_array, antenna_pairs, loops)
    for corr_loop in result:
        for corr in corr_loop:
            assert isinstance(corr, NP.ndarray)

def test_corrs_list_on_loops_values(correlations_array, antenna_pairs, loops):
    result = VI.corrs_list_on_loops(correlations_array, antenna_pairs, loops)
    
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
            inpcorr = correlations_array[pair_index]
            if conj:
                inpcorr = inpcorr.T.conj()
            assert NP.allclose(corr_loop[i], inpcorr)

def test_advariant(corrs_list1, polaxes, advariant1):
    # Call the function
    result = VI.advariant(corrs_list1, pol_axes=polaxes)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    

    # Check if the result has the expected shape
    assert result.shape == advariant1.shape

    # Check if the values are as expected
    assert NP.allclose(result, advariant1)

@pytest.mark.parametrize("expected_shape", [(1, 2, 2, 2)])
def test_advariants_multiple_loops(corrs_lol, expected_shape, advariants_on_list):
    # Call the function
    result = VI.advariants_multiple_loops(corrs_lol)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    
    
    # Check if the result has the expected shape
    assert result.shape == advariants_on_list.shape

    # Check if the values are as expected
    assert NP.allclose(result, advariants_on_list)

def test_vectors_from_advariants(advariants_on_list, vectors_from_advariants):
    result = VI.vector_from_advariant(advariants_on_list)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    
    
    # Check if the result has the expected shape
    assert result.shape == vectors_from_advariants.shape

    # Check if the values are as expected
    assert NP.allclose(result, vectors_from_advariants)

def test_minkowski_dot_products(vectors_from_advariants, minkowski_dot_products):
    z2 = vectors_from_advariants[:,:2,:]
    z1 = vectors_from_advariants[:,[2],:]
    mdp22 = VI.minkowski_dot(z2)
    mdp21 = VI.minkowski_dot(z2, z1)
    mdp22_expected, mdp21_expected = minkowski_dot_products
    assert mdp22.shape == mdp22_expected.shape
    assert mdp21.shape == mdp21_expected.shape
    assert NP.allclose(mdp22, mdp22_expected)
    assert NP.allclose(mdp21, mdp21_expected)

def test_complete_minkowski_dots(vectors_from_advariants, complete_minkowski_dots):
    result = VI.complete_minkowski_dots(vectors_from_advariants)
    assert result.size == complete_minkowski_dots.size
    assert result.size == (4*5)//2 + 2*(vectors_from_advariants.shape[-2]-2)*4
    assert NP.allclose(result, complete_minkowski_dots)    

def test_remove_scaling_factor_minkoski_dots(complete_minkowski_dots, minkoski_dots_scaling_factor_removed):
    result = VI.remove_scaling_factor_minkoski_dots(complete_minkowski_dots)
    assert NP.allclose(result, minkoski_dots_scaling_factor_removed)
    assert NP.allclose(NP.sum(result**2), 1.0)

def test_invariance(corrs_list1, corrs_list2, corrs_list3, complex_gains):
    corrs_in = [corrs_list1, corrs_list2, corrs_list3]
    advars_in = VI.advariants_multiple_loops(corrs_in)
    z4v_in = VI.vector_from_advariant(advars_in)
    mdp_in = VI.complete_minkowski_dots(z4v_in)
    ci_in = VI.remove_scaling_factor_minkoski_dots(mdp_in)

    preinds = NP.concatenate([NP.zeros((3,1), dtype=int), 1+NP.arange(6, dtype=int).reshape(3,2)], axis=-1)
    postinds = NP.roll(preinds, -1, axis=-1)
    corrs_out = [VI.corrupt_visibilities(NP.array(corrs_in[loopi]), complex_gains[preinds[loopi]], complex_gains[postinds[loopi]]) for loopi in range(len(corrs_in))]
    advars_out = VI.advariants_multiple_loops(corrs_out)
    z4v_out = VI.vector_from_advariant(advars_out)
    mdp_out = VI.complete_minkowski_dots(z4v_out)
    ci_out = VI.remove_scaling_factor_minkoski_dots(mdp_out)

    assert NP.allclose(ci_in, ci_out)
