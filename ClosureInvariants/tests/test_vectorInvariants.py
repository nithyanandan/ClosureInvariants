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
    z2 = vectors_from_advariants[0,:2,:]
    z1 = vectors_from_advariants[0,[2],:]
    mdp22 = VI.minkowski_dot(z2)
    mdp21 = VI.minkowski_dot(z2, z1)
    mdp22_expected, mdp21_expected = minkowski_dot_products
    assert NP.allclose(mdp22, mdp22_expected)
    assert NP.allclose(mdp21, mdp21_expected)

