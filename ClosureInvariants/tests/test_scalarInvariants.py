import pytest
import numpy as NP
from .. import scalarInvariants as SI

def test_copol_corrs_list_on_loops_shape(copol_correlations_array, antenna_pairs, loops):
    result = SI.corrs_list_on_loops(copol_correlations_array, antenna_pairs, loops)
    assert len(result) == len(loops)
    for corr_loop in result:
        assert len(corr_loop) == len(loops[0])

def test_copol_advariant(copol_corrs_list1, copol_advariant1):
    # Call the function
    result = SI.advariant(copol_corrs_list1)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    

    # Check if the result has the expected shape
    assert result.shape == copol_advariant1.shape

    # Check if the values are as expected
    assert NP.allclose(result, copol_advariant1)

def test_copol_advariants_multiple_loops(copol_corrs_lol, copol_advariants_on_list):
    # Call the function
    result = SI.advariants_multiple_loops(copol_corrs_lol)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    
    
    # Check if the result has the expected shape
    assert result.shape == copol_advariants_on_list.shape

    # Check if the values are as expected
    assert NP.allclose(result, copol_advariants_on_list)