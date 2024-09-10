import pytest
import numpy as NP
from .. import graphUtils as GU
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

def test_copol_advariants_multiple_loops(copol_xc_lol, copol_advariant_loops):
    # Call the function
    result = SI.advariants_multiple_loops(copol_xc_lol)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    
    
    # Check if the result has the expected shape
    assert result.shape == copol_advariant_loops.shape

    # Check if the values are as expected
    assert NP.allclose(result, copol_advariant_loops)

def test_copol_advariants_multiple_loops_old(copol_corrs_lol, copol_advariants_on_list):
    # Call the function
    result = SI.advariants_multiple_loops(copol_corrs_lol)

    # Check if the result is a numpy array
    assert isinstance(result, NP.ndarray)    
    
    # Check if the result has the expected shape
    assert result.shape == copol_advariants_on_list.shape

    # Check if the values are as expected
    assert NP.allclose(result, copol_advariants_on_list)

@pytest.mark.parametrize(
    "advariants, normaxis, normwts, normpower, expected_shape",
    [
        (NP.random.randn(3,4,5) + 1j*NP.random.randn(3,4,5), -1, None, 2, (3, 4, 10)),
        (NP.random.randn(3,4,5) + 1j*NP.random.randn(3,4,5), -1, NP.random.rand(3,4,5), 2, (3,4,10)),
        (NP.random.randn(3,4,5) + 1j*NP.random.randn(3,4,5), -1, NP.random.rand(3,4,5), 1, (3,4,10)),
        (NP.random.randn(3,4,5) + 1j*NP.random.randn(3,4,5), -1, NP.zeros((3,4,5)), 2, (3,4,10)),
    ]
)
def test_invariants_from_advariants_method1(advariants, normaxis, normwts, normpower, expected_shape):
    if normwts is not None and NP.all(normwts == 0):
        normwts[:,:,0] = 1  # For the specific test case where normwts is 1 for the first element and 0 for the rest

    invariants = SI.invariants_from_advariants_method1(advariants, normaxis, normwts=normwts, normpower=normpower)
    assert invariants.shape == expected_shape
    assert NP.all(NP.isfinite(invariants))

@pytest.mark.parametrize(
    "advariants, normaxis, normwts, normpower, exception_type",
    [
        ([1, 2, 3], 0, None, 2, TypeError),
        (NP.random.randn(3,4,5) + 1j * NP.random.randn(3,4,5), 1, [1,2,3], 2, TypeError),
        (NP.random.randn(3,4,5) + 1j * NP.random.randn(3,4,5), 1, NP.random.rand(3,4,5), '2', TypeError),
    ]
)
def test_invariants_from_advariants_method1_exceptions(advariants, normaxis, normwts, normpower, exception_type):
    with pytest.raises(exception_type):
        SI.invariants_from_advariants_method1(advariants, normaxis, normwts=normwts, normpower=normpower)

@pytest.mark.parametrize(
    "normwts, normpower",
    [
        (None, 2),
        (NP.random.rand(13,6), 2),
        (NP.random.rand(13,6), 1),
        (NP.zeros((13,6)), 2)
    ]
)
def test_scalar_invariance(copol_xc, copol_complex_gains, example_ids, baseid_ind, normwts, normpower):
    bl_axis = -1
    element_axis = -1
    element_pairs = [(example_ids[i], example_ids[j]) for i in range(len(example_ids)) for j in range(i + 1, len(example_ids))]
    triads_indep = GU.generate_triads(example_ids, baseid_ind)
    copol_xc_lol = SI.corrs_list_on_loops(copol_xc, element_pairs, triads_indep, bl_axis=bl_axis)
    advars_in = SI.advariants_multiple_loops(copol_xc_lol)

    normax = -1
    if normwts is not None and NP.all(normwts == 0):
        normwts[:,0] = 1  # For the specific test case where normwts is 1 for the first element and 0 for the rest
    ci_in = SI.invariants_from_advariants_method1(advars_in, normax, normwts=normwts, normpower=normpower)

    prefactor_gains = NP.take(copol_complex_gains, NP.array(element_pairs)[:,0], axis=element_axis) # A collection of g_a
    postfactor_gains = NP.take(copol_complex_gains, NP.array(element_pairs)[:,1], axis=element_axis) # A collection of g_b
    copol_xc_mod = SI.corrupt_visibilities(copol_xc, prefactor_gains, postfactor_gains)
    copol_xc_lol_mod = SI.corrs_list_on_loops(copol_xc_mod, element_pairs, triads_indep, bl_axis=bl_axis)
    advars_out = SI.advariants_multiple_loops(copol_xc_lol_mod)
    ci_out = SI.invariants_from_advariants_method1(advars_out, normax, normwts=normwts, normpower=normpower)

    assert NP.allclose(ci_in, ci_out)

    scale_factor = NP.abs(copol_complex_gains)**2

    assert NP.allclose(scale_factor[...,[baseid_ind]], advars_out/advars_in)

def test_point_scalar_invariance(copol_point_xc, copol_complex_gains, example_ids, baseid_ind):
    bl_axis = -1
    element_axis = -1
    element_pairs = [(example_ids[i], example_ids[j]) for i in range(len(example_ids)) for j in range(i + 1, len(example_ids))]
    triads_indep = GU.generate_triads(example_ids, baseid_ind)

    normax = -1
    normwts = 'max'
    normpower = 1

    prefactor_gains = NP.take(copol_complex_gains, NP.array(element_pairs)[:,0], axis=element_axis) # A collection of g_a
    postfactor_gains = NP.take(copol_complex_gains, NP.array(element_pairs)[:,1], axis=element_axis) # A collection of g_b
    copol_xc_mod = SI.corrupt_visibilities(copol_point_xc, prefactor_gains, postfactor_gains)
    copol_xc_lol_mod = SI.corrs_list_on_loops(copol_xc_mod, element_pairs, triads_indep, bl_axis=bl_axis)
    advars_in = SI.advariants_multiple_loops(copol_xc_lol_mod)
    ci_in = SI.invariants_from_advariants_method1(advars_in, normax, normwts=normwts, normpower=normpower)

    ci_out = NP.concatenate([NP.ones(advars_in.shape), NP.zeros(advars_in.shape)], axis=-1)

    assert ci_in.shape == ci_out.shape
    assert NP.allclose(ci_in, ci_out)

def test_point_closure_phases_from_advariants(copol_point_xc, copol_complex_gains, example_ids, baseid_ind):
    bl_axis = -1
    element_axis = -1
    element_pairs = [(example_ids[i], example_ids[j]) for i in range(len(example_ids)) for j in range(i + 1, len(example_ids))]
    triads_indep = GU.generate_triads(example_ids, baseid_ind)

    prefactor_gains = NP.take(copol_complex_gains, NP.array(element_pairs)[:,0], axis=element_axis) # A collection of g_a
    postfactor_gains = NP.take(copol_complex_gains, NP.array(element_pairs)[:,1], axis=element_axis) # A collection of g_b
    copol_xc_mod = SI.corrupt_visibilities(copol_point_xc, prefactor_gains, postfactor_gains)
    copol_xc_lol_mod = SI.corrs_list_on_loops(copol_xc_mod, element_pairs, triads_indep, bl_axis=bl_axis)
    advars_in = SI.advariants_multiple_loops(copol_xc_lol_mod)
    cp_in = SI.closurePhases_from_advariants(advars_in)

    cp_out = NP.zeros(advars_in.shape)

    assert NP.allclose(cp_in, cp_out)

@pytest.mark.parametrize(
    "normwts, normpower",
    [
        (None, 2),
        (NP.random.rand(3,3), 2),
        (NP.random.rand(3,3), 1),
        (NP.zeros((3,3)), 2)
    ]
)
def test_scalar_invariance_old(copol_corrs_list1, copol_corrs_list2, copol_corrs_list3, copol_complex_gains, normwts, normpower):
    corrs_in = [copol_corrs_list1, copol_corrs_list2, copol_corrs_list3]
    advars_in = SI.advariants_multiple_loops(corrs_in)
    normax = 0
    if normwts is not None and NP.all(normwts == 0):
        normwts[0,:] = 1  # For the specific test case where normwts is 1 for the first element and 0 for the rest
    ci_in = SI.invariants_from_advariants_method1(advars_in, normax, normwts=normwts, normpower=normpower)

    irun = 0
    preinds = NP.concatenate([NP.zeros((3,1), dtype=int), 1+NP.arange(6, dtype=int).reshape(3,2)], axis=-1)
    postinds = NP.roll(preinds, -1, axis=-1)
    corrs_out = [SI.corrupt_visibilities(NP.array(corrs_in[loopi]), copol_complex_gains[irun,preinds[loopi]][...,NP.newaxis], copol_complex_gains[irun,postinds[loopi]][...,NP.newaxis]) for loopi in range(len(corrs_in))]
    advars_out = SI.advariants_multiple_loops(corrs_out)
    ci_out = SI.invariants_from_advariants_method1(advars_out, normax, normwts=normwts, normpower=normpower)

    assert NP.allclose(ci_in, ci_out)