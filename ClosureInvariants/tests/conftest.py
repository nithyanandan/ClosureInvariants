import pytest
import numpy as NP

@pytest.fixture
def example_ids():
    return [1, 2, 3, 4]

@pytest.fixture
def example_ids_strings():
    return ['A', 'B', 'C', 'D']

@pytest.fixture(params=[('scalar',), ('matrix',)])
def arrtype(request):
    return request.param[0]

@pytest.fixture
def realimag_scalar_array():
    return NP.array([1+2j, 3-4j])

@pytest.fixture
def realimag_matrix_array():
    return NP.array([[[1+2j, 3+4j], [5+6j, 7+8j]], [[9+10j, 11+12j], [13+14j, 15+16j]]])

@pytest.fixture
def real_scalar_array():
    return NP.array([[1, 2], [3, -4]]).ravel()

@pytest.fixture
def real_matrix_array():
    return NP.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]).ravel()

@pytest.fixture
def antenna_ids():
    return [0, 1, 2, 3]

@pytest.fixture
def antenna_pairs(antenna_ids):
    return [(antenna_ids[i], antenna_ids[j]) for i in range(len(antenna_ids)) for j in range(i + 1, len(antenna_ids))]

@pytest.fixture
def loops(antenna_ids):
    return [[antenna_ids[i], antenna_ids[j], antenna_ids[k]] for i in range(len(antenna_ids)) for j in range(i + 1, len(antenna_ids)) for k in range(j + 1, len(antenna_ids))]

@pytest.fixture
def polaxes():
    return (-2,-1)

@pytest.fixture
def pol_correlations_array(antenna_pairs):
    real = NP.arange(len(antenna_pairs)*2*2)
    imag = real[::-1]
    return (real+1j*imag).reshape(-1,2,2)

@pytest.fixture
def pol_corrs_list1():
    return [NP.array([[1+2j, 3+4j], 
                      [5+6j, 7+8j]]), 
            NP.array([[9+10j, 11+12j], 
                      [13+14j, 15+16j]]), 
            NP.array([[17+18j, 19+20j], 
                      [21+22j, 23+24j]])]

@pytest.fixture
def pol_corrs_list2():
    return [NP.array([[25+26j, 27+28j], 
                      [29+30j, 31+32j]]), 
            NP.array([[33+34j, 35+36j], 
                      [37+38j, 39+40j]]), 
            NP.array([[41+42j, 43+44j], 
                      [45+46j, 47+48j]])]

@pytest.fixture
def pol_corrs_list3():
    return [NP.array([[49+50j, 51+52j], 
                      [53+54j, 55+56j]]), 
            NP.array([[57+58j, 59+60j], 
                      [61+62j, 63+64j]]), 
            NP.array([[65+66j, 67+68j], 
                      [69+70j, 71+72j]])]

@pytest.fixture
def pol_advariant1(pol_corrs_list1):
    return (pol_corrs_list1[0]@NP.linalg.inv(pol_corrs_list1[1].T.conj())@pol_corrs_list1[2])[NP.newaxis,...]

@pytest.fixture
def pol_advariant2(pol_corrs_list2):
    return (pol_corrs_list2[0]@NP.linalg.inv(pol_corrs_list2[1].T.conj())@pol_corrs_list2[2])[NP.newaxis,...]

@pytest.fixture
def pol_advariant3(pol_corrs_list3):
    return (pol_corrs_list3[0]@NP.linalg.inv(pol_corrs_list3[1].T.conj())@pol_corrs_list3[2])[NP.newaxis,...]

@pytest.fixture
def pol_corrs_lol(pol_corrs_list1, pol_corrs_list2, pol_corrs_list3):
    return [pol_corrs_list1, pol_corrs_list2, pol_corrs_list3]

@pytest.fixture
def pol_advariants_on_list(pol_advariant1, pol_advariant2, pol_advariant3):
    return NP.concatenate([pol_advariant1, pol_advariant2, pol_advariant3], axis=0)[NP.newaxis,...]

@pytest.fixture
def vectors_from_pol_advariants():
    return NP.asarray([[[ -8.5 +5.j,  -8.5 +5.j, 3.5+3.5j, 4.5-4.5j],
                        [-44.5+41.j, -44.5+41.j, 3.5+3.5j, 4.5-4.5j],
                        [-80.5+77.j, -80.5+77.j, 3.5+3.5j, 4.5-4.5j]]])

@pytest.fixture
def minkowski_dot_products():
    mdp_22 = NP.array([[-32.5, -32.5, 8., 8., -32.5, 8., 8., -32.5, -32.5, -32.5]])
    mdp_21 = NP.array([[-32.5, 8., -32.5, 8., 8., -32.5, 8., -32.5]])
    return (mdp_22, mdp_21)

@pytest.fixture
def complete_minkowski_dots(minkowski_dot_products):
    mdp_22, mdp_21 = minkowski_dot_products
    return NP.concatenate([mdp_22, mdp_21], axis=-1)

@pytest.fixture
def minkoski_dots_scaling_factor_removed(complete_minkowski_dots):
    return complete_minkowski_dots / NP.sqrt(NP.sum(complete_minkowski_dots**2, axis=-1, keepdims=True))

@pytest.fixture
def pol_complex_gains():
    nants = 10
    mean_gain_scale = 3.0
    randseed = None
    rng = NP.random.default_rng(randseed)
    gains = rng.normal(loc=1.0, scale=NP.sqrt(0.5)/mean_gain_scale, size=(nants,2,2)).astype(NP.float64) + 1j * rng.normal(loc=1.0, scale=NP.sqrt(0.5)/mean_gain_scale, size=(nants,2,2)).astype(NP.float64) # shape is (...,n_antennas,2,2)
    return gains

@pytest.fixture
def copol_correlations_array(antenna_pairs):
    real = NP.arange(len(antenna_pairs))
    imag = real[::-1]
    return real+1j*imag

@pytest.fixture
def copol_corrs_list1():
    return [NP.array([1+2j, 3+4j, 5+6j]),
            NP.array([7+8j, 9+10j, 11+12j]),  
            NP.array([13+14j, 15+16j, 17+18j])]

@pytest.fixture
def copol_corrs_list2():
    return [NP.array([25+26j, 27+28j, 29+30j]),
            NP.array([31+32j, 33+34j, 35+36j]), 
            NP.array([37+38j, 39+40j, 41+42j])] 

@pytest.fixture
def copol_corrs_list3():
    return [NP.array([49+50j, 51+52j, 53+54j]),
            NP.array([55+56j, 57+58j, 59+60j]),  
            NP.array([61+62j, 63+64j, 65+66j])]

