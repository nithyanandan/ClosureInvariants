import numpy as NP
from astroutils import mathops as MO
from typing import List, Tuple, Union

# def corrs_list_on_loops(corrs: NP.ndarray, 
#                         ant_pairs: Union[List[Tuple[int,int]], List[List[int,int]], 
#                                          List[Tuple[str,str]], List[List[str,str]]], 
#                         loops: Union[List[Union[List[Union[int, str]], Tuple[Union[int, str], ...]]], 
#                                      Tuple[Union[List[Union[int, str]], Tuple[Union[int, str], ...]], ...]], 
#                         bl_axis: int = -3, 
#                         pol_axes: Union[List[int], Tuple[int, int]] = (-2, -1)) -> List[List[NP.ndarray]]:
def corrs_list_on_loops(corrs: NP.ndarray, 
                        ant_pairs: List[Union[Tuple[Union[int, str], Union[int, str]], 
                                              List[Union[int, str]]]], 
                        loops: List[Union[List[Union[int, str]], Tuple[Union[int, str], ...]]], 
                        bl_axis: int = -3, 
                        pol_axes: Union[List[int], Tuple[int, int]] = (-2, -1)) -> List[List[NP.ndarray]]:
    """
    Generate triads of antenna correlations based on loops of antenna pairs.

    Parameters
    ----------
    corrs : numpy.ndarray
        Array of correlations associated with antenna pairs.
    ant_pairs : list of tuples or list of lists
        List of antenna pairs. Each pair is represented as a tuple or a list of two elements.
    loops : list of lists or tuple of lists/tuples
        List of loops of antenna indices. Each loop is represented as a list or a tuple of antenna indices.
    bl_axis : int, optional
        Axis corresponding to the number of antenna pairs in the correlations array. Default is -3.
    pol_axes : list or tuple of int, optional
        Axes corresponding to the polarization axes in the correlations array. Default is (-2, -1).

    Returns
    -------
    list of lists of numpy.ndarray
        List of 3-element tuples of correlations, one tuple for each loop.

    Raises
    ------
    TypeError
        If the input ant_pairs or loops are not of valid types.
        If the input pol_axes is not a list, tuple, or numpy array.
    ValueError
        If the input ant_pairs has invalid shape.
        If the input loops has invalid shape.
        If the input corrs and ant_pairs do not have the same number of baselines.
        If the input pol_axes does not have exactly two elements.

    Notes
    -----
    This function generates sets of antenna correlations by picking three sets of antenna pairs from the input list of antenna pairs, 
    based on the specified loops of antenna indices.

    Examples
    --------
    >>> corrs = np.random.randn(10, 2, 2)  # Example correlations array
    >>> ant_pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Example antenna pairs
    >>> loops = [[0, 1, 2], [2, 3, 4]]  # Example loops of antenna indices
    >>> corrs_list_on_loops(corrs, ant_pairs, loops)
    [[[correlation1], [correlation2], [correlation3]],
     [[correlation4], [correlation5], [correlation6]]]
    """

    if not isinstance(ant_pairs, (list,NP.ndarray)):
        raise TypeError('Input ant_pairs must be a list or numpy array')
    ant_pairs = NP.asarray(ant_pairs)
    if ant_pairs.ndim == 1:
        if ant_pairs.size != 2:
            raise ValueError('Input ant_pairs contains invalid shape')
    elif ant_pairs.ndim == 2:
        if ant_pairs.shape[-1] != 2:
            raise ValueError('Input ant_pairs contains invalid shape')
    else:
        raise ValueError('Input ant_pairs contains invalid shape')
    ant_pairs = NP.asarray(ant_pairs).reshape(-1,2)
    
    if not isinstance(bl_axis, int):
        raise TypeError('Input bl_axis must be an integer')
        
    if not isinstance(corrs, NP.ndarray):
        raise TypeError('Input corrs must be a numpy array')
    if corrs.shape[bl_axis] != ant_pairs.shape[0]:
        raise ValueError('Input corrs and ant_pairs do not have same number of baselines')
    
    if not isinstance(loops, (list,NP.ndarray)):
        raise TypeError('Input loops must be a list or numpy array')
    loops = NP.asarray(loops)
    if loops.ndim == 1:
        loops = loops.reshape(1,-1)
    elif loops.ndim != 2:
        raise ValueError('Input loops contains invalid shape')

    if not isinstance(pol_axes, (list,tuple,NP.ndarray)):
        raise TypeError('Input pol_axes must be a list, tuple, or numpy array')
    if len(pol_axes) != 2:
        raise ValueError('Input pol_axes must be a two-element sequence')
    else:
        for pax in pol_axes:
            if not isinstance(pax, int):
                raise TypeError('Input pol_axes must be a two-element sequence of integers')
    pol_axes = NP.array(pol_axes).ravel()
    tmpind = NP.where(pol_axes < 0)[0]
    if tmpind.size > 0:
        pol_axes[tmpind] += corrs.ndim # Convert to a positive value for the polarization axes
        
    corrs_lol = []
    for loopi, loop in enumerate(loops):
        corrs_loop = []
        for i in range(len(loop)):
            bl_ind = NP.where((ant_pairs[:,0] == loop[i]) & (ant_pairs[:,1]==loop[(i+1)%loop.size]))[0]
            if bl_ind.size == 1:
                corr = NP.copy(NP.take(corrs, bl_ind, axis=bl_axis))
            elif bl_ind.size == 0: # Check for reversed pair
                bl_ind = NP.where((ant_pairs[:,0] == loop[(i+1)%loop.size]) & (ant_pairs[:,1]==loop[i]))[0]
                if bl_ind.size == 0:
                    raise IndexError('Specified antenna pair ({0:0d},{1:0d}) not found in input ant_pairs'.format(loop[i], loop[(i+1)%loop.size]))
                elif bl_ind.size == 1: # Take Hermitian
                    corr = MO.hermitian(NP.take(corrs, bl_ind, axis=bl_axis), axes=pol_axes)
                elif bl_ind.size > 1:
                    raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i+1)%loop.size]))
            elif bl_ind.size > 1:
                raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i+1)%loop.size]))
        
            corr = NP.take(corr, 0, axis=bl_axis)
            corrs_loop += [corr]
        corrs_lol += [corrs_loop]
        
    return corrs_lol

def advariant(corrs_list: List[NP.ndarray], pol_axes: Union[List[int], tuple] = (-2, -1)) -> NP.ndarray:
    """
    Construct the advariant from a list of odd-numbered correlations forming a closed odd-edged loop.

    Parameters
    ----------
    corrs_list : List of numpy.ndarray
        List of odd-numbered correlations forming the edges of a closed odd-edged loop.
    pol_axes : List or tuple of int, optional
        Axes on which the hat operation is applied. Default is (-2, -1).

    Returns
    -------
    numpy.ndarray
        Advariant constructed from the list of correlations.

    Raises
    ------
    TypeError
        If the input corrs_list is not a list.
        If any element of corrs_list is not a numpy array.
    ValueError
        If the input corrs_list does not have an odd number of elements.
        If the last two dimensions of each numpy array in corrs_list are not equal.
        If the last two dimensions of each numpy array in corrs_list are not (1, 1) or (2, 2).
        If the shapes of numpy arrays in corrs_list are not identical.

    Notes
    -----
    The advariant is constructed by taking the matrix product of correlations from an odd-numbered list of 
    correlations that forms the edges of a closed odd-edged loop. Every alternate correlation, starting from 
    the second element, undergoes the hat operation, which includes taking the Hermitian conjugate and finding 
    the matrix inverse along the specified polarization axes.

    Examples
    --------
    >>> import numpy as np
    >>> from your_module import advariant
    >>> corrs_list = [
    ...     np.array([[1+2j, 3+4j], [5+6j, 7+8j]]),
    ...     np.array([[9+10j, 11+12j], [13+14j, 15+16j]]),
    ...     np.array([[17+18j, 19+20j], [21+22j, 23+24j]])
    ... ]
    >>> advariant(corrs_list)
    array([[ -4.+0.5j  -5.+1.5j]
           [-12.+8.5j -13.+9.5j]])
    """
    print(type(corrs_list))
    if not isinstance(corrs_list, list):
        raise TypeError('Input corrs_list must be a list')
    nedges = len(corrs_list)
    if nedges%2 == 0:
        raise ValueError('Input corrs_list must be a list made of odd number of elements for an advariant to be constructed')

    if not isinstance(pol_axes, (list,tuple,NP.ndarray)):
        raise TypeError('Input pol_axes must be a list, tuple, or numpy array')
    if len(pol_axes) != 2:
        raise ValueError('Input pol_axes must be a two-element sequence')
    else:
        for pax in pol_axes:
            if not isinstance(pax, int):
                raise TypeError('Input pol_axes must be a two-element sequence of integers')
    pol_axes = NP.array(pol_axes).ravel()
    
    advar = None
    matgrp = None
    for edgei,corr in enumerate(corrs_list):  
        if not isinstance(corr, NP.ndarray):
            raise TypeError('Element {0:0d} of corrs_list must be a numpy array'.format(edgei))

        if edgei == 0:
            tmpind = NP.where(pol_axes < 0)[0]
            if tmpind.size > 0:
                pol_axes[tmpind] += corr.ndim # Convert to a positive value for the polarization axes

            expected_pol_axes = corr.ndim-2 + NP.arange(2) # For inverse, they have to be the last two axes
            if not NP.array_equal(pol_axes, expected_pol_axes):
                raise ValueError('For advariant calculation, pol_axes must be the last two axes because of inherent assumptions about the axes over which matrix multiplication is performed')
        inv_axes = NP.copy(pol_axes)
        if corr.ndim == 1:
            corr = corr[...,NP.newaxis,NP.newaxis] # shape=(...,n=1,n=1)
        elif corr.ndim >= 2:            
            shape2 = NP.asarray(corr.shape[-2:])
            if (shape2[0] != shape2[1]):
                raise ValueError('The last two dimensions of each numpy array that forms the items in the input corrs_list must be equal')
            elif (shape2[0] != 1) and (shape2[0] != 2):
                raise ValueError('The last two dimensions of each numpy array that forms the items in the input corrs_list must be (1,1) or (2,2) for GL(1,C) and GL(2,C) matrices, respctively')
            if corr.ndim == 2:
                corr = corr[NP.newaxis,:,:] # shape=(ncorr=1,n=(1/2),n=(1/2))
                inv_axes += 1
        if edgei == 0:
            matgrp = NP.copy(shape2[0])
            advar = NP.copy(corr)
        else:
            if not NP.array_equal(shape2[0], matgrp):
                raise ValueError('Shape of list element {0:0d} not indentical to that of list element 0'.format(edgei))
            if edgei%2 == 0:
                advar = advar@corr
            else:
                advar = advar @ MO.hat(corr, axes=inv_axes)
                
    return advar

def advariants_multiple_loops(corrs_lol: List[List[NP.ndarray]], pol_axes: Union[List[int], tuple] = (-2, -1)) -> NP.ndarray:
    """
    Calculate advariants on multiple loops from a list of lists of odd-numbered correlations forming closed odd-edged loops.

    Parameters
    ----------
    corrs_lol : List of lists of numpy.ndarray
        List of lists of odd-numbered correlations forming closed odd-edged loops.
    pol_axes : List or tuple of int, optional
        Axes on which the hat operation is applied. Default is (-2, -1).

    Returns
    -------
    numpy.ndarray
        Advariants calculated for each loop in the input list.

    Raises
    ------
    TypeError
        If the input corrs_lol is not a list of lists.
    ValueError
        If the input corrs_lol contains invalid shapes of numpy arrays.
        If the input corrs_lol is empty.

    Notes
    -----
    This function calculates advariants on multiple loops by calling the advariant function for each loop in the input 
    list of lists of odd-numbered correlations. The hat operation, which includes taking the Hermitian conjugate and 
    finding the matrix inverse along the specified polarization axes, is applied during the advariant calculation.

    Examples
    --------
    >>> import numpy as np
    >>> from your_module import advariants_multiple_loops
    >>> corrs_lol = [
    ...     [np.array([[1+2j, 3+4j], [5+6j, 7+8j]]), np.array([[9+10j, 11+12j], [13+14j, 15+16j]]), np.array([[17+18j, 19+20j], [21+22j, 23+24j]])],
    ...     [np.array([[25+26j, 27+28j], [29+30j, 31+32j]]), np.array([[33+34j, 35+36j], [37+38j, 39+40j]]), np.array([[41+42j, 43+44j], [45+46j, 47+48j]])]
    ... ]
    >>> advariants_multiple_loops(corrs_lol)
    array([[[[ -4. +0.5j  -5. +1.5j]
             [-12. +8.5j -13. +9.5j]]
            [[-40.+36.5j -41.+37.5j]
             [-48.+44.5j -49.+45.5j]]]])
    """
    if not isinstance(corrs_lol, list):
        raise TypeError('Input corrs_lol must be a list of lists')
    advars_list = []
    for ind, corrs_list in enumerate(corrs_lol):
        advars_list += [advariant(corrs_list, pol_axes=pol_axes)]
    return NP.moveaxis(NP.array(advars_list), 0, -3) # Move the loop axis to third from the end 

def vector_from_advariant(advars):
    """
    Generate 4-vectors from advariants.

    This function takes advariants, which are either 2x2 or 1x1 complex-valued matrices, and generates 4-vectors.

    Parameters:
    advars (numpy.ndarray): A numpy array representing advariants. It can be a 2D array with shape (...,2,2) for GL(2,C) matrices or a 2D array with shape (...,1,1) for GL(1,C) matrices.

    Returns:
    numpy.ndarray: A numpy array representing 4-vectors derived from the advariants. For GL(2,C) matrices, the output has shape (...,4), where the last axis contains the components of the 4-vectors. For GL(1,C) matrices, the input advariants are returned as they already represent 4-vectors.

    Examples:
    >>> import numpy as NP
    >>> advars1 = NP.array([[1+0j, 2+1j], [2-1j, 3+0j]])  # GL(2,C) matrix
    >>> vector_from_advariant(advars1)
    array([[ 2.+0.j, 2.+0.j, -1.+0.j, -1.+0.j]])

    >>> advars2 = NP.array([[4+3j]])  # GL(1,C) matrix
    >>> vector_from_advariant(advars2)
    array([[4+3j]])
    """
    if not isinstance(advars, NP.ndarray):
        raise TypeError('Input advars must be a numpy array')
    shape2 = advars.shape[-2:]
    matgrp = NP.copy(shape2[0])
    
    if (advars.shape[-1] != 2) and (advars.shape[-1] != 1):
        raise ValueError('Input advariant shape incompatible with GL(1,C) or GL(2,C) matrices')
    elif advars.shape[-1] == 2:
        if advars.shape[-1] != advars.shape[-2]:
            raise ValueError('Advariants must be 2x2 matrices in the last two dimensions')
        # Determine Pauli matrices
        pauli_sig_0 = NP.identity(matgrp, dtype=complex).reshape(tuple(NP.ones(advars.ndim-2, dtype=int))+tuple(shape2)) # Psigma_0: shape=(...,n=(1/2),n=(1/2))
        pauli_sig_1 = NP.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex).reshape(tuple(NP.ones(advars.ndim-2, dtype=int))+tuple(shape2)) # Psigma_1: shape=(...,n=(1/2),n=(1/2))
        pauli_sig_2 = NP.asarray([[0.0, -1j], [1j, 0.0]], dtype=complex).reshape(tuple(NP.ones(advars.ndim-2, dtype=int))+tuple(shape2)) # Psigma_2: shape=(...,n=(1/2),n=(1/2))
        pauli_sig_3 = NP.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex).reshape(tuple(NP.ones(advars.ndim-2, dtype=int))+tuple(shape2)) # Psigma_: shape=(...,n=(1/2),n=(1/2))

        # Determine components of 4-vectors
        z0 = 0.5 * NP.trace(advars@pauli_sig_0, axis1=-2, axis2=-1)
        z1 = 0.5 * NP.trace(advars@pauli_sig_1, axis1=-2, axis2=-1)
        z2 = 0.5 * NP.trace(advars@pauli_sig_2, axis1=-2, axis2=-1)
        z3 = 0.5 * NP.trace(advars@pauli_sig_3, axis1=-2, axis2=-1)

        z_4vect = NP.concatenate([z0[...,NP.newaxis], z1[...,NP.newaxis], z2[...,NP.newaxis], z3[...,NP.newaxis]], axis=-1) # shape=(...,4)
        return z_4vect
    else:
        if advars.shape[-1] != advars.shape[-2]:
            raise ValueError('Advariants must be 1x1 matrices in the last two dimensions')
        return advars # GL(1,C) already    