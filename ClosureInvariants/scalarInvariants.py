import numpy as NP
from typing import List, Tuple, Union

def corrupt_visibilities(vis: NP.ndarray, g_a: NP.ndarray, g_b: NP.ndarray) -> NP.ndarray:
    """
    Corrupt visibilities with complex gains, g_a as a pre-factor and g_b as a post-factor.

    Parameters:
    vis : numpy.ndarray
        The input array representing visibilities of shape (...,Nbl).
    g_a : numpy.ndarray
        The complex antenna gains as a pre-factor of shape (...,Nbl).
    g_b : numpy.ndarray
        The complex antenna gains as a post-factor of shape (...,Nbl).

    Returns:
    numpy.ndarray
        The corrupted visibilities.

    Raises:
    TypeError: If vis, g_a, or g_b is not a numpy array.
    ValueError: If inputs have incompatible dimensions or shapes.

    Examples:
    >>> import numpy as NP
    >>> vis = NP.array([1+2j, 3+4j, 5+6j, 7+8j])
    >>> g_a = NP.array([0.9+0.1j, 1.0, 1.1, 0.95+0.05j])
    >>> g_b = np.array([1+0.2j, 1.0, 1.2, 0.9+0.1j])
    >>> corrupt_visibilities(vis, g_a, g_b)
    array([0.7 + 1.76j, 3 + 4j, 6.6 + 7.92j, 4.83 + 6.53j])
    """
    if not isinstance(vis, NP.ndarray):
        raise TypeError('Input vis must be a numpy array')
    if not isinstance(g_a, NP.ndarray):
        raise TypeError('Input g_a must be a numpy array')
    if not isinstance(g_b, NP.ndarray):
        raise TypeError('Input g_b must be a numpy array')
    if vis.ndim != g_a.ndim:
        raise ValueError('Inputs vis and g_a must have same number of dimensions')
    if vis.ndim != g_b.ndim:
        raise ValueError('Inputs vis and g_b must have same number of dimensions')
    if g_a.ndim != g_b.ndim:
        raise ValueError('Inputs g_a and g_b must have same number of dimensions')
    return g_a * vis * g_b.conj()

def corrs_list_on_loops(corrs: NP.ndarray, 
                        ant_pairs: List[Union[Tuple[Union[int, str], Union[int, str]], 
                                              List[Union[int, str]]]], 
                        loops: List[Union[List[Union[int, str]], Tuple[Union[int, str], ...]]], 
                        bl_axis: int = -1) -> List[List[NP.ndarray]]:
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
        Axis corresponding to the number of antenna pairs in the correlations array. Default is -1.

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
    >>> corrs = np.random.randn(10)  # Example correlations array
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
                elif bl_ind.size == 1: # Take conjugate
                    corr = NP.take(corrs.conj(), bl_ind, axis=bl_axis)
                elif bl_ind.size > 1:
                    raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i+1)%loop.size]))
            elif bl_ind.size > 1:
                raise IndexError('{0:0d} indices found for antenna pair ({1:0d},{2:0d}) in input ant_pairs'.format(bl_ind, loop[i], loop[(i+1)%loop.size]))
        
            corr = NP.take(corr, 0, axis=bl_axis)
            corrs_loop += [corr]
        corrs_lol += [corrs_loop]
        
    return corrs_lol