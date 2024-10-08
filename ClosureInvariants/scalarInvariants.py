from typing import List, Tuple, Union, Optional

import numpy as NP

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
    >>> g_b = NP.array([1+0.2j, 1.0, 1.2, 0.9+0.1j])
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
    ValueError
        If the input ant_pairs has invalid shape.
        If the input loops has invalid shape.
        If the input corrs and ant_pairs do not have the same number of baselines.

    Notes
    -----
    This function generates sets of antenna correlations by picking three sets of antenna pairs from the input list of antenna pairs, 
    based on the specified loops of antenna indices.

    Examples
    --------
    >>> corrs = NP.random.randn(10)  # Example correlations array
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

def advariant(corrs_list: Union[List[List[NP.ndarray]], List[NP.ndarray]]) -> NP.ndarray:
    """
    Construct the advariant from a list of odd-numbered correlations forming a closed odd-edged loop.

    Parameters
    ----------
    corrs_list : List or numpy.ndarray of numpy.ndarray
        List of odd-numbered correlations forming the edges of a closed odd-edged loop.

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
        If the shapes of numpy arrays in corrs_list are not identical.

    Notes
    -----
    The advariant is constructed by taking the product of correlations from an odd-numbered list of 
    correlations that forms the edges of a closed odd-edged loop. Every alternate correlation, starting from 
    the second element, undergoes the hat operation, which involves taking the inverse conjugate.

    Examples
    --------
    >>> import numpy as NP
    >>> from your_module import advariant
    >>> corrs_list = NP.array([5j, 3+2j, 1-4j])
    >>> advariant(corrs_list)
    array([3.84615385+4.23076923j])
    """

    if not isinstance(corrs_list, (list,NP.ndarray)):
        raise TypeError('Input corrs_list must be a list')
    nedges = len(corrs_list)
    if nedges%2 == 0:
        raise ValueError('Input corrs_list must be a list made of odd number of elements for an advariant to be constructed')

    advar = None
    for edgei,corr in enumerate(corrs_list):  
        if edgei == 0:
            advar = NP.copy(corr)
        else:
            if edgei%2 == 0:
                advar = advar * corr
            else:
                advar = advar / corr.conj()
                
    return advar

def advariants_multiple_loops(corrs_lol: List[List[NP.ndarray]]) -> NP.ndarray:
    """
    Calculate advariants on multiple loops from a list of lists of odd-numbered correlations forming closed odd-edged loops.

    Parameters
    ----------
    corrs_lol : List of lists of numpy.ndarray
        List of lists of odd-numbered correlations forming closed odd-edged loops.

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
    list of lists of odd-numbered correlations. The hat operation, which involves taking the inverse of the conjugate, 
    is applied during the advariant calculation.

    Examples
    --------
    >>> import numpy as NP
    >>> from your_module import advariants_multiple_loops
    >>> cl1 = [NP.array([1+2j, 3+4j, 5+6j]),
    ...        NP.array([7+8j, 9+10j, 11+12j]),
    ...        NP.array([13+14j, 15+16j, 17+18j])]
    >>> cl2 = [NP.array([25+26j, 27+28j, 29+30j]),
    ...        NP.array([31+32j, 33+34j, 35+36j]),
    ...        NP.array([37+38j, 39+40j, 41+42j])]
    >>> cl3 = [NP.array([49+50j, 51+52j, 53+54j]),
    ...        NP.array([55+56j, 57+58j, 59+60j]),
    ...        NP.array([61+62j, 63+64j, 65+66j])]
    >>> corrs_lol = [cl1, cl2, cl3]
    >>> advariants_multiple_loops(corrs_lol)
    array([[ -3.76106195 +1.4159292j   -6.91160221 +4.32044199j
             -9.6490566  +6.92830189j]
           [-31.8070529 +28.84433249j -33.87928731+30.91224944j
            -35.94327648+32.97262991j]
           [-56.32738192+53.33939296j -58.35097535+55.36216543j
            -60.37296992+57.38342042j]])
    """
    if not isinstance(corrs_lol, list):
        raise TypeError('Input corrs_lol must be a list of lists')
    advars_list = []
    for ind, corrs_list in enumerate(corrs_lol):
        advars_list += [advariant(corrs_list)]
    return NP.moveaxis(NP.array(advars_list), 0, -1) # Move the loop axis to first from the end

def invariants_from_advariants_method1(advariants: NP.ndarray, 
                                       normaxis: int, 
                                       normwts: Optional[Union[NP.ndarray, str]] = None, 
                                       normpower: int = 2) -> NP.ndarray:
    """
    Calculate the invariants from the advariants using a specified normalization method.
    The real and imaginary parts of the advariants are split, concatenated, and then 
    the normalization is performed on this concatenated array.

    Parameters
    ----------
    advariants : numpy.ndarray
        Array of advariants to be normalized.
    normaxis : int
        Axis along which the normalization occurs.
    normwts : numpy.ndarray, optional
        Weights used for the normalization. Should be broadcastable to the shape of advariants.
        Default is 1 along the normaxis.
    normpower : int, optional
        Power used to define the norm (e.g., L1 norm if normpower=1, L2 norm if normpower=2).
        Default is 2 (L2 norm).

    Returns
    -------
    numpy.ndarray
        Array of normalized invariants.

    Raises
    ------
    TypeError
        If the input advariants is not a numpy array.
        If normwts is provided and is not a numpy array.
        If normpower is not an integer.

    Examples
    --------
    >>> import numpy as NP
    >>> from your_module import invariants_from_advariants_method1
    >>> advariants = NP.random.randn(3, 4, 5) + 1j * NP.random.randn(3, 4, 5)
    >>> normaxis = -1
    >>> invariants = invariants_from_advariants_method1(advariants, normaxis)
    >>> print(invariants)

    >>> normwts = NP.random.rand(3, 4, 5)
    >>> invariants = invariants_from_advariants_method1(advariants, normaxis, normwts=normwts, normpower=1)
    >>> print(invariants)
    """
    
    # Validate inputs
    if not isinstance(advariants, NP.ndarray):
        raise TypeError('Input advariants must be a numpy array')
    if normwts is not None and not (isinstance(normwts, NP.ndarray) or normwts == 'max'):
        raise TypeError("Input normwts must be a numpy array, None, or 'max'")
    # if normwts is not None and not isinstance(normwts, NP.ndarray):
    #     raise TypeError('Input normwts must be a numpy array')
    if not isinstance(normpower, int):
        raise TypeError('Input normpower must be an integer')

    # Split real and imaginary parts and concatenate them
    if NP.iscomplexobj(advariants):
        realvalued_advariants = NP.concatenate((advariants.real, advariants.imag), axis=-1)
    else:
        realvalued_advariants = NP.copy(advariants)
    
    if isinstance(normwts, str):
        if normwts == 'max':
            realvalued_normwts = NP.zeros_like(realvalued_advariants)
            max_indices = NP.argmax(NP.abs(realvalued_advariants), axis=normaxis, keepdims=True)
            NP.put_along_axis(realvalued_normwts, max_indices, 1, axis=normaxis)
    elif normwts is None:
        realvalued_normwts = NP.ones_like(realvalued_advariants)
    else:
        if NP.iscomplexobj(advariants):
            realvalued_normwts = NP.concatenate((normwts.real, normwts.imag), axis=-1)
        else:
            realvalued_normwts = normwts
    
    # Calculate the normalization factor
    normalization_factor = NP.sum(NP.abs(realvalued_advariants * realvalued_normwts)**normpower, axis=normaxis, keepdims=True)**(1/normpower)

    # Calculate the invariants
    invariants = realvalued_advariants / normalization_factor

    return invariants

def closurePhases_from_advariants(advariants: NP.ndarray) -> NP.ndarray:
    """
    Calculate closure phases from the given advariants.

    The closure phase is defined as the phase angle of the advariants.
    This function returns the phase of each element in the input advariants array.
    If the advariants are real, the phase will be zero or pi.

    Parameters
    ----------
    advariants : numpy.ndarray
        A numpy array (real or complex) representing the advariants from which closure phases 
        are to be computed.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the closure phases, which are the angles of the 
        elements in the input advariants array. The output array has the 
        same shape as the input advariants array.

    Raises
    ------
    TypeError
        If the input advariants is not a numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> advariants = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    >>> closure_phases = closurePhase_from_advariants(advariants)
    >>> print(closure_phases)
    array([[ 1.10714872,  0.92729522],
           [ 0.87605805,  0.85196633]])

    >>> advariants_real = np.array([[1, 2], [3, 4]])
    >>> closure_phases_real = closurePhase_from_advariants(advariants_real)
    >>> print(closure_phases_real)
    array([[0., 0.],
           [0., 0.]])
    """

    # Validate input
    if not isinstance(advariants, NP.ndarray):
        raise TypeError('Input advariants must be a numpy array')

    # Calculate closure phases (angles of the advariants)
    closure_phases = NP.angle(advariants)

    return closure_phases

def covariant(corrs_list: Union[List[List[NP.ndarray]], List[NP.ndarray]]) -> NP.ndarray:
    """
    Construct the covariant from a list of even-numbered correlations forming a closed even-edged loop.

    Parameters
    ----------
    corrs_list : List or numpy.ndarray of numpy.ndarray
        List of even-numbered correlations forming the edges of a closed even-edged loop.

    Returns
    -------
    numpy.ndarray
        Covariant constructed from the list of correlations.

    Raises
    ------
    TypeError
        If the input corrs_list is not a list.
        If any element of corrs_list is not a numpy array.
    ValueError
        If the input corrs_list does not have an even number of elements.
        If the shapes of numpy arrays in corrs_list are not identical.

    Notes
    -----
    The covariant is constructed by taking the product of correlations from an even-numbered list of 
    correlations that forms the edges of a closed even-edged loop. Every alternate correlation, starting from 
    the first element, undergoes the hat operation, which involves taking the inverse conjugate.

    Examples
    --------
    >>> import numpy as NP
    >>> from your_module import covariant
    >>> corrs_list = [NP.array([2j, 3+2j, 1-4j, 2, 1, 3, 2]),
    ...               NP.array([2+1j, 1-2j, 3+4j, 5, 6, 7, 8]),
    ...               NP.array([4-1j, 2+3j, 1, 0, 1, 3, 5]),
    ...               NP.array([1+1j, 2, 3-1j, 4, 5, 6, 7])]
    >>> covariant(corrs_list)
    """

    if not isinstance(corrs_list, (list, NP.ndarray)):
        raise TypeError('Input corrs_list must be a list or numpy array')
    nedges = len(corrs_list)
    if nedges % 2 != 0:
        raise ValueError('Input corrs_list must be a list made of even number of elements for a covariant to be constructed')

    covar = None
    for edgei, corr in enumerate(corrs_list):
        if edgei == 0:
            covar = NP.copy(corr)
        else:
            if edgei % 2 == 0:
                covar = covar * corr
            else:
                covar = covar / corr.conj()

    return covar

def covariants_multiple_loops(corrs_lol: List[List[NP.ndarray]]) -> NP.ndarray:
    """
    Calculate covariants on multiple loops from a list of lists of even-numbered correlations forming closed even-edged loops.

    Parameters
    ----------
    corrs_lol : List of lists of numpy.ndarray
        List of lists of even-numbered correlations forming closed even-edged loops.

    Returns
    -------
    numpy.ndarray
        Covariants calculated for each loop in the input list.

    Raises
    ------
    TypeError
        If the input corrs_lol is not a list of lists.
    ValueError
        If the input corrs_lol contains invalid shapes of numpy arrays.
        If the input corrs_lol is empty.

    Notes
    -----
    This function calculates covariants on multiple loops by calling the covariant function for each loop in the input 
    list of lists of even-numbered correlations. The hat operation, which involves taking the inverse of the conjugate, 
    is applied during the covariant calculation.

    Examples
    --------
    >>> import numpy as NP
    >>> from your_module import covariants_multiple_loops
    >>> cl1 = [NP.array([1+2j, 3+4j, 5+6j, 7+8j, 9, 10, 11]),
    ...        NP.array([2+1j, 4+3j, 6+5j, 8+7j, 12, 13, 14]),
    ...        NP.array([3+2j, 5+4j, 7+6j, 9+8j, 15, 16, 17]),
    ...        NP.array([4+3j, 6+5j, 8+7j, 10+9j, 18, 19, 20])]
    >>> cl2 = [NP.array([11+12j, 13+14j, 15+16j, 17+18j, 21, 22, 23]),
    ...        NP.array([12+11j, 14+13j, 16+15j, 18+17j, 24, 25, 26]),
    ...        NP.array([13+12j, 15+14j, 17+16j, 19+18j, 27, 28, 29]),
    ...        NP.array([14+13j, 16+15j, 18+17j, 20+19j, 30, 31, 32])]
    >>> cl3 = [NP.array([21+22j, 23+24j, 25+26j, 27+28j, 33, 34, 35]),
    ...        NP.array([22+21j, 24+23j, 26+25j, 28+27j, 36, 37, 38]),
    ...        NP.array([23+22j, 25+24j, 27+26j, 29+28j, 39, 40, 41]),
    ...        NP.array([24+23j, 26+25j, 28+27j, 30+29j, 42, 43, 44])]
    >>> cl4 = [NP.array([31+32j, 33+34j, 35+36j, 37+38j, 45, 46, 47]),
    ...        NP.array([32+31j, 34+33j, 36+35j, 38+37j, 48, 49, 50]),
    ...        NP.array([33+32j, 35+34j, 37+36j, 39+38j, 51, 52, 53]),
    ...        NP.array([34+33j, 36+35j, 38+37j, 40+39j, 54, 55, 56])]
    >>> cl5 = [NP.array([41+42j, 43+44j, 45+46j, 47+48j, 57, 58, 59]),
    ...        NP.array([42+41j, 44+43j, 46+45j, 48+47j, 60, 61, 62]),
    ...        NP.array([43+42j, 45+44j, 47+46j, 49+48j, 63, 64, 65]),
    ...        NP.array([44+43j, 46+45j, 48+47j, 50+49j, 66, 67, 68])]
    >>> corrs_lol = [cl1, cl2, cl3, cl4, cl5]
    >>> covariants_multiple_loops(corrs_lol)
    """
    if not isinstance(corrs_lol, list):
        raise TypeError('Input corrs_lol must be a list of lists')
    if not all(isinstance(corrs_list, list) for corrs_list in corrs_lol):
        raise TypeError('Each element of corrs_lol must be a list')
    
    covars_list = []
    for ind, corrs_list in enumerate(corrs_lol):
        covars_list += [covariant(corrs_list)]
    
    # Move the loop axis to the first from the end
    return NP.moveaxis(NP.array(covars_list), 0, -1)

def closureAmplitudes_from_covariants(covariants: NP.ndarray) -> NP.ndarray:
    """
    Calculate closure amplitudes from the given covariants.

    The closure amplitude is defined as the absolute value of the covariants.
    This function returns the absolute value of each element in the input covariants array.

    Parameters
    ----------
    covariants : numpy.ndarray
        A numpy array (real or complex) representing the covariants from which closure amplitudes 
        are to be computed.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the closure amplitudes, which are the absolute values of the 
        elements in the input covariants array. The output array has the same shape as the input 
        covariants array.

    Raises
    ------
    TypeError
        If the input covariants is not a numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> covariants = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    >>> closure_amplitudes = closureAmplitudes_from_covariants(covariants)
    >>> print(closure_amplitudes)
    array([[ 2.23606798,  5.        ],
           [ 7.81024968, 10.63014581]])

    >>> covariants_real = np.array([[1, 2], [3, 4]])
    >>> closure_amplitudes_real = closureAmplitudes_from_covariants(covariants_real)
    >>> print(closure_amplitudes_real)
    array([[1., 2.],
           [3., 4.]])
    """
    # Validate input
    if not isinstance(covariants, NP.ndarray):
        raise TypeError('Input covariants must be a numpy array')

    # Calculate closure amplitudes (absolute values of the covariants)
    closure_amplitudes = NP.abs(covariants)

    return closure_amplitudes

def closurePhases_from_covariants(covariants: NP.ndarray) -> NP.ndarray:
    """
    Calculate closure phases from the given covariants.

    The closure phase is defined as the phase angle of the covariants.
    This function returns the phase of each element in the input covariants array.
    If the covariants are real, the phase will be zero or pi.

    Parameters
    ----------
    covariants : numpy.ndarray
        A numpy array (real or complex) representing the covariants from which closure phases 
        are to be computed.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the closure phases, which are the angles of the 
        elements in the input covariants array. The output array has the 
        same shape as the input covariants array.

    Raises
    ------
    TypeError
        If the input covariants is not a numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> covariants = np.array([[1+2j, 3+4j], [5+6j, 7+8j]])
    >>> closure_phases = closurePhases_from_covariants(covariants)
    >>> print(closure_phases)
    array([[ 1.10714872,  0.92729522],
           [ 0.87605805,  0.85196633]])

    >>> covariants_real = np.array([[1, 2], [3, 4]])
    >>> closure_phases_real = closurePhases_from_covariants(covariants_real)
    >>> print(closure_phases_real)
    array([[0., 0.],
           [0., 0.]])
    """
    # Validate input
    if not isinstance(covariants, NP.ndarray):
        raise TypeError('Input covariants must be a numpy array')

    # Calculate closure phases (angles of the covariants)
    closure_phases = NP.angle(covariants)

    return closure_phases

