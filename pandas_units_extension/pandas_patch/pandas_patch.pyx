
cimport numpy as cnp

def item_from_zerodim_new(val: object) -> object:
    """
    If the value is a zerodim ndarray (NOT subclass), return the item it contains.

    Parameters
    ----------
    val : object

    Returns
    -------
    object

    Examples
    --------
    >>> item_from_zerodim(1)
    1
    >>> item_from_zerodim('foobar')
    'foobar'
    >>> item_from_zerodim(np.array(1))
    1
    >>> item_from_zerodim(np.array([1]))
    array([1])
    """
    if cnp.PyArray_IsZeroDim(val) and cnp.PyArray_CheckExact(val):
        return cnp.PyArray_ToScalar(cnp.PyArray_DATA(val), val)
    return val