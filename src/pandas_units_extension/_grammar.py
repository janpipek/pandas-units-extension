"""
Parsing of the ``astropy`` dtype-string grammar.

The grammar is::

    astropy{<selector>}[<type-specific params>]

where both the ``{selector}`` and the ``[params]`` parts are optional:

- ``astropy``                 -> bare umbrella, type inferred from the data
- ``astropy{quantity}``       -> concrete type, no parameters
- ``astropy{quantity}[km/s]`` -> concrete type with parameters
- ``astropy[km/s]``           -> type inferred from the parameters

This module is intentionally free of any pandas or astropy imports so that the
parser stays a pure, cheaply-testable function.
"""

import re
from typing import NamedTuple

# ``astropy`` followed by an optional ``{selector}`` and an optional ``[params]``.
# The selector must be non-empty; the params may be empty (``astropy{quantity}[]``).
_PATTERN: re.Pattern[str] = re.compile(
    r"^astropy(?:\{(?P<selector>[^}]+)\})?(?:\[(?P<params>.*)\])?$"
)


class ParsedDtype(NamedTuple):
    """
    The result of parsing an ``astropy`` dtype string.

    Attributes
    ----------
    selector : str or None
        The type selector inside the braces (e.g. ``"quantity"``), or ``None``
        for the bare ``astropy`` string.
    params : str or None
        The type-specific parameters inside the square brackets (e.g.
        ``"km/s"``), or ``None`` when no brackets were given. May be the empty
        string for ``astropy{quantity}[]``.

    Examples
    --------
    >>> from pandas_units_extension._grammar import parse_dtype_string
    >>> parsed = parse_dtype_string("astropy{quantity}[km/s]")
    >>> parsed.selector
    'quantity'
    >>> parsed.params
    'km/s'
    """

    selector: str | None
    params: str | None


def parse_dtype_string(string: str) -> ParsedDtype:
    """
    Parse an ``astropy`` dtype string into its selector and parameters.

    Parameters
    ----------
    string : str
        The dtype string to parse.

    Returns
    -------
    ParsedDtype
        The parsed selector and parameters.

    Raises
    ------
    TypeError
        If the string is not a valid ``astropy`` dtype string, i.e. anything not
        starting with ``astropy`` (so that pandas' dtype registry falls through
        to other dtypes).

    Notes
    -----
    A string with parameters but no selector (``astropy[km/s]``) is valid; the
    concrete type is inferred from the parameters by the caller.

    Examples
    --------
    >>> parse_dtype_string("astropy{quantity}[km/s]")
    ParsedDtype(selector='quantity', params='km/s')
    >>> parse_dtype_string("astropy[km/s]")
    ParsedDtype(selector=None, params='km/s')
    >>> parse_dtype_string("astropy")
    ParsedDtype(selector=None, params=None)
    >>> parse_dtype_string("float64")
    Traceback (most recent call last):
        ...
    TypeError: Cannot parse 'float64' as an astropy dtype string.
    """
    if not isinstance(string, str):
        raise TypeError(f"'parse_dtype_string' expects a string, got {type(string)}")

    match: re.Match[str] | None = _PATTERN.match(string)
    if match is None:
        raise TypeError(f"Cannot parse '{string}' as an astropy dtype string.")

    return ParsedDtype(selector=match["selector"], params=match["params"])
