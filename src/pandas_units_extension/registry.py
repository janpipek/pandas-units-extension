"""
Registry mapping astropy types to their pandas extension implementations.

A single :class:`AstropyTypeSpec` describes one astropy-backed extension type
(e.g. a :class:`~astropy.units.Quantity`) and ties together the selector
keyword used in the dtype grammar, the native astropy type, and the concrete
pandas dtype/array/accessor classes. The registry powers three surfaces:

- dtype-string parsing (``astropy{quantity}[...]`` -> the concrete dtype),
- native-object ingest (:func:`~pandas_units_extension.base.from_astropy`),
- accessor dispatch (``series.astropy`` -> the right accessor subclass).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pandas.api.extensions import ExtensionArray

    from .base import AstropyDtype, AstropyExtensionArray


@dataclass(frozen=True, slots=True)
class AstropyTypeSpec:
    """
    Description of one astropy-backed extension type.

    Attributes
    ----------
    selector : str
        The keyword used in the dtype grammar, e.g. ``"quantity"``.
    native_type : type
        The native astropy type this spec handles, e.g. ``astropy.units.Quantity``.
    dtype_cls : type
        The concrete :class:`AstropyDtype` subclass, e.g. ``QuantityDtype``.
    array_cls : type
        The concrete :class:`AstropyExtensionArray` subclass, e.g.
        ``QuantityExtensionArray``.
    series_accessor_cls : type or None
        The concrete ``AstropySeriesAccessor`` subclass surfaced as
        ``series.astropy`` for this type, or ``None`` if the type only supports
        the common accessor methods.
    parse_params : Callable[[str | None], AstropyDtype]
        Build a concrete dtype instance from the ``[params]`` part of a dtype
        string (``None`` when no brackets were given), e.g.
        ``"km/s" -> QuantityDtype(u.Unit("km/s"))``.

    Examples
    --------
    >>> import astropy.units as u
    >>> from pandas_units_extension.registry import get_by_selector
    >>> spec = get_by_selector("quantity")
    >>> spec.selector
    'quantity'
    >>> spec.native_type is u.Quantity
    True
    >>> spec.array_cls.__name__
    'QuantityExtensionArray'
    """

    selector: str
    native_type: type
    dtype_cls: type["AstropyDtype"]
    array_cls: type["AstropyExtensionArray"]
    series_accessor_cls: type | None
    parse_params: Callable[[str | None], "AstropyDtype"]


# Registration order is preserved so that inference is deterministic.
_SPECS: list[AstropyTypeSpec] = []
_BY_SELECTOR: dict[str, AstropyTypeSpec] = {}


def register_astropy_type(spec: AstropyTypeSpec) -> None:
    """
    Register an astropy-backed extension type.

    Parameters
    ----------
    spec : AstropyTypeSpec
        The specification to register.

    Raises
    ------
    ValueError
        If a spec with the same selector is already registered.
    """
    if spec.selector in _BY_SELECTOR:
        raise ValueError(
            f"An astropy type with selector '{spec.selector}' is already registered."
        )
    _SPECS.append(spec)
    _BY_SELECTOR[spec.selector] = spec


def get_by_selector(selector: str) -> AstropyTypeSpec:
    """
    Look up a spec by its selector keyword.

    Parameters
    ----------
    selector : str
        The selector keyword, e.g. ``"quantity"``.

    Returns
    -------
    AstropyTypeSpec

    Raises
    ------
    KeyError
        If no spec with the given selector is registered.

    Examples
    --------
    >>> from pandas_units_extension.registry import get_by_selector
    >>> get_by_selector("quantity").selector
    'quantity'
    """
    return _BY_SELECTOR[selector]


def infer_from_object(obj: object) -> AstropyTypeSpec | None:
    """
    Find the spec whose native type matches a native astropy object.

    Parameters
    ----------
    obj : object
        A native astropy object (e.g. a ``Quantity``).

    Returns
    -------
    AstropyTypeSpec or None
        The first registered spec whose ``native_type`` is a superclass of
        ``type(obj)``, or ``None`` if none match.

    Examples
    --------
    >>> import astropy.units as u
    >>> from pandas_units_extension.registry import infer_from_object
    >>> infer_from_object(1 * u.m).selector
    'quantity'
    >>> infer_from_object(42) is None
    True
    """
    return next((spec for spec in _SPECS if isinstance(obj, spec.native_type)), None)


def infer_from_params(params: str) -> "AstropyDtype | None":
    """
    Build a concrete dtype from the ``[params]`` of a selector-less dtype string.

    Used for the ``astropy[<params>]`` form: the concrete type is inferred by
    trying each registered type's ``parse_params`` in registration order and
    returning the dtype produced by the first one that accepts ``params``.

    Parameters
    ----------
    params : str
        The parameter string from inside the square brackets, e.g. ``"km/s"``.

    Returns
    -------
    AstropyDtype or None
        The concrete dtype from the first type that parses ``params``, or
        ``None`` if no registered type accepts them.

    Examples
    --------
    >>> from pandas_units_extension.registry import infer_from_params
    >>> infer_from_params("km/s")
    QuantityDtype("km / s")
    >>> infer_from_params("definitely not a unit") is None
    True
    """
    for spec in _SPECS:
        try:
            return spec.parse_params(params)
        except (ValueError, TypeError):
            continue
    return None


def infer_from_array(arr: "ExtensionArray") -> AstropyTypeSpec | None:
    """
    Find the spec whose array class matches an extension array.

    Parameters
    ----------
    arr : ExtensionArray
        A pandas extension array.

    Returns
    -------
    AstropyTypeSpec or None
        The first registered spec whose ``array_cls`` ``arr`` is an instance of,
        or ``None`` if none match.

    Examples
    --------
    >>> import astropy.units as u
    >>> from pandas_units_extension import QuantityExtensionArray
    >>> from pandas_units_extension.registry import infer_from_array
    >>> infer_from_array(QuantityExtensionArray([1, 2, 3], u.m)).selector
    'quantity'
    """
    return next((spec for spec in _SPECS if isinstance(arr, spec.array_cls)), None)
