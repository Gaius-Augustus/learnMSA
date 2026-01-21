from collections.abc import Sequence
from typing import Annotated

import numpy as np
from pydantic import BeforeValidator, PlainSerializer


def nd_array_before_validator(x):
    """Custom before validation logic for numpy arrays."""
    if isinstance(x, str):
        import ast
        x_list = ast.literal_eval(x)
        x = np.array(x_list)
    if isinstance(x, list):
        x = np.array(x)
    return x


def nd_array_serializer(x):
    """Custom serialization logic for numpy arrays."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


NPArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_before_validator),
    PlainSerializer(nd_array_serializer, return_type=list),
]


def get_value(param, head: int, index: int | None = None) -> float:
    """Get the value of a parameter for a specific head and index.

    Args:
        param: The parameter which can be a float, a sequence of floats,
               a sequence of sequences of floats, or a numpy array.
        head: The head index.
        index: The index within the head, if applicable.
    """
    if isinstance(param, float):
        return param
    elif isinstance(param, np.ndarray):
        if param.ndim == 0:
            return float(param.item())
        elif param.ndim == 1:
            return float(param[head])
        elif param.ndim == 2:
            if index is None:
                raise ValueError("Index must be provided for 2D arrays.")
            return float(param[head][index])
        else:
            raise ValueError(
                "Numpy arrays with more than 2 dimensions are not supported."
            )
    elif isinstance(param, Sequence) and not isinstance(param, str):
        if all(isinstance(x, float) for x in param):
            return param[head]
        elif (
            all(
                isinstance(x, Sequence) and not isinstance(x, str)
                for x in param
            )
            and all(all(isinstance(y, float) for y in x) for x in param)
        ):
            if index is None:
                raise ValueError("Index must be provided for nested sequences.")
            return param[head][index]
    raise ValueError("Invalid parameter type.")


def get_emission_dist(
    param: Sequence[float] | Sequence[Sequence[float]] |
          Sequence[Sequence[Sequence[float]]] | NPArray | None,
    head: int,
    index: int | None = None,
    default: Sequence[float] | NPArray | None = None
) -> Sequence[float] | NPArray:
    """Get the emission distribution for a specific head and optional index.

    Args:
        param: The emission parameter which can be:
            - None: use default
            - Sequence[float]: same distribution for all heads/positions
            - Sequence[Sequence[float]]: head-specific, position-independent
            - Sequence[Sequence[Sequence[float]]]: fully specified
        head: The head index.
        index: The match state index within the head, if applicable.
        default: Default distribution to use if param is None.

    Returns:
        A sequence of floats representing the emission distribution.
    """
    if param is None:
        if default is None:
            raise ValueError("No emission distribution provided and no default.")
        return default

    # Case 1: Single distribution for all
    if all(isinstance(x, (float, int)) for x in param):
        # Type assertion: at this point we know param is Sequence[float]
        return param  # type: ignore[return-value]

    # Case 2 or 3: Nested sequences
    inner = param[head]  # type: ignore[index]

    # Case 2: Head-specific, position-independent
    if all(isinstance(x, (float, int)) for x in inner):  # type: ignore[arg-type]
        return inner  # type: ignore[return-value]

    # Case 3: Fully specified - need index
    if index is None:
        raise ValueError(
            "Index must be provided for position-specific emissions."
        )
    return inner[index]  # type: ignore[index, return-value]
