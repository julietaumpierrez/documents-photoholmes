# %%
from typing import Any, Tuple

import numpy as np
from numpy.typing import ArrayLike


# %%
def baseN_to_base10(t: Tuple, N: int) -> int:
    tN = sum([N**i * x for i, x in enumerate(t)])
    return tN


# %%
T = 2
n = 2 * T - 1
max_val = n - 1

tuples: list[np.ndarray] = []

tuple_codes = list()
for x0 in range(n):
    for x1 in range(n):
        for x2 in range(n):
            for x3 in range(n):
                x = baseN_to_base10((x0, x1, x2, x3), n)
                x_c = baseN_to_base10(
                    (max_val - x0, max_val - x1, max_val - x2, max_val - x3), n
                )
                x_r = baseN_to_base10((x3, x2, x1, x0), n)
                x_cr = baseN_to_base10(
                    (max_val - x3, max_val - x2, max_val - x1, max_val - x0), n
                )

                if (
                    x in tuple_codes
                    or x_c in tuple_codes
                    or x_r in tuple_codes
                    or x_cr in tuple_codes
                ):
                    continue
                else:
                    tuples.append(np.array((x0, x1, x2, x3)))
                    tuple_codes.append(x)
# %%
coh, cov = np.zeros((2, len(tuples)))
# %%
