import numpy as np
from scipy.special import binom

def bernstein_coeff_ordern_new(n, tmin, tmax, t_actual):
    l = tmax - tmin
    t = (t_actual - tmin) / l  # Normalizing the time to [0, 1]

    # Bernstein polynomial coefficients
    P = np.array([binom(n, i) * (1 - t)**(n - i) * t**i for i in range(n + 1)]).squeeze().T

    # First derivative of the Bernstein polynomial (Pdot)
    Pdot = np.array([
        n * (binom(n - 1, i - 1) * (1 - t)**(n - i) * t**(i - 1) if i > 0 else 0) -
        n * (binom(n - 1, i) * (1 - t)**(n - i - 1) * t**i if i < n else 0)
        for i in range(n + 1)
    ]).squeeze().T / l

    # Second derivative of the Bernstein polynomial (Pddot)
    Pddot = np.array([
        n * (n - 1) * (
            (binom(n - 2, i - 2) * (1 - t)**(n - i) * t**(i - 2) if i > 1 else 0) -
            2 * (binom(n - 2, i - 1) * (1 - t)**(n - i - 1) * t**(i - 1) if 0 < i < n else 0) +
            (binom(n - 2, i) * (1 - t)**(n - i - 2) * t**i if i < n - 1 else 0)
        )
        for i in range(n + 1)
    ]).squeeze().T / (l**2)

    return P, Pdot, Pddot
