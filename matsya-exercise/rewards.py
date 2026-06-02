import numpy as np
from HARK.metric import MetricObject
import functools


def utility_fix(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if np.ndim(args[0]) == 0:
            if args[0] < 0.0:
                return np.nan
            else:
                return func(*[np.array([args[0]])] + list(args[1:]), **kwargs)[0]
        else:
            out = func(*args, **kwargs)
            neg = args[0] < 0.0
            out[neg] = np.nan
            return out

    return wrapper

# ==============================================================================
# ============== Log-CRRA Labor utility functions ==============================
# ==============================================================================
#
# u(c, h) = log(c) + psi * (1 - h)^(1 - theta) / (1 - theta)
#
# c  : consumption (must be > 0)
# h  : labor supply, fraction of time working, h in [0, 1)
# 1-h: leisure
# psi: weight on the leisure term (must be > 0)
# theta: curvature of the leisure term (inverse Frisch elasticity)
#         When theta = 1, the leisure term collapses to psi * log(1 - h).
#
# The utility is separable: the consumption term uses log utility (CRRA
# with rho = 1), and the leisure term is CRRA utility over leisure (1 - h)
# with curvature theta.  Separability means each partial derivative and its
# inverse are independent of the other argument -- a property exploited in
# the EGM solver's first-order conditions.
# ==============================================================================

@utility_fix
def LogCRRALaborutility(c, h, psi, theta):
    """
    Evaluates the log-CRRA labor utility function.

        u(c, h) = log(c) + psi * (1 - h)^(1 - theta) / (1 - theta)

    The special case theta = 1 returns log(c) + psi * log(1 - h).

    Parameters
    ----------
    c : float or np.ndarray
        Consumption. Must be strictly positive.
    h : float or np.ndarray
        Labor supply (fraction of time worked). Must lie in [0, 1).
    psi : float
        Weight on the leisure term. Must be positive.
    theta : float
        Curvature of the leisure term (inverse Frisch elasticity of labor
        supply). Higher values make labor supply less elastic to wages.

    Returns
    -------
    float or np.ndarray
        Utility value.

    Tests
    -----
    >>> float(LogCRRALaborutility(1.0, 0.0, 1.0, 2.0))  # log(1) + 1*(1)^(-1)/(-1) = -1.0
    -1.0
    >>> import numpy as np; bool(np.isclose(LogCRRALaborutility(np.e, 0.0, 1.0, 1.0), 1.0))
    True
    """
    if theta == 1:
        return np.log(c) + psi * np.log(1.0 - h)
    return np.log(c) + psi * (1.0 - h) ** (1.0 - theta) / (1.0 - theta)


def LogCRRALaborutilityPc(c, h, psi, theta):
    """
    Marginal utility of consumption: du/dc = 1/c.

    Because the utility is separable, this does not depend on h, psi, or theta.

    Parameters
    ----------
    c : float or np.ndarray
        Consumption. Must be strictly positive.
    h : float or np.ndarray
        Labor supply. Unused; included for a consistent call signature.
    psi : float
        Weight on the leisure term. Unused.
    theta : float
        Curvature of the leisure term. Unused.

    Returns
    -------
    float or np.ndarray
        Marginal utility of consumption (always positive).
    """
    return 1.0 / c


def LogCRRALaborutilityPh(c, h, psi, theta):
    """
    Marginal utility of labor: du/dh = -psi * (1 - h)^(-theta).

    Always negative: more labor means less leisure, which lowers utility.
    Because the utility is separable, this does not depend on c.

    Parameters
    ----------
    c : float or np.ndarray
        Consumption. Unused; included for a consistent call signature.
    h : float or np.ndarray
        Labor supply. Must lie in [0, 1).
    psi : float
        Weight on the leisure term.
    theta : float
        Curvature of the leisure term.

    Returns
    -------
    float or np.ndarray
        Marginal utility of labor (always negative for psi, theta > 0).
    """
    return -psi * (1.0 - h) ** (-theta)


def LogCRRALaborutilityPcc(c, h, psi, theta):
    """
    Second derivative of utility w.r.t. consumption: d^2u/dc^2 = -1/c^2.

    Parameters
    ----------
    c : float or np.ndarray
        Consumption. Must be strictly positive.
    h : float or np.ndarray
        Labor supply. Unused.
    psi : float
        Weight on the leisure term. Unused.
    theta : float
        Curvature of the leisure term. Unused.

    Returns
    -------
    float or np.ndarray
        Second derivative of utility w.r.t. consumption (always negative).
    """
    return -(c ** (-2.0))


def LogCRRALaborutilityPhh(c, h, psi, theta):
    """
    Second derivative of utility w.r.t. labor: d^2u/dh^2 = -psi*theta*(1-h)^(-theta-1).

    Always negative for psi, theta > 0 (strict concavity in h), which
    guarantees an interior optimum exists whenever the budget constraint allows.

    Parameters
    ----------
    c : float or np.ndarray
        Consumption. Unused.
    h : float or np.ndarray
        Labor supply. Must lie in [0, 1).
    psi : float
        Weight on the leisure term.
    theta : float
        Curvature of the leisure term.

    Returns
    -------
    float or np.ndarray
        Second derivative of utility w.r.t. labor (always negative).
    """
    return -psi * theta * (1.0 - h) ** (-theta - 1.0)


def LogCRRALaborutilityPc_inv(uc, psi, theta):
    """
    Inverse of the marginal utility of consumption.

    Solves 1/c = uc for c, yielding c = 1/uc.

    Because the utility is separable, this does not require knowing h.

    Parameters
    ----------
    uc : float or np.ndarray
        Marginal utility of consumption. Must be strictly positive.
    psi : float
        Weight on the leisure term. Unused.
    theta : float
        Curvature of the leisure term. Unused.

    Returns
    -------
    float or np.ndarray
        Consumption level corresponding to the given marginal utility.
    """
    return 1.0 / uc


def LogCRRALaborutilityPh_inv(uh, psi, theta):
    """
    Inverse of the marginal utility of labor.

    Solves -psi*(1-h)^(-theta) = uh for h, yielding:

        h = 1 - (-uh / psi)^(-1/theta)

    Because uh = du/dh < 0 (for any interior h), the term (-uh/psi) is always
    strictly positive, so the power is well-defined. Because the utility is
    separable, this does not require knowing c.

    Parameters
    ----------
    uh : float or np.ndarray
        Marginal utility of labor. Must be strictly negative.
    psi : float
        Weight on the leisure term. Must be positive.
    theta : float
        Curvature of the leisure term. Must be positive.

    Returns
    -------
    float or np.ndarray
        Labor supply level h in [0, 1) corresponding to the given marginal utility.
    """
    return 1.0 - (-uh / psi) ** (-1.0 / theta)


def LogCRRALaborutilityPc_invP(uc, psi, theta):
    """
    Derivative of the inverse of the marginal utility of consumption.

    d/d(uc) [1/uc] = -uc^(-2).

    This is used in EGM procedures that require the Jacobian of the
    inverse-MU mapping (e.g. when constructing MargValueFuncCRRA analogs).

    Parameters
    ----------
    uc : float or np.ndarray
        Marginal utility of consumption. Must be strictly positive.
    psi : float
        Weight on the leisure term. Unused.
    theta : float
        Curvature of the leisure term. Unused.

    Returns
    -------
    float or np.ndarray
        Derivative of the inverse MU_c function (always negative).
    """
    return -(uc ** (-2.0))


def LogCRRALaborutilityPh_invP(uh, psi, theta):
    """
    Derivative of the inverse of the marginal utility of labor.

    Starting from h(uh) = 1 - (-uh/psi)^(-1/theta), differentiating gives:

        dh/d(uh) = -(1 / (theta * psi)) * (-uh / psi)^(-1/theta - 1)

    Parameters
    ----------
    uh : float or np.ndarray
        Marginal utility of labor. Must be strictly negative.
    psi : float
        Weight on the leisure term. Must be positive.
    theta : float
        Curvature of the leisure term. Must be positive.

    Returns
    -------
    float or np.ndarray
        Derivative of the inverse MU_h function (always positive, since
        a less negative uh -- higher wage shadow value -- implies more labor).
    """
    return -(1.0 / (theta * psi)) * (-uh / psi) ** (-1.0 / theta - 1.0)


###############################################################################


class UtilityFunction(MetricObject):
    distance_criteria = ["eval_func"]

    def __init__(self, eval_func, der_func=None, inv_func=None):
        self.eval_func = eval_func
        self.der_func = der_func
        self.inv_func = inv_func

    def __call__(self, *args, **kwargs):
        return self.eval_func(*args, **kwargs)

    def derivative(self, *args, **kwargs):
        if self.der_func is None:
            raise NotImplementedError("No derivative function available")
        return self.der_func(*args, **kwargs)

    def inverse(self, *args, **kwargs):
        if self.inv_func is None:
            raise NotImplementedError("No inverse function available")
        return self.inv_func(*args, **kwargs)

    def der(self, *args, **kwargs):
        return self.derivative(*args, **kwargs)

    def inv(self, *args, **kwargs):
        return self.inverse(*args, **kwargs)


class UtilityFuncLogCRRALabor(UtilityFunction):
    """
    A class for the separable log-CRRA labor utility function:

        u(c, h) = log(c) + psi * (1 - h)^(1 - theta) / (1 - theta)

    where c is consumption, h is labor supply (fraction of time worked),
    and (1 - h) is leisure.

    The utility is separable: the consumption part is log utility (CRRA with
    rho = 1), and the leisure part is CRRA utility over leisure (1 - h) with
    curvature theta.  Separability means each partial derivative and its
    inverse can be evaluated independently of the other argument.

    Parameters
    ----------
    psi : float
        Weight on the leisure term. Controls the importance of leisure
        relative to consumption. Must be positive.
    theta : float
        Curvature of the leisure term (inverse Frisch elasticity of labor
        supply). Higher values make labor supply less responsive to wages.
        When theta = 1, the leisure term becomes psi * log(1 - h).
    """

    distance_criteria = ["psi", "theta"]

    def __init__(self, psi, theta):
        self.psi = psi
        self.theta = theta

    def __call__(self, c, h):
        """
        Evaluate u(c, h) = log(c) + psi * (1 - h)^(1 - theta) / (1 - theta).

        Parameters
        ----------
        c : float or np.ndarray
            Consumption. Must be strictly positive.
        h : float or np.ndarray
            Labor supply. Must lie in [0, 1).

        Returns
        -------
        float or np.ndarray
            Utility.
        """
        return LogCRRALaborutility(c, h, self.psi, self.theta)

    def derivative(self, c, h, order=1, axis=0):
        """
        Partial derivative of u(c, h) with respect to c (axis=0) or h (axis=1).

            axis=0:  du/dc = 1/c
            axis=1:  du/dh = -psi * (1 - h)^(-theta)

        Parameters
        ----------
        c : float or np.ndarray
            Consumption. Must be strictly positive.
        h : float or np.ndarray
            Labor supply. Must lie in [0, 1).
        axis : int
            0 for the partial w.r.t. c, 1 for the partial w.r.t. h.

        Returns
        -------
        float or np.ndarray
            Partial derivative.

        Raises
        ------
        ValueError
            If axis is not 0 or 1.
        """
        if order == 1:
            if axis == 0:
                return LogCRRALaborutilityPc(c, h, self.psi, self.theta)
            elif axis == 1:
                return LogCRRALaborutilityPh(c, h, self.psi, self.theta)
            else:
                raise ValueError(f"axis must be 0 (du/dc) or 1 (du/dh), got {axis}")
        elif order == 2:
            if axis == 0:
                return LogCRRALaborutilityPcc(c, h, self.psi, self.theta)
            elif axis == 1:
                return LogCRRALaborutilityPhh(c, h, self.psi, self.theta)
            else:
                raise ValueError(f"axis must be 0 (d^2u/dc^2) or 1 (d^2u/dh^2), got {axis}")
        else:
            raise ValueError(f"Derivative of order {order} not supported")

    def inverse(self, v, axis=0, order=(1, 0)):
        """
        Inverse (and related operations) for the marginal utility of c or h.

        Mirrors the ``order`` convention used by ``UtilityFuncCRRA.inverse``:
        the first element of ``order`` counts how many times the utility was
        differentiated before inverting ("P"s before ``_inv``), and the second
        element counts how many times the inverse is differentiated afterward
        ("P"s after ``_inv``).  The ``axis`` argument then selects which
        argument's first-order condition to work with.

        Supported combinations:

            order=(1, 0), axis=0  →  (du/dc)^{-1}:   c = 1/uc
            order=(1, 0), axis=1  →  (du/dh)^{-1}:   h = 1 - (-uh/psi)^{-1/theta}
            order=(1, 1), axis=0  →  d/d(uc) [1/uc]  =  -uc^{-2}
            order=(1, 1), axis=1  →  d/d(uh) [h(uh)] =  -(1/(theta*psi))*(-uh/psi)^{-1/theta - 1}

        Because the utility is separable, neither inverse requires the other
        argument (c is not needed to recover h, and vice versa).

        Parameters
        ----------
        v : float or np.ndarray
            Value at which to evaluate the (derivative of the) inverse.
            For axis=0: marginal utility of consumption (must be > 0).
            For axis=1: marginal utility of labor (must be < 0).
        axis : int
            0 to operate on the consumption first-order condition.
            1 to operate on the labor first-order condition.
        order : tuple of int, optional
            A 2-tuple following the ``UtilityFuncCRRA`` convention.
            Default is (1, 0), i.e. the plain inverse of the marginal utility.

        Returns
        -------
        float or np.ndarray
            Result corresponding to the requested ``order`` and ``axis``.

        Raises
        ------
        ValueError
            If ``order`` or ``axis`` is not supported.
        """
        if order == (1, 0):
            if axis == 0:
                return LogCRRALaborutilityPc_inv(v, self.psi, self.theta)
            elif axis == 1:
                return LogCRRALaborutilityPh_inv(v, self.psi, self.theta)
            else:
                raise ValueError(f"axis must be 0 or 1, got {axis}")
        elif order == (1, 1):
            if axis == 0:
                return LogCRRALaborutilityPc_invP(v, self.psi, self.theta)
            elif axis == 1:
                return LogCRRALaborutilityPh_invP(v, self.psi, self.theta)
            else:
                raise ValueError(f"axis must be 0 or 1, got {axis}")
        else:
            raise ValueError(f"order {order} is not supported; use (1,0) or (1,1)")

    def derinv(self, v, axis=0, order=(1, 0)):
        """
        Alias for ``self.inverse``. See ``inverse`` for full documentation.

        The default ``order=(1, 0)`` returns the plain inverse of the marginal
        utility, matching the convention of ``UtilityFuncCRRA.derinv``.
        """
        return self.inverse(v, axis=axis, order=order)
