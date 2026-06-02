"""
Subclasses of AgentType representing consumers who make decisions about how much
labor to supply, as well as a consumption-saving decision.

It currently only has
one model: labor supply on the intensive margin (unit interval) with CRRA utility
from a composite good (of consumption and leisure), with transitory and permanent
productivity shocks.  Agents choose their quantities of labor and consumption after
observing both of these shocks, so the transitory shock is a state variable.
"""

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
    get_TranShkGrid_from_TranShkDstn,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)
from HARK.interpolation import (
    BilinearInterp,
    ConstantFunction,
    LinearInterp,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    ValueFuncCRRA,
    VariableLowerBoundFunc2D,
)
from HARK.distributions import (
    DiscreteDistribution,
    MarkovProcess,
    Uniform,
    make_tauchen_ar1,
)
from HARK.metric import MetricObject
from HARK.rewards import CRRAutilityP, CRRAutilityP_inv
from rewards import LogCRRALaborutilityPc_inv
from HARK.utilities import make_assets_grid

plt.ion()


class ConsumerLaborSolution(MetricObject):
    """
    A class for representing one period of the solution to a Consumer Labor problem.

    Parameters
    ----------
    cFunc : function
        The consumption function for this period, defined over normalized
        bank balances and the transitory productivity shock: cNrm = cFunc(bNrm,TranShk).
    LbrFunc : function
        The labor supply function for this period, defined over normalized
        bank balances: Lbr = LbrFunc(bNrm,TranShk).
    vFunc : function
        The beginning-of-period value function for this period, defined over
        normalized bank balances: v = vFunc(bNrm,TranShk).
    vPfunc : function
        The beginning-of-period marginal value (of bank balances) function for
        this period, defined over normalized bank balances: vP = vPfunc(bNrm,TranShk).
    bNrmMin: float
        The minimum allowable bank balances for this period, as a function of
        the transitory shock. cFunc, LbrFunc, etc are undefined for bNrm < bNrmMin(TranShk).
    """

    distance_criteria = ["cFunc", "LbrFunc"]

    def __init__(self, cFunc=None, LbrFunc=None, vFunc=None, vPfunc=None, bNrmMin=None):
        if cFunc is not None:
            self.cFunc = cFunc
        if LbrFunc is not None:
            self.LbrFunc = LbrFunc
        if vFunc is not None:
            self.vFunc = vFunc
        if vPfunc is not None:
            self.vPfunc = vPfunc
        if bNrmMin is not None:
            self.bNrmMin = bNrmMin


def make_log_polynomial_LbrCost(T_cycle, LbrCostCoeffs):
    r"""
    Construct the age-varying cost of working LbrCost using polynomial coefficients
    (over t_cycle) for (log) LbrCost.

    .. math::
        \text{LbrCost}_{t}=\exp(\sum \text{LbrCostCoeffs}_n t^{n})

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal period's in the agent's problem.
    LbrCostCoeffs : [float]
        List or array of arbitrary length, representing polynomial coefficients
        of t = 0,...,T_cycle, which determine (log) LbrCost.

    Returns
    -------
    LbrCost : [float]
        List of age-dependent labor utility cost parameters.
    """
    N = len(LbrCostCoeffs)
    age_vec = np.arange(T_cycle)
    LbrCostBase = np.zeros(T_cycle)
    for n in range(N):
        LbrCostBase += LbrCostCoeffs[n] * age_vec**n
    LbrCost = np.exp(LbrCostBase).tolist()
    return LbrCost


###############################################################################


def make_labor_intmarg_solution_terminal(
    CRRA, aXtraGrid, LbrCost, WageRte, TranShkGrid
):
    """
    Constructs the terminal period solution and solves for optimal consumption
    and labor when there is no future.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    t = -1
    TranShkGrid_T = TranShkGrid[t]
    LbrCost_T = LbrCost[t]
    WageRte_T = WageRte[t]

    # Add a point at b_t = 0 to make sure that bNrmGrid goes down to 0
    bNrmGrid = np.insert(aXtraGrid, 0, 0.0)
    bNrmCount = bNrmGrid.size
    TranShkCount = TranShkGrid_T.size

    # Replicated bNrmGrid for each transitory shock theta_t
    bNrmGridTerm = np.tile(np.reshape(bNrmGrid, (bNrmCount, 1)), (1, TranShkCount))
    TranShkGridTerm = np.tile(TranShkGrid_T, (bNrmCount, 1))
    # Tile the grid of transitory shocks for the terminal solution.

    # Array of labor (leisure) values for terminal solution
    LsrTerm = np.minimum(
        (LbrCost_T / (1.0 + LbrCost_T))
        * (bNrmGridTerm / (WageRte_T * TranShkGridTerm) + 1.0),
        1.0,
    )
    LsrTerm[0, 0] = 1.0
    LbrTerm = 1.0 - LsrTerm

    # Calculate market resources in terminal period, which is consumption
    mNrmTerm = bNrmGridTerm + LbrTerm * WageRte_T * TranShkGridTerm
    cNrmTerm = mNrmTerm  # Consume everything we have

    # Make a bilinear interpolation to represent the labor and consumption functions
    LbrFunc_terminal = BilinearInterp(LbrTerm, bNrmGrid, TranShkGrid_T)
    cFunc_terminal = BilinearInterp(cNrmTerm, bNrmGrid, TranShkGrid_T)

    # Compute the effective consumption value using consumption value and labor value at the terminal solution
    xEffTerm = LsrTerm**LbrCost_T * cNrmTerm
    vNvrsFunc_terminal = BilinearInterp(xEffTerm, bNrmGrid, TranShkGrid_T)
    vFunc_terminal = ValueFuncCRRA(vNvrsFunc_terminal, CRRA)

    # Using the envelope condition at the terminal solution to estimate the marginal value function
    vPterm = LsrTerm**LbrCost_T * CRRAutilityP(xEffTerm, rho=CRRA)
    vPnvrsTerm = CRRAutilityP_inv(vPterm, rho=CRRA)
    # Evaluate the inverse of the CRRA marginal utility function at a given marginal value, vP

    # Get the Marginal Value function
    vPnvrsFunc_terminal = BilinearInterp(vPnvrsTerm, bNrmGrid, TranShkGrid_T)
    vPfunc_terminal = MargValueFuncCRRA(vPnvrsFunc_terminal, CRRA)

    # Trivial function that return the same real output for any input
    bNrmMin_terminal = ConstantFunction(0.0)

    # Make and return the terminal period solution
    solution_terminal = ConsumerLaborSolution(
        cFunc=cFunc_terminal,
        LbrFunc=LbrFunc_terminal,
        vFunc=vFunc_terminal,
        vPfunc=vPfunc_terminal,
        bNrmMin=bNrmMin_terminal,
    )
    return solution_terminal


def solve_ConsLaborIntMarg(
    solution_next,
    PermShkDstn,
    TranShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    TranShkGrid,
    vFuncBool,
    CubicBool,
    WageRte,
    LbrCost,
):
    """
    Solves one period of the consumption-saving model with endogenous labor supply
    on the intensive margin by using the endogenous grid method to invert the first
    order conditions for optimal composite consumption and between consumption and
    leisure, obviating any search for optimal controls.

    Parameters
    ----------
    solution_next : ConsumerLaborSolution
        The solution to the next period's problem; must have the attributes
        vPfunc and bNrmMinFunc representing marginal value of bank balances and
        minimum (normalized) bank balances as a function of the transitory shock.
    PermShkDstn: [np.array]
        Discrete distribution of permanent productivity shocks.
    TranShkDstn: [np.array]
        Discrete distribution of transitory productivity shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion over the composite good.
    Rfree : float
        Risk free interest rate on assets retained at the end of the period.
    PermGroFac : float
        Expected permanent income growth factor for next period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  Currently not handled, must be None.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    TranShkGrid: np.array
        Grid of transitory shock values to use as a state grid for interpolation.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.  Not yet handled, must be False.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.
        Cubic interpolation is not yet handled, must be False.
    WageRte: float
        Wage rate per unit of labor supplied.
    LbrCost: float
        Cost parameter for supplying labor: :math:`u_t = U(x_t)`, :math:`x_t = c_t z_t^{LbrCost}`,
        where :math:`z_t` is leisure :math:`= 1 - Lbr_t`.

    Returns
    -------
    solution_now : ConsumerLaborSolution
        The solution to this period's problem, including a consumption function
        cFunc, a labor supply function LbrFunc, and a marginal value function vPfunc;
        each are defined over normalized bank balances and transitory prod shock.
        Also includes bNrmMinNow, the minimum permissible bank balances as a function
        of the transitory productivity shock.
    """
    # Make sure the inputs for this period are valid: CRRA > LbrCost/(1+LbrCost)
    # and CubicBool = False. CRRA condition is met automatically when CRRA >= 1.
    frac = 1.0 / (1.0 + LbrCost)
    if CRRA <= frac * LbrCost:
        raise ValueError("CRRA must be strictly greater than alpha/(1+alpha).")
    if BoroCnstArt is not None:
        raise ValueError("Model cannot handle artificial borrowing constraint yet.")
    if CubicBool is True:
        raise ValueError("Model cannot handle cubic interpolation yet.")
    if vFuncBool is True:
        raise ValueError("Model cannot compute the value function yet.")

    # Unpack next period's solution and the productivity shock distribution, and define the inverse (marginal) utilty function
    vPfunc_next = solution_next.vPfunc
    TranShkPrbs = TranShkDstn.pmv
    TranShkVals = TranShkDstn.atoms.flatten()
    PermShkPrbs = PermShkDstn.pmv
    PermShkVals = PermShkDstn.atoms.flatten()
    TranShkCount = TranShkPrbs.size
    PermShkCount = PermShkPrbs.size

    def uPinv(X):
        return CRRAutilityP_inv(X, rho=CRRA)

    # Make tiled versions of the grid of a_t values and the components of the shock distribution
    aXtraCount = aXtraGrid.size
    bNrmGrid = aXtraGrid  # Next period's bank balances before labor income

    # Replicated axtraGrid of b_t values (bNowGrid) for each transitory (productivity) shock
    bNrmGrid_rep = np.tile(np.reshape(bNrmGrid, (aXtraCount, 1)), (1, TranShkCount))

    # Replicated transitory shock values for each a_t state
    TranShkVals_rep = np.tile(
        np.reshape(TranShkVals, (1, TranShkCount)), (aXtraCount, 1)
    )

    # Replicated transitory shock probabilities for each a_t state
    TranShkPrbs_rep = np.tile(
        np.reshape(TranShkPrbs, (1, TranShkCount)), (aXtraCount, 1)
    )

    # Construct a function that gives marginal value of next period's bank balances *just before* the transitory shock arrives
    # Next period's marginal value at every transitory shock and every bank balances gridpoint
    vPNext = vPfunc_next(bNrmGrid_rep, TranShkVals_rep)

    # Integrate out the transitory shocks (in TranShkVals direction) to get expected vP just before the transitory shock
    vPbarNext = np.sum(vPNext * TranShkPrbs_rep, axis=1)

    # Transformed marginal value through the inverse marginal utility function to "decurve" it
    vPbarNvrsNext = uPinv(vPbarNext)

    # Linear interpolation over b_{t+1}, adding a point at minimal value of b = 0.
    vPbarNvrsFuncNext = LinearInterp(
        np.insert(bNrmGrid, 0, 0.0), np.insert(vPbarNvrsNext, 0, 0.0)
    )

    # "Recurve" the intermediate marginal value function through the marginal utility function
    vPbarFuncNext = MargValueFuncCRRA(vPbarNvrsFuncNext, CRRA)

    # Get next period's bank balances at each permanent shock from each end-of-period asset values
    # Replicated grid of a_t values for each permanent (productivity) shock
    aNrmGrid_rep = np.tile(np.reshape(aXtraGrid, (aXtraCount, 1)), (1, PermShkCount))

    # Replicated permanent shock values for each a_t value
    PermShkVals_rep = np.tile(
        np.reshape(PermShkVals, (1, PermShkCount)), (aXtraCount, 1)
    )

    # Replicated permanent shock probabilities for each a_t value
    PermShkPrbs_rep = np.tile(
        np.reshape(PermShkPrbs, (1, PermShkCount)), (aXtraCount, 1)
    )
    bNrmNext = (Rfree / (PermGroFac * PermShkVals_rep)) * aNrmGrid_rep

    # Calculate marginal value of end-of-period assets at each a_t gridpoint
    # Get marginal value of bank balances next period at each shock
    vPbarNext = (PermGroFac * PermShkVals_rep) ** (-CRRA) * vPbarFuncNext(bNrmNext)

    # Take expectation across permanent income shocks
    EndOfPrdvP = (
        DiscFac
        * Rfree
        * LivPrb
        * np.sum(vPbarNext * PermShkPrbs_rep, axis=1, keepdims=True)
    )

    # Compute scaling factor for each transitory shock
    TranShkScaleFac_temp = (
        frac
        * (WageRte * TranShkGrid) ** (LbrCost * frac)
        * (LbrCost ** (-LbrCost * frac) + LbrCost**frac)
    )

    # Flip it to be a row vector
    TranShkScaleFac = np.reshape(TranShkScaleFac_temp, (1, TranShkGrid.size))

    # Use the first order condition to compute an array of "composite good" x_t values corresponding to (a_t,theta_t) values
    xNow = (np.dot(EndOfPrdvP, TranShkScaleFac)) ** (-1.0 / (CRRA - LbrCost * frac))

    # Transform the composite good x_t values into consumption c_t and leisure z_t values
    TranShkGrid_rep = np.tile(
        np.reshape(TranShkGrid, (1, TranShkGrid.size)), (aXtraCount, 1)
    )
    xNowPow = xNow**frac  # Will use this object multiple times in math below

    # Find optimal consumption from optimal composite good
    cNrmNow = (((WageRte * TranShkGrid_rep) / LbrCost) ** (LbrCost * frac)) * xNowPow

    # Find optimal leisure from optimal composite good
    LsrNow = (LbrCost / (WageRte * TranShkGrid_rep)) ** frac * xNowPow

    # The zero-th transitory shock is TranShk=0, and the solution is to not work: Lsr = 1, Lbr = 0.
    cNrmNow[:, 0] = uPinv(EndOfPrdvP.flatten())
    LsrNow[:, 0] = 1.0

    # Agent cannot choose to work a negative amount of time. When this occurs, set
    # leisure to one and recompute consumption using simplified first order condition.
    # Find where labor would be negative if unconstrained
    violates_labor_constraint = LsrNow > 1.0
    EndOfPrdvP_temp = np.tile(
        np.reshape(EndOfPrdvP, (aXtraCount, 1)), (1, TranShkCount)
    )
    cNrmNow[violates_labor_constraint] = uPinv(
        EndOfPrdvP_temp[violates_labor_constraint]
    )
    LsrNow[violates_labor_constraint] = 1.0  # Set up z=1, upper limit

    # Calculate the endogenous bNrm states by inverting the within-period transition
    aNrmNow_rep = np.tile(np.reshape(aXtraGrid, (aXtraCount, 1)), (1, TranShkGrid.size))
    bNrmNow = (
        aNrmNow_rep
        - WageRte * TranShkGrid_rep
        + cNrmNow
        + WageRte * TranShkGrid_rep * LsrNow
    )

    # Add an extra gridpoint at the absolute minimal valid value for b_t for each TranShk;
    # this corresponds to working 100% of the time and consuming nothing.
    bNowArray = np.concatenate(
        (np.reshape(-WageRte * TranShkGrid, (1, TranShkGrid.size)), bNrmNow), axis=0
    )
    # Consume nothing
    cNowArray = np.concatenate((np.zeros((1, TranShkGrid.size)), cNrmNow), axis=0)
    # And no leisure!
    LsrNowArray = np.concatenate((np.zeros((1, TranShkGrid.size)), LsrNow), axis=0)
    LsrNowArray[0, 0] = 1.0  # Don't work at all if TranShk=0, even if bNrm=0
    LbrNowArray = 1.0 - LsrNowArray  # Labor is the complement of leisure

    # Get (pseudo-inverse) marginal value of bank balances using end of period
    # marginal value of assets (envelope condition), adding a column of zeros
    # zeros on the left edge, representing the limit at the minimum value of b_t.
    vPnvrsNowArray = np.concatenate(
        (np.zeros((1, TranShkGrid.size)), uPinv(EndOfPrdvP_temp))
    )

    # Construct consumption and marginal value functions for this period
    bNrmMinNow = LinearInterp(TranShkGrid, bNowArray[0, :])

    # Loop over each transitory shock and make a linear interpolation to get lists
    # of optimal consumption, labor and (pseudo-inverse) marginal value by TranShk
    cFuncNow_list = []
    LbrFuncNow_list = []
    vPnvrsFuncNow_list = []
    for j in range(TranShkGrid.size):
        # Adjust bNrmNow for this transitory shock, so bNrmNow_temp[0] = 0
        bNrmNow_temp = bNowArray[:, j] - bNowArray[0, j]

        # Make consumption function for this transitory shock
        cFuncNow_list.append(LinearInterp(bNrmNow_temp, cNowArray[:, j]))

        # Make labor function for this transitory shock
        LbrFuncNow_list.append(LinearInterp(bNrmNow_temp, LbrNowArray[:, j]))

        # Make pseudo-inverse marginal value function for this transitory shock
        vPnvrsFuncNow_list.append(LinearInterp(bNrmNow_temp, vPnvrsNowArray[:, j]))

    # Make linear interpolation by combining the lists of consumption, labor and marginal value functions
    cFuncNowBase = LinearInterpOnInterp1D(cFuncNow_list, TranShkGrid)
    LbrFuncNowBase = LinearInterpOnInterp1D(LbrFuncNow_list, TranShkGrid)
    vPnvrsFuncNowBase = LinearInterpOnInterp1D(vPnvrsFuncNow_list, TranShkGrid)

    # Construct consumption, labor, pseudo-inverse marginal value functions with
    # bNrmMinNow as the lower bound.  This removes the adjustment in the loop above.
    cFuncNow = VariableLowerBoundFunc2D(cFuncNowBase, bNrmMinNow)
    LbrFuncNow = VariableLowerBoundFunc2D(LbrFuncNowBase, bNrmMinNow)
    vPnvrsFuncNow = VariableLowerBoundFunc2D(vPnvrsFuncNowBase, bNrmMinNow)

    # Construct the marginal value function by "recurving" its pseudo-inverse
    vPfuncNow = MargValueFuncCRRA(vPnvrsFuncNow, CRRA)

    # Make a solution object for this period and return it
    solution = ConsumerLaborSolution(
        cFunc=cFuncNow, LbrFunc=LbrFuncNow, vPfunc=vPfuncNow, bNrmMin=bNrmMinNow
    )
    return solution


###############################################################################


def solve_ConsLaborIntMargSep(
    solution_next,
    PermShkDstn,
    TranShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    TranShkGrid,
    vFuncBool,
    CubicBool,
    WageRte,
    Psi,
    LsrCurv,
):
    r"""
    Solves one period of the consumption-saving model with endogenous
    intensive-margin labor supply, using a **separable** period utility:

    .. math::
        u(c_t, h_t) = u_{\text{CRRA}}(c_t) + \psi \, \frac{(1 - h_t)^{1 - \theta}}{1 - \theta}

    where :math:`u_{\text{CRRA}}` is CRRA utility over consumption (log utility
    when ``CRRA == 1``, which is the Dupor et al. 2023 specification),
    :math:`\psi` (``Psi``) is the leisure weight, and :math:`\theta`
    (``LsrCurv``) governs the curvature of leisure utility (inverse Frisch
    elasticity).

    Because the utility is **separable** in :math:`c` and :math:`h`, the
    Euler equation and the intratemporal MRS condition decouple and can be
    solved sequentially via EGM. This mirrors the two-stage decomposition
    (``cons-noshocks`` followed by ``labor-supply``) in the Dupor 2023 stage
    notebook, but is implemented as a single one-period solver to keep the
    same call signature as :func:`solve_ConsLaborIntMarg`.

    Algorithm
    ---------
    1. Compute :math:`\text{EndOfPrdvP}(a_t) = \beta R (1-\mathsf{D})
       \,\mathbb{E}[(\Gamma\psi)^{-\rho} v'_{b,t+1}(b_{t+1}, \theta_{t+1})]`
       on the exogenous end-of-period asset grid ``aXtraGrid``.  This step
       is identical to the original :func:`solve_ConsLaborIntMarg`.
    2. **Consumption first-order condition (Euler).**
       :math:`u'(c_t) = \text{EndOfPrdvP}(a_t)` yields
       :math:`c_t^* = (u')^{-1}(\text{EndOfPrdvP}(a_t))`.  Note this does
       **not** depend on :math:`\theta_t` or :math:`h_t` thanks to
       separability.
    3. **Intratemporal MRS condition.**
       :math:`\psi (1-h_t)^{-\theta} = w \theta_t \, u'(c_t^*)` yields
       :math:`h_t^*(a_t, \theta_t) = 1 - \left(\frac{w \theta_t \,
       u'(c_t^*)}{\psi}\right)^{-1/\theta}`.  Clamp to :math:`[0, 1]`.
    4. Recover the endogenous bank-balance grid:
       :math:`b_t = a_t + c_t^* - w \theta_t h_t^*`.
    5. Build interpolating policy and marginal-value functions on
       :math:`(b_t, \theta_t)`, anchored by the lower-bound function
       :math:`b_{\min}(\theta_t) = -w \theta_t` (the full-labor, zero-
       consumption corner).

    Parameters
    ----------
    solution_next : ConsumerLaborSolution
        Solution to next period's problem; must expose ``vPfunc`` and
        ``bNrmMin`` over ``(b_{t+1}, theta_{t+1})``.
    PermShkDstn : DiscreteDistribution
        Discrete distribution of permanent productivity shocks.
    TranShkDstn : DiscreteDistribution
        Discrete distribution of transitory productivity shocks.
    LivPrb : float
        Survival probability (next-period existence).
    DiscFac : float
        Intertemporal discount factor :math:`\beta`.
    CRRA : float
        Coefficient of relative risk aversion on consumption.  Set
        ``CRRA = 1`` to recover the Dupor 2023 log-consumption specification.
    Rfree : float
        Gross risk-free rate on retained assets.
    PermGroFac : float
        Expected permanent income growth factor.
    BoroCnstArt : float or None
        Artificial borrowing constraint.  Not yet handled; must be ``None``.
    aXtraGrid : np.ndarray
        Grid of "extra" end-of-period assets (above the natural minimum).
    TranShkGrid : np.ndarray
        Grid of transitory shock values used as a state for interpolation.
        ``TranShkGrid[0]`` is assumed to be ``0.0`` (unemployment node).
    vFuncBool : bool
        Whether to compute the value function.  Not yet implemented; must
        be ``False``.
    CubicBool : bool
        Whether to use cubic interpolation.  Not yet implemented; must be
        ``False``.
    WageRte : float
        Wage rate per efficiency unit of labor supplied.
    Psi : float
        Weight on the leisure utility term, :math:`\psi > 0`.
    LsrCurv : float
        Curvature of the leisure utility term, :math:`\theta > 0`.
        Equals the inverse Frisch elasticity of labor supply.

    Returns
    -------
    solution_now : ConsumerLaborSolution
        Contains ``cFunc``, ``LbrFunc``, ``vPfunc``, and ``bNrmMin``, each
        defined over normalized bank balances :math:`b_t` and the
        transitory productivity shock :math:`\theta_t`.
    """
    # ---------- Input validation (mirrors solve_ConsLaborIntMarg) ----------
    if BoroCnstArt is not None:
        raise ValueError("Model cannot handle artificial borrowing constraint yet.")
    if CubicBool is True:
        raise ValueError("Model cannot handle cubic interpolation yet.")
    if vFuncBool is True:
        raise ValueError("Model cannot compute the value function yet.")
    if Psi <= 0.0:
        raise ValueError("Psi (leisure weight) must be strictly positive.")
    if LsrCurv <= 0.0:
        raise ValueError("LsrCurv (leisure curvature) must be strictly positive.")

    # ---------- Unpack next-period solution and shock distributions ----------
    vPfunc_next = solution_next.vPfunc
    TranShkPrbs = TranShkDstn.pmv
    TranShkVals = TranShkDstn.atoms.flatten()
    PermShkPrbs = PermShkDstn.pmv
    PermShkVals = PermShkDstn.atoms.flatten()
    TranShkCount = TranShkPrbs.size
    PermShkCount = PermShkPrbs.size

    def uPinv(X):
        # Inverse marginal utility of consumption: c such that u'(c) = X
        return LogCRRALaborutilityPc_inv(X, Psi, LsrCurv)

    # =====================================================================
    # Step 1: Build expected marginal value over future transitory shocks
    # (Identical to solve_ConsLaborIntMarg.)
    # =====================================================================
    aXtraCount = aXtraGrid.size
    bNrmGrid = aXtraGrid  # next period's bank balances before labor income

    bNrmGrid_rep = np.tile(np.reshape(bNrmGrid, (aXtraCount, 1)), (1, TranShkCount))
    TranShkVals_rep = np.tile(
        np.reshape(TranShkVals, (1, TranShkCount)), (aXtraCount, 1)
    )
    TranShkPrbs_rep = np.tile(
        np.reshape(TranShkPrbs, (1, TranShkCount)), (aXtraCount, 1)
    )

    # Marginal value at every (b_{t+1}, theta_{t+1}) gridpoint
    vPNext = vPfunc_next(bNrmGrid_rep, TranShkVals_rep)

    # Integrate over transitory shocks to get pre-transitory expected MV
    vPbarNext = np.sum(vPNext * TranShkPrbs_rep, axis=1)

    # Transform through inverse MU to decurve, then linearly interpolate,
    # then recurve via MargValueFuncCRRA.  When CRRA == 1 this is the
    # composition (1/c)(c(b)) used in HARK's log-utility solvers.
    vPbarNvrsNext = uPinv(vPbarNext)
    vPbarNvrsFuncNext = LinearInterp(
        np.insert(bNrmGrid, 0, 0.0), np.insert(vPbarNvrsNext, 0, 0.0)
    )
    vPbarFuncNext = MargValueFuncCRRA(vPbarNvrsFuncNext, CRRA)

    # =====================================================================
    # Step 2: Integrate over permanent shocks, applying the discount &
    # growth factors. Result: EndOfPrdvP(a_t) on the exogenous a-grid.
    # =====================================================================
    aNrmGrid_rep = np.tile(np.reshape(aXtraGrid, (aXtraCount, 1)), (1, PermShkCount))
    PermShkVals_rep = np.tile(
        np.reshape(PermShkVals, (1, PermShkCount)), (aXtraCount, 1)
    )
    PermShkPrbs_rep = np.tile(
        np.reshape(PermShkPrbs, (1, PermShkCount)), (aXtraCount, 1)
    )
    bNrmNext = (Rfree / (PermGroFac * PermShkVals_rep)) * aNrmGrid_rep
    vPbarNext_perm = (PermGroFac * PermShkVals_rep) ** (-CRRA) * vPbarFuncNext(bNrmNext)
    EndOfPrdvP = (
        DiscFac
        * Rfree
        * LivPrb
        * np.sum(vPbarNext_perm * PermShkPrbs_rep, axis=1, keepdims=True)
    )  # shape: (aXtraCount, 1)

    # =====================================================================
    # Step 3 (NEW): Solve the Euler equation for consumption.
    #     u'(c_t) = EndOfPrdvP(a_t)  =>  c_t = (u')^{-1}(EndOfPrdvP)
    # Because utility is *separable*, this is independent of h_t and
    # theta_t -- the same c* applies across all transitory shock columns.
    # =====================================================================
    TranShkGrid_rep = np.tile(
        np.reshape(TranShkGrid, (1, TranShkGrid.size)), (aXtraCount, 1)
    )
    EndOfPrdvP_tile = np.tile(EndOfPrdvP, (1, TranShkGrid.size))
    cNrmNow = uPinv(EndOfPrdvP_tile)  # shape: (aXtraCount, TranShkGrid.size)

    # =====================================================================
    # Step 4 (NEW): Apply the intratemporal MRS to recover labor supply.
    #     psi * (1 - h)^(-LsrCurv) = WageRte * theta_t * u'(c_t)
    # <=> 1 - h = (WageRte * theta_t * u'(c_t) / psi)^(-1/LsrCurv)
    # Equivalently, using u'(c_t) = EndOfPrdvP_tile:
    # =====================================================================
    with np.errstate(divide="ignore", invalid="ignore"):
        LsrNow = (WageRte * TranShkGrid_rep * EndOfPrdvP_tile / Psi) ** (
            -1.0 / LsrCurv
        )
    # Where TranShk == 0 the FOC has no interior solution (no productive
    # time to sell). Set leisure = 1, labor = 0; consumption is unchanged.
    LsrNow[:, 0] = 1.0

    # If unconstrained FOC implies leisure > 1 (would require negative
    # labor), clamp to the corner h = 0. Because utility is separable,
    # consumption need not be recomputed.
    LsrNow = np.minimum(LsrNow, 1.0)
    LbrNow = 1.0 - LsrNow

    # =====================================================================
    # Step 5: Recover endogenous bank-balance grid via within-period budget
    #     a_t = b_t + WageRte * theta_t * h_t - c_t
    # =>  b_t = a_t + c_t - WageRte * theta_t * h_t
    # =====================================================================
    aNrmNow_rep = np.tile(np.reshape(aXtraGrid, (aXtraCount, 1)), (1, TranShkGrid.size))
    bNrmNow = aNrmNow_rep + cNrmNow - WageRte * TranShkGrid_rep * LbrNow

    # =====================================================================
    # Step 6: Anchor the lower edge of the bNrm grid at the
    #     full-labor / zero-consumption corner: b_min(theta) = -w * theta.
    # With log utility, c=0 is only approached asymptotically; the anchor
    # gives the interpolation a well-defined left endpoint.
    # =====================================================================
    bNowArray = np.concatenate(
        (np.reshape(-WageRte * TranShkGrid, (1, TranShkGrid.size)), bNrmNow), axis=0
    )
    cNowArray = np.concatenate((np.zeros((1, TranShkGrid.size)), cNrmNow), axis=0)
    LsrNowArray = np.concatenate((np.zeros((1, TranShkGrid.size)), LsrNow), axis=0)
    LsrNowArray[0, 0] = 1.0  # at theta = 0 we never work, even at b = 0
    LbrNowArray = 1.0 - LsrNowArray
    vPnvrsNowArray = np.concatenate(
        (np.zeros((1, TranShkGrid.size)), uPinv(EndOfPrdvP_tile))
    )

    # =====================================================================
    # Step 7: Build the policy and marginal-value interpolants.
    # =====================================================================
    bNrmMinNow = LinearInterp(TranShkGrid, bNowArray[0, :])

    cFuncNow_list = []
    LbrFuncNow_list = []
    vPnvrsFuncNow_list = []
    for j in range(TranShkGrid.size):
        # Shift each column so its left endpoint is 0 (the lower bound
        # is re-attached afterward via VariableLowerBoundFunc2D).
        bNrmNow_temp = bNowArray[:, j] - bNowArray[0, j]
        cFuncNow_list.append(LinearInterp(bNrmNow_temp, cNowArray[:, j]))
        LbrFuncNow_list.append(LinearInterp(bNrmNow_temp, LbrNowArray[:, j]))
        vPnvrsFuncNow_list.append(LinearInterp(bNrmNow_temp, vPnvrsNowArray[:, j]))

    cFuncNowBase = LinearInterpOnInterp1D(cFuncNow_list, TranShkGrid)
    LbrFuncNowBase = LinearInterpOnInterp1D(LbrFuncNow_list, TranShkGrid)
    vPnvrsFuncNowBase = LinearInterpOnInterp1D(vPnvrsFuncNow_list, TranShkGrid)

    cFuncNow = VariableLowerBoundFunc2D(cFuncNowBase, bNrmMinNow)
    LbrFuncNow = VariableLowerBoundFunc2D(LbrFuncNowBase, bNrmMinNow)
    vPnvrsFuncNow = VariableLowerBoundFunc2D(vPnvrsFuncNowBase, bNrmMinNow)

    vPfuncNow = MargValueFuncCRRA(vPnvrsFuncNow, CRRA)

    solution = ConsumerLaborSolution(
        cFunc=cFuncNow, LbrFunc=LbrFuncNow, vPfunc=vPfuncNow, bNrmMin=bNrmMinNow
    )
    return solution


###############################################################################


# Make a dictionary of constructors for the intensive margin labor model
LaborIntMargConsumerType_constructors_default = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "LbrCost": make_log_polynomial_LbrCost,
    "TranShkGrid": get_TranShkGrid_from_TranShkDstn,
    "solution_terminal": make_labor_intmarg_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
LaborIntMargConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
LaborIntMargConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}


# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
LaborIntMargConsumerType_IncShkDstn_default = {
    "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
    "PermShkCount": 16,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
    "TranShkCount": 15,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.0,  # Unemployment benefits replacement rate while working
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
}

# Default parameters to make aXtraGrid using make_assets_grid
LaborIntMargConsumerType_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 80.0,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 200,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameter to make LbrCost using make_log_polynomial_LbrCost
LaborIntMargConsumerType_LbrCost_default = {
    "LbrCostCoeffs": [-1.0]  # Polynomial age coeffs on log labor utility cost
}

# Make a dictionary to specify an intensive margin labor supply choice consumer type
LaborIntMargConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "pseudo_terminal": False,  # Terminal period really does exist
    "constructors": LaborIntMargConsumerType_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "WageRte": [1.0],  # Wage rate paid on labor income
    "BoroCnstArt": None,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
}
LaborIntMargConsumerType_simulation_default = {
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    "NewbornTransShk": False,  # Whether Newborns have transitory shock
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
    "neutral_measure": False,  # Whether to use permanent income neutral measure (see Harmenberg 2021)
}
LaborIntMargConsumerType_default = {}
LaborIntMargConsumerType_default.update(LaborIntMargConsumerType_IncShkDstn_default)
LaborIntMargConsumerType_default.update(LaborIntMargConsumerType_aXtraGrid_default)
LaborIntMargConsumerType_default.update(LaborIntMargConsumerType_LbrCost_default)
LaborIntMargConsumerType_default.update(LaborIntMargConsumerType_solving_default)
LaborIntMargConsumerType_default.update(LaborIntMargConsumerType_simulation_default)
LaborIntMargConsumerType_default.update(LaborIntMargConsumerType_kNrmInitDstn_default)
LaborIntMargConsumerType_default.update(LaborIntMargConsumerType_pLvlInitDstn_default)
init_labor_intensive = LaborIntMargConsumerType_default


class LaborIntMargConsumerType(IndShockConsumerType):
    r"""
    A class representing agents who make a decision each period about how much
    to consume vs save and how much labor to supply (as a fraction of their time).
    They get CRRA utility from a composite good :math:`x_t = c_t*z_t^alpha`, and discount
    future utility flows at a constant factor.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(b_t,\theta_{t}) &= \max_{c_t,L_{t}}u_{t}(c_t,L_t) + \DiscFac (1 - \DiePrb_{t+1}) \mathbb{E}_{t} \left[ (\PermGroFac_{t+1} \psi_{t+1})^{1-\CRRA} v_{t+1}(b_{t+1},\theta_{t+1}) \right], \\
        & \text{s.t.}  \\
        m_{t} &= b_{t} + L_{t}\theta_{t} \text{WageRte}_{t}, \\
        a_t &= m_t - c_t, \\
        b_{t+1} &= a_t \Rfree_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}), \\
        (\psi_{t+1},\theta_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi]=\mathbb{E}[\theta] &= 1, \\
        u_{t}(c,L) &= \frac{(c (1-L)^{\alpha_t})^{1-\CRRA}}{1-\CRRA} \\
        \end{align*}


    Constructors
    ------------
    IncShkDstn: Constructor, :math:`\psi`, :math:`\theta`
        The agent's income shock distributions.

        Its default constructor is :func:`HARK.Calibration.Income.IncomeProcesses.construct_lognormal_income_process_unemployment`
    aXtraGrid: Constructor
        The agent's asset grid.

        Its default constructor is :func:`HARK.utilities.make_assets_grid`
    LbrCost: Constructor, :math:`\alpha`
        The agent's labor cost function.

        Its default constructor is :func:`HARK.ConsumptionSaving.ConsLaborModel.make_log_polynomial_LbrCost`

    Solving Parameters
    ------------------
    cycles: int
        0 specifies an infinite horizon model, 1 specifies a finite model.
    T_cycle: int
        Number of periods in the cycle for this agent type.
    CRRA: float, default=2.0, :math:`\rho`
        Coefficient of Relative Risk Aversion. Must be greater than :math:`\max_{t}({\frac{\alpha_t}{\alpha_t+1}})`
    Rfree: float or list[float], time varying, :math:`\mathsf{R}`
        Risk Free interest rate. Pass a list of floats to make Rfree time varying.
    DiscFac: float, :math:`\beta`
        Intertemporal discount factor.
    LivPrb: list[float], time varying, :math:`1-\mathsf{D}`
        Survival probability after each period.
    WageRte: list[float], time varying
        Wage rate paid on labor income.
    PermGroFac: list[float], time varying, :math:`\Gamma`
        Permanent income growth factor.

    Simulation Parameters
    ---------------------
    AgentCount: int
        Number of agents of this kind that are created during simulations.
    T_age: int
        Age after which to automatically kill agents, None to ignore.
    T_sim: int, required for simulation
        Number of periods to simulate.
    track_vars: list[strings]
        List of variables that should be tracked when running the simulation.
        For this agent, the options are 'Lbr', 'PermShk', 'TranShk', 'aLvl', 'aNrm', 'bNrm', 'cNrm', 'mNrm', 'pLvl', and 'who_dies'.

        PermShk is the agent's permanent income shock

        TranShk is the agent's transitory income shock

        aLvl is the nominal asset level

        aNrm is the normalized assets

        bNrm is the normalized resources without this period's labor income

        cNrm is the normalized consumption

        mNrm is the normalized market resources

        pLvl is the permanent income level

        Lbr is the share of the agent's time spent working

        who_dies is the array of which agents died
    aNrmInitMean: float
        Mean of Log initial Normalized Assets.
    aNrmInitStd: float
        Std of Log initial Normalized Assets.
    pLvlInitMean: float
        Mean of Log initial permanent income.
    pLvlInitStd: float
        Std of Log initial permanent income.
    PermGroFacAgg: float
        Aggregate permanent income growth factor (The portion of PermGroFac attributable to aggregate productivity growth).
    PerfMITShk: boolean
        Do Perfect Foresight MIT Shock (Forces Newborns to follow solution path of the agent they replaced if True).
    NewbornTransShk: boolean
        Whether Newborns have transitory shock.

    Attributes
    ----------
    solution: list[Consumer solution object]
        Created by the :func:`.solve` method. Finite horizon models create a list with T_cycle+1 elements, for each period in the solution.
        Infinite horizon solutions return a list with T_cycle elements for each period in the cycle.

        Visit :class:`HARK.ConsumptionSaving.ConsLaborModel.ConsumerLaborSolution` for more information about the solution.

    history: Dict[Array]
        Created by running the :func:`.simulate()` method.
        Contains the variables in track_vars. Each item in the dictionary is an array with the shape (T_sim,AgentCount).
        Visit :class:`HARK.core.AgentType.simulate` for more information.
    """

    IncShkDstn_default = LaborIntMargConsumerType_IncShkDstn_default
    aXtraGrid_default = LaborIntMargConsumerType_aXtraGrid_default
    LbrCost_default = LaborIntMargConsumerType_LbrCost_default
    solving_default = LaborIntMargConsumerType_solving_default
    simulation_default = LaborIntMargConsumerType_simulation_default

    default_ = {
        "params": LaborIntMargConsumerType_default,
        "solver": solve_ConsLaborIntMarg,
        "model": "ConsLaborIntMarg.yaml",
        "track_vars": ["aNrm", "cNrm", "Lbr", "mNrm", "pLvl"],
    }

    time_vary_ = copy(IndShockConsumerType.time_vary_)
    time_vary_ += ["WageRte", "LbrCost", "TranShkGrid"]
    time_inv_ = copy(IndShockConsumerType.time_inv_)

    def pre_solve(self):
        self.construct("solution_terminal")

    def calc_bounding_values(self):  # pragma: nocover
        """
        NOT YET IMPLEMENTED FOR THIS CLASS
        """
        raise NotImplementedError()

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):  # pragma: nocover
        """
        NOT YET IMPLEMENTED FOR THIS CLASS
        """
        raise NotImplementedError()

    def get_states(self):
        """
        Calculates updated values of normalized bank balances and permanent income
        level for each agent.  Uses pLvlNow, aNrmNow, PermShkNow.  Calls the get_states
        method for the parent class, then erases mNrmNow, which cannot be calculated
        until after get_controls in this model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.get_states(self)
        # Delete market resource calculation
        self.state_now["mNrm"][:] = np.nan

    def get_controls(self):
        """
        Calculates consumption and labor supply for each consumer of this type
        using the consumption and labor functions in each period of the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        LbrNow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these] = self.solution[t].cFunc(
                self.state_now["bNrm"][these], self.shocks["TranShk"][these]
            )  # Assign consumption values
            MPCnow[these] = self.solution[t].cFunc.derivativeX(
                self.state_now["bNrm"][these], self.shocks["TranShk"][these]
            )  # Assign marginal propensity to consume values (derivative)
            LbrNow[these] = self.solution[t].LbrFunc(
                self.state_now["bNrm"][these], self.shocks["TranShk"][these]
            )  # Assign labor supply
        self.controls["cNrm"] = cNrmNow
        self.MPCnow = MPCnow
        self.controls["Lbr"] = LbrNow

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Make an array of wage rates by age
        Wage = np.zeros(self.AgentCount)
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            Wage[these] = self.WageRte[t]
        LbrEff = self.controls["Lbr"] * self.shocks["TranShk"]
        yNrmNow = LbrEff * Wage
        mNrmNow = self.state_now["bNrm"] + yNrmNow
        aNrmNow = mNrmNow - self.controls["cNrm"]

        self.state_now["LbrEff"] = LbrEff
        self.state_now["mNrm"] = mNrmNow
        self.state_now["aNrm"] = aNrmNow
        self.state_now["yNrm"] = yNrmNow
        super().get_poststates()

    def plot_cFunc(self, t, bMin=None, bMax=None, ShkSet=None):
        """
        Plot the consumption function by bank balances at a given set of transitory shocks.

        Parameters
        ----------
        t : int
            Time index of the solution for which to plot the consumption function.
        bMin : float or None
            Minimum value of bNrm at which to begin the plot.  If None, defaults
            to the minimum allowable value of bNrm for each transitory shock.
        bMax : float or None
            Maximum value of bNrm at which to end the plot.  If None, defaults
            to bMin + 20.
        ShkSet : [float] or None
            Array or list of transitory shocks at which to plot the consumption
            function.  If None, defaults to the TranShkGrid for this time period.

        Returns
        -------
        None
        """
        if ShkSet is None:
            ShkSet = self.TranShkGrid[t]

        for j in range(len(ShkSet)):
            TranShk = ShkSet[j]
            bMin_temp = self.solution[t].bNrmMin(TranShk) if bMin is None else bMin
            bMax_temp = bMin_temp + 20.0 if bMax is None else bMax

            B = np.linspace(bMin_temp, bMax_temp, 300)
            C = self.solution[t].cFunc(B, TranShk * np.ones_like(B))
            plt.plot(B, C)
        plt.xlabel(r"Beginning of period normalized bank balances $b_t$")
        plt.ylabel(r"Normalized consumption level $c_t$")
        plt.ylim([0.0, None])
        plt.xlim(bMin, bMax)
        plt.show(block=False)

    def plot_LbrFunc(self, t, bMin=None, bMax=None, ShkSet=None):
        """
        Plot the labor supply function by bank balances at a given set of transitory shocks.

        Parameters
        ----------
        t : int
            Time index of the solution for which to plot the labor supply function.
        bMin : float or None
            Minimum value of bNrm at which to begin the plot.  If None, defaults
            to the minimum allowable value of bNrm for each transitory shock.
        bMax : float or None
            Maximum value of bNrm at which to end the plot.  If None, defaults
            to bMin + 20.
        ShkSet : [float] or None
            Array or list of transitory shocks at which to plot the labor supply
            function.  If None, defaults to the TranShkGrid for this time period.

        Returns
        -------
        None
        """
        if ShkSet is None:
            ShkSet = self.TranShkGrid[t]

        for j in range(len(ShkSet)):
            TranShk = ShkSet[j]
            bMin_temp = self.solution[t].bNrmMin(TranShk) if bMin is None else bMin
            bMax_temp = bMin_temp + 20.0 if bMax is None else bMax

            B = np.linspace(bMin_temp, bMax_temp, 300)
            L = self.solution[t].LbrFunc(B, TranShk * np.ones_like(B))
            plt.plot(B, L)
        plt.xlabel(r"Beginning of period normalized bank balances $b_t$")
        plt.ylabel(r"Labor supply $\ell_t$")
        plt.ylim([-0.001, 1.001])
        plt.xlim(bMin, bMax)
        plt.show(block=False)

    def check_conditions(self, verbose=None):
        raise NotImplementedError()  # pragma: nocover

    def calc_limiting_values(self):
        raise NotImplementedError()  # pragma: nocover


###############################################################################

# Make a dictionary for intensive margin labor supply model with finite lifecycle
init_labor_lifecycle = init_labor_intensive.copy()
init_labor_lifecycle["PermGroFac"] = [
    1.01,
    1.01,
    1.01,
    1.01,
    1.01,
    1.02,
    1.02,
    1.02,
    1.02,
    1.02,
]
init_labor_lifecycle["PermShkStd"] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
init_labor_lifecycle["TranShkStd"] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
init_labor_lifecycle["LivPrb"] = [
    0.99,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
    0.1,
]  # Living probability decreases as time moves forward.
init_labor_lifecycle["WageRte"] = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]  # Wage rate in a lifecycle
init_labor_lifecycle["Rfree"] = 10 * [1.03]
# Assume labor cost coeffs is a polynomial of degree 1
init_labor_lifecycle["LbrCostCoeffs"] = np.array([-2.0, 0.4])
init_labor_lifecycle["T_cycle"] = 10
# init_labor_lifecycle['T_retire']   = 7 # IndexError at line 774 in interpolation.py.
init_labor_lifecycle["T_age"] = (
    11  # Make sure that old people die at terminal age and don't turn into newborns!
)


###############################################################################
# ============== Separable log-CRRA-labor sibling class =====================
###############################################################################
#
# Everything below mirrors the LaborIntMargConsumerType machinery above, but
# for the *separable* utility u(c, h) = u_CRRA(c) + Psi*(1-h)^(1-LsrCurv)/(1-LsrCurv).
# Setting CRRA = 1 recovers the Dupor et al. (2023) specification.
#
# The original LbrCost parameter is replaced by two scalars per period:
#   Psi     -- weight on the leisure utility term (> 0)
#   LsrCurv -- curvature of the leisure utility term (= inverse Frisch
#              elasticity, > 0)
###############################################################################


def make_labor_intmarg_sep_solution_terminal(
    CRRA, aXtraGrid, Psi, LsrCurv, WageRte, TranShkGrid
):
    r"""
    Construct the terminal-period solution for the separable log-CRRA-labor
    model.  The terminal agent has no future, so they solve the static problem

    .. math::
        \max_{c, h} \; u_{\text{CRRA}}(c)
            + \psi \, \frac{(1-h)^{1-\theta}}{1-\theta}
        \quad\text{s.t.}\quad c = b + w \vartheta\, h, \; h \in [0, 1]

    Combining the intratemporal first-order condition with the budget yields a
    monotone-in-:math:`h` transcendental equation, which has either an interior
    root or a corner at :math:`h = 0`.  Specifically, let

    .. math::
        G(h) := \psi (1-h)^{-\theta} (b + w\vartheta h) - w\vartheta.

    Because :math:`G(0) = \psi b - w\vartheta` and :math:`G` is monotonically
    increasing in :math:`h \in [0, 1)`, the corner :math:`h^* = 0` is optimal
    whenever :math:`\psi b \geq w\vartheta`; otherwise there is a unique
    interior root that this function solves by vectorized Newton iterations.
    When :math:`\theta = 1` the closed form
    :math:`h^* = (1 - \psi b / (w\vartheta)) / (1 + \psi)` is used instead.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion on consumption.
    aXtraGrid : np.ndarray
        Grid of "extra" end-of-period assets (above the natural minimum).
    Psi : list[float]
        Time-varying weight on the leisure utility term; the terminal value
        ``Psi[-1]`` is used here.
    LsrCurv : list[float]
        Time-varying curvature of the leisure utility term; ``LsrCurv[-1]``
        is used here.
    WageRte : list[float]
        Time-varying wage rate; ``WageRte[-1]`` is used here.
    TranShkGrid : list[np.ndarray]
        Time-varying transitory shock grids; ``TranShkGrid[-1]`` is used here.

    Returns
    -------
    solution_terminal : ConsumerLaborSolution
        Terminal-period solution, with ``cFunc``, ``LbrFunc``, ``vPfunc``, and
        ``bNrmMin`` defined over :math:`(b, \vartheta)`.  ``vFunc`` is not
        produced (consistent with ``vFuncBool = False`` in the period solver).
    """
    t = -1
    TranShkGrid_T = TranShkGrid[t]
    Psi_T = Psi[t]
    LsrCurv_T = LsrCurv[t]
    WageRte_T = WageRte[t]

    # Add a point at b = 0 to make sure the b-grid goes down to 0
    bNrmGrid = np.insert(aXtraGrid, 0, 0.0)
    bNrmCount = bNrmGrid.size
    TranShkCount = TranShkGrid_T.size

    # Tile b- and theta-grids into (bNrmCount, TranShkCount) arrays
    bNrmTerm = np.tile(np.reshape(bNrmGrid, (bNrmCount, 1)), (1, TranShkCount))
    TranShkTerm = np.tile(
        np.reshape(TranShkGrid_T, (1, TranShkCount)), (bNrmCount, 1)
    )
    labor_inc_per_h = WageRte_T * TranShkTerm  # marginal labor income

    # Corner condition: at h = 0 the FOC G(0) = Psi*b - w*theta is non-negative
    # iff Psi*b >= w*theta. That's the rich-agent corner where labor is zero.
    corner = Psi_T * bNrmTerm >= labor_inc_per_h
    LbrTerm = np.zeros_like(bNrmTerm)

    interior = ~corner & (TranShkTerm > 0.0)

    if abs(LsrCurv_T - 1.0) < 1e-12:
        # ----- Closed-form solution when LsrCurv == 1 (log-leisure) -----
        # FOC: psi/(1-h) * (b + w*theta*h) = w*theta
        # =>  h = (1 - psi*b/(w*theta)) / (1 + psi)
        with np.errstate(divide="ignore", invalid="ignore"):
            h_int = (1.0 - Psi_T * bNrmTerm / labor_inc_per_h) / (1.0 + Psi_T)
        LbrTerm = np.where(interior, np.clip(h_int, 0.0, 1.0 - 1e-12), 0.0)
    else:
        # ----- Vectorized Newton iterations on G(h) = 0 for h in (0, 1) -----
        h = np.where(interior, 0.5, 0.0)  # initial guess
        for _ in range(60):
            one_minus_h = 1.0 - h
            # Avoid 0/negative inside the power for non-interior cells.
            one_minus_h_safe = np.where(interior, one_minus_h, 1.0)
            pow_term = one_minus_h_safe ** (-LsrCurv_T)
            G = Psi_T * pow_term * (bNrmTerm + labor_inc_per_h * h) - labor_inc_per_h
            # G'(h) = Psi*[LsrCurv*(1-h)^(-LsrCurv-1)*(b + w*theta*h)
            #              + (1-h)^(-LsrCurv) * w*theta]
            Gp = Psi_T * (
                LsrCurv_T * one_minus_h_safe ** (-LsrCurv_T - 1.0)
                * (bNrmTerm + labor_inc_per_h * h)
                + pow_term * labor_inc_per_h
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                step = np.where(interior, G / Gp, 0.0)
            h_new = np.clip(h - step, 0.0, 1.0 - 1e-12)
            if np.max(np.abs(h_new - h)) < 1e-12:
                h = h_new
                break
            h = h_new
        LbrTerm = np.where(interior, h, 0.0)

    LsrTerm = 1.0 - LbrTerm
    # Terminal consumption is everything available (consume all in last period)
    cNrmTerm = bNrmTerm + labor_inc_per_h * LbrTerm

    # ----- Build interpolating policy functions -----
    LbrFunc_terminal = BilinearInterp(LbrTerm, bNrmGrid, TranShkGrid_T)
    cFunc_terminal = BilinearInterp(cNrmTerm, bNrmGrid, TranShkGrid_T)

    # ----- Build the pseudo-inverse marginal value function -----
    # Envelope: v'_b(b, theta) = u'(c*(b, theta))
    # Pseudo-inverse: vPnvrs(b, theta) = (u')^{-1}(v'_b) = c*(b, theta)
    # When CRRA = 1 this is exactly the consumption function.
    vPnvrsTerm = cNrmTerm.copy()
    # At the (b=0, theta=0) corner consumption is zero; the boundary point
    # is informational only -- linear interpolation handles it gracefully.
    vPnvrsFunc_terminal = BilinearInterp(vPnvrsTerm, bNrmGrid, TranShkGrid_T)
    vPfunc_terminal = MargValueFuncCRRA(vPnvrsFunc_terminal, CRRA)

    bNrmMin_terminal = ConstantFunction(0.0)

    solution_terminal = ConsumerLaborSolution(
        cFunc=cFunc_terminal,
        LbrFunc=LbrFunc_terminal,
        vPfunc=vPfunc_terminal,
        bNrmMin=bNrmMin_terminal,
    )
    return solution_terminal


###############################################################################
# Parameter dictionaries for the separable variant
###############################################################################

# Constructors -- same as the parent's *except* drop `LbrCost` (no longer used)
# and redirect `solution_terminal` to the separable-utility variant.
LaborIntMargSepConsumerType_constructors_default = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "TranShkGrid": get_TranShkGrid_from_TranShkDstn,
    "solution_terminal": make_labor_intmarg_sep_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
}

# These three dictionaries are unchanged from the parent and are reused.
LaborIntMargSepConsumerType_kNrmInitDstn_default = LaborIntMargConsumerType_kNrmInitDstn_default
LaborIntMargSepConsumerType_pLvlInitDstn_default = LaborIntMargConsumerType_pLvlInitDstn_default
LaborIntMargSepConsumerType_IncShkDstn_default = LaborIntMargConsumerType_IncShkDstn_default
LaborIntMargSepConsumerType_aXtraGrid_default = LaborIntMargConsumerType_aXtraGrid_default

# Default labor-utility parameters (replaces LbrCost_default from parent).
# Values match the Dupor 2023 YAML calibration (psi = 1.0, theta = 2.0).
LaborIntMargSepConsumerType_LaborUtil_default = {
    "Psi": [1.0],      # Leisure utility weight (time-varying list, len = T_cycle)
    "LsrCurv": [2.0],  # Leisure utility curvature (inverse Frisch elasticity)
}

# Solving defaults -- same skeleton as parent, with CRRA = 1.0 for Dupor's log
# specification and no LbrCostCoeffs (since LbrCost is not used).
LaborIntMargSepConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,
    "T_cycle": 1,
    "pseudo_terminal": False,
    "constructors": LaborIntMargSepConsumerType_constructors_default,
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 1.0,            # log utility over consumption (Dupor spec)
    "Rfree": [1.03],
    "DiscFac": 0.96,
    "LivPrb": [0.98],
    "PermGroFac": [1.01],
    "WageRte": [1.0],
    "BoroCnstArt": None,
    "vFuncBool": False,
    "CubicBool": False,
}

LaborIntMargSepConsumerType_simulation_default = LaborIntMargConsumerType_simulation_default

# Merge everything into a single flat parameter dictionary
LaborIntMargSepConsumerType_default = {}
LaborIntMargSepConsumerType_default.update(LaborIntMargSepConsumerType_IncShkDstn_default)
LaborIntMargSepConsumerType_default.update(LaborIntMargSepConsumerType_aXtraGrid_default)
LaborIntMargSepConsumerType_default.update(LaborIntMargSepConsumerType_LaborUtil_default)
LaborIntMargSepConsumerType_default.update(LaborIntMargSepConsumerType_solving_default)
LaborIntMargSepConsumerType_default.update(LaborIntMargSepConsumerType_simulation_default)
LaborIntMargSepConsumerType_default.update(LaborIntMargSepConsumerType_kNrmInitDstn_default)
LaborIntMargSepConsumerType_default.update(LaborIntMargSepConsumerType_pLvlInitDstn_default)
init_labor_intensive_sep = LaborIntMargSepConsumerType_default


class LaborIntMargSepConsumerType(LaborIntMargConsumerType):
    r"""
    A sibling of :class:`LaborIntMargConsumerType` in which the period utility
    is **separable** in consumption and leisure:

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\DiePrb}{\mathsf{D}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \begin{align*}
        v_t(b_t, \theta_t) &= \max_{c_t, h_t}
            \; u_{\text{CRRA}}(c_t)
            + \psi_t \frac{(1 - h_t)^{1 - \vartheta_t}}{1 - \vartheta_t}
            + \DiscFac (1 - \DiePrb_{t+1}) \,
              \mathbb{E}_{t} \left[
                (\PermGroFac_{t+1} \psi_{t+1})^{1-\CRRA}
                v_{t+1}(b_{t+1}, \theta_{t+1})
              \right] \\
        \text{s.t.} \quad
        m_{t} &= b_{t} + h_{t}\theta_{t}\,\text{WageRte}_{t}, \\
        a_{t} &= m_{t} - c_{t}, \\
        b_{t+1} &= a_{t}\,\Rfree_{t+1} / (\PermGroFac_{t+1} \psi_{t+1}), \\
        (\psi_{t+1}, \theta_{t+1}) &\sim F_{t+1}, \quad
            \mathbb{E}[\psi] = \mathbb{E}[\theta] = 1.
        \end{align*}

    With ``CRRA = 1`` this reduces to the Dupor et al. (2023) specification
    :math:`\log(c_t) + \psi\,(1-h_t)^{1-\theta}/(1-\theta)`.

    Differences from :class:`LaborIntMargConsumerType`
    --------------------------------------------------
    1.  Period utility is *separable* in consumption and leisure -- in the
        parent class the two are combined into a single CRRA composite
        :math:`x_t = c_t z_t^{\alpha_t}`.
    2.  The labor cost parameter ``LbrCost`` (= :math:`\alpha_t`) is replaced
        by two scalars per period: ``Psi`` (= :math:`\psi`, leisure weight)
        and ``LsrCurv`` (= :math:`\vartheta`, leisure curvature / inverse
        Frisch elasticity).
    3.  The period solver is :func:`solve_ConsLaborIntMargSep`, which inverts
        the consumption Euler equation and the intratemporal MRS condition
        **independently** thanks to separability.
    4.  No restriction analogous to ``CRRA > LbrCost/(1+LbrCost)`` is needed.

    Solving parameters
    ------------------
    Same as :class:`LaborIntMargConsumerType`, except:

    Psi : list[float], time varying, :math:`\psi`
        Weight on the leisure utility term; must be strictly positive.
    LsrCurv : list[float], time varying, :math:`\vartheta`
        Curvature of the leisure utility term (inverse Frisch elasticity);
        must be strictly positive.  ``LsrCurv = 1`` collapses the leisure
        term to :math:`\psi \log(1 - h)`.

    All other parameters (``CRRA``, ``Rfree``, ``DiscFac``, ``LivPrb``,
    ``PermGroFac``, ``WageRte``, ``BoroCnstArt``, ``aXtraGrid``,
    ``TranShkGrid``, income-shock parameters, simulation parameters)
    behave identically to the parent class.

    Inherited methods
    -----------------
    The simulation methods ``get_states``, ``get_controls``, ``get_poststates``,
    and the plotting helpers ``plot_cFunc``, ``plot_LbrFunc`` are inherited
    unchanged from the parent.  They depend only on the
    :class:`ConsumerLaborSolution` interface (``cFunc``, ``LbrFunc``,
    ``bNrmMin``), which is produced identically by both solvers.
    """

    IncShkDstn_default = LaborIntMargSepConsumerType_IncShkDstn_default
    aXtraGrid_default = LaborIntMargSepConsumerType_aXtraGrid_default
    LaborUtil_default = LaborIntMargSepConsumerType_LaborUtil_default
    solving_default = LaborIntMargSepConsumerType_solving_default
    simulation_default = LaborIntMargSepConsumerType_simulation_default

    # Explicitly do NOT carry over the parent's `LbrCost_default`: setting it
    # to None signals that the separable variant does not use LbrCost at all.
    LbrCost_default = None

    default_ = {
        "params": LaborIntMargSepConsumerType_default,
        "solver": solve_ConsLaborIntMargSep,
        "model": "ConsLaborIntMargSep.yaml",
        "track_vars": ["aNrm", "cNrm", "Lbr", "mNrm", "pLvl"],
    }

    # Override the parent's time_vary_ list: replace `LbrCost` with the two
    # labor-utility parameters `Psi` and `LsrCurv`.
    time_vary_ = copy(IndShockConsumerType.time_vary_)
    time_vary_ += ["WageRte", "Psi", "LsrCurv", "TranShkGrid"]
    time_inv_ = copy(IndShockConsumerType.time_inv_)


###############################################################################
# Finite-lifecycle default dictionary for the separable variant
###############################################################################

init_labor_lifecycle_sep = init_labor_intensive_sep.copy()
init_labor_lifecycle_sep["PermGroFac"] = [
    1.01, 1.01, 1.01, 1.01, 1.01, 1.02, 1.02, 1.02, 1.02, 1.02,
]
init_labor_lifecycle_sep["PermShkStd"] = [0.1] * 10
init_labor_lifecycle_sep["TranShkStd"] = [0.1] * 10
init_labor_lifecycle_sep["LivPrb"] = [
    0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
]
init_labor_lifecycle_sep["WageRte"] = [1.0] * 10
init_labor_lifecycle_sep["Rfree"] = [1.03] * 10
init_labor_lifecycle_sep["Psi"] = [1.0] * 10
init_labor_lifecycle_sep["LsrCurv"] = [2.0] * 10
init_labor_lifecycle_sep["T_cycle"] = 10
init_labor_lifecycle_sep["T_age"] = 11


###############################################################################
# ============== Markov-income (Dupor 2023) sibling class ===================
###############################################################################
#
# This block builds yet another sibling of LaborIntMargConsumerType in which:
#
#   (a) the agent's productivity x follows an AR(1) in logs, discretized into
#       a Markov chain via HARK's `make_tauchen_ar1` (no permanent / transitory
#       shock separation);
#   (b) the regional mutual fund is replaced by a single risk-free asset with
#       gross return `Rfree`;
#   (c) the fiscal authority is exogenous: an income tax rate `Tau` and a
#       lump-sum transfer `Trnsfr` are model parameters;
#   (d) intermediate firms are exogenous: a per-capita real dividend `Div`
#       arrives each period, of which a fraction `(1 - Omega) * (x / xBar)`
#       is paid directly to the household (then taxed at rate `Tau`).
#
# After consolidating the tax schedule T_t = -T + tau*(wxh + (1-omega)*delta*D),
# the household's flow-of-funds becomes (paper eq. 4.4 + 4.6, simplified):
#
#   c_t + a_{t+1} = Trnsfr
#                 + (1 - Tau) * WageRte * x_t * h_t
#                 + (1 - Tau) * (1 - Omega) * (x_t / xBar) * Div
#                 + Rfree * a_t,
#   a_{t+1} >= 0.
#
# Defining the "bank balances before labor income" state
#   b_t = Rfree * a_t + Trnsfr + (1 - Tau) * (1 - Omega) * (x_t / xBar) * Div
# preserves the within-period budget identity used by the existing solvers:
#   a_{t+1} = b_t + (1 - Tau) * WageRte * x_t * h_t - c_t.
###############################################################################


def make_dupor_productivity_process(
    MrkvCount=7, MrkvAR1=0.955, MrkvSigma=0.015 ** 0.5, MrkvBound=3.0
):
    r"""
    Discretize an AR(1) log-productivity process via Tauchen's method
    (via HARK's ``make_tauchen_ar1``) and exponentiate to a productivity grid.

    The log grid is shifted by :math:`-\sigma_y^2/2`, where
    :math:`\sigma_y^2 = \sigma_\eta^2 / (1 - \rho^2)`, so that the unconditional
    mean of the level process satisfies :math:`\mathbb{E}[x] \approx 1`,
    matching the YAML calibration :math:`\bar{x} = 1`.

    Parameters
    ----------
    MrkvCount : int
        Number of discrete productivity states.
    MrkvAR1 : float
        AR(1) persistence of log productivity (:math:`\rho`).
    MrkvSigma : float
        Standard deviation of the AR(1) innovation (:math:`\sigma_\eta`).
    MrkvBound : float
        Tauchen bound, in multiples of the unconditional standard deviation
        of log :math:`x`.

    Returns
    -------
    xGrid : np.ndarray, shape (MrkvCount,)
        Productivity levels (always positive).
    MrkvArray : np.ndarray, shape (MrkvCount, MrkvCount)
        Markov transition matrix; ``MrkvArray[i, j] = Pr(x' = j | x = i)``.
    """
    log_x, MrkvArray = make_tauchen_ar1(
        N=MrkvCount, sigma=MrkvSigma, ar_1=MrkvAR1, bound=MrkvBound
    )
    sigma_y_sq = MrkvSigma ** 2 / (1.0 - MrkvAR1 ** 2)
    log_x_centered = log_x - 0.5 * sigma_y_sq
    xGrid = np.exp(log_x_centered)
    return xGrid, MrkvArray


def make_stationary_MrkvDstn(MrkvArray):
    r"""
    Compute the stationary distribution :math:`\pi` of a Markov chain by
    solving :math:`\pi^\top \Gamma = \pi^\top` (i.e., :math:`\pi` is the left
    eigenvector of ``MrkvArray`` associated with eigenvalue 1).

    Parameters
    ----------
    MrkvArray : np.ndarray, shape (J, J)
        Markov transition matrix; ``MrkvArray[i, j] = Pr(x' = j | x = i)``.
        Rows must sum to 1.

    Returns
    -------
    MrkvInitDstn : HARK.distributions.DiscreteDistribution
        Discrete distribution over Markov state indices ``[0, 1, ..., J-1]``
        with probabilities equal to the stationary distribution.  Suitable
        for drawing initial Markov states for newborn agents.
    """
    MrkvArray = np.asarray(MrkvArray, dtype=float)
    eigvals, eigvecs = np.linalg.eig(MrkvArray.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()
    J = MrkvArray.shape[0]
    return DiscreteDistribution(pmv=pi, atoms=np.arange(J, dtype=int))


###############################################################################


def solve_ConsLaborIntMargSepMrkv(
    solution_next,
    MrkvArray,
    xGrid,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
    WageRte,
    Psi,
    LsrCurv,
    Tau,
    Trnsfr,
    Div,
    Omega,
    xBar,
):
    r"""
    Solve one period of the Markov-income, separable-utility intensive-margin
    labor model (Dupor et al. 2023, simplified household problem).

    The household's productivity :math:`x_t` follows a Markov chain on the
    grid ``xGrid`` with transition matrix ``MrkvArray``.  Period utility is

    .. math::
        u(c_t, h_t) = u_{\text{CRRA}}(c_t)
            + \psi\,\frac{(1 - h_t)^{1 - \theta}}{1 - \theta},

    where :math:`\psi` (``Psi``) is the leisure weight and :math:`\theta`
    (``LsrCurv``) is the leisure curvature.  With ``CRRA = 1`` this matches
    the Dupor 2023 log-consumption specification.

    The budget constraint is

    .. math::
        c_t + a_{t+1} = T + (1-\tau)\bigl[w x_t + (1-\omega)\,\delta(x_t)\, D\bigr]
            h_t + (1+r) a_t,
        \quad a_{t+1} \geq 0,

    where :math:`\delta(x_t) = x_t / \bar{x}`.  The dividend term is now
    *proportional to labor supply* :math:`h_t`, reflecting the equilibrium
    intermediate-goods market result that, when the labor market clears,
    the level of dividends paid out is itself proportional to aggregate
    labor.  The household takes :math:`D` as given but earns its share
    only by supplying labor.  Parameters:  ``Tau`` :math:`= \tau`,
    ``Trnsfr`` :math:`= T`, ``Omega`` :math:`= \omega`, ``Div`` :math:`= D`,
    ``xBar`` :math:`= \bar{x}`, and ``Rfree`` :math:`= 1 + r`.

    It is convenient to define the *effective per-hour after-tax income*

    .. math::
        \tilde w(x_i) := (1-\tau)\bigl[w x_i + (1-\omega)\,\delta(x_i)\, D\bigr],

    so the period budget collapses to
    :math:`c_t + a_{t+1} = b_t + \tilde w(x_t) h_t`
    with :math:`b_t := T + (1+r) a_t`.

    Algorithm (EGM, parallels :func:`solve_ConsLaborIntMargSep`):

    1. Pre-compute next-period bank balances
       :math:`b_{t+1}(a_{t+1}) = R a_{t+1} + T`
       on ``aXtraGrid``.  Note that, because the dividend is now earned
       only when working, :math:`b_{t+1}` does **not** depend on
       :math:`x'`.
    2. Evaluate :math:`v'_{b,t+1}(b_{t+1}(a_{t+1}), x')` for each Markov
       state using the list of next-period marginal-value functions.
    3. For each current Markov state :math:`x_i`, take the Markov expectation
       :math:`\mathbb{E}[v'_{b,t+1} \mid x_i] = \sum_{x'} \Gamma_{i, x'}
       v'_{b,t+1}(\cdot, x')` and form ``EndOfPrdvP(a_{t+1}, x_i)``.
    4. Euler:  :math:`c^* = (u')^{-1}(\text{EndOfPrdvP})` -- separability
       means this does **not** depend on :math:`x_i` or :math:`h`.
    5. Intratemporal MRS:  :math:`h^* = 1 - \bigl(\psi c^* /
       \tilde w(x_i)\bigr)^{1/\theta}`, clamped to :math:`[0, 1]`.
    6. Endogenous :math:`b_t = a_{t+1} + c^* - \tilde w(x_i) h^*`.
    7. Build 1-D interpolants over :math:`b` for each Markov state and pack
       them into lists.

    Parameters
    ----------
    solution_next : ConsumerLaborSolution
        Next period's solution.  Its ``vPfunc`` must be a *list* of
        ``MrkvCount`` callables :math:`v'_{b, t+1}(\cdot, x')`, one per
        Markov state.
    MrkvArray : np.ndarray, shape (MrkvCount, MrkvCount)
        Transition matrix; ``MrkvArray[i, j] = Pr(x' = j | x = i)``.
    xGrid : np.ndarray, shape (MrkvCount,)
        Productivity levels.
    LivPrb : float
        Survival probability.
    DiscFac : float
        Intertemporal discount factor :math:`\beta`.
    CRRA : float
        CRRA on consumption (``CRRA = 1`` for Dupor's log spec).
    Rfree : float
        Gross risk-free rate :math:`1 + r`.
    BoroCnstArt : None
        Artificial borrowing constraint.  Must be ``None``.
    aXtraGrid : np.ndarray
        Grid of end-of-period assets above the natural minimum.
    vFuncBool, CubicBool : bool
        Must both be ``False`` (not yet implemented).
    WageRte : float
        Real wage rate :math:`w`.
    Psi, LsrCurv : float
        Leisure utility parameters.
    Tau : float
        Proportional income-tax rate :math:`\tau \in [0, 1)`.
    Trnsfr : float
        Lump-sum transfer :math:`T`.
    Div : float
        Exogenous per-capita dividend :math:`D`.
    Omega : float
        Dividend retention share :math:`\omega \in [0, 1]`.
    xBar : float
        Mean productivity normalization :math:`\bar{x}`.

    Returns
    -------
    solution_now : ConsumerLaborSolution
        ``cFunc``, ``LbrFunc``, ``vPfunc``, and ``bNrmMin`` are *lists of
        length* ``MrkvCount``.  ``cFunc[i]``, ``LbrFunc[i]`` are 1-D
        interpolants over :math:`b`; ``vPfunc[i]`` is a
        ``MargValueFuncCRRA`` wrapping the consumption interpolant;
        ``bNrmMin[i]`` is a scalar lower bound on :math:`b` for that
        Markov state.
    """
    if BoroCnstArt is not None:
        raise ValueError("Model cannot handle artificial borrowing constraint yet.")
    if CubicBool is True:
        raise ValueError("Model cannot handle cubic interpolation yet.")
    if vFuncBool is True:
        raise ValueError("Model cannot compute the value function yet.")
    if Psi <= 0.0:
        raise ValueError("Psi (leisure weight) must be strictly positive.")
    if LsrCurv <= 0.0:
        raise ValueError("LsrCurv (leisure curvature) must be strictly positive.")
    if not (0.0 <= Tau < 1.0):
        raise ValueError("Tau (income-tax rate) must lie in [0, 1).")
    if not (0.0 <= Omega <= 1.0):
        raise ValueError("Omega (dividend retention) must lie in [0, 1].")

    MrkvCount = xGrid.size
    aXtraCount = aXtraGrid.size
    vPfunc_next = solution_next.vPfunc  # list of length MrkvCount

    def uPinv(X):
        return CRRAutilityP_inv(X, rho=CRRA)

    # ---------------------------------------------------------------
    # Step 1-2: Build next period's bank balances and evaluate v'_{b,t+1}.
    #   b_{t+1}(a') = R*a' + Trnsfr      (dividend now earned per hour worked,
    #                                     so it is NOT part of autonomous b)
    # bNrmNext is shape (aXtraCount,) -- independent of x'.
    # vPnext has shape (aXtraCount, MrkvCount), columns indexed by x'.
    # ---------------------------------------------------------------
    bNrmNext = Rfree * aXtraGrid + Trnsfr

    vPnext = np.empty((aXtraCount, MrkvCount))
    for xp_idx in range(MrkvCount):
        vPnext[:, xp_idx] = vPfunc_next[xp_idx](bNrmNext)

    # ---------------------------------------------------------------
    # Step 3: Markov expectation conditional on the current state.
    # EndOfPrdvP[:, i] = beta * R * LivPrb * sum_j Gamma_{i,j} * vPnext[:, j]
    # which, in matrix form, is vPnext @ MrkvArray.T.
    # Shape: (aXtraCount, MrkvCount), columns indexed by CURRENT x.
    # ---------------------------------------------------------------
    EndOfPrdvP = DiscFac * Rfree * LivPrb * (vPnext @ MrkvArray.T)

    cFunc_list = []
    LbrFunc_list = []
    vPfunc_list = []
    bNrmMin_list = []

    for x_idx in range(MrkvCount):
        x_now = xGrid[x_idx]
        EndOfPrdvP_x = EndOfPrdvP[:, x_idx]

        # Effective after-tax wage per unit of labor at this productivity.
        # An extra hour of work yields wage income (1-Tau)*w*x plus the
        # household's share of dividends (1-Tau)*(1-Omega)*(x/xBar)*Div.
        eff_wage = (1.0 - Tau) * (
            WageRte * x_now + (1.0 - Omega) * (x_now / xBar) * Div
        )

        # Step 4: Euler equation
        cNrmNow = uPinv(EndOfPrdvP_x)

        # Step 5: Intratemporal MRS for labor supply.
        # psi*(1-h)^(-LsrCurv) = eff_wage * u'(c) = eff_wage * EndOfPrdvP_x
        # => (1-h) = (eff_wage * EndOfPrdvP_x / Psi)^(-1/LsrCurv)
        # If eff_wage <= 0 (e.g. x == 0), there is no productive time to
        # sell: h = 0 (corner).
        if eff_wage > 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                LsrNow = (eff_wage * EndOfPrdvP_x / Psi) ** (-1.0 / LsrCurv)
            LsrNow = np.minimum(LsrNow, 1.0)
        else:
            LsrNow = np.ones_like(cNrmNow)
        LbrNow = 1.0 - LsrNow

        # Step 6: Endogenous b on the EGM grid.
        # a' = b + eff_wage*h - c  =>  b = a' + c - eff_wage*h
        bNrmNow = aXtraGrid + cNrmNow - eff_wage * LbrNow

        # Anchor lower boundary at "full labor, zero consumption" corner.
        # Below this b, the agent cannot satisfy c >= 0 even at h = 1.
        b_lower = -eff_wage
        bArr = np.insert(bNrmNow, 0, b_lower)
        cArr = np.insert(cNrmNow, 0, 0.0)
        # At the corner: full labor (unless eff_wage = 0).
        LbrCorner = 1.0 if eff_wage > 0.0 else 0.0
        LbrArr = np.insert(LbrNow, 0, LbrCorner)
        vPnvrsArr = np.insert(uPinv(EndOfPrdvP_x), 0, 0.0)

        # Step 7: Build 1-D interpolants for this Markov state.
        cFunc_x = LinearInterp(bArr, cArr)
        LbrFunc_x = LinearInterp(bArr, LbrArr)
        vPnvrsFunc_x = LinearInterp(bArr, vPnvrsArr)
        vPfunc_x = MargValueFuncCRRA(vPnvrsFunc_x, CRRA)

        cFunc_list.append(cFunc_x)
        LbrFunc_list.append(LbrFunc_x)
        vPfunc_list.append(vPfunc_x)
        bNrmMin_list.append(b_lower)

    return ConsumerLaborSolution(
        cFunc=cFunc_list,
        LbrFunc=LbrFunc_list,
        vPfunc=vPfunc_list,
        bNrmMin=bNrmMin_list,
    )


###############################################################################


def make_labor_intmarg_sep_mrkv_solution_terminal(
    CRRA, aXtraGrid, Psi, LsrCurv, WageRte, xGrid, Tau, Div, Omega, xBar,
):
    r"""
    Construct the terminal-period solution for the Markov-income, separable-
    utility variant.  The terminal agent has no future, so for each Markov
    state :math:`x_i` they solve

    .. math::
        \max_{c, h} \; u_{\text{CRRA}}(c) + \psi (1-h)^{1-\theta}/(1-\theta)
        \quad \text{s.t.} \quad c = b + \tilde w(x_i)\, h, \; h \in [0, 1],

    where the effective after-tax per-hour income is

    .. math::
        \tilde w(x_i) := (1-\tau)\bigl[w x_i + (1-\omega)(x_i/\bar x)\,D\bigr].

    The intratemporal FOC plus the budget yields the monotone-in-:math:`h`
    transcendental equation

    .. math::
        G(h) := \psi (1-h)^{-\theta} \bigl(b + \tilde w(x_i)\, h\bigr)
                - \tilde w(x_i) = 0,

    with a corner at :math:`h^* = 0` when :math:`\psi b \geq \tilde w(x_i)`
    and a unique interior root otherwise (closed-form when ``LsrCurv == 1``,
    vectorized Newton iterations otherwise).

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion on consumption.
    aXtraGrid : np.ndarray
        End-of-period assets-above-minimum grid (re-used as the b-grid here).
    Psi, LsrCurv : list[float]
        Time-varying leisure parameters; ``[-1]`` element is used.
    WageRte : list[float]
        Time-varying wage rate; ``[-1]`` element is used.
    xGrid : np.ndarray, shape (MrkvCount,)
        Productivity grid (time-invariant).
    Tau : list[float]
        Time-varying income-tax rate; ``[-1]`` element is used.
    Div, Omega, xBar : list[float]
        Time-varying dividend level, dividend retention share, and mean-
        productivity normaliser; ``[-1]`` elements are used.

    Returns
    -------
    solution_terminal : ConsumerLaborSolution
        With ``cFunc``, ``LbrFunc``, ``vPfunc``, ``bNrmMin`` stored as
        *lists of length* ``MrkvCount``.
    """
    Psi_T = Psi[-1]
    LsrCurv_T = LsrCurv[-1]
    WageRte_T = WageRte[-1]
    Tau_T = Tau[-1]
    Div_T = Div[-1]
    Omega_T = Omega[-1]
    xBar_T = xBar[-1]
    MrkvCount = xGrid.size

    # Use the same b-grid for every Markov state (with a 0 prepended)
    bNrmGrid = np.insert(aXtraGrid, 0, 0.0)
    bNrmCount = bNrmGrid.size

    cFunc_list = []
    LbrFunc_list = []
    vPfunc_list = []
    bNrmMin_list = []

    for x_idx in range(MrkvCount):
        x_now = xGrid[x_idx]
        # Effective after-tax wage per hour: wage income plus the
        # dividend share that accrues per hour worked.
        labor_inc_per_h = (1.0 - Tau_T) * (
            WageRte_T * x_now + (1.0 - Omega_T) * (x_now / xBar_T) * Div_T
        )
        bArr = bNrmGrid.copy()

        # If x = 0 (only possible if Tauchen grid includes a zero state) the
        # agent has no productive time: h = 0, c = b.
        if labor_inc_per_h <= 0.0:
            LbrTerm = np.zeros(bNrmCount)
        else:
            # Corner condition: psi*b >= labor_inc_per_h  =>  h* = 0.
            corner = Psi_T * bArr >= labor_inc_per_h
            interior = ~corner

            if abs(LsrCurv_T - 1.0) < 1e-12:
                # Closed form for log leisure
                with np.errstate(divide="ignore", invalid="ignore"):
                    h_int = (1.0 - Psi_T * bArr / labor_inc_per_h) / (1.0 + Psi_T)
                LbrTerm = np.where(interior, np.clip(h_int, 0.0, 1.0 - 1e-12), 0.0)
            else:
                # Vectorized Newton iterations on G(h) = 0 for h in (0, 1).
                h = np.where(interior, 0.5, 0.0)
                for _ in range(60):
                    one_minus_h = 1.0 - h
                    one_minus_h_safe = np.where(interior, one_minus_h, 1.0)
                    pow_term = one_minus_h_safe ** (-LsrCurv_T)
                    G = (
                        Psi_T * pow_term * (bArr + labor_inc_per_h * h)
                        - labor_inc_per_h
                    )
                    Gp = Psi_T * (
                        LsrCurv_T * one_minus_h_safe ** (-LsrCurv_T - 1.0)
                        * (bArr + labor_inc_per_h * h)
                        + pow_term * labor_inc_per_h
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        step = np.where(interior, G / Gp, 0.0)
                    h_new = np.clip(h - step, 0.0, 1.0 - 1e-12)
                    if np.max(np.abs(h_new - h)) < 1e-12:
                        h = h_new
                        break
                    h = h_new
                LbrTerm = np.where(interior, h, 0.0)

        cTerm = bArr + labor_inc_per_h * LbrTerm

        # Build 1D interpolants for this Markov state
        cFunc_x = LinearInterp(bArr, cTerm)
        LbrFunc_x = LinearInterp(bArr, LbrTerm)
        # Pseudo-inverse marginal value: vPnvrs(b) = (u')^{-1}(v'_b) = c*(b)
        vPnvrsFunc_x = LinearInterp(bArr, cTerm)
        vPfunc_x = MargValueFuncCRRA(vPnvrsFunc_x, CRRA)

        cFunc_list.append(cFunc_x)
        LbrFunc_list.append(LbrFunc_x)
        vPfunc_list.append(vPfunc_x)
        bNrmMin_list.append(0.0)

    return ConsumerLaborSolution(
        cFunc=cFunc_list,
        LbrFunc=LbrFunc_list,
        vPfunc=vPfunc_list,
        bNrmMin=bNrmMin_list,
    )


###############################################################################
# Parameter dictionaries for the Markov-income variant
###############################################################################

# Compute the default productivity grid + transition matrix at module load.
# Calibration matches Tables/benchmark-parameters.tex:
#   rho   = 0.955             (persistence of x, quarterly)
#   sigma = sqrt(0.015)       (sd of innovation; table reports variance 1.5%)
# Users who want a different AR(1) calibration can override `xGrid` and
# `MrkvArray` directly in the parameter dictionary.
_default_xGrid, _default_MrkvArray = make_dupor_productivity_process(
    MrkvCount=7, MrkvAR1=0.955, MrkvSigma=0.015 ** 0.5, MrkvBound=3.0
)

# Constructors: drop the perm/trans income-shock constructors entirely, and
# point `solution_terminal` at the new Markov-aware terminal solver.
LaborIntMargSepMrkvConsumerType_constructors_default = {
    "aXtraGrid": make_assets_grid,
    "solution_terminal": make_labor_intmarg_sep_mrkv_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
}

# Defaults for the Markov productivity process
LaborIntMargSepMrkvConsumerType_Mrkv_default = {
    "MrkvArray": _default_MrkvArray,
    "xGrid": _default_xGrid,
    # Initial distribution over Markov states for newborns.  If left as
    # ``None``, the simulator will lazily compute the stationary distribution
    # of ``MrkvArray`` via ``make_stationary_MrkvDstn`` when ``initialize_sim``
    # is called.  Users can pre-populate this with any
    # ``HARK.distributions.DiscreteDistribution`` whose atoms are integer
    # indices into ``xGrid``.
    "MrkvInitDstn": None,
}

# Leisure-utility defaults.
#   Psi      = 1.0    Leisure weight, RECALIBRATED for THIS model's utility
#                     normalization to deliver an ergodic mean hours of ~0.42
#                     (the Dupor 2023 target).  NOTE: the paper's tabulated
#                     psi = 5.8 does NOT transfer to this log-consumption /
#                     log-leisure specification -- with Div = 0 it yields mean
#                     hours of only ~0.086.  See dupor_minimal_demo.ipynb, which
#                     prints the realized mean hours as a calibration check.
#   LsrCurv  = 1.0    = theta; 1/theta = Frisch elasticity ~1.0 (Beraja et al.
#                     2019); theta = 1 gives the log-leisure closed-form path.
LaborIntMargSepMrkvConsumerType_LaborUtil_default = {
    "Psi": [1.0],
    "LsrCurv": [1.0],
}

# Fiscal + dividend defaults (Dupor 2023 benchmark, quarterly).
# Values are from Tables/benchmark-parameters.tex unless noted.
LaborIntMargSepMrkvConsumerType_DuporEcon_default = {
    "Tau": [0.0],       # Proportional income-tax rate. NOT in the benchmark
                        # table -- in the paper, tau is determined endogenously
                        # in equilibrium from the government budget constraint.
                        # Defaulted to 0 here (no proportional tax distortion);
                        # users should override with a calibrated value to
                        # study fiscal effects.
    "Trnsfr": [0.03],   # Lump-sum transfer T (paper Table: T = 0.03).
    "Div": [0.0],       # Per-capita real dividend D. Endogenous in the paper;
                        # set to 0 here as the user's "exogenous D" input.
                        # Note: the household receives (1-Omega)*delta(x)*Div
                        # per HOUR worked (intermediate-goods market clearing),
                        # so non-zero Div interacts with the labor margin.
    "Omega": [0.32],    # Dividend retention share omega (paper Table: 0.32).
    "xBar": [1.0],      # Mean productivity normalization; the productivity
                        # grid is centered so E[x] approx 1.
}

# Asset-grid defaults (reused unchanged from the parent class)
LaborIntMargSepMrkvConsumerType_aXtraGrid_default = (
    LaborIntMargConsumerType_aXtraGrid_default
)
LaborIntMargSepMrkvConsumerType_kNrmInitDstn_default = (
    LaborIntMargConsumerType_kNrmInitDstn_default
)
LaborIntMargSepMrkvConsumerType_pLvlInitDstn_default = (
    LaborIntMargConsumerType_pLvlInitDstn_default
)

# Solving defaults: no PermShk/TranShk/PermGroFac/income-process params here.
# Calibration matches Tables/benchmark-parameters.tex (Dupor 2023 benchmark).
# The model period is a QUARTER; the table's "annual real rate = 4%" target
# corresponds to a quarterly gross rate Rfree = 1.04 ** (1/4).
LaborIntMargSepMrkvConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,
    "T_cycle": 1,
    "pseudo_terminal": False,
    "constructors": LaborIntMargSepMrkvConsumerType_constructors_default,
    # PRIMITIVE PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 1.0,                # log utility on consumption (Dupor spec)
    "Rfree": [1.04 ** 0.25],    # gross quarterly real rate -> annual = 4%
    "DiscFac": 0.985,           # mean discount factor bar(beta) (Table)
    "LivPrb": [1.0],            # no mortality in the baseline Dupor calibration
    "WageRte": [1.0],
    "BoroCnstArt": None,
    "vFuncBool": False,
    "CubicBool": False,
}

# Simulation defaults (most carry over; we drop PermGroFacAgg-style fields
# that are inseparable from the perm-income machinery).
LaborIntMargSepMrkvConsumerType_simulation_default = {
    "AgentCount": 10000,
    "T_age": None,
    "PerfMITShk": False,
    "neutral_measure": False,
}

# Merge everything into the flat parameter dictionary
LaborIntMargSepMrkvConsumerType_default = {}
LaborIntMargSepMrkvConsumerType_default.update(
    LaborIntMargSepMrkvConsumerType_aXtraGrid_default
)
LaborIntMargSepMrkvConsumerType_default.update(
    LaborIntMargSepMrkvConsumerType_Mrkv_default
)
LaborIntMargSepMrkvConsumerType_default.update(
    LaborIntMargSepMrkvConsumerType_LaborUtil_default
)
LaborIntMargSepMrkvConsumerType_default.update(
    LaborIntMargSepMrkvConsumerType_DuporEcon_default
)
LaborIntMargSepMrkvConsumerType_default.update(
    LaborIntMargSepMrkvConsumerType_solving_default
)
LaborIntMargSepMrkvConsumerType_default.update(
    LaborIntMargSepMrkvConsumerType_simulation_default
)
LaborIntMargSepMrkvConsumerType_default.update(
    LaborIntMargSepMrkvConsumerType_kNrmInitDstn_default
)
LaborIntMargSepMrkvConsumerType_default.update(
    LaborIntMargSepMrkvConsumerType_pLvlInitDstn_default
)
init_labor_intensive_sep_mrkv = LaborIntMargSepMrkvConsumerType_default


class LaborIntMargSepMrkvConsumerType(LaborIntMargSepConsumerType):
    r"""
    A Markov-income, separable-utility consumer that implements the
    *simplified* Dupor et al. (2023) household problem.

    Differences from :class:`LaborIntMargSepConsumerType`
    ----------------------------------------------------
    1.  **Income process.** Productivity :math:`x_t` follows an AR(1) in logs

        .. math::
            \log x_{t+1} = \rho \log x_t + \eta_{t+1},
            \quad \eta_{t+1} \sim \mathcal{N}(0, \sigma_\eta^2),

        discretized into a Markov chain via HARK's
        :func:`HARK.distributions.make_tauchen_ar1`.  There are no
        permanent / transitory shocks, no permanent-income normalisation,
        and no permanent-income growth factor :math:`\Gamma`.

    2.  **Mutual fund stand-in.** The regional mutual fund of the paper is
        replaced by a single risk-free asset with gross return
        ``Rfree`` :math:`= 1 + r`.

    3.  **Fiscal authority.** The income-tax schedule and lump-sum transfer
        are exogenous parameters:  ``Tau`` :math:`= \tau`,  ``Trnsfr``
        :math:`= T`.

    4.  **Intermediate firms.**  The household receives an exogenous per-
        capita real dividend stream ``Div`` :math:`= D` in proportion
        :math:`\delta(x_t) = x_t / \bar{x}`, after a retention share
        ``Omega`` :math:`= \omega` and the income tax.  Following the
        paper's intermediate-goods-market result -- that, when the labor
        market clears, the aggregate dividend is itself proportional to
        aggregate labor supply -- the household's dividend payout is
        *scaled by its own labor supply* :math:`h_t`.

    Budget constraint
    -----------------

    .. math::
        c_t + a_{t+1} = T + (1-\tau)\bigl[w x_t + (1-\omega)\,\delta(x_t)\, D\bigr]
            h_t + (1+r) a_t, \quad a_{t+1} \geq 0,

    with :math:`\delta(x_t) = x_t / \bar{x}`.  Defining the
    *effective after-tax per-hour income* :math:`\tilde w(x_t) =
    (1-\tau)[w x_t + (1-\omega)\delta(x_t)\, D]`, the budget collapses
    to :math:`c_t + a_{t+1} = T + (1+r) a_t + \tilde w(x_t) h_t`.

    Period utility
    --------------

    .. math::
        u(c_t, h_t) = u_{\text{CRRA}}(c_t)
            + \psi\,\frac{(1 - h_t)^{1 - \theta}}{1 - \theta}.

    With ``CRRA = 1`` this matches the Dupor 2023 log-consumption spec.

    Solution layout
    ---------------
    Each ``solution[t]`` is a :class:`ConsumerLaborSolution` whose
    ``cFunc``, ``LbrFunc``, ``vPfunc``, and ``bNrmMin`` attributes are
    *lists of length* ``MrkvCount`` (the number of discrete productivity
    states).  To query the policy at Markov state ``i``:

    .. code-block:: python

        c_at_b = agent.solution[t].cFunc[i](b)
        h_at_b = agent.solution[t].LbrFunc[i](b)

    Parameters
    ----------
    See :data:`init_labor_intensive_sep_mrkv` for the full default dict.
    Key new parameters: ``MrkvArray``, ``xGrid``, ``Tau``, ``Trnsfr``,
    ``Div``, ``Omega``, ``xBar``.  Removed relative to the parent:
    ``PermShkStd``, ``PermShkCount``, ``TranShkStd``, ``TranShkCount``,
    ``UnempPrb``, ``IncUnemp``, ``T_retire``, ``UnempPrbRet``,
    ``IncUnempRet``, ``PermGroFac``, ``PermGroFacAgg``, ``NewbornTransShk``.

    Simulation
    ----------
    Forward Monte Carlo simulation is supported via the standard HARK
    workflow:

    .. code-block:: python

        agent = LaborIntMargSepMrkvConsumerType(cycles=0)
        agent.solve()
        agent.T_sim = 1000
        agent.track_vars = ["aLvl", "cLvl", "Lbr", "Mrkv", "xLvl", "yLvl"]
        agent.initialize_sim()
        agent.simulate()
        # Aggregate paths
        C_path = agent.history["cLvl"].mean(axis=1)
        L_path = agent.history["Lbr"].mean(axis=1)

    Because the Dupor income process is a discrete Markov chain (no
    permanent-income normalisation), all simulator quantities are stored
    as *levels*:

    * ``aLvl`` -- end-of-period asset level
    * ``bLvl`` -- pre-labor cash-on-hand
    * ``mLvl`` -- post-labor market resources
    * ``yLvl`` -- post-tax labor income
    * ``xLvl`` -- realised productivity ``xGrid[Mrkv]``
    * ``cLvl`` -- consumption choice
    * ``Lbr``  -- labor supply choice (in [0, 1])
    * ``Mrkv`` -- current discrete productivity index

    Sequence-space Jacobians (SSJ)
    ------------------------------
    See the module-level :func:`sketch_SSJ_usage_for_DuporAgent` docstring
    for a recipe explaining how to feed an instance of this class to
    :func:`HARK.SSJutils.make_basic_SSJ_matrices`.  The short version:
    set ``cycles = 0`` and ``T_cycle = 1``, write a small YAML model file
    describing the transition system, then call

    .. code-block:: python

        from HARK.SSJutils import make_basic_SSJ_matrices
        J_C_T = make_basic_SSJ_matrices(
            agent, shock="Trnsfr", outcomes=["cLvl", "Lbr"],
            grids={"aLvl": {"N": 200, "min": 0.0, "max": 50.0},
                   "Mrkv":  {"N": agent.MrkvArray.shape[0]}},
        )
    """

    aXtraGrid_default = LaborIntMargSepMrkvConsumerType_aXtraGrid_default
    LaborUtil_default = LaborIntMargSepMrkvConsumerType_LaborUtil_default
    DuporEcon_default = LaborIntMargSepMrkvConsumerType_DuporEcon_default
    Mrkv_default = LaborIntMargSepMrkvConsumerType_Mrkv_default
    solving_default = LaborIntMargSepMrkvConsumerType_solving_default
    simulation_default = LaborIntMargSepMrkvConsumerType_simulation_default

    # Inherited LbrCost_default = None (from parent) signals no LbrCost usage.
    IncShkDstn_default = None  # no continuous income-shock distribution

    default_ = {
        "params": LaborIntMargSepMrkvConsumerType_default,
        "solver": solve_ConsLaborIntMargSepMrkv,
        "model": "ConsLaborIntMargSepMrkv.yaml",
        "track_vars": ["aLvl", "bLvl", "mLvl", "cLvl", "Lbr", "Mrkv", "xLvl", "yLvl"],
    }

    # The Markov solver consumes Tau/Trnsfr/Div/Omega/xBar/Psi/LsrCurv per
    # period and WageRte per period; xGrid and MrkvArray are time-invariant.
    time_vary_ = copy(IndShockConsumerType.time_vary_)
    # Drop permanent-income time-vary entries inherited from the parent that
    # do not apply to this model.
    for _drop in ("PermShkStd", "TranShkStd", "PermGroFac", "LbrCost"):
        if _drop in time_vary_:
            time_vary_.remove(_drop)
    time_vary_ += [
        "WageRte", "Psi", "LsrCurv", "Tau", "Trnsfr", "Div", "Omega", "xBar",
    ]
    time_inv_ = copy(IndShockConsumerType.time_inv_)
    if "PermGroFacAgg" in time_inv_:
        time_inv_.remove("PermGroFacAgg")
    # ``MrkvArray`` and ``xGrid`` are time-invariant (one transition matrix
    # and one productivity grid used in every period of the cycle).
    time_inv_ += ["MrkvArray", "xGrid"]

    # The simulator works in *levels* (no permanent-income normalisation).
    state_vars = ["aLvl", "bLvl", "mLvl", "yLvl", "xLvl", "Mrkv"]
    shock_vars_ = ["Mrkv"]
    # ``MrkvInitDstn`` is lazily built from ``MrkvArray`` in initialize_sim;
    # don't list it here, otherwise ``reset_rng`` will choke on its default
    # ``None`` value before construction.
    distributions = ["kNrmInitDstn"]

    # ------------------------------------------------------------------
    # Simulation methods
    # ------------------------------------------------------------------
    def _make_MrkvInitDstn(self):
        """Lazy-construct ``MrkvInitDstn`` from the stationary distribution
        of ``MrkvArray`` if the user has not supplied one explicitly."""
        if not hasattr(self, "MrkvInitDstn") or self.MrkvInitDstn is None:
            self.MrkvInitDstn = make_stationary_MrkvDstn(self.MrkvArray)
        return self.MrkvInitDstn

    def initialize_sim(self):
        """Set up the simulator: zero out shock arrays, then defer to
        :meth:`AgentType.initialize_sim` which builds blank state arrays
        and calls :meth:`sim_birth` for the full population."""
        self.shocks["Mrkv"] = np.zeros(self.AgentCount, dtype=int)
        self._make_MrkvInitDstn()
        # Skip the IndShock/LaborIntMargConsumerType chain (which expects
        # PermShk/TranShk and pLvl) and go straight to AgentType.
        from HARK.core import AgentType
        AgentType.initialize_sim(self)
        # AgentType.initialize_sim sets every state_now slot to a float array
        # via np.empty(AgentCount).  Force Mrkv to int so downstream indexing
        # (and `np.bincount`-style operations) doesn't choke.
        self.state_now["Mrkv"] = self.state_now["Mrkv"].astype(int)

    def sim_birth(self, which_agents):
        """Initialise newborn agents.

        Draws initial asset *levels* from ``kNrmInitDstn`` (reinterpreted as
        a level distribution in this no-permanent-income model) and initial
        Markov productivity states from the stationary distribution of
        ``MrkvArray`` (or from ``MrkvInitDstn`` if the user provides one).
        """
        N = int(np.sum(which_agents))
        if N == 0:
            return
        # Reinterpret kNrmInitDstn as the initial-asset-level distribution.
        # In the Dupor model there is no permanent-income normaliser, so
        # "kNrm" is simply the agent's starting financial-asset level.
        self.state_now["aLvl"][which_agents] = self.kNrmInitDstn.draw(N)
        # All other states are computed during the period; just zero them
        # for now (the array slots already exist from initialize_sim).
        for var in ("bLvl", "mLvl", "yLvl", "xLvl"):
            self.state_now[var][which_agents] = 0.0
        # Draw initial Markov state from the stationary distribution.
        MrkvInit = self._make_MrkvInitDstn()
        MrkvInit.seed = self.RNG.integers(0, 2**31 - 1)
        self.state_now["Mrkv"][which_agents] = MrkvInit.draw(N).astype(int)
        self.shocks["Mrkv"][which_agents] = self.state_now["Mrkv"][which_agents]
        # Standard HARK bookkeeping for newborns.
        self.t_age[which_agents] = 0
        if not getattr(self, "PerfMITShk", False):
            self.t_cycle[which_agents] = 0

    def sim_death(self):
        """Use scalar :math:`1-\\mathrm{LivPrb}` to determine deaths each
        period (mortality is not Markov-state-dependent in this model).
        Identical in spirit to :meth:`IndShockConsumerType.sim_death`."""
        DiePrb_by_t_cycle = 1.0 - np.asarray(self.LivPrb)
        idx = self.t_cycle - 1 if self.cycles == 1 else self.t_cycle
        DiePrb = DiePrb_by_t_cycle[idx]
        DeathShks = Uniform(seed=self.RNG.integers(0, 2**31 - 1)).draw(
            N=self.AgentCount
        )
        who_dies = DeathShks < DiePrb
        if self.T_age is not None:
            too_old = self.t_age >= self.T_age
            who_dies = np.logical_or(who_dies, too_old)
        return who_dies

    def get_shocks(self):
        """Draw new Markov productivity state for each agent via
        :class:`HARK.distributions.MarkovProcess`.  Newborns and agents
        at ``t_sim == 0`` retain the state assigned in :meth:`sim_birth`."""
        MrkvPrev = self.shocks["Mrkv"]
        # Don't transition agents who were just born this period (their
        # Markov state was already drawn from MrkvInitDstn).
        dont_change = self.t_age == 0
        if self.t_sim == 0:
            dont_change[:] = True

        proc = MarkovProcess(
            self.MrkvArray, seed=self.RNG.integers(0, 2**31 - 1)
        )
        MrkvNow = proc.draw(MrkvPrev).astype(int)
        MrkvNow[dont_change] = MrkvPrev[dont_change]
        self.shocks["Mrkv"] = MrkvNow

    def _per_period_param(self, name):
        """Helper: pull the period-:math:`t` value of a time-varying
        parameter for each agent, using ``t_cycle`` as the index."""
        arr = np.asarray(getattr(self, name))
        return arr[self.t_cycle]

    def get_states(self):
        """Compute pre-labor cash on hand :math:`b_t` for each agent:

        .. math::
            b_t = (1+r_{t-1}) a_{t-1} + T_t.

        Because the dividend is now earned *per hour worked* (the
        intermediate-goods market equilibrium implies :math:`D_t \\propto
        H_t`), it does **not** enter the autonomous bank balance
        :math:`b_t` -- it shows up later in :meth:`get_poststates` as
        part of labor income :math:`y_t`.

        Time-varying parameters are looked up at ``t_cycle`` (the index of
        the period being simulated).  Interest factor ``Rfree`` is looked
        up at ``t_cycle - 1`` to match the solver's convention that
        ``Rfree[t]`` compounds savings from period ``t`` to period ``t+1``.
        """
        Mrkv = self.shocks["Mrkv"]
        xVals = np.asarray(self.xGrid)[Mrkv]

        Rfree_arr = np.asarray(self.Rfree)
        idx_R = self.t_cycle - 1 if self.cycles == 1 else self.t_cycle
        R = Rfree_arr[idx_R]

        Trnsfr = self._per_period_param("Trnsfr")

        bLvl = R * self.state_prev["aLvl"] + Trnsfr
        self.state_now["bLvl"] = bLvl
        self.state_now["xLvl"] = xVals
        self.state_now["Mrkv"] = Mrkv.copy()

    def get_controls(self):
        """Apply state-:math:`x`-specific policy functions
        ``solution[t].cFunc[j]`` and ``solution[t].LbrFunc[j]`` to each
        agent based on their current Markov state and period."""
        cLvl = np.full(self.AgentCount, np.nan)
        Lbr = np.full(self.AgentCount, np.nan)
        Mrkv = self.shocks["Mrkv"]
        J = self.MrkvArray.shape[0]

        for t in range(self.T_cycle):
            right_t = self.t_cycle == t
            if not np.any(right_t):
                continue
            soln = self.solution[t]
            for j in range(J):
                these = right_t & (Mrkv == j)
                if not np.any(these):
                    continue
                b_here = self.state_now["bLvl"][these]
                cLvl[these] = soln.cFunc[j](b_here)
                Lbr[these] = soln.LbrFunc[j](b_here)

        # Numerical guardrails: keep labor in [0, 1] and consumption >= 0.
        Lbr = np.clip(Lbr, 0.0, 1.0)
        cLvl = np.maximum(cLvl, 0.0)
        self.controls["cLvl"] = cLvl
        self.controls["Lbr"] = Lbr

    def get_poststates(self):
        """Compute end-of-period asset level :math:`a_t` and the
        intermediate quantities :math:`y_t` (labor income, *including*
        the dividend share earned per hour worked) and :math:`m_t`
        (post-labor market resources):

        .. math::
            y_t &= (1-\\tau_t)\\bigl[w_t x_t
                + (1-\\omega_t)(x_t/\\bar x_t)\\, D_t\\bigr] h_t, \\\\
            m_t &= b_t + y_t, \\quad
            a_t = m_t - c_t.
        """
        WageRte = self._per_period_param("WageRte")
        Tau = self._per_period_param("Tau")
        Omega = self._per_period_param("Omega")
        Div = self._per_period_param("Div")
        xBar = self._per_period_param("xBar")

        xVals = self.state_now["xLvl"]
        Lbr = self.controls["Lbr"]
        eff_wage = (1.0 - Tau) * (
            WageRte * xVals + (1.0 - Omega) * (xVals / xBar) * Div
        )
        yLvl = eff_wage * Lbr
        mLvl = self.state_now["bLvl"] + yLvl
        aLvl = mLvl - self.controls["cLvl"]
        # Enforce the natural borrowing constraint a >= 0.
        aLvl = np.maximum(aLvl, 0.0)

        self.state_now["yLvl"] = yLvl
        self.state_now["mLvl"] = mLvl
        self.state_now["aLvl"] = aLvl

    # Plotting helpers from the parent assume 2-D policy functions; disable
    # them here so users get a clear error if they call them by mistake.
    def plot_cFunc(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "plot_cFunc is not implemented for the Markov-income variant; "
            "the policy function is a list of 1-D interpolants over Mrkv."
        )

    def plot_LbrFunc(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "plot_LbrFunc is not implemented for the Markov-income variant; "
            "the policy function is a list of 1-D interpolants over Mrkv."
        )

    # ------------------------------------------------------------------
    # Sequence-Space Jacobian convenience wrapper
    # ------------------------------------------------------------------
    def make_SSJ(
        self,
        shock,
        outcomes,
        grids=None,
        eps=1e-4,
        T_max=300,
        solved=False,
        offset=False,
        verbose=False,
        **kwargs,
    ):
        r"""
        Construct sequence-space Jacobian matrices for this agent by
        delegating to :func:`HARK.SSJutils.make_basic_SSJ_matrices`.

        This is a thin wrapper that fills in sensible defaults for the
        ``grids`` argument and checks that the agent is in the
        one-period-infinite-horizon configuration required by the SSJ
        algorithm.

        Parameters
        ----------
        shock : str
            Name of the parameter to perturb (e.g. ``"Trnsfr"``, ``"Tau"``,
            ``"Div"``, ``"Rfree"``).
        outcomes : str or list of str
            Outcome variables to differentiate (e.g. ``"cLvl"``, ``"Lbr"``,
            ``"aLvl"``).
        grids : dict, optional
            Grid specification passed straight through to HARK.  If
            ``None``, a default grid over ``aLvl`` (200 nodes, [0, 50])
            and ``Mrkv`` (``MrkvCount`` nodes) is used.
        eps, T_max, solved, offset, verbose, **kwargs
            Forwarded to :func:`make_basic_SSJ_matrices` (see its docstring
            for full descriptions).

        Returns
        -------
        SSJ : np.ndarray or list of np.ndarray
            One SSJ per outcome.  Shape ``(T_max, T_max)`` each.

        Notes
        -----
        Requires that the agent has a symbolic model file (the YAML named
        in ``default_["model"]``).  See
        :func:`sketch_SSJ_usage_for_DuporAgent` for the YAML sketch.
        """
        if self.cycles != 0 or self.T_cycle != 1:
            raise ValueError(
                "make_SSJ requires a one-period infinite-horizon agent "
                "(cycles=0, T_cycle=1); got cycles={}, T_cycle={}.".format(
                    self.cycles, self.T_cycle
                )
            )
        if grids is None:
            grids = {
                "aLvl": {"N": 200, "min": 0.0, "max": 50.0},
                "Mrkv": {"N": int(self.MrkvArray.shape[0])},
            }
        # Lazy import to avoid a hard dependency on the latest HARK SSJ API
        # at module import time.
        from HARK.SSJutils import make_basic_SSJ_matrices

        return make_basic_SSJ_matrices(
            self,
            shock=shock,
            outcomes=outcomes,
            grids=grids,
            eps=eps,
            T_max=T_max,
            solved=solved,
            offset=offset,
            verbose=verbose,
            **kwargs,
        )


###############################################################################
# Sequence-Space Jacobian (SSJ) usage recipe
###############################################################################


def sketch_SSJ_usage_for_DuporAgent():
    r"""
    Recipe for using HARK's :mod:`HARK.SSJutils` to construct sequence-space
    Jacobians (SSJs) for an instance of :class:`LaborIntMargSepMrkvConsumerType`.

    The SSJ machinery from Auclert, Bardoczy, Rognlie, and Straub (2021) is
    available in HARK via two main entry points:

    * :func:`HARK.SSJutils.make_basic_SSJ_matrices`  -- one-period
      infinite-horizon (stationary) heterogeneous-agent models.
    * :func:`HARK.SSJutils.make_flat_LC_SSJ_matrices` -- life-cycle models
      with flat demographic dynamics.

    For the Dupor steady-state household problem, ``make_basic_SSJ_matrices``
    is the appropriate choice.  It requires the agent to:

    1. Be a *one-period infinite-horizon* model, i.e. ``cycles == 0`` and
       ``T_cycle == 1``.  By default this class is configured with
       ``cycles = 1, T_cycle = 1`` (matching the parent
       :class:`LaborIntMargConsumerType`), so switch to ``cycles = 0`` before
       calling SSJ:

       .. code-block:: python

           agent = LaborIntMargSepMrkvConsumerType(cycles=0, T_cycle=1)
           agent.solve()

    2. Carry a *symbolic model file* (a YAML descriptor) used by
       :meth:`HARK.core.AgentType.initialize_sym` to build the transition
       matrices the SSJ algorithm operates on.  The class advertises
       ``model = "ConsLaborIntMargSepMrkv.yaml"`` in ``default_``, but that
       YAML must actually exist on the import path.  A minimal sketch:

       .. code-block:: yaml

           # ConsLaborIntMargSepMrkv.yaml  (very rough sketch)
           symbols:
               states:    [aLvl, Mrkv]
               controls:  [cLvl, Lbr]
               outcomes:  [cLvl, Lbr, yLvl, aLvl]
               shocks:    [Mrkv_next]
           equations:
               # Markov transition for productivity
               - Mrkv_next ~ Categorical(MrkvArray[Mrkv, :])
               - xLvl  = xGrid[Mrkv]
               - bLvl  = Rfree*aLvl + Trnsfr        # autonomous cash on hand
               - cLvl  = cFunc[Mrkv](bLvl)
               - Lbr   = LbrFunc[Mrkv](bLvl)
               # Dividend is earned per hour worked (intermediate-goods
               # market clearing => D proportional to labor supply):
               - yLvl  = (1-Tau)*(WageRte*xLvl
                                  + (1-Omega)*(xLvl/xBar)*Div) * Lbr
               - aLvl' = bLvl + yLvl - cLvl

       Writing a full HARK-compliant YAML for this model is a one-time
       up-front cost, after which the entire SSJ toolbox becomes available.

    3. Supply ``grids`` describing the discretisation of each *arrival*
       state variable (everything other than what is normalised out).  In
       this model there is no permanent-income normalisation, so both
       ``aLvl`` and ``Mrkv`` need explicit grids:

       .. code-block:: python

           grids = {
               "aLvl": {"N": 200, "min": 0.0, "max": 50.0},
               "Mrkv": {"N": agent.MrkvArray.shape[0]},
           }

    Once those three pieces are in place, building an SSJ matrix for a
    given (shock, outcome) pair is one call:

    .. code-block:: python

        from HARK.SSJutils import make_basic_SSJ_matrices

        # SSJ of consumption with respect to a lump-sum transfer
        J_C_T = make_basic_SSJ_matrices(
            agent,
            shock="Trnsfr",
            outcomes=["cLvl", "Lbr", "aLvl"],
            grids=grids,
            T_max=300,
            verbose=True,
        )
        # J_C_T is a list [J_c, J_h, J_a] each of shape (T_max, T_max).

    Common ``shock`` choices for the Dupor agent:

    * ``"Trnsfr"`` -- fiscal lump-sum transfer T.
    * ``"Tau"``    -- proportional income-tax rate :math:`\tau`.
    * ``"Div"``    -- aggregate dividend D.
    * ``"Omega"``  -- dividend retention share :math:`\omega`.
    * ``"Rfree"``  -- ex-post interest factor (pass ``offset=True`` so HARK
      knows ``Rfree[t]`` represents the t+1 interest factor).
    * ``"WageRte"`` -- per-period wage rate.

    All of these are scalar (or singleton-list) parameters that
    ``make_basic_SSJ_matrices`` will perturb by ``eps`` to compute the
    fake-news matrix.

    Alternative paths if you do *not* want to write the YAML right away
    -----------------------------------------------------------------

    The Monte-Carlo simulation methods added to
    :class:`LaborIntMargSepMrkvConsumerType` can be used to *manually*
    construct an impulse-response column of the SSJ:

    1. Solve the stationary model (``cycles=0``) and simulate for many
       periods to reach the steady-state distribution; record steady-state
       aggregates :math:`\bar C, \bar L, \bar A`.
    2. Construct a finite-horizon copy of the agent with ``T_cycle = T_max``
       and ``cycles = 1``, where the chosen shock is perturbed by ``eps``
       at exactly one period :math:`s` and equals its steady-state value
       in all others.
    3. Solve this finite-horizon problem *backwards* from the stationary
       solution, then simulate the cohort forward starting from the
       stationary distribution.
    4. The column :math:`J[:, s]` of the SSJ equals
       :math:`(\\bar Y_t^{(\\text{shock})} - \\bar Y_t) / \\epsilon`.

    This is functionally what :func:`HARK.SSJutils.calc_shock_response_manually`
    does, but it can also be implemented directly on top of
    :meth:`simulate` if you prefer to avoid the symbolic-simulator
    infrastructure.
    """
    return None


###############################################################################
# Finite-lifecycle default dictionary for the Markov-income variant
###############################################################################

init_labor_lifecycle_sep_mrkv = init_labor_intensive_sep_mrkv.copy()
init_labor_lifecycle_sep_mrkv["LivPrb"] = [
    0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
]
init_labor_lifecycle_sep_mrkv["WageRte"] = [1.0] * 10
init_labor_lifecycle_sep_mrkv["Rfree"] = [1.04 ** 0.25] * 10  # quarterly
init_labor_lifecycle_sep_mrkv["Psi"] = [5.8] * 10
init_labor_lifecycle_sep_mrkv["LsrCurv"] = [1.0] * 10
init_labor_lifecycle_sep_mrkv["Tau"] = [0.0] * 10
init_labor_lifecycle_sep_mrkv["Trnsfr"] = [0.03] * 10
init_labor_lifecycle_sep_mrkv["Div"] = [0.0] * 10
init_labor_lifecycle_sep_mrkv["Omega"] = [0.32] * 10
init_labor_lifecycle_sep_mrkv["xBar"] = [1.0] * 10
init_labor_lifecycle_sep_mrkv["T_cycle"] = 10
init_labor_lifecycle_sep_mrkv["T_age"] = 11
