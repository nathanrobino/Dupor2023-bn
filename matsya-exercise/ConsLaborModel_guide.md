# Guide to `ConsLaborModel.py`

This document explains the structure and logic of `ConsLaborModel.py` for readers
familiar with dynamic programming in economics but new to Python. The file
implements a **consumption–saving model with endogenous intensive-margin labor
supply**, solved by backward induction using the **Endogenous Grid Method (EGM)**.

---

## Table of Contents

1. [The Economic Model](#1-the-economic-model)
2. [How the File Is Organized](#2-how-the-file-is-organized)
3. [Python Concepts You Will Encounter](#3-python-concepts-you-will-encounter)
4. [Component-by-Component Walkthrough](#4-component-by-component-walkthrough)
   - [4.1 `ConsumerLaborSolution`](#41-consumerlaborsolution)
   - [4.2 `make_log_polynomial_LbrCost`](#42-make_log_polynomial_lbrcost)
   - [4.3 `make_labor_intmarg_solution_terminal`](#43-make_labor_intmarg_solution_terminal)
   - [4.4 `solve_ConsLaborIntMarg`](#44-solve_conslaborintmarg)
   - [4.5 Parameter Dictionaries](#45-parameter-dictionaries)
   - [4.6 `LaborIntMargConsumerType`](#46-laborintmargconsumertype)
5. [Variable Naming Conventions](#5-variable-naming-conventions)
6. [Key NumPy Patterns](#6-key-numpy-patterns)

---

## 1. The Economic Model

### Preferences

Each period the agent gets utility from a **composite good** \( x_t \) that bundles
consumption \( c_t \) and leisure \( z_t = 1 - \ell_t \):

$$
u_t(c_t, \ell_t) = \frac{\left(c_t \, z_t^{\alpha_t}\right)^{1-\rho}}{1-\rho}
$$

where \( \rho > 0 \) is the CRRA coefficient (`CRRA`) and \( \alpha_t \geq 0 \) is
the **labor cost parameter** (`LbrCost`), which can vary with age. A higher
\( \alpha_t \) means leisure enters more strongly into utility, so the agent works
less at a given wage.

> **Tip for MATLAB users:** This is just a standard CRRA function evaluated at
> \( x_t = c_t \cdot z_t^{\alpha_t} \) instead of at \( c_t \) alone.

### Budget Constraints

The within-period flow of funds is:

| Variable | Meaning |
|---|---|
| \( b_t \) | **Bank balances** at the start of period \( t \), *before* earning labor income (normalized by permanent income \( p_t \)) |
| \( \theta_t \) | **Transitory productivity shock** (mean 1) |
| \( \ell_t \in [0,1] \) | Fraction of time spent **working** |
| \( m_t = b_t + \ell_t \cdot w_t \cdot \theta_t \) | **Market resources** after earning labor income |
| \( a_t = m_t - c_t \) | End-of-period **assets** (savings) |

Between periods, assets grow at the risk-free rate \( R \) and are deflated by
permanent income growth \( \Gamma_{t+1} \psi_{t+1} \):

$$
b_{t+1} = \frac{R \, a_t}{\Gamma_{t+1} \psi_{t+1}}
$$

where \( \psi_{t+1} \) is the **permanent income shock** (mean 1).

### Bellman Equation

$$
v_t(b_t, \theta_t) = \max_{c_t,\, \ell_t} \;
u_t(c_t, \ell_t)
+ \beta \,(1-D_{t+1})\,
\mathbb{E}_t\!\left[
  (\Gamma_{t+1}\psi_{t+1})^{1-\rho}\,
  v_{t+1}(b_{t+1}, \theta_{t+1})
\right]
$$

subject to the budget constraints above and \( c_t \geq 0,\; \ell_t \in [0,1] \).

Because the agent observes \( \theta_t \) **before** choosing \( c_t \) and
\( \ell_t \), the transitory shock is a **state variable** (not just an
expectation); this is what makes the state space two-dimensional
\((b_t,\, \theta_t)\).

### First-Order Conditions and EGM

Optimality over the composite good \( x_t \) gives a single FOC whose
closed-form inverse is used to map an exogenous grid of end-of-period assets
\( \{a_t\} \) directly to the implied \( b_t \) values—this is the **Endogenous
Grid Method**. No root-finding loop is needed.

---

## 2. How the File Is Organized

```
ConsLaborModel.py
│
├── ConsumerLaborSolution            # Data container: stores cFunc, LbrFunc, vPfunc, ...
│
├── make_log_polynomial_LbrCost      # Helper: builds age-varying α_t from polynomial coefficients
│
├── make_labor_intmarg_solution_terminal   # Builds the terminal-period solution (T)
│
├── solve_ConsLaborIntMarg           # Solves one interior period by EGM (called T-1, T-2, ...)
│
├── Parameter dictionaries           # Default values for all model parameters
│   ├── LaborIntMargConsumerType_IncShkDstn_default
│   ├── LaborIntMargConsumerType_aXtraGrid_default
│   ├── LaborIntMargConsumerType_LbrCost_default
│   ├── LaborIntMargConsumerType_solving_default
│   ├── LaborIntMargConsumerType_simulation_default
│   └── LaborIntMargConsumerType_default   (all of the above merged)
│
└── LaborIntMargConsumerType         # Agent class: holds parameters, calls solver, simulates
    ├── pre_solve()
    ├── get_states()
    ├── get_controls()
    ├── get_poststates()
    ├── plot_cFunc()
    └── plot_LbrFunc()
```

The **solver** (`solve_ConsLaborIntMarg`) and **agent class**
(`LaborIntMargConsumerType`) are kept separate deliberately. The solver is a
pure function: given next period's solution and this period's parameters, it
returns this period's solution. The class manages parameters, loops the solver
across time periods, and simulates agents.

---

## 3. Python Concepts You Will Encounter

### Classes

Think of a **class** as a MATLAB `struct` that can also carry functions
(called *methods*). You create an instance with `MyClass(arg1, arg2)`. Attributes
are accessed with dot notation: `solution.cFunc`.

```python
# Defining a class
class Foo:
    def __init__(self, x):   # __init__ is the constructor
        self.x = x           # self.x is an instance attribute

obj = Foo(3)
print(obj.x)  # → 3
```

### Inheritance

`LaborIntMargConsumerType` inherits from `IndShockConsumerType`. This means it
gets all of the parent class's methods for free and only *overrides* (replaces)
the ones that differ—`get_states`, `get_controls`, `get_poststates`. In MATLAB
terms, it is like calling a base script and then patching specific sub-functions.

### Dictionaries

A **dict** is like a named list in R or a struct in MATLAB:

```python
params = {"CRRA": 2.0, "DiscFac": 0.96}
params["Rfree"] = 1.03   # add a key
rho = params["CRRA"]     # retrieve a value
```

### `np.tile` and Array Broadcasting

MATLAB users will recognize `repmat`. `np.tile(A, (rows, cols))` replicates
array `A` that many times along each dimension. Much of the solver uses tiled
arrays so that vectorized operations can be applied across the entire
state–shock grid simultaneously (no explicit loops over grid points).

---

## 4. Component-by-Component Walkthrough

### 4.1 `ConsumerLaborSolution`

```python
class ConsumerLaborSolution(MetricObject):
    distance_criteria = ["cFunc", "LbrFunc"]
    def __init__(self, cFunc=None, LbrFunc=None, vFunc=None,
                 vPfunc=None, bNrmMin=None): ...
```

**What it is:** A plain data container for one period's solution. After the
solver runs, the solution for period \( t \) is stored as an instance of this
class.

**Key attributes:**

| Attribute | Type | Meaning |
|---|---|---|
| `cFunc` | 2-D interpolant | \( \hat{c}(b_t, \theta_t) \) — optimal normalized consumption |
| `LbrFunc` | 2-D interpolant | \( \hat{\ell}(b_t, \theta_t) \) — optimal labor supply |
| `vPfunc` | 2-D interpolant | \( v'(b_t, \theta_t) \) — marginal value of bank balances |
| `vFunc` | 2-D interpolant | \( v(b_t, \theta_t) \) — value function (terminal period only) |
| `bNrmMin` | 1-D interpolant | Minimum feasible \( b_t \) as a function of \( \theta_t \) |

`distance_criteria` tells the HARK toolkit which attributes to compare when
checking whether the solution has converged (for infinite-horizon problems).

---

### 4.2 `make_log_polynomial_LbrCost`

```python
def make_log_polynomial_LbrCost(T_cycle, LbrCostCoeffs):
```

**Purpose:** Converts a list of polynomial coefficients into a list of
age-varying \( \alpha_t \) values using:

$$
\alpha_t = \exp\!\left(\sum_{n=0}^{N-1} \text{LbrCostCoeffs}_n \cdot t^n\right),
\quad t = 0, 1, \ldots, T-1
$$

**Example:** `LbrCostCoeffs = [-2.0, 0.4]` gives a linearly increasing (in log)
labor cost that starts low (cheap leisure when young) and rises with age.

The output is a plain Python **list** of length `T_cycle`, one \( \alpha_t \) per
period. This is the value stored as `self.LbrCost` on the agent.

---

### 4.3 `make_labor_intmarg_solution_terminal`

```python
def make_labor_intmarg_solution_terminal(CRRA, aXtraGrid, LbrCost, WageRte, TranShkGrid):
```

**Purpose:** Constructs the terminal-period solution analytically. In the
terminal period there is no future, so the agent consumes all resources. The
labor supply satisfies a closed-form expression derived from the static FOC:

$$
z_T^* = \min\!\left(
  \frac{\alpha_T}{1+\alpha_T}
  \left(\frac{b_T}{w_T \theta_T} + 1\right),\; 1
\right)
$$

This function builds two-dimensional (`BilinearInterp`) interpolating functions
over the grid \((b_T, \theta_T)\) and packages them into a
`ConsumerLaborSolution` object.

> **Why a function, not a method on the class?** HARK separates the
> *construction* of the terminal solution from the agent class so that users
> can swap in a different terminal condition without subclassing.

---

### 4.4 `solve_ConsLaborIntMarg`

```python
def solve_ConsLaborIntMarg(solution_next, PermShkDstn, TranShkDstn, ...):
```

This is the **core solver**. It is called once per period, working backwards from
the terminal period. Here is the algorithm in plain language:

#### Step 1 — Expected marginal value of next-period bank balances

Take next period's marginal value function `vPfunc_next(b_{t+1}, θ_{t+1})` and
integrate over the *transitory* shock to get the expected marginal value *before*
the shock is realized:

$$
\bar{v}^{\prime}(b_{t+1}) = \sum_j \pi^{\theta}_j \cdot v'_{t+1}(b_{t+1},\, \theta_j)
$$

This is computed on a grid by calling `vPfunc_next` on all combinations of
\( b \) and \( \theta \) simultaneously (vectorized, using `np.tile`).

#### Step 2 — Integrate over permanent shocks

Apply the Euler equation to translate end-of-period assets \( a_t \) into an
expected marginal value of saving:

$$
\mathbb{E}\!\left[\text{EndOfPrd}\,v'_a\right]
= \beta R \,(1-D)\sum_k \pi^{\psi}_k
  (\Gamma\psi_k)^{-\rho}\,\bar{v}^{\prime}\!\left(\frac{R\,a_t}{\Gamma\psi_k}\right)
$$

#### Step 3 — Invert the FOC for the composite good

The FOC for the composite good \( x_t = c_t z_t^{\alpha} \) has an analytic
solution. A scaling factor that depends on \( (w_t, \theta_t, \alpha_t, \rho) \)
converts `EndOfPrdvP` into optimal \( x_t^* \), and then into \( c_t^* \) and
\( z_t^* \) separately.

#### Step 4 — Enforce constraints

If the unconstrained solution implies \( z_t > 1 \) (would require negative
labor), clamp \( z_t = 1 \) (no work) and recompute consumption from the
simplified Euler equation.

#### Step 5 — Recover endogenous grid of \( b_t \)

Invert the within-period budget constraint to find the \( b_t \) value that
would have led each \( (a_t, \theta_t) \) pair:

$$
b_t = a_t - w_t \theta_t + c_t + w_t \theta_t z_t
$$

This is the "endogenous" part of EGM: the \( b_t \) grid is an *output* of the
computation, not an input.

#### Step 6 — Interpolate policy functions

For each transitory shock value \( \theta_j \), build a linear interpolation
of \( c^*(b_t) \) and \( \ell^*(b_t) \) over the endogenous \( b_t \) grid.
Combine the shock-specific interpolants into a 2-D function using
`LinearInterpOnInterp1D`, then wrap with `VariableLowerBoundFunc2D` to handle
the varying lower bound of the state space.

#### Returns

A `ConsumerLaborSolution` containing `cFuncNow`, `LbrFuncNow`, `vPfuncNow`,
and `bNrmMinNow`.

---

### 4.5 Parameter Dictionaries

Several dictionaries group parameters by purpose. They are later merged into
a single flat dictionary `LaborIntMargConsumerType_default` (equivalent to
`init_labor_intensive`).

| Dictionary | Contents |
|---|---|
| `..._IncShkDstn_default` | Income shock process parameters (std devs, grid counts, unemployment) |
| `..._aXtraGrid_default` | Asset grid parameters (min, max, count, nesting factor) |
| `..._LbrCost_default` | Polynomial coefficients for \( \alpha_t \) |
| `..._solving_default` | Model structure: CRRA, Rfree, DiscFac, LivPrb, WageRte, … |
| `..._simulation_default` | Simulation settings: AgentCount, T_age, … |
| `..._kNrmInitDstn_default` | Initial distribution of normalized capital |
| `..._pLvlInitDstn_default` | Initial distribution of permanent income |

There is also `init_labor_lifecycle` at the bottom of the file — a variant of
the default dictionary configured for a 10-period finite life-cycle model with
declining survival probabilities and an age-varying labor cost.

---

### 4.6 `LaborIntMargConsumerType`

```python
class LaborIntMargConsumerType(IndShockConsumerType):
```

This class inherits everything from `IndShockConsumerType` (a standard
consumption–saving agent in HARK) and overrides three simulation methods to
accommodate the labor supply choice.

#### Class-level attributes

```python
time_vary_ = [..., "WageRte", "LbrCost", "TranShkGrid"]
```

Attributes listed in `time_vary_` are treated as **lists** (one element per
period), while those in `time_inv_` are scalar. This is how HARK handles
time-varying parameters in a lifecycle or cyclic model.

#### `pre_solve()`

Called once before backward induction begins. Constructs the terminal period
solution by calling `make_labor_intmarg_solution_terminal` through HARK's
`construct()` machinery.

#### `get_states()`

Calls the parent class's `get_states()` (which computes `bNrm` and `pLvl`),
then intentionally sets `mNrm = NaN` because market resources depend on labor
supply, which is not yet known.

#### `get_controls()`

Loops over all time periods and, for each agent at period \( t \), evaluates the
policy functions:

```python
cNrmNow[these] = self.solution[t].cFunc(bNrm[these], TranShk[these])
LbrNow[these]  = self.solution[t].LbrFunc(bNrm[these], TranShk[these])
```

> **MATLAB analogy:** This is like evaluating a function handle stored in a
> cell array, e.g. `cFunc{t}(bNrm, TranShk)`.

#### `get_poststates()`

Computes end-of-period quantities after controls are known:

```
LbrEff = Lbr * TranShk       (effective labor units)
yNrm   = LbrEff * WageRte    (labor income)
mNrm   = bNrm + yNrm         (market resources)
aNrm   = mNrm - cNrm         (savings)
```

#### `plot_cFunc()` / `plot_LbrFunc()`

Convenience methods that evaluate and plot the consumption or labor function for
period `t` at a set of transitory shock values `ShkSet`. Each shock value
produces one curve in the figure.

---

## 5. Variable Naming Conventions

The codebase follows consistent naming conventions:

| Suffix / Prefix | Meaning |
|---|---|
| `Nrm` | Normalized by permanent income \( p_t \) |
| `Lvl` | Level (not normalized) |
| `b` | Bank balances at the **start** of period, before labor income |
| `m` | Market resources after labor income |
| `a` | End-of-period assets (savings) |
| `Lbr` | Labor supply \( \ell_t \in [0,1] \) |
| `Lsr` | Leisure \( z_t = 1 - \ell_t \) |
| `TranShk` | Transitory income/productivity shock \( \theta_t \) |
| `PermShk` | Permanent income shock \( \psi_t \) |
| `vP` | Marginal value \( v'(\cdot) \) |
| `vPnvrs` | Pseudo-inverse of marginal value \( (v')^{-1}(\cdot) \equiv (u')^{-1}(v'(\cdot)) \) |
| `_rep` | Tiled/replicated array (used for vectorized operations over a grid) |
| `_list` | List of interpolating functions, one per shock node |
| `_T` | Terminal period quantity |
| `Func` | An interpolating function object |

---

## 6. Key NumPy Patterns

### Creating a tiled grid (cf. `repmat`)

```python
# In MATLAB: repmat(bNrmGrid(:), 1, TranShkCount)
bNrmGrid_rep = np.tile(bNrmGrid.reshape(-1, 1), (1, TranShkCount))
# Shape: (aXtraCount, TranShkCount)

# In MATLAB: repmat(TranShkVals(:)', aXtraCount, 1)
TranShkVals_rep = np.tile(TranShkVals.reshape(1, -1), (aXtraCount, 1))
# Shape: (aXtraCount, TranShkCount)
```

### Vectorized expectation (cf. `sum(..., 2)`)

```python
# E[f(b, θ)] over θ, for each b
vPbarNext = np.sum(vPNext * TranShkPrbs_rep, axis=1)
# In R: rowSums(vPNext * TranShkPrbs_rep)
```

### Prepending a point to an array (`np.insert`)

```python
# Add a zero at the front of a 1-D array
bNrmGrid_ext = np.insert(bNrmGrid, 0, 0.0)
# Equivalent to: [0.0; bNrmGrid] in MATLAB or c(0, bNrmGrid) in R
```

### Boolean indexing to enforce constraints

```python
violates = LsrNow > 1.0          # logical array, same shape as LsrNow
LsrNow[violates] = 1.0           # set all offending entries to 1
cNrmNow[violates] = uPinv(...)   # recompute consumption at those entries
# In MATLAB: LsrNow(violates) = 1; etc.
```

---

*This guide covers the full code in `ConsLaborModel.py` as it exists in the
`matsya-exercise/` folder. The upstream HARK version of this file lives in
`HARK/ConsumptionSaving/ConsLaborModel.py` and may differ slightly.*
