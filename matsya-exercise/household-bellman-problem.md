# Household Bellman Problem — Dupor et al. (2023)

Excerpted from `Subfiles/model.tex` (Section 4), written as a **partial-equilibrium household problem** with explicit stage decomposition following the SolvingMicroDSOPs / dolo-plus pedagogical style. The regional mutual fund, intermediate firms, and all aggregate prices are exogenous.

---

## Timing Within a Period

The usual analysis packs many events into a single complex decision. Following the approach of SolvingMicroDSOPs, we decompose everything that happens within a period into distinct operations so that each element can be understood in isolation.

1. **Arrival perch** $[\prec]$: The household enters the period with assets $a$ from last period's savings decision. The Markov productivity state $x$ has not yet been revealed.

2. **Shock realization** (arrival $\to$ decision transition): Nature draws the new productivity $x' \sim \Pi(\cdot \mid x)$, where $x$ is last period's realised productivity. Non-labour market resources are then determined:

$$
m \coloneqq (1+r)\,a + (1-\tau)\,\frac{x'}{\bar{x}}\,(1-\omega)\,D + T
$$

3. **Decision perch** $[\sim]$: The household observes $(m, x')$ and jointly chooses consumption $c$ and labour supply $h$. The full budget available for spending is market resources plus after-tax labour income:

$$
c + a' = m + (1-\tau)\,w\,x'\,h
$$

4. **Continuation perch** $[\succ]$: End-of-period assets $a' = m + (1-\tau)\,w\,x'\,h - c$ are determined. The household carries $(a', x')$ into the next period's arrival perch.

---

## Stage Flow Diagram

```
  ┌──────────────┐   Π(x'|x)    ┌──────────────┐  max_{c,h}   ┌──────────────┐
  │  Arrival [<]  │─────────────→│  Decision [~] │────────────→│  Contin. [>]  │
  │    (a, x)     │  m = f(a,x') │    (m, x')    │ a'=m+…−c   │   (a', x')    │
  └──────────────┘               └──────────────┘              └──────────────┘
        ↑                                                             │
        └───────────────────── next period ←──────────────────────────┘
```

---

## State and Control Spaces

| Perch | Variable | Domain | Description |
|:------|:---------|:-------|:------------|
| Arrival $[\prec]$ | $a$ | $\mathsf{X}_a \subseteq \mathbb{R}_+$ | Beginning-of-period assets |
| Arrival $[\prec]$ | $x$ | $\mathsf{X}_x$ (finite) | Last period's Markov productivity (conditioning variable for $x'$) |
| Exogenous | $x'$ | $\mathsf{X}_x$ | Markov productivity draw (resolves at arrival $\to$ decision) |
| Decision $[\sim]$ | $m$ | $\mathsf{X}_m \subseteq \mathbb{R}_+$ | Non-labour market resources (constructed) |
| Decision $[\sim]$ | $x'$ | $\mathsf{X}_x$ | Realised productivity |
| Control | $c$ | $\mathbb{R}_+$ | Consumption |
| Control | $h$ | $[0, 1]$ | Labour supply (fraction of time endowment) |
| Continuation $[\succ]$ | $a'$ | $\mathsf{X}_a$ | End-of-period assets (endogenous poststate) |
| Continuation $[\succ]$ | $x'$ | $\mathsf{X}_x$ | Realised productivity (exogenous carry-forward) |

The Markov index $x$ appears at both arrival and continuation perches: the current realisation $x'$ passes through as a sufficient statistic for next period's Markov draw.

---

## Step 1: Arrival $\to$ Decision Transition

The household arrives with assets $a$ and last period's Markov index $x$. Nature draws the new productivity:

$$
x' \sim \Pi(\cdot \mid x) \tag{shock}
$$

Non-labour market resources are constructed from the return on assets, the exogenous dividend share, and the lump-sum transfer:

$$
m \coloneqq \underbrace{(1+r)\,a}_{\text{asset return}} + \underbrace{(1-\tau)\,\frac{x'}{\bar{x}}\,(1-\omega)\,D}_{\text{after-tax profit-sharing income}} + \underbrace{T}_{\text{lump-sum transfer}} \tag{arvl→dcsn}
$$

The variable $m$ bundles all non-labour resources into a single sufficient statistic, analogous to "market resources" (or "cash-on-hand") in SolvingMicroDSOPs. After this transition the decision-perch state is $(m, x')$.

**Information at the decision perch:** the household observes both $m$ and $x'$ before choosing $(c, h)$.

---

## Step 2: The Decision Problem (Bellman Equation)

Given $(m, x')$, the household solves:

$$
\mathrm{v}(m, x') = \max_{c,\, h} \left\{ u(c, h) + \beta\, \mathrm{v}_{[\succ]}(a', x') \right\} \tag{4.3}
$$

subject to the budget constraint:

$$
a' = m + (1-\tau)\,w\,x'\,h - c \tag{4.4}
$$

and the bounds:

$$
c > 0, \qquad h \in [0, 1], \qquad a' \geq 0 \tag{4.5}
$$

### Period Utility

$$
u(c, h) = \log(c) + \psi\,\frac{(1-h)^{1-\theta}}{1-\theta}
$$

where $\log(c)$ gives unit elasticity of intertemporal substitution, and $\theta$ governs the Frisch elasticity of labour supply.

### The Joint $(c, h)$ Choice

Unlike the standard buffer-stock model (which has a single control $c$), this problem has two controls chosen simultaneously. The intratemporal first-order condition for $h$ is:

$$
\psi\,(1-h)^{-\theta} = \frac{(1-\tau)\,w\,x'}{c}
$$

which pins down $h$ as a function of $c$ and $x'$:

$$
h = 1 - \left(\frac{c\,\psi}{(1-\tau)\,w\,x'}\right)^{1/\theta}
$$

This analytical reduction means the problem can (in principle) be reduced to a single effective control $c$, with $h$ determined by the intratemporal condition. However, the EGM inversion step becomes non-standard because labour income $w\,x'\,h$ depends on the control.

---

## Step 3: Decision $\to$ Continuation Transition

Given optimal choices $(c^*, h^*)$, end-of-period assets are:

$$
a' = m + (1-\tau)\,w\,x'\,h^* - c^* \tag{dcsn→cntn}
$$

The continuation-perch state is $(a', x')$. The household carries both into the next period's arrival perch, where $a'$ becomes next period's $a$ and $x'$ becomes next period's conditioning variable for the Markov draw.

---

## Step 4: Continuation $\to$ Arrival Mover (Expectation)

The arrival value function integrates over the pre-decision Markov shock:

$$
\mathrm{v}_{[\prec]}(a, x) = \mathbb{E}_{x' \mid x}\!\left[\,\mathrm{v}\!\bigl(m(a, x'),\, x'\bigr)\,\right] = \sum_{x' \in \mathsf{X}_x} \Pi(x' \mid x)\;\mathrm{v}\!\bigl(m(a, x'),\, x'\bigr) \tag{dcsn→arvl mover}
$$

Because the Markov shock is **pre-decision** (resolves at arrival $\to$ decision), the expectation operator sits at the `dcsn_to_arvl_mover`, not inside the $\max$. The decision-perch problem is deterministic conditional on $(m, x')$.

The marginal (envelope) condition for the arrival value is:

$$
\mathrm{v}'_{[\prec]}(a, x) = \mathbb{E}_{x' \mid x}\!\left[\,(1+r)\,/\,c^*(m(a,x'), x')\,\right]
$$

This marginal value is the object that the Euler equation from the previous period's decision perch must satisfy.

---

## Exogenous Parameters (from GE)

These are determined in general equilibrium but are fixed from the household's perspective:

| Symbol | Meaning | Origin |
|:-------|:--------|:-------|
| $w$ | Real wage | Labour-market clearing (intermediate firms) |
| $r$ | Real return on savings | Mutual-fund no-arbitrage: $1+r = (1+R)/(1+\pi)$ |
| $D$ | Per-capita real dividends | Intermediate firms' profits |
| $\omega$ | Dividend retention share | Mutual-fund structure |
| $\tau$ | Proportional income-tax rate | Government budget constraint |
| $T$ | Lump-sum transfer | Government budget constraint |
| $\bar{x}$ | Mean productivity | Normalisation |

---

## Preference and Process Parameters

| Symbol | Meaning |
|:-------|:--------|
| $\beta$ | Discount factor (two types per region, solved separately) |
| $\psi$ | Leisure utility weight |
| $\theta$ | Leisure curvature (Frisch elasticity parameter) |
| $\Pi(\cdot \mid x)$ | Markov transition matrix for $x$ (discretised AR(1): $\log x' = \rho \log x + \eta$) |
| $\rho$ | Productivity persistence |
| $\sigma_\eta$ | Productivity innovation s.d. |

---

## Structural Summary

| Design question | Dupor2023 answer | DDSL rationale |
|:----------------|:-----------------|:---------------|
| Shock timing | Pre-decision (Markov $x'$ resolves at arrival $\to$ decision) | Expectation in `dcsn_to_arvl_mover`, not inside `max` |
| Decision variables | Joint $(c, h)$ | Single decision perch, two controls |
| State at decision | $(m, x')$ — constructed | $m$ bundles asset return + dividends + transfer |
| Continuation state | $(a', x')$ | $a'$ is endogenous poststate; $x'$ is exogenous carry-forward |
| Budget constraint | $a' = m + (1-\tau)wx'h - c$ | `dcsn_to_cntn_transition` |
| Borrowing constraint | $a' \geq 0$ | Poststate domain bound |
| PE vs GE | $w, r, D, T$ exogenous | Clean interface: these become equilibrium objects in GE extension |

---

*Source*: Dupor, Karabarbounis, Kudlyak, and Mehkari (2023), "Regional Consumption Responses and the Aggregate Fiscal Multiplier," Section 4.
*Stage decomposition style*: Following Carroll (2024), "Solving Microeconomic Dynamic Stochastic Optimization Problems," Section 1 ("The Problem").
