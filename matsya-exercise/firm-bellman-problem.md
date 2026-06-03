# Firm Bellman Problem - Dupor et al. (2023)

Excerpted from `Subfiles/model.tex` (Section 4), written as a **partial-equilibrium firm problem** with explicit stage decomposition following the SolvingMicroDSOPs / dolo-plus pedagogical style.  The household distribution, the regional mutual fund, the government, and all of the cross-regional aggregate variables (other-region prices, outputs, inflation) are exogenous from any individual firm's point of view.

The paper has **two layers of firms** in each region *i*:

1. **Final-good firms** &mdash; one per region, perfectly competitive, with CES production over intermediate inputs. Their problem is *static* (no Bellman) and is summarised first for completeness.
2. **Intermediate-good firms** &mdash; a continuum indexed by *(i, j)*, monopolistically competitive, with linear technology in labour and **Calvo** price stickiness. This is the genuinely dynamic problem, and is the focus of the rest of the document.

---

## Part A &mdash; Final-good firm (static)

Each region *i* has a single representative final-good producer who combines intermediates from all regions via the CES technology of equation (4.11):

$$
Y_{i} \;=\; \left[\; \sum_{i'=1}^{N} \gamma_{ii'}^{\,1/\epsilon} \int_{j} v_{i,i',j}^{\,(\epsilon-1)/\epsilon} \, \mathrm{d}j \;\right]^{\epsilon/(\epsilon-1)}
$$

with home-bias weights *gamma\_{ii'}* summing to 1 over *i'*. Cost-minimisation against the given vector of intermediate prices *p\_{i',j}* yields the standard CES demand schedule (eq. 4.12):

$$
v_{i,i',j} \;=\; \gamma_{ii'} \left[\frac{p_{i',j}}{P_i}\right]^{-\epsilon} Y_i
$$

and zero-profit competition pins down the price aggregator (eq. 4.13):

$$
P_i \;=\; \left[\; \sum_{i'=1}^{N} \gamma_{ii'} \int_j p_{i',j}^{\,1-\epsilon}\, \mathrm{d}j \;\right]^{1/(1-\epsilon)}.
$$

No state variables, no controls (the final-good firm passes demand straight through to the intermediate sector), and no value function. The whole problem reduces to two algebraic identities. **Skip ahead to Part B for the recursive problem.**

---

## Part B &mdash; Intermediate-good firm (dynamic Calvo)

### Setup

Firm *(i, j)* is a monopolistically-competitive producer of intermediate variety *j* located in region *i*. It has

* **linear-in-labour technology**:  *y\_{i,j,t} = L\_{i,j,t}*  (eq. 4.14),
* **demand** from final-good firms in *all* regions via CES:
  $y_{i,j,t} = \sum_{i'} \mu_{i'} v_{i',i,j,t} = \sum_{i'} \mu_{i'} \gamma_{i'i} \big[p_{i,j,t}/P_{i',t}\big]^{-\epsilon} Y_{i',t}$  (eq. 4.15),
* **Calvo price stickiness**: in every period each firm independently draws a reset opportunity with probability *lambda*. With probability *1 - lambda* it must keep its existing nominal price.
* **discount factor**: the firm discounts using the (regional) mutual-fund stochastic discount factor, $\tilde\beta_{i,t+1} = 1/(1+r_{i,t+1})$.

Since the technology, demand and discount factor are *identical across firms within a region* and depend on *(i, j)* only through the firm's own price *p\_{i,j,t}*, all firms that draw the reset opportunity in period *t* choose the same reset price *p\*\_{i,t}*. Idiosyncratic firm heterogeneity collapses to **cohort heterogeneity**: which period each firm last reset.

### Timing Within a Period

The usual analysis packs many events into a single complex decision. Following the household markdown's approach, we decompose everything that happens within a period into distinct operations so that each element can be understood in isolation.

1. **Arrival perch** prec: The firm enters the period carrying last period's nominal price *p\_{i,j,t-1}*, and observes the **regional aggregate state** *s\_i = (w\_i, P\_i, pi\_i, Y\_i, Y\_{-i}, Q\_{-i,i}, r\_i)*. Neither the Calvo draw nor the household demand has resolved yet.
2. **Calvo lottery** (arrival -> decision transition): Nature draws *zeta\_{j,t}* in {**reset**, **stuck**}, with `Pr[reset] = lambda` and `Pr[stuck] = 1 - lambda`, independently across firms.
3. **Decision perch** dec:
   * If *zeta = stuck*, the firm has no choice; the **effective decision variable is degenerate**: *p\_{i,j,t} = p\_{i,j,t-1}*.
   * If *zeta = reset*, the firm chooses *p\*\_{i,j,t}* to maximise the discounted PV of profits over all future periods in which this same price might still be in effect.
4. **Production and sales** (decision -> continuation transition): Given the realised price *p\_{i,j,t}*, demand *y\_{i,j,t}* is read off the CES schedule. The firm hires *L\_{i,j,t} = y\_{i,j,t}* units of labour at wage *w\_{i,t}*, sells at price *p\_{i,j,t}*, and remits real profits *d\_{i,j,t} = (p\_{i,j,t}/P\_{i,t}) y\_{i,j,t} - w\_{i,t} y\_{i,j,t}* (eq. 4.20) to the mutual fund (and, indirectly, to households).
5. **Continuation perch** cntn: The firm carries the (just-realised) price *p\_{i,j,t}* into the next period's arrival perch.

### Stage Flow Diagram

```
                Calvo: lambda          max p*        produce, sell, pay wages
                       (1 - lambda)    (no choice)
+--------------+   zeta in {R,S}    +--------------+   y, L, d   +--------------+
| Arrival [prec] |---------------->| Decision [dec] |----------->| Cont. [cntn] |
|  (p_{t-1}, s)  |                 |   (p_{t-1}, s,|             |   (p_t, s)   |
|                |                 |        zeta)  |             |              |
+--------------+                   +--------------+             +--------------+
        ^                                                                |
        +------------------- next period (s -> s') ----------------------+
```

The aggregate state *s* evolves exogenously from the firm's perspective; the firm's *own* state is just its lagged price.

### State and Control Spaces

| Perch             | Variable                              | Domain               | Description                                                                            |
| ----------------- | ------------------------------------- | -------------------- | -------------------------------------------------------------------------------------- |
| Arrival prec      | *p\_{t-1}*                            | positive             | Lagged nominal price the firm carries in                                               |
| Arrival prec      | *s*                                   | aggregate state set  | Vector *(w\_i, P\_i, pi\_i, Y\_i, Y\_{-i}, Q\_{-i,i}, r\_i)* taken as given            |
| Exogenous         | *zeta*                                | {reset, stuck}       | Calvo draw realised at arrival -> decision                                             |
| Decision dec      | *(p\_{t-1}, s, zeta)*                 | -                    | Information at the decision perch                                                      |
| Control           | *p\**                                 | positive (if reset)  | Reset price, chosen only if *zeta = reset*; if *zeta = stuck*, *p\_t = p\_{t-1}* (no choice) |
| Implied (no choice) | *L = y(p\_t, s),  d = (p\_t/P) y - w L* | nonnegative          | Labour and dividends are residuals once price and *s* are fixed                        |
| Continuation cntn | *p\_t*                                | positive             | Carry-forward price (either *p\** or *p\_{t-1}*)                                       |

The aggregate state *s* appears at both arrival and continuation perches: from the firm's perspective it follows an exogenous law of motion that is determined in the GE block, not by the firm itself.

### Step 1: Arrival -> Decision Transition

The firm arrives with price *p\_{t-1}* and aggregate state *s*. Two things happen.

**(a) Calvo draw.** Nature draws

$$
\zeta_{j,t} \sim \text{Bernoulli}(\lambda), \qquad
\text{reset with prob } \lambda, \;\; \text{stuck with prob } 1-\lambda.
$$

*(Independent across firms; resolves at arrival -> decision.)*

**(b) Aggregate update.** The exogenous law of motion *s\_t -> s\_{t+1}*  is given by the rest of the model (household block, mutual fund, monetary authority, government, and the other firms). From the perspective of a *single* intermediate firm, *s* is a sufficient statistic, exactly analogous to *m* (market resources) in the household problem.

**Information at the decision perch:** the firm observes *p\_{t-1}*, *s*, and *zeta* before choosing *p\**.

### Step 2: The Decision Problem (Bellman Equation)

Define the firm's per-period real flow profit at price *p* and aggregate state *s* as

$$
\pi(p, s) \;\equiv\; \frac{p}{P_i}\, y(p, s) \;-\; w_i\, y(p, s)
\;=\; \left(\frac{p}{P_i} - w_i\right) y(p, s),
$$

where the demand function *y(p, s)* is given by the CES aggregator:

$$
y(p, s) \;=\; \sum_{i'} \mu_{i'} \gamma_{i'i}\left[\frac{p}{P_{i'}}\right]^{-\epsilon} Y_{i'}.
$$

Let *V\_{stuck}(p, s)* denote the value to a firm that arrives with price *p* and state *s* and is **not** allowed to reset, and *V\_{reset}(s)* the value to a firm that **is** allowed to reset. The dynamic-programming pair is

$$
V_{stuck}(p, s) \;=\; \pi(p, s) \;+\; \tilde\beta_i \, \mathbb{E}_{s'\mid s}\!\Big[\,\lambda\, V_{reset}(s') \;+\; (1-\lambda)\, V_{stuck}\!\big(p / (1+\pi_i'),\, s'\big)\Big],
$$

$$
V_{reset}(s) \;=\; \max_{p^*}\Big\{\, \pi(p^*, s) \;+\; \tilde\beta_i\, \mathbb{E}_{s'\mid s}\!\big[\,\lambda\, V_{reset}(s') \;+\; (1-\lambda)\, V_{stuck}\!\big(p^* / (1+\pi_i'),\, s'\big)\big]\,\Big\}.
$$

The *p / (1+pi\_i')* term inside *V\_{stuck}* records that a nominal price held over to next period is mechanically *deflated* in real terms by next period's inflation. Equivalently one can write the problem in *real* prices, in which case the law of motion of the stuck firm's relative price is $\tilde p_t = \tilde p_{t-1} / (1+\pi_t)$.

The combined value at the arrival perch is

$$
V(p, s) \;=\; \lambda\, V_{reset}(s) \;+\; (1-\lambda)\, V_{stuck}(p, s).
$$

#### Paper's closed-form FOC (equations 4.16-4.19)

Because the demand schedule is CES with elasticity *epsilon* and technology is linear, the reset-price first-order condition admits the well-known closed-form ratio of two infinite sums of discounted future marginal-cost-weighted and demand-weighted aggregates (eq. 4.17):

$$
\frac{p_{i,t}^{*}}{P_{i,t}} \;=\; \frac{\epsilon}{\epsilon-1}\,\cdot\, \frac{\sum_{i'} \mu_{i'} \gamma_{i'i}\, Q_{i',i,t}^{\epsilon}\,\big[\, w_{i,t} Y_{i',t} + (1-\lambda)\, \tilde\beta_i\, (1+\pi_{i',t+1})^{1+\epsilon}\, \mathcal{X}_{i',i,t+1}\,\big]}{\sum_{i'} \mu_{i'} \gamma_{i'i}\, Q_{i',i,t}^{\epsilon}\,\big[\, Y_{i',t} + (1-\lambda)\, \tilde\beta_i\, (1+\pi_{i',t+1})^{\epsilon}\, \mathcal{Z}_{i',i,t+1}\,\big]}
$$

with two recursive auxiliary objects (eq. 4.18, 4.19):

$$
\mathcal{X}_{i',i,t} \;=\; w_{i,t} Y_{i',t} \;+\; (1-\lambda)\, \tilde\beta_i\, (1+\pi_{i',t+1})^{1+\epsilon}\, \mathcal{X}_{i',i,t+1},
$$

$$
\mathcal{Z}_{i',i,t} \;=\; Y_{i',t} \;+\; (1-\lambda)\, \tilde\beta_i\, (1+\pi_{i',t+1})^{\epsilon}\, \mathcal{Z}_{i',i,t+1}.
$$

*Mechanically*: *X* is the discounted future stream of marginal-cost-weighted demand, *Z* is the discounted future stream of demand. Their ratio (times the markup *epsilon/(epsilon-1)*) is the optimal reset price. The recursions *are not value functions*; they are summary statistics of the discounted future aggregates that appear in the firm's FOC.

### Step 3: Decision -> Continuation Transition

Given the *realised* price *p\_t* (either *p\** or *p\_{t-1}*, depending on *zeta*) and aggregate state *s*, the implied flows are

$$
y_{i,j,t} \;=\; \sum_{i'} \mu_{i'} \gamma_{i'i}\left[\frac{p_{i,j,t}}{P_{i',t}}\right]^{-\epsilon} Y_{i',t}, \qquad
L_{i,j,t} \;=\; y_{i,j,t},
$$

$$
d_{i,j,t} \;=\; \frac{p_{i,j,t}}{P_{i,t}}\, y_{i,j,t} \;-\; w_{i,t}\, y_{i,j,t}.
$$

*(Eqs. 4.14 and 4.20.)* Per-capita real dividends in region *i* are *D\_{i,t} = int\_j d\_{i,j,t}*; these become an input to the household's and mutual fund's problems.

The continuation-perch state is *(p\_t, s)*.

### Step 4: Continuation -> Arrival Mover (Expectation)

Between *t* and *t+1*, two exogenous things happen: the aggregate state evolves according to the GE block, and the price erodes in *real* terms by the regional inflation factor:

$$
p_{t} \;\longmapsto\; p_{t} \;\;\text{(nominal carry, unchanged)} \qquad \text{but in *real* terms} \;\; \tilde p_{t+1} = \tilde p_t / (1+\pi_{i,t+1}).
$$

The arrival value at *t+1* integrates over the joint distribution of *s'* and the next-period Calvo draw:

$$
V(p, s) \;=\; \mathbb{E}_{s'\mid s}\!\Big[\, \lambda\, V_{reset}(s') \;+\; (1-\lambda)\, V_{stuck}\!\big(p, s'\big)\,\Big].
$$

Because the Calvo draw is **pre-decision** at the arrival -> decision transition of the next period, it does *not* appear inside the max in *V\_{reset}* &mdash; only the expectation over the aggregate state does. (Same logic as the household's pre-decision Markov shock.)

---

## Exogenous Aggregate Inputs (from GE)

These are determined in general equilibrium but are fixed from any single intermediate firm's perspective:

| Symbol                                       | Meaning                                                                            | Origin                                                                |
| -------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| *w\_i*                                       | Real wage in region *i*                                                            | Labour-market clearing (firm-side aggregate labour demand = supply)   |
| *r\_i*                                       | Real return; defines SDF $\tilde\beta_i = 1/(1+r_i)$                                | Mutual-fund no-arbitrage:  *1 + r\_i = (1+R)/(1+pi\_i) = (q' + (1-tau\_d) omega D)/q* |
| *pi\_i*                                      | Regional inflation                                                                 | Aggregation of own-region reset prices via *P\_i* identity (eq. 4.13) |
| *P\_i*                                       | Regional price aggregator                                                          | Eq. 4.13                                                              |
| *Y\_{i'}*  for *i' = 1, ..., N*              | Per-capita demand for each region's final good                                     | Final-good market clearing: *Y\_i = C\_i + G\_i*                      |
| *Q\_{i',i}*                                  | Real exchange rate *P\_{i'}/P\_i*                                                  | Cross-region inflation accumulation                                   |
| *mu\_{i'}*, *gamma\_{i'i}*, *epsilon*, *alpha* | Population weights, trade-bias weights, CES elasticity, home-bias                  | Calibration                                                           |
| *lambda*                                     | Calvo reset probability                                                            | Calibration                                                           |

---

## Structural Summary

| Design question         | Dupor2023 answer                                                          | DDSL rationale                                                                                                |
| ----------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Shock timing            | Pre-decision (Calvo *zeta* resolves at arrival -> decision)               | Lottery sits in the arrival -> decision mover, *V\_{stuck}* and *V\_{reset}* are two perches behind one max   |
| Decision variable       | *p\** (only if *zeta = reset*; else degenerate)                           | Discrete-continuous structure: control collapses to a single point on the *stuck* branch                      |
| Idiosyncratic state     | **Just** *p\_{t-1}* (last-reset price)                                    | All within-region heterogeneity is *cohort-of-last-reset*; no continuum of types                              |
| Aggregate state         | *s = (w\_i, P\_i, pi\_i, Y\_i, Y\_{-i}, Q\_{-i,i}, r\_i)*                 | Taken as given by the firm; evolves in the rest of the GE block                                               |
| Demand                  | CES across *all* regions, elasticity *epsilon*                            | Cross-region terms enter only through the aggregate state                                                     |
| Technology              | *y = L*                                                                   | Eliminates a state variable (no capital, no productivity shock)                                               |
| Discount factor         | Stochastic, $\tilde\beta_i = 1/(1+r_i)$                                   | Same SDF the mutual fund uses; firms are nested under the fund                                                |
| Profits                 | *d = (p/P) y - w y*                                                       | Real profit per firm; aggregated to *D\_i = int d\_j* and fed back to the household and mutual-fund problems  |
| Closed-form FOC         | Yes &mdash; ratio of two recursions (eqs. 4.17-4.19)                      | Linear technology + CES demand + Calvo = standard NK reset-price formula                                      |
| PE vs GE                | *w\_i, r\_i, pi\_i, Y\_{i'}, Q* exogenous                                 | Clean interface for plugging the firm block into a sequence-space Jacobian system                             |

---

## Aggregation across firms (for the GE block)

Because all firms that reset in period *t* choose the same *p\*\_{i,t}*, the *cross-sectional* distribution of nominal prices within region *i* at time *t* is a mixture over reset cohorts:

* a mass *lambda* with price *p\*\_{i,t}*,
* a mass *(1 - lambda) lambda* with price *p\*\_{i,t-1}*,
* a mass *(1 - lambda)^2 lambda* with price *p\*\_{i,t-2}*,
* etc.

This produces the standard recursive identities of equations (4.23)-(4.24) which the GE block uses to compute the regional output index *Y\_i*, dividends *D\_i*, and price aggregator *P\_i* without ever having to keep track of an infinite-dimensional firm distribution. The "firm side" of the GE problem is therefore a *low-dimensional* set of forward-looking aggregate equations (the NKPC system), in stark contrast to the household side which carries a genuinely infinite-dimensional ergodic distribution *phi\_i*.

---

*Source*: Dupor, Karabarbounis, Kudlyak, and Mehkari (2023), "Regional Consumption Responses and the Aggregate Fiscal Multiplier," Section 4, equations (4.11)-(4.20).

*Stage decomposition style*: Following Carroll (2024), "Solving Microeconomic Dynamic Stochastic Optimization Problems," Section 1 ("The Problem"), adapted to a discrete-continuous (Calvo) firm problem.
