# Safe fixes to the matsya-exercise HARK model

Five high-confidence fixes to the HARK household implementation (the 06-02
update), and how they were produced. See PR nathanrobino/Dupor2023-bn#1.

## The fixes
1. **Dead Sep solver** -- `solve_ConsLaborIntMargSep` called
   `LogCRRALaborutilityPc_inv(X)` but the function needs `(uc, psi, theta)`
   (TypeError); fixed to `(X, Psi, LsrCurv)`.
2. **Calibration** -- the Markov default `Psi=5.8` gave ergodic mean hours
   ~0.086, not the documented 42%; recalibrated to `Psi=1.0` for this
   log-consumption / log-leisure normalization (-> ~0.42 hours).
3. **Silent dividend no-op** -- the demo passed `D=1` but the parameter is
   `Div`, so the dividend was ignored; removed the bogus kwarg (runs the
   `Div=0` default).
4. **Reproducibility** -- `track_vars` + an `econ-ark>=0.17` assertion + a
   mean-hours check in the demo; bumped `environment.yml` pins
   (python>=3.11 / numpy>=2 / econ-ark>=0.17).
5. **Doctest** -- `rewards.py` `LogCRRALaborutility` doctest corrected
   (value 1.0, not 2.0) and made numpy-2 repr-robust.

Verified on econ-ark 0.17.2: the demo notebook runs end-to-end
(`ergodic mean hours = 0.421`), the doctest passes, and the Sep solver solves.

## How these were produced (reproducible method)
- **Claude Code** on model `claude-opus-4-8`, run via the `/effort` command
  with the **max** effort level and **ultracode** enabled (multi-agent
  workflow orchestration: parallel review agents + a reconciler).
- **DP / DDSL consultation** via the Econ-ARK **Matsya** research copilot,
  session **`dupor-improve`** -- e.g. `matsya "<question>" --session dupor-improve`.

## Not addressed here (modeling judgment calls left to the author)
- The dividend enters via an `h`-scaled effective wage (`eff_wage*h`) rather
  than as autonomous income (differs from the paper / the DDSL / the code's
  own design comment).
- The dividend-tax convention (the repo's two paper transcriptions disagree).
- Guide scope, the parent Sep solver's CRRA consistency, and the
  `ConsLaborModel copy.py` filename / HARK provenance.
