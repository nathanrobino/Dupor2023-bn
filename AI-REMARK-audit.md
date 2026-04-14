# REMARK Compliance Audit — Dupor2023-bn

## Prompts used

1. **"Does this repository meet the requirement for a REMARK found at
   https://github.com/econ-ark/REMARK/blob/main/STANDARD.md"**
   — Asked the AI to audit the repo against the full REMARK standard.

2. Follow-up discussion refined the assessment to focus on Tier 1 (baseline)
   vs. Tier 3 (published) requirements.

---

## AI summary of REMARK tiers

REMARKs are organized into three tiers. **Tier 1 (Docker REMARK)** is the
baseline: the repo needs a tagged release, a `Dockerfile`, `reproduce.sh`,
`README.md` (≥ 50 non-empty lines), `LICENSE`, and
`binder/environment.yml`. **Tier 2 (Reproducible)** adds a longer README
(≥ 100 non-empty lines), a `REMARK.md` with a tier field, and a
`CITATION.cff`. **Tier 3 (Published)** adds a Zenodo DOI tied to a specific
git tag.

---

## Compliance judgment — Tier 1 (baseline)

**Partially compliant.**

| Requirement | Status | Detail |
|---|---|---|
| Tagged release | **FAIL** | `git tag -l` returns nothing |
| `Dockerfile` | Pass | |
| `reproduce.sh` | Pass | |
| `README.md` present | Pass | |
| `README.md` ≥ 50 non-empty lines | **Borderline** | 53 non-empty lines |
| `LICENSE` | Pass | |
| `binder/environment.yml` | Pass | |

The repo has every required *file* but lacks the single most important
process requirement: a tagged release.

### Additional issues the AI flagged

- `REMARK.md` references two files that do not exist in the repo
  (`HANK-and-SAM-tutorial.ipynb`, `dashboard/app.ipynb`).
- `REMARK.md` and `CITATION.cff` still reference the upstream repo
  (`econ-ark/HAFiscal`) rather than `nathanrobino/Dupor2023-bn`.
- The `.gitignore` now hides `Code/`, `Data/`, `Figures/`, `Tables/`, and
  `Subfiles/` from GitHub, which would break reproducibility if someone
  clones the public repo and runs `reproduce.sh`.

---

## Prioritized action list

1. **Create a tagged release** (e.g. `git tag -a v0.1.0 -m "Initial REMARK submission"` then `git push origin v0.1.0`). This is the only hard blocker for Tier 1.
2. **Revisit `.gitignore` hiding of core directories.** Directories like `Code/` and `Data/` must be visible on GitHub for `reproduce.sh` to work when cloned. Either revert those ignore rules or restructure reproduction so it can fetch the data/code externally.
3. **Update `REMARK.md`**: remove references to nonexistent notebooks/dashboards, update the `github_repo_url` to this repo's URL, and add a `tier: 1` field.
4. **Update `CITATION.cff`**: change `repository-code` to point to this repo.
5. **Expand `README.md`** to ≥ 100 non-empty lines (required for Tier 2 and above; also gives more margin for the Tier 1 minimum of 50).
