# Independent review of `differential_evolution` — start here

An independent adversarial code review of `vrbaj/differential_evolution` at commit `ed6ff80`,
performed with Claude Opus 5 (Claude Code) in a clean Python 3.12 environment, then
cross-checked end to end by a **second independent Claude Opus 5 pass** on a different machine.
No connection to the Codex refactor — neither reviewer had access to that session.

**53 numbered findings.** 48 of them are machine-verified by the four scripts in this bundle;
6 more are performance measurements; the rest are directly checkable facts about the
repository. Nothing is asserted from memory, and every literature claim is quoted from the
primary paper.

**What the second pass did.** It re-cloned `ed6ff80`, re-ran all three original scripts on a
different Python build, re-read the source module by module, and pulled the SHADE (CEC 2013)
and L-SHADE (CEC 2014) papers to settle the three literature questions the first pass had
flagged as uncertain.

* Everything reproduced — all 30 runtime assertions, all 14 repository checks, and the
  quantitative results to the digit (64 tests, 88 % coverage, mypy 56 errors in 9 files,
  ruff 22 findings with the same composition, the PyPI name taken at v1.12.0).
* Two of the three uncertain claims were **confirmed** (L-SHADE's fixed `p = 0.11` and its
  weighted Lehmer mean for `M_CR`). The third — the `DirectedMutation` citation — was
  **narrowed but not closed**: a candidate primary source was found, but neither pass could
  read it, so whether its equation matches the implementation is still up to you.
* Four findings were **added** (M1d, M1e, M1f, M10), collected as issue 13, and one positive
  was added: all 15 benchmark formulas are mathematically correct.
* One fix suggestion was **wrong** and has been corrected in place (the tanh schedule
  defaults in M5).

Everything from the second pass is marked **[2nd pass]** in `REVIEW.md`.

## What is in this bundle

| File | What it is | Read it when |
|---|---|---|
| `REVIEW.md` | The full report: 53 findings by severity, each with a reproduction, plus §11, a table saying exactly how every single finding was verified | You want the whole picture. Start at §0, the executive summary, then the revision note. |
| `ISSUES.md` | The 13 highest-impact findings rewritten as standalone GitHub issue bodies, ready to paste | You would rather triage them one at a time than read a wall of text |
| `repro.py` | 19 runtime assertions: C1–C4, H1–H10, M5, M8, M9, D5 | You want to see the bugs happen on your own machine |
| `repro2.py` | 11 more runtime assertions: M1a, M1b, M3, M4, M6, M7, D2, D6, D7, P4, P6 | Same, for the algorithm-fidelity and API findings |
| `audit.py` | 14 repository / packaging / API-surface checks (K1–K8, D1, D3, D4, D8, T1, T2, T4), plus the P1–P6 performance measurements and a lint/mypy summary | You want the publication-readiness picture |
| `verify_extra.py` | The 5 second-pass checks: M1d, M1e, M1f, M10, and the benchmark-formula sweep behind C3 | You want the SHADE/L-SHADE fidelity findings that are quoted against the papers |

## Running the checks

```bash
conda create -n differential_evolution python=3.12 -y
conda activate differential_evolution
cd /path/to/differential_evolution        # your checkout, at ed6ff80 or later
pip install -e .
python repro.py
python repro2.py
python audit.py            # add --offline to skip the live PyPI name lookup
python verify_extra.py
```

`repro.py`, `repro2.py` and `verify_extra.py` need nothing but the standard library and the
package itself. `audit.py` additionally uses `ruff`, `mypy` and `coverage` if they are
installed, and skips those sections if they are not. Everything runs in well under a minute
apart from the performance section of `audit.py` (~30 s).

Each check prints `observed` / `expected` / `PASS|FAIL`, so the same files work as an
acceptance checklist: fix something, re-run, watch the FAIL turn into PASS.

Current state at `ed6ff80`: `repro.py` 19 FAIL, `repro2.py` 11 FAIL, `audit.py` 14 FAIL,
`verify_extra.py` 4 FAIL and 1 PASS (the benchmark formulas are correct).

## If you only have ten minutes

Read §0 of `REVIEW.md`, then these four:

* **C1** — `TentInitializer` puts 147 of 200 coordinates exactly on the lower bound.
* **C2** — one `inf` fitness turns the whole SHADE/L-SHADE memory into `NaN`, after which
  the optimizer silently does nothing for the rest of the budget and still reports
  `success=True`.
* **C3** — 13 of the 15 exported benchmark functions silently ignore coordinates beyond the
  first two, so a "10-D Ackley" run reports `fun = 0.0`.
* **K1/K2** — the PyPI name `differential-evolution` is already taken, and there is no
  LICENSE file. Both block the publication plan regardless of the code.

If you have another ten minutes and the paper matters more than the release, read **issue 13**
next. It collects the five places where `SHADE` and `LSHADE` diverge from Tanabe & Fukunaga,
each one quoted against the paper, and it is the set a referee who knows these algorithms will
check first.

## Things the review found that are genuinely good

Listed in §9 of `REVIEW.md`, and worth saying up front because the rest of the report is
relentless by request:

* The Sobol implementation is **correct** — cross-checked against `scipy.stats.qmc.Sobol`.
  It differs from scipy only because it uses Bratley–Fox rather than Joe–Kuo direction
  numbers, and it passes the 1-D equidistribution property in every dimension up to 40.
* L-SHADE reaches `2.6e-48` median on a 10-D sphere at 50 000 evaluations — reference-quality
  convergence.
* The documentation builds with **zero warnings** under `sphinx -W --keep-going`.
* `ALGORITHM_AUDIT.md` and `docs/discrepancies.rst` are better than most published DE library
  documentation, and `discrepancies.rst` — separating deliberate choices from open questions
  — is a real differentiator worth keeping and expanding.
* **All 15 benchmark formulas are correct** — evaluated at their published global optima,
  zero mismatches. C3 is a missing dimension guard, not bad mathematics, which makes it far
  cheaper to fix than it first looks. (Second pass.)
* The component decomposition itself is the right one. The §4 criticisms are about protocol
  width and typing, not about the factoring.

## The three literature questions — now settled

The first pass flagged three claims as unverified. The second pass pulled both papers and
resolved all three; the relevant text is quoted in `REVIEW.md` M2 and H6, and in issue 13.

1. **The `M_CR` update rule in L-SHADE — CONFIRMED.** L-SHADE 2014 Eq. (7) defines one
   weighted Lehmer mean and states that "`S` refers to either `S_CR` or `S_F`". SHADE 2013
   keeps them apart: Eq. (17) arithmetic for `M_CR`, Eq. (18) Lehmer for `M_F`. The library
   uses the SHADE rule in both classes, so `LSHADE` is wrong here.
2. **The fixed `p = 0.11` in L-SHADE — CONFIRMED.** Eq. (3) selects `x_pbest` from the top
   `N × p` members with `p` a fixed hyper-parameter, swept over `{0.05, …, 0.15}` in the
   paper's parameter table with 0.11 reported for `D = 30`. SHADE 2013 Eq. (20) is the
   randomised `p_i = rand[2/NP, 0.2]`, which the library uses in both classes.
3. **The `DirectedMutation` source — STILL OPEN, but narrowed.** The cited DOI really is the
   trigonometric-mutation paper, so the citation as written is wrong; that much is certain.
   The first pass concluded that no primary source exists and the operator should be renamed.
   That now looks less likely — Fan & Lampinen do have a separate paper titled *A directed
   mutation operation for the differential evolution algorithm*, and there is also Fan,
   Lampinen & Dulikravich, *Improvements to Mutation Donor Formulation of Differential
   Evolution*, EUROGEN 2003 — but **neither pass could read either paper.** So the formula
   match is unverified. Get the original, compare it with `mutation.py`, and let that decide
   between fixing the DOI and renaming the operator.

## The one thing still open

The `DirectedMutation` provenance. The paper *"A directed mutation operation for the
differential evolution algorithm"* is attested by a ResearchGate record and by secondary
citations in DE surveys, but it is not indexed in Crossref and neither pass could open the full
text — so its volume / issue / pages / DOI are unknown **and, more importantly, it has not been
verified that its equation is the one in `mutation.py`**.

This is the single place in the bundle where the review points at a source it has not read, and
it is flagged as such in `REVIEW.md` H6 and §11. Everything else there is machine-verified,
quoted directly from a primary source, or a directly checkable fact about the repository.
