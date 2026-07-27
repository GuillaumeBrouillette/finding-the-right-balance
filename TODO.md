# TODO before making this repository public

Working notes for the cleanup pass. **Delete this file once the list is done.**

This repo was extracted on 2026-07-27 from the private working repo
(`Relative-Distance-Selection-for-RAG`), which holds the paper source, the full
result set and the project history. Only the code and the small result tables
came across, in a single fresh commit. The extraction was scoped by what the
paper's `% [provenance]` comments actually cite — the code was **not** audited
line by line, so treat the list below as a starting point, not an exhaustive one.

## README is stale in places

The README was written during the revision cycle and still reflects an earlier
version of the paper. Known issues:

1. **Seg-Score is presented as a headline method** (lines 5, 41, ~459: "the
   geometric RNG-Score / Seg-Score rerankers"). The final paper mentions it once,
   to say it is *excluded* from the analyses. Decide whether to keep the code and
   drop it from the docs, or drop both.
2. **RQ numbering does not match the paper.** README headings use RQ1–RQ5 from an
   older scheme; the "Derived analyses" section already has to say "paper RQ3",
   "paper RQ5", "paper RQ6" to compensate. Simplest fix: drop the RQ labels and
   use descriptive headings.
3. **"Experiments to run (revision runbook)"** (~lines 335–449) is internal
   reviewer-response material: DGX commands, "reviewer major 1/2/4/5", pending
   runs, dropped-from-scope items. It should not ship publicly — delete it, or
   replace it with a short "reproducing the paper" section.
4. **The intro points at that runbook** (lines 11–14) as "the authoritative list
   of commands to reproduce the paper" — update when item 3 is done.
5. **CE integration strategy names** (`S1-Blend`, `S2-Semimetric`) do not appear
   verbatim in the paper; check the naming is consistent with the final text.

## Repository housekeeping

6. Repo is **private**. Make it public when the preprint is out, and add the
   arXiv/preprint URL to the README citation block (currently `note = {Preprint}`
   with no URL).
7. The GitHub **repository description** is not set yet.
8. Local folder is `finding-the-right-balance-code`; the GitHub repo is
   `finding-the-right-balance`. Harmless, but rename the folder if it bothers you.

## What is and is not here

- `results/` holds the summary and `analysis_*` tables of the 25 run directories
  the paper keeps, plus two small `*_per_query.csv` fixtures (2026-06-12 SciFact
  and HotpotQA) so the `analyze_regimes.py` pipeline can be run end to end. The
  large per-query tables (several hundred MB each) are gitignored and live only
  in the working repo.
- Datasets are not shipped; the loaders download them on first use.
- Deliberately excluded, kept in the working repo's local `archive/` folder:
  `recsys_diversification.py` and its results (the dropped MovieLens RQ7),
  `diagnostic_retrieval.py` (debug tool), and the on-hold bi-encoder training
  code (`train_biencoder.py`, `evaluate_biencoder.py`, `training/`).

## Verified at extraction time

- `python tests/test_dedup.py` and `python tests/test_ndcg_duplicates.py` pass
  (9 assertions total).
- All 16 top-level modules import cleanly with no missing intra-project
  dependencies.
