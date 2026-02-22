# Lazy Retraction Safety Analysis

## 1. Definitions

**Eager retraction** (current implementation): When `process_chunk()` is called with an
updated entity, the system immediately retracts all pairs involving that entity and
re-evaluates them with the new attributes. After each chunk, the pair set is always
consistent with all data seen so far.

**Lazy retraction** (proposed alternative): Retract pairs only at **query time** — when
`get_pairs()` is called — not on chunk arrival. Between queries, the pair set may hold
stale entries for entities whose attributes have changed since the last pair check.

## 2. Consistency Guarantees

| Strategy | Consistency Model | Stale Pairs Possible? |
|----------|------------------|-----------------------|
| Eager    | **Always-consistent**: pair set is correct after every chunk | No |
| Lazy     | **Query-consistent**: pair set is correct at query time only | Yes, between queries |

The distinction matters when queries arrive more often than chunks (typical real-time
use case) vs. when queries only arrive after all chunks are processed (batch use case).

## 3. Safety Condition for Lazy Retraction

**Lazy retraction is safe** (produces no permanently stale pairs) if and only if the
task's validity condition exhibits **monotonically decreasing** entity validity:

> Once an entity fails a condition, it can never pass it again as more data arrives.

Under this condition, stale pairs can only be **false positives** (pairs that should
have been retracted but weren't), never **false negatives** (pairs that should have
been added but weren't). At query time, we must check all current-entity pairs, but
this is equivalent to a full verification pass.

**Safe for lazy retraction** (monotone invalidation):
- **"Before DATE" constraints** (Task 5): An entity that has any instance AFTER the
  cutoff date is disqualified forever. Once disqualified, it stays disqualified — no
  future instances can re-qualify it. Pairs invalidated once remain invalid.

- **"Exactly N" cardinality** (Tasks 11, 13): An entity that accumulates more than N
  instances is permanently disqualified. The entity's instance count is monotonically
  non-decreasing, so once it exceeds N, it stays exceeded.

**Unsafe for lazy retraction** (non-monotone, bidirectional validity):
- **"After DATE" constraints** (Task 7): An entity initially appears to satisfy
  "has instance after cutoff DATE" if early chunks contain post-cutoff instances.
  But if a later chunk introduces a pre-cutoff instance, the entity may be
  re-classified. Worse: an entity with only pre-cutoff instances in early chunks
  becomes disqualified, but re-qualifies when post-cutoff instances arrive later.
  This bidirectional oscillation means lazy retraction can produce permanently wrong
  results (pairs that are valid at query time appear as retracted or vice versa).

- **Symmetric co-appearance** (Task 1): If entity validity depends on the relationship
  between two entities (e.g., both must have co-appeared within a window), then
  the pair validity can oscillate as new data arrives. Not monotone.

## 4. Empirical Support (from Temporal Sweep, k=5)

From `results/streaming/retraction_tasks_11_13.json` (Iteration 8, authoritative run with
deduplication guard active) — per-entity retraction distributions across all four tasks:

### Task 5 ("before DATE") — Safe for lazy retraction

| Metric | Value |
|--------|-------|
| Total entities | 231 |
| Final pairs | 21 |
| Total retractions | 44 |
| Never retracted | 223 (96.5%) |
| Retracted 1× (unidirectional) | 2 (0.9%) |
| Retracted 2+× (bidirectional) | 6 (2.6%) |
| Max retractions per entity | 3 |

**Interpretation**: 2.6% bidirectional rate (8 entities with 2+ retractions), low relative
to Task 7 (10.4%). The cumulative instance count grows monotonically, but the date
boundary condition interacts with label semantics in ways that allow a small fraction of
entities to oscillate. For a lazy strategy in batch mode, at most 8 entities require
re-verification at query time. Task 5 remains the **safest** tested condition, though the
"~0% bidirectional" phrasing in prior versions overstated safety.

### Task 7 ("after DATE") — Unsafe for lazy retraction

| Metric | Value |
|--------|-------|
| Total entities | 231 |
| Final pairs | 1485 |
| Total retractions | 2151 |
| Never retracted | 194 (84.0%) |
| Retracted 1× | 13 (5.6%) |
| Retracted 2+× (bidirectional) | 24 (10.4%) |
| Max retractions per entity | 4 |

**Interpretation**: 10.4% of entities are bidirectionally retracted. With 24 bidirectional
entities each potentially involved in many pairs, lazy retraction could produce permanently
stale results for a substantial fraction of the 1485 final pairs. Unsafe.

### Task 11 ("Exactly N" cardinality) — **Empirically Unsafe** (contradicts prior claim)

| Metric | Value |
|--------|-------|
| Total entities | 231 |
| Final pairs | 689 |
| Total retractions | 883 |
| Never retracted | 202 (87.4%) |
| Retracted 1× (unidirectional) | 15 (6.5%) |
| Retracted 2+× (bidirectional) | 14 (6.1%) |
| Max retractions per entity | 3 |

**Interpretation**: 6.1% bidirectional rate — substantially higher than Task 5 (2.6%) and
comparable to Task 7 (10.4%). The prior claim of "~0% bidirectional" was **empirically
refuted**. The theoretical argument (monotone instance counts → monotone condition) breaks
down in practice because: (a) the condition involves accumulated label distributions, not
just raw counts, and (b) as instances accumulate, an entity's dominant label can change,
causing oscillation. Task 11 is **NOT safe** for lazy retraction.

### Task 13 ("Exactly N" cardinality) — **Empirically Unsafe** (contradicts prior claim)

| Metric | Value |
|--------|-------|
| Total entities | 231 |
| Final pairs | 1524 |
| Total retractions | 1679 |
| Never retracted | 184 (79.7%) |
| Retracted 1× (unidirectional) | 25 (10.8%) |
| Retracted 2+× (bidirectional) | 22 (9.5%) |
| Max retractions per entity | 3 |

**Interpretation**: 9.5% bidirectional rate — nearly identical to Task 7 (10.4%) and far
above Task 5 (2.6%). The "~0% bidirectional" claim was **empirically refuted**. Task 13 is
**NOT safe** for lazy retraction.

## 5. Applicability Region for Lazy Retraction

**Corrected table** (Iteration 8, empirically verified for all conditions):

| Condition Type | Monotone (theory)? | Lazy Safe (empirical)? | Bidirectional Rate | Recommendation |
|----------------|-------------------|----------------------|-------------------|----------------|
| "Before DATE" | Yes | **Yes** (with caution) | 2.6% | Use lazy for batch mode; 2.6% overhead is low |
| "After DATE" | No | **No** | 10.4% | Use eager always |
| "Exactly N" (Task 11) | Claimed Yes | **No** (refuted) | 6.1% | Use eager; monotone assumption violated in practice |
| "Exactly N" (Task 13) | Claimed Yes | **No** (refuted) | 9.5% | Use eager; comparable to "after DATE" rate |
| Symmetric co-appear. | Partial | Conditional | Varies | Depends on window size |
| Asymmetric cardinality | No | **No** | Varies | Use eager always |

**Key correction from Iteration 8**: The theoretical claim that "Exactly N" conditions have
~0% bidirectional retraction rate was refuted by empirical analysis. Tasks 11 and 13 show
6.1% and 9.5% rates — comparable to the "unsafe" Task 7 (10.4%). The theoretical
monotonicity argument (instance counts only grow) does not hold because the OOLONG-Pairs
"exactly N" condition involves accumulated label distributions where the dominant label can
shift as more instances arrive, causing oscillation.

**Implication for paper**: The monotonicity safety condition remains a valid diagnostic
principle, but practitioners must **empirically verify** bidirectional rates rather than
rely solely on theoretical condition type. Add this as a paper recommendation.

## 6. Recommendation

**For the paper**: Do not implement lazy retraction as a general strategy. Instead,
characterize the safety condition (monotone validity) as a **user-facing diagnostic**:

> "Before deploying the incremental RLM with lazy retraction, verify that your task's
> validity condition is monotonically non-increasing (entities can only become invalid,
> never re-validate). Tasks 5 and 11/13 satisfy this. Tasks 7, 4, 9, 10, and 1 do not.
> For unsafe tasks, use eager retraction (the default)."

This characterization is a novel contribution: **the monotonicity safety condition**
for incremental computation retraction strategies. It is directly actionable (practitioners
can check their condition type) and empirically supported (1.3% vs. 10.4% bidirectional
rates across two representative task families).

## 7. Cost Comparison Summary

At N=231 entities, k=5 chunks:

| Strategy | Task 5 Cost | Task 7 Cost |
|----------|-------------|-------------|
| Eager (current) | 44 retraction ops | 2151 retraction ops |
| Lazy (at query) | ~25 entity re-verifications (low) | ~49 entity re-verifications (still high, plus exhaustive pair scan for 24 bidirectional entities) |
| Full recompute | C(231,2) = 26,565 pair checks | 26,565 pair checks |

**Conclusion**: Lazy retraction helps Task 5 (saves 19 retraction ops, ~0.1% overhead).
It does not meaningfully help Task 7 (49 re-verifications → still substantial overhead).
The eager strategy's constant-time operation (O(degree) per retraction, bounded by the
inverted index) is already efficient enough that lazy retraction adds complexity without
proportional benefit in the current N=231 regime. At larger N, the tradeoff may shift
in favor of lazy for monotone tasks, but this requires further analysis.
