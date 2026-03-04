"""
Incremental computation primitives for Dynamic RLM.

This module provides helper classes that can be injected into the REPL
environment to support incremental computation patterns:

1. EntityCache — stores entity classifications with dependency tracking
2. PairTracker — tracks valid pairs with retraction support
3. IncrementalState — combines both for a complete incremental pipeline

These are made available in the REPL when incremental mode is enabled,
giving the LLM concrete tools to implement the incremental protocol.

Key novelty: the retraction mechanism handles non-monotonic discovery.
When new data invalidates a cached classification (e.g., "exactly one X"
constraint is violated by a new entity also having X), the affected pairs
are automatically retracted and re-evaluated.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ["EntityCache", "PairTracker", "IncrementalState"]


class EntityCache:
    """Cache entity classifications with versioning.

    Each entity has:
    - id: unique identifier
    - attributes: dict of classifications/properties
    - source_chunk: which chunk it was first seen in
    - last_updated: which chunk last modified it

    Supports O(1) lookup by entity ID and O(k) iteration over
    entities from a specific chunk.

    Limitation: No deletion support. Entities are retained indefinitely once
    added. For streaming scenarios with entity disappearance, stale pairs may
    persist. If entity disappearance is needed, the caller must explicitly call
    pair_tracker.retract_entity() and then remove the entity from the cache
    manually (not supported by this class in the current implementation).
    """

    def __init__(self):
        self._entities: dict[str, dict[str, Any]] = {}
        self._by_chunk: dict[int, set[str]] = {}  # chunk_index -> entity_ids
        self._version: int = 0

    def add(self, entity_id: str, attributes: dict[str, Any], chunk_index: int) -> bool:
        """Add or update an entity. Returns True if this was an update (not new)."""
        is_update = entity_id in self._entities
        self._entities[entity_id] = {
            "attributes": attributes,
            "source_chunk": chunk_index
            if not is_update
            else self._entities[entity_id]["source_chunk"],
            "last_updated": chunk_index,
        }
        if chunk_index not in self._by_chunk:
            self._by_chunk[chunk_index] = set()
        self._by_chunk[chunk_index].add(entity_id)
        self._version += 1
        return is_update

    def get(self, entity_id: str) -> dict[str, Any] | None:
        """Get entity attributes."""
        entry = self._entities.get(entity_id)
        return entry["attributes"] if entry else None

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Get all entity attributes."""
        return {eid: e["attributes"] for eid, e in self._entities.items()}

    def get_from_chunk(self, chunk_index: int) -> set[str]:
        """Get entity IDs that were added OR updated in a given chunk.

        Iteration 10 docstring fix: this method was previously documented as
        "entity IDs first seen in chunk_index" but `add()` unconditionally writes
        to `_by_chunk[chunk_index]` for both new entities and updates — so an entity
        first seen in chunk 0 but updated in chunk 3 appears in both
        `_by_chunk[0]` and `_by_chunk[3]`.

        For entities introduced for the first time in a specific chunk, use
        `get_new_in_chunk()` which filters by `source_chunk`.
        """
        return self._by_chunk.get(chunk_index, set())

    def get_new_in_chunk(self, chunk_index: int) -> set[str]:
        """Get entity IDs first introduced (source_chunk == chunk_index).

        Unlike get_from_chunk(), this excludes entities that existed before
        chunk_index and were merely updated in it. Useful for computing only
        the truly new entities from a given chunk without counting updates.
        """
        return {
            eid for eid in self._by_chunk.get(chunk_index, set())
            if self._entities.get(eid, {}).get("source_chunk") == chunk_index
        }

    def get_ids(self) -> set[str]:
        """Get all entity IDs."""
        return set(self._entities.keys())

    def __len__(self) -> int:
        return len(self._entities)

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self._entities

    def __repr__(self) -> str:
        return f"EntityCache({len(self._entities)} entities, {len(self._by_chunk)} chunks)"


class PairTracker:
    """Track valid entity pairs with retraction support.

    Key feature: when an entity's classification changes, all pairs
    involving that entity can be efficiently found and re-evaluated.

    Maintains an inverted index: entity_id -> set of pairs involving it.
    This enables O(degree) retraction per entity update instead of O(n²) full scan.
    """

    def __init__(self):
        self._pairs: set[tuple[str, str]] = set()
        self._entity_pairs: dict[str, set[tuple[str, str]]] = {}  # inverted index
        # Pairs removed by retraction but not yet re-added. Diagnostic only —
        # correctness does not depend on this set. Note: permanently-invalidated
        # pairs accumulate here indefinitely (O(n²) worst case at large scale with
        # high retraction rates). Call clear_retracted() to reclaim memory if needed.
        self._retracted: set[tuple[str, str]] = set()
        self._retraction_count: int = 0

    def add_pair(self, id1: str, id2: str) -> None:
        """Add a valid pair (stored in canonical order)."""
        pair = (min(id1, id2), max(id1, id2))
        self._pairs.add(pair)
        self._entity_pairs.setdefault(id1, set()).add(pair)
        self._entity_pairs.setdefault(id2, set()).add(pair)
        # If this was previously retracted, un-retract it
        self._retracted.discard(pair)

    def remove_pair(self, id1: str, id2: str) -> None:
        """Remove a pair."""
        pair = (min(id1, id2), max(id1, id2))
        self._pairs.discard(pair)
        if id1 in self._entity_pairs:
            self._entity_pairs[id1].discard(pair)
        if id2 in self._entity_pairs:
            self._entity_pairs[id2].discard(pair)

    def retract_entity(self, entity_id: str) -> set[tuple[str, str]]:
        """Retract all pairs involving an entity.

        Returns the set of retracted pairs (they need to be re-evaluated
        with the entity's new classification).

        This is the key operation for non-monotonic incremental computation.
        Also cleans up the partner entity's inverted index to prevent stale
        references that would cause double-counting when the partner is also
        retracted in the same chunk.
        """
        affected = self._entity_pairs.get(entity_id, set()).copy()
        for pair in affected:
            self._pairs.discard(pair)
            self._retracted.add(pair)
            # Clean up partner's inverted index to prevent double-counting
            partner = pair[1] if pair[0] == entity_id else pair[0]
            if partner in self._entity_pairs:
                self._entity_pairs[partner].discard(pair)
        self._retraction_count += len(affected)
        # Clear this entity's pair index (will be rebuilt on re-evaluation)
        self._entity_pairs[entity_id] = set()
        return affected

    def get_pairs(self) -> set[tuple[str, str]]:
        """Get all currently valid pairs."""
        return self._pairs.copy()

    def get_pairs_for_entity(self, entity_id: str) -> set[tuple[str, str]]:
        """Get all pairs involving an entity."""
        return self._entity_pairs.get(entity_id, set()).copy()

    @property
    def retraction_count(self) -> int:
        """Total retractions performed."""
        return self._retraction_count

    @property
    def retracted_pairs(self) -> set[tuple[str, str]]:
        """Pairs retracted and not yet re-added. Diagnostic only.

        Note: permanently-invalidated pairs accumulate here until
        clear_retracted() is called. Use retraction_count for a
        bounded telemetry metric.
        """
        return self._retracted.copy()

    def clear_retracted(self) -> int:
        """Clear the retracted-pairs diagnostic set and return count cleared.

        Call periodically to prevent unbounded memory growth in long-running
        streaming scenarios with high retraction rates.
        """
        n = len(self._retracted)
        self._retracted.clear()
        return n

    def has_pair(self, id1: str, id2: str) -> bool:
        """Check if a pair exists (order-independent)."""
        canonical = (min(id1, id2), max(id1, id2))
        return canonical in self._pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __contains__(self, pair: tuple[str, str]) -> bool:
        """Check if a canonical pair exists."""
        return pair in self._pairs

    def __repr__(self) -> str:
        return f"PairTracker({len(self._pairs)} pairs, {self._retraction_count} retractions)"


class IncrementalState:
    """Combined incremental computation state.

    This is injected into the REPL environment as `_incremental` when
    incremental mode is active. It provides:
    - entity_cache: EntityCache for classifications
    - pair_tracker: PairTracker with retraction support
    - chunk_log: record of what was processed per chunk
    - stats: computation statistics for benchmarking

    The LLM can use these directly in REPL code:
    ```repl
    _incremental.entity_cache.add("user_123", {"type": "person"}, chunk_idx=1)
    _incremental.pair_tracker.add_pair("user_123", "user_456")
    ```
    """

    def __init__(self):
        self.entity_cache = EntityCache()
        self.pair_tracker = PairTracker()
        self.chunk_log: list[dict[str, Any]] = []
        self._total_new_entities: int = 0
        self._total_pair_checks: int = 0
        self._total_retractions: int = 0
        self._noop_retractions: int = 0       # retracted then immediately re-added (no-op)
        self._permanent_retractions: int = 0  # retracted and NOT re-added (genuine invalidation)
        self._processed_chunk_indices: dict[int, dict[str, Any]] = {}  # chunk_index -> cached_stats

    def process_chunk(
        self,
        chunk_index: int,
        new_entities: dict[str, dict[str, Any]],
        pair_checker: Any = None,
        monotone_attrs: set[str] | None = None,
    ) -> dict[str, Any]:
        """Process a new chunk of entities incrementally.

        Args:
            chunk_index: Index of the arriving chunk
            new_entities: {entity_id: attributes} for entities in the new chunk
            pair_checker: Optional callable(attrs1, attrs2) -> bool for pair validation
            monotone_attrs: Optional set of attribute names that are monotone increasing.
                For these attributes, once an entity has a truthy value in any prior chunk,
                that value is preserved even if the current chunk's data would downgrade it
                (e.g., "qualifying=True" from chunk 0 is preserved when the entity appears
                in chunk 3 with only non-qualifying labels).

                Optimization: if ALL monotone_attrs values are unchanged after merging
                (the entity's effective state is identical to what was cached), the entity
                is NOT added to updated_ids. This skips the O(degree) retraction +
                O(n) updated-entity sweep for that entity — eliminating no-op retraction
                cycles at zero correctness cost.

                Use monotone_attrs for "at least one qualifying label" predicate types
                (existential predicates). Do NOT use for "exactly N" or cardinality-
                constrained predicates, which are genuinely non-monotone.

        Returns:
            Dict with processing stats for this chunk, including:
                - noop_retractions: pairs retracted then immediately re-added (no cost savings)
                - permanent_retractions: pairs retracted and not re-added (genuine invalidations)

        Complexity: O((k + u) · n) per chunk, where:
            k = number of new entities in this chunk
            u = number of updated entities (entities seen in prior chunks, now updated)
            n = total entities accumulated so far

        The new-entity loop is O(k · n) — the main savings driver (k ≪ n).
        The updated-entity sweep is O(u · n) per chunk. When u is small relative
        to k (entities rarely re-appear across chunks), the incremental approach
        remains efficient. However, if u is large (e.g., 10%+ of entities are
        updated per chunk in high-churn streaming settings), the updated-entity
        sweep can dominate and may exceed the cost of full recompute (O(n²/k)
        per chunk). In that regime, reduce chunk granularity to lower u, or
        switch to full recompute.

        The theoretical savings claim O(k · n) vs O(n²) holds when u ≪ k.
        For OOLONG-Pairs (N=231, u≈0 per chunk), this is satisfied. For general
        streaming applications, measure u/k before relying on the savings estimate.

        With monotone_attrs: the updated-entity sweep is additionally suppressed for
        entities whose effective (pair-checker-relevant) state is unchanged after the
        monotone merge. This brings the amortized cost closer to O(k · n) even when
        entities reappear across chunks, provided the monotone attributes dominate the
        pair-checker condition.

        Note: when monotone_attrs is provided, the attrs dicts in new_entities may be
        mutated (truthy cached values written back). Callers who reuse the dict after
        calling process_chunk() should be aware of this side effect.
        """
        # Idempotency guard: if this chunk was already processed, return cached stats.
        # This converts Failure Mode C (model re-executes process_chunk multiple times
        # per turn due to max_iterations > 1) from "correct but O(u·n) wasteful per
        # redundant call" to "correct and O(1) per redundant call".
        # Re-processing a chunk requires calling reset() first.
        if chunk_index in self._processed_chunk_indices:
            warnings.warn(
                f"process_chunk({chunk_index}) called more than once. "
                f"Returning cached stats. Re-processing requires reset().",
                stacklevel=2,
            )
            return self._processed_chunk_indices[chunk_index]

        # Snapshot existing IDs BEFORE adding new entities. If an entity ID in
        # new_entities already exists, it's treated as an update (updated_ids),
        # not a new entity, and the updated-entity sweep handles re-evaluation.
        existing_ids = self.entity_cache.get_ids()
        updated_ids = set()
        new_ids = set()

        # 1. Add/update entities, with optional monotone attribute merge.
        #
        # SAFETY INVARIANT: When monotone_attrs is provided, pair_checker must depend
        # exclusively on the declared monotone attributes. If pair_checker reads other
        # entity attributes, set monotone_attrs=None to preserve correct retraction.
        # Rationale: The no-op update optimization (below) skips retraction when only
        # monotone attrs are present and unchanged. If pair_checker depends on a
        # non-monotone attr that changed, the optimization silently skips a necessary
        # retraction, producing incorrect pair results.
        #
        # Monotone merge logic (when monotone_attrs is provided):
        # For each entity that already exists in the cache (is an update), read the
        # cached attribute values BEFORE writing. For each attribute in monotone_attrs:
        #   - If old_val is truthy AND new_val is falsy → preserve old_val in attrs.
        #     (The cached truthy value is retained; the current chunk's downgrade is ignored.)
        # After merging: if ALL monotone_attrs values are unchanged (boolean-equivalent),
        # mark the entity as a no-op update and skip adding it to updated_ids.
        # This eliminates O(degree) retraction + O(n) updated-entity sweep for that entity.
        for eid, attrs in new_entities.items():
            is_noop_update = False  # True only for updates where effective state is unchanged

            if monotone_attrs:
                cached_attrs = self.entity_cache.get(eid)  # read BEFORE updating
                if cached_attrs is not None:  # entity exists → will be an update
                    monotone_all_same = True
                    for attr in monotone_attrs:
                        old_val = cached_attrs.get(attr)
                        new_val = attrs.get(attr)
                        if old_val and not new_val:
                            # Monotone merge: preserve cached truthy value
                            attrs[attr] = old_val
                            # After merge, attrs[attr] == old_val → boolean-unchanged
                        elif bool(new_val) != bool(old_val):
                            # Genuine change in this monotone attribute
                            monotone_all_same = False
                    if monotone_all_same:
                        # All monotone attributes are unchanged after merge.
                        # This entity's effective state (for pair checking) is the same
                        # as it was in the cache → no retraction needed.
                        is_noop_update = True

            was_update = self.entity_cache.add(eid, attrs, chunk_index)
            if was_update:
                if not is_noop_update:
                    updated_ids.add(eid)
                # else: skip retraction for this entity (no-op update)
            else:
                new_ids.add(eid)

        # 2. Handle retractions for updated entities
        retracted_pairs = set()
        for eid in updated_ids:
            retracted = self.pair_tracker.retract_entity(eid)
            retracted_pairs |= retracted
        # Count deduplicated retractions (partner cleanup in retract_entity
        # prevents double-counting, but we use set size for belt-and-suspenders)
        self._total_retractions += len(retracted_pairs)

        # 3. Check pairs incrementally (if checker provided)
        new_pairs = 0
        pair_checks = 0
        chunk_noop_retractions = 0
        chunk_permanent_retractions = 0
        if pair_checker:
            # New × existing (excluding other new entities)
            for new_id in new_ids:
                new_attrs = self.entity_cache.get(new_id)
                for existing_id in existing_ids:
                    existing_attrs = self.entity_cache.get(existing_id)
                    pair_checks += 1
                    if pair_checker(new_attrs, existing_attrs):
                        self.pair_tracker.add_pair(new_id, existing_id)
                        new_pairs += 1

            # New × new
            new_list = sorted(new_ids)
            for i, id1 in enumerate(new_list):
                for id2 in new_list[i + 1 :]:
                    pair_checks += 1
                    attrs1 = self.entity_cache.get(id1)
                    attrs2 = self.entity_cache.get(id2)
                    if pair_checker(attrs1, attrs2):
                        self.pair_tracker.add_pair(id1, id2)
                        new_pairs += 1

            # Re-evaluate retracted pairs — distinguishing no-op from permanent retractions.
            # A "no-op retraction" is one where the pair is immediately re-added after
            # re-evaluation (the attribute change did not affect pair validity).
            # A "permanent retraction" is one where the pair is NOT re-added (genuinely
            # invalidated by the new entity state).
            for p in retracted_pairs:
                attrs1 = self.entity_cache.get(p[0])
                attrs2 = self.entity_cache.get(p[1])
                if attrs1 and attrs2:
                    pair_checks += 1
                    if pair_checker(attrs1, attrs2):
                        self.pair_tracker.add_pair(p[0], p[1])
                        new_pairs += 1
                        chunk_noop_retractions += 1
                    else:
                        chunk_permanent_retractions += 1
                else:
                    # One entity missing from cache (should not happen in normal operation)
                    chunk_permanent_retractions += 1
            self._noop_retractions += chunk_noop_retractions
            self._permanent_retractions += chunk_permanent_retractions

            # For updated entities, also check against ALL other entities
            # (not just retracted pairs) — the new classification may create
            # NEW valid pairs that didn't exist before.
            # Bug fix (Iteration 12): track already-checked canonical pairs across
            # updated-entity sweeps to avoid double-counting when two updated entities
            # interact. Previously, (A,B) was checked once in A's sweep and once in
            # B's sweep, inflating the pair_checks counter (though correctness was
            # preserved by add_pair's idempotency).
            all_ids = self.entity_cache.get_ids()
            checked_in_updated_sweep: set[tuple[str, str]] = set()
            for updated_id in updated_ids:
                updated_attrs = self.entity_cache.get(updated_id)
                for other_id in all_ids:
                    if other_id == updated_id:
                        continue
                    # Skip new entities — updated × new already covered
                    # in the new × existing loop (new_id × existing_id where
                    # updated entities are in existing_ids)
                    if other_id in new_ids:
                        continue
                    canonical = (min(updated_id, other_id), max(updated_id, other_id))
                    # Skip if this pair was already checked via retraction
                    if canonical in retracted_pairs:
                        continue
                    # Skip if this pair was already checked in a prior iteration of
                    # the updated-entity sweep (deduplication across updated entities)
                    if canonical in checked_in_updated_sweep:
                        continue
                    checked_in_updated_sweep.add(canonical)
                    other_attrs = self.entity_cache.get(other_id)
                    pair_checks += 1
                    if pair_checker(updated_attrs, other_attrs):
                        self.pair_tracker.add_pair(updated_id, other_id)
                        new_pairs += 1

        # Warn if updated-entity sweep dominates (quality-of-life heuristic).
        # When len(updated_ids) >> len(new_ids), the O(u × n) sweep dominates
        # cost. Consider reducing chunk granularity or switching to full recompute.
        if updated_ids and new_ids and len(updated_ids) > len(new_ids) * 2:
            warnings.warn(
                f"High update ratio: {len(updated_ids)} updates vs {len(new_ids)} new entities "
                f"at chunk {chunk_index}. The O(u×n) updated-entity sweep dominates cost. "
                f"Consider reducing chunk granularity or switching to full recompute.",
                stacklevel=2,
            )

        self._total_new_entities += len(new_ids)
        self._total_pair_checks += pair_checks

        stats = {
            "chunk_index": chunk_index,
            "new_entities": len(new_ids),
            "updated_entities": len(updated_ids),
            "pair_checks": pair_checks,
            "new_pairs": new_pairs,
            "retracted_pairs": len(retracted_pairs),
            "noop_retractions": chunk_noop_retractions if pair_checker else 0,
            "permanent_retractions": chunk_permanent_retractions if pair_checker else 0,
            "total_pairs": len(self.pair_tracker),
            "total_entities": len(self.entity_cache),
        }
        self.chunk_log.append(stats)
        # Cache stats for idempotency guard — future calls with same chunk_index
        # return this cached result instantly (O(1)) without re-executing the sweep.
        self._processed_chunk_indices[chunk_index] = stats
        return stats

    def get_stats(self) -> dict[str, Any]:
        """Get cumulative statistics.

        Retraction breakdown:
        - total_retractions: all retraction events (PairTracker.retraction_count)
        - noop_retractions: pairs retracted then immediately re-added (effective no-ops;
          the entity's attribute change did not affect pair validity)
        - permanent_retractions: pairs retracted and NOT re-added (genuine invalidations
          where the entity's new state fails the pair_checker condition)

        Note: total_retractions ≈ noop_retractions + permanent_retractions when a
        pair_checker is provided. Without a pair_checker, noop/permanent counters stay 0
        since re-evaluation is skipped.

        For V3 monotone_attrs runs: expect noop_retractions >> permanent_retractions
        (monotone merge preserves qualifying=True → pairs re-added immediately).
        For V2 runs (no monotone fix): permanent_retractions > 0 (entities downgraded →
        pairs not re-added → missing from final results).
        """
        return {
            "total_entities": len(self.entity_cache),
            "total_pairs": len(self.pair_tracker),
            "total_new_entities_processed": self._total_new_entities,
            "total_pair_checks": self._total_pair_checks,
            "total_retractions": self._total_retractions,
            "noop_retractions": self._noop_retractions,
            "permanent_retractions": self._permanent_retractions,
            "chunks_processed": len(self.chunk_log),
        }

    def apply_edits(
        self,
        edits: dict[str, dict[str, Any]],
        pair_checker: Any = None,
        edit_chunk_index: int = -1,
        merge: bool = False,
    ) -> dict[str, Any]:
        """Apply entity attribute edits and retract/re-evaluate affected pairs.

        For dynamic context scenarios where entity attributes change between turns
        (document edits, streaming corrections, etc.).

        This is the first-class API for non-monotonic context updates. It handles:
        1. Updating entity attributes in the cache
        2. Retracting all pairs involving edited entities
        3. Re-evaluating retracted pairs with updated attributes
        4. Discovering new pairs created by the attribute change

        Complexity:
            Phase 1 (update + retract): O(E × P_avg) where E = len(edits) and
                P_avg = avg pairs per entity.
            Phase 2 (re-evaluate retracted): O(R) where R = total retracted pairs.
            Phase 3 (new pair discovery): O(E × N) where N = total entities.
                For small edit batches (E < 20), this is negligible. For bulk edits
                (E > 100), consider batching or using process_chunk() with a
                synthetic edit chunk instead.

        Args:
            edits: {entity_id: new_attributes} for entities to modify
            pair_checker: Optional callable(attrs1, attrs2) -> bool
            edit_chunk_index: Chunk index to record for the edit (default -1)
            merge: If True, merge new attributes into existing attributes instead
                of replacing them. Existing attributes not present in the edit are
                preserved. Default False (full replacement, caller must provide
                complete attributes).

        Returns:
            Stats dict with: entities_edited, total_retracted, pairs_readded,
            new_pairs_from_edits, pairs_before, pairs_after, precision_preserved.
        """
        pairs_before = len(self.pair_tracker)

        # Phase 1: Update all entities and collect ALL retracted pairs (deduplicated)
        all_retracted: set[tuple[str, str]] = set()
        for eid, new_attrs in edits.items():
            if merge:
                old_attrs = self.entity_cache.get(eid) or {}
                merged = {**old_attrs, **new_attrs}
                self.entity_cache.add(eid, merged, chunk_index=edit_chunk_index)
            else:
                self.entity_cache.add(eid, new_attrs, chunk_index=edit_chunk_index)
            retracted = self.pair_tracker.retract_entity(eid)
            all_retracted |= retracted

        total_retracted = len(all_retracted)
        self._total_retractions += total_retracted

        pairs_readded = 0
        new_pairs_from_edits = 0
        pair_checks = 0

        if pair_checker:
            # Phase 2: Re-evaluate all retracted pairs with updated attributes
            for p in all_retracted:
                a1 = self.entity_cache.get(p[0])
                a2 = self.entity_cache.get(p[1])
                pair_checks += 1
                if a1 and a2 and pair_checker(a1, a2):
                    self.pair_tracker.add_pair(p[0], p[1])
                    pairs_readded += 1

            # Phase 3: Check for NEW pairs (edited entities × all existing)
            edited_ids = set(edits.keys())
            checked_in_edit_sweep: set[tuple[str, str]] = set()
            for eid in edited_ids:
                updated_attrs = self.entity_cache.get(eid)
                if not updated_attrs:
                    continue
                for other_id in self.entity_cache.get_ids():
                    if other_id == eid:
                        continue
                    if self.pair_tracker.has_pair(eid, other_id):
                        continue
                    # Deduplicate: if both entities are edited, only check once
                    canonical = (min(eid, other_id), max(eid, other_id))
                    if canonical in checked_in_edit_sweep:
                        continue
                    checked_in_edit_sweep.add(canonical)
                    other_attrs = self.entity_cache.get(other_id)
                    pair_checks += 1
                    if other_attrs and pair_checker(updated_attrs, other_attrs):
                        self.pair_tracker.add_pair(eid, other_id)
                        new_pairs_from_edits += 1

        self._total_pair_checks += pair_checks
        pairs_after = len(self.pair_tracker)
        permanent = total_retracted - pairs_readded
        self._permanent_retractions += max(0, permanent)
        self._noop_retractions += pairs_readded

        return {
            "entities_edited": len(edits),
            "total_retracted": total_retracted,
            "pairs_readded": pairs_readded,
            "new_pairs_from_edits": new_pairs_from_edits,
            "pair_checks": pair_checks,
            "permanent_retractions": max(0, permanent),
            "pairs_before": pairs_before,
            "pairs_after": pairs_after,
        }

    def verify_lossless(self, expected_entity_ids: set[str]) -> dict[str, Any]:
        """Verify that entity_cache contains exactly the expected entities.

        This is the programmatic proof that the REPL state is lossless: after
        processing chunks 0..k, the EntityCache must contain EXACTLY the union
        of all entity IDs seen in those chunks — no drops, no phantom additions.

        The cache is lossless by construction (EntityCache.add() never removes),
        but this method provides runtime verification for external auditing.

        Args:
            expected_entity_ids: The complete set of entity IDs that should be
                in the cache after processing all chunks so far.

        Returns:
            Dict with: is_lossless, missing_ids, extra_ids, expected_count,
            cached_count, chunks_processed.
        """
        cached_ids = self.entity_cache.get_ids()
        # Convert both to comparable types (str)
        cached_str = {str(x) for x in cached_ids}
        expected_str = {str(x) for x in expected_entity_ids}
        missing = sorted(expected_str - cached_str)
        extra = sorted(cached_str - expected_str)
        return {
            "is_lossless": len(missing) == 0 and len(extra) == 0,
            "missing_ids": missing,
            "extra_ids": extra,
            "expected_count": len(expected_str),
            "cached_count": len(cached_str),
            "chunks_processed": len(self.chunk_log),
        }

    def memory_usage(self) -> dict[str, Any]:
        """Report memory usage in bytes for each component.

        Uses sys.getsizeof for shallow container sizes plus deep traversal
        of entity attributes and pair tuples. Tracks already-counted object
        IDs to avoid double-counting shared references (e.g., entity ID
        strings stored in _entities, _by_chunk, and _entity_pairs are the
        SAME Python objects due to interning).

        This gives an accurate lower bound on the Python-level memory footprint.

        Useful for addressing the "memory will blow up" concern: shows that
        REPL state memory is negligible compared to LLM context window sizes.
        """
        import sys

        # Track already-counted objects to avoid double-counting shared references.
        # Python string interning means the same entity ID string stored in
        # _entities, _by_chunk, and _entity_pairs is the SAME object in memory.
        _counted_ids: set[int] = set()

        def _sizeof_unique(obj: Any) -> int:
            """Return sys.getsizeof(obj) only if not already counted."""
            obj_id = id(obj)
            if obj_id in _counted_ids:
                return 0
            _counted_ids.add(obj_id)
            return sys.getsizeof(obj)

        # Entity cache: dict of dicts with nested attributes
        entity_bytes = _sizeof_unique(self.entity_cache._entities)
        for eid, entry in self.entity_cache._entities.items():
            entity_bytes += _sizeof_unique(eid)  # key string
            entity_bytes += _sizeof_unique(entry)  # entry dict
            attrs = entry.get("attributes", {})
            entity_bytes += _sizeof_unique(attrs)
            for attr_k, attr_v in attrs.items():
                entity_bytes += _sizeof_unique(attr_k)
                entity_bytes += _sizeof_unique(attr_v)
                # Deep-traverse list attributes (e.g., instances lists)
                if isinstance(attr_v, list):
                    for item in attr_v:
                        entity_bytes += _sizeof_unique(item)
                        if isinstance(item, dict):
                            for k2, v2 in item.items():
                                entity_bytes += _sizeof_unique(k2) + _sizeof_unique(v2)

        # By-chunk index (entity ID strings already counted above via _sizeof_unique)
        chunk_index_bytes = _sizeof_unique(self.entity_cache._by_chunk)
        for chunk_set in self.entity_cache._by_chunk.values():
            chunk_index_bytes += _sizeof_unique(chunk_set)
            for eid in chunk_set:
                chunk_index_bytes += _sizeof_unique(eid)

        # Pair tracker: set of tuples
        pair_bytes = _sizeof_unique(self.pair_tracker._pairs)
        for p in self.pair_tracker._pairs:
            pair_bytes += _sizeof_unique(p)

        # Inverted index: dict of entity_id -> set of tuples
        # Entity ID strings and pair tuples may already be counted above.
        index_bytes = _sizeof_unique(self.pair_tracker._entity_pairs)
        for eid, pair_set in self.pair_tracker._entity_pairs.items():
            index_bytes += _sizeof_unique(eid)
            index_bytes += _sizeof_unique(pair_set)
            for p in pair_set:
                index_bytes += _sizeof_unique(p)

        # Retracted pairs set
        retracted_bytes = _sizeof_unique(self.pair_tracker._retracted)
        for p in self.pair_tracker._retracted:
            retracted_bytes += _sizeof_unique(p)

        # Chunk log and processed indices
        meta_bytes = sys.getsizeof(self.chunk_log)
        for entry in self.chunk_log:
            meta_bytes += sys.getsizeof(entry)
        meta_bytes += sys.getsizeof(self._processed_chunk_indices)

        total = entity_bytes + chunk_index_bytes + pair_bytes + index_bytes + retracted_bytes + meta_bytes

        return {
            "entity_cache_bytes": entity_bytes,
            "chunk_index_bytes": chunk_index_bytes,
            "pair_tracker_bytes": pair_bytes,
            "inverted_index_bytes": index_bytes,
            "retracted_set_bytes": retracted_bytes,
            "metadata_bytes": meta_bytes,
            "total_bytes": total,
            "total_kb": round(total / 1024, 1),
            "total_mb": round(total / (1024 * 1024), 3),
            "component_breakdown": {
                "entity_cache": round(entity_bytes / 1024, 1),
                "chunk_index": round(chunk_index_bytes / 1024, 1),
                "pair_set": round(pair_bytes / 1024, 1),
                "inverted_index": round(index_bytes / 1024, 1),
                "retracted": round(retracted_bytes / 1024, 1),
                "metadata": round(meta_bytes / 1024, 1),
            },
            "counts": {
                "entities": len(self.entity_cache),
                "pairs": len(self.pair_tracker),
                "retracted": len(self.pair_tracker._retracted),
            },
        }

    def rebuild_pairs(self, pair_checker: Any) -> dict[str, Any]:
        """Rebuild all pairs from entity cache using the pair_checker.

        Proves the two-tier architecture: pair state is fully derivable from
        entity state. The pair tracker (O(n²)) is an optimization cache that
        can be evicted and rebuilt on-demand from the entity cache (O(n)).

        This method:
        1. Saves the current pair set
        2. Clears the pair tracker
        3. Re-evaluates all C(n,2) entity pairs using pair_checker
        4. Returns comparison between original and rebuilt pairs

        Args:
            pair_checker: callable(attrs1, attrs2) -> bool

        Returns:
            Dict with: original_count, rebuilt_count, match (bool),
            missing_pairs (in original but not rebuilt),
            extra_pairs (in rebuilt but not original).
        """
        # Save original pairs
        original_pairs = self.pair_tracker.get_pairs()
        original_count = len(original_pairs)

        # Clear pair tracker
        self.pair_tracker = PairTracker()

        # Rebuild from entity cache
        # Note: entity_cache.get() returns the attributes dict directly
        ids = sorted(self.entity_cache.get_ids())
        rebuild_checks = 0
        for i, id1 in enumerate(ids):
            attrs1 = self.entity_cache.get(id1) or {}
            for id2 in ids[i + 1 :]:
                attrs2 = self.entity_cache.get(id2) or {}
                rebuild_checks += 1
                if pair_checker(attrs1, attrs2):
                    self.pair_tracker.add_pair(id1, id2)

        rebuilt_pairs = self.pair_tracker.get_pairs()
        rebuilt_count = len(rebuilt_pairs)

        missing = original_pairs - rebuilt_pairs
        extra = rebuilt_pairs - original_pairs

        return {
            "original_count": original_count,
            "rebuilt_count": rebuilt_count,
            "match": len(missing) == 0 and len(extra) == 0,
            "missing_pairs": len(missing),
            "extra_pairs": len(extra),
            "rebuild_checks": rebuild_checks,
            "entity_count": len(ids),
        }

    def reset(self) -> None:
        """Reset all state to allow reprocessing from scratch.

        Clears entity cache, pair tracker, chunk log, and all counters.
        Required if you need to re-process a chunk that was already processed
        (the idempotency guard in process_chunk() returns cached stats without
        reprocessing; reset() clears the cache so fresh processing occurs).

        Use with caution: resetting discards all accumulated incremental state.
        For selective re-processing, consider creating a new IncrementalState
        instead of resetting.
        """
        self.entity_cache = EntityCache()
        self.pair_tracker = PairTracker()
        self.chunk_log = []
        self._total_new_entities = 0
        self._total_pair_checks = 0
        self._total_retractions = 0
        self._noop_retractions = 0
        self._permanent_retractions = 0
        self._processed_chunk_indices = {}

    def __repr__(self) -> str:
        return (
            f"IncrementalState("
            f"{len(self.entity_cache)} entities, "
            f"{len(self.pair_tracker)} pairs, "
            f"{len(self.chunk_log)} chunks)"
        )
