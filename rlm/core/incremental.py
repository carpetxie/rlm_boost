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

from typing import Any


class EntityCache:
    """Cache entity classifications with versioning.

    Each entity has:
    - id: unique identifier
    - attributes: dict of classifications/properties
    - source_chunk: which chunk it was first seen in
    - last_updated: which chunk last modified it

    Supports O(1) lookup by entity ID and O(k) iteration over
    entities from a specific chunk.
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
        """Get entity IDs first seen in a given chunk."""
        return self._by_chunk.get(chunk_index, set())

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
        self._retracted: set[tuple[str, str]] = set()  # pairs removed by retraction
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
        """
        affected = self._entity_pairs.get(entity_id, set()).copy()
        for pair in affected:
            self._pairs.discard(pair)
            self._retracted.add(pair)
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
        """Pairs that were retracted and not yet re-added."""
        return self._retracted.copy()

    def __len__(self) -> int:
        return len(self._pairs)

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

    def process_chunk(
        self,
        chunk_index: int,
        new_entities: dict[str, dict[str, Any]],
        pair_checker: Any = None,
    ) -> dict[str, Any]:
        """Process a new chunk of entities incrementally.

        Args:
            chunk_index: Index of the arriving chunk
            new_entities: {entity_id: attributes} for entities in the new chunk
            pair_checker: Optional callable(attrs1, attrs2) -> bool for pair validation

        Returns:
            Dict with processing stats for this chunk
        """
        existing_ids = self.entity_cache.get_ids()
        updated_ids = set()
        new_ids = set()

        # 1. Add/update entities
        for eid, attrs in new_entities.items():
            was_update = self.entity_cache.add(eid, attrs, chunk_index)
            if was_update:
                updated_ids.add(eid)
            else:
                new_ids.add(eid)

        # 2. Handle retractions for updated entities
        retracted_pairs = set()
        for eid in updated_ids:
            retracted = self.pair_tracker.retract_entity(eid)
            retracted_pairs |= retracted
            self._total_retractions += len(retracted)

        # 3. Check pairs incrementally (if checker provided)
        new_pairs = 0
        pair_checks = 0
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

            # Re-evaluate retracted pairs
            for p in retracted_pairs:
                attrs1 = self.entity_cache.get(p[0])
                attrs2 = self.entity_cache.get(p[1])
                if attrs1 and attrs2:
                    pair_checks += 1
                    if pair_checker(attrs1, attrs2):
                        self.pair_tracker.add_pair(p[0], p[1])
                        new_pairs += 1

            # For updated entities, also check against ALL other entities
            # (not just retracted pairs) — the new classification may create
            # NEW valid pairs that didn't exist before
            all_ids = self.entity_cache.get_ids()
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
                    # Skip if this pair was already checked via retraction
                    canonical = (min(updated_id, other_id), max(updated_id, other_id))
                    if canonical in retracted_pairs:
                        continue
                    other_attrs = self.entity_cache.get(other_id)
                    pair_checks += 1
                    if pair_checker(updated_attrs, other_attrs):
                        self.pair_tracker.add_pair(updated_id, other_id)
                        new_pairs += 1

        self._total_new_entities += len(new_ids)
        self._total_pair_checks += pair_checks

        stats = {
            "chunk_index": chunk_index,
            "new_entities": len(new_ids),
            "updated_entities": len(updated_ids),
            "pair_checks": pair_checks,
            "new_pairs": new_pairs,
            "retracted_pairs": len(retracted_pairs),
            "total_pairs": len(self.pair_tracker),
            "total_entities": len(self.entity_cache),
        }
        self.chunk_log.append(stats)
        return stats

    def get_stats(self) -> dict[str, Any]:
        """Get cumulative statistics."""
        return {
            "total_entities": len(self.entity_cache),
            "total_pairs": len(self.pair_tracker),
            "total_new_entities_processed": self._total_new_entities,
            "total_pair_checks": self._total_pair_checks,
            "total_retractions": self._total_retractions,
            "chunks_processed": len(self.chunk_log),
        }

    def __repr__(self) -> str:
        return (
            f"IncrementalState("
            f"{len(self.entity_cache)} entities, "
            f"{len(self.pair_tracker)} pairs, "
            f"{len(self.chunk_log)} chunks)"
        )
