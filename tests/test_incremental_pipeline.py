"""
Tests for the incremental computation pipeline.

Validates:
1. EntityCache — add, update, retrieval, chunk tracking
2. PairTracker — add, retract, inverted index
3. IncrementalState — end-to-end incremental processing with retractions
4. HistoryManager — message pruning strategies
5. Integration — persistent REPL with incremental primitives
"""

from rlm.core.history_manager import HistoryManager
from rlm.core.incremental import EntityCache, IncrementalState, PairTracker

# =============================================================================
# EntityCache Tests
# =============================================================================


class TestEntityCache:
    def test_add_and_get(self):
        cache = EntityCache()
        cache.add("u1", {"type": "person", "location": "NYC"}, chunk_index=0)
        assert cache.get("u1") == {"type": "person", "location": "NYC"}
        assert len(cache) == 1

    def test_update_returns_true(self):
        cache = EntityCache()
        assert cache.add("u1", {"type": "person"}, chunk_index=0) is False  # new
        assert cache.add("u1", {"type": "org"}, chunk_index=1) is True  # update

    def test_chunk_tracking(self):
        cache = EntityCache()
        cache.add("u1", {"x": 1}, chunk_index=0)
        cache.add("u2", {"x": 2}, chunk_index=0)
        cache.add("u3", {"x": 3}, chunk_index=1)
        assert cache.get_from_chunk(0) == {"u1", "u2"}
        assert cache.get_from_chunk(1) == {"u3"}
        assert cache.get_from_chunk(2) == set()

    def test_get_ids(self):
        cache = EntityCache()
        cache.add("a", {}, 0)
        cache.add("b", {}, 0)
        assert cache.get_ids() == {"a", "b"}

    def test_contains(self):
        cache = EntityCache()
        cache.add("u1", {}, 0)
        assert "u1" in cache
        assert "u2" not in cache


# =============================================================================
# PairTracker Tests
# =============================================================================


class TestPairTracker:
    def test_add_pair(self):
        tracker = PairTracker()
        tracker.add_pair("u1", "u2")
        assert len(tracker) == 1
        assert ("u1", "u2") in tracker.get_pairs()

    def test_canonical_order(self):
        tracker = PairTracker()
        tracker.add_pair("u2", "u1")
        tracker.add_pair("u1", "u2")
        assert len(tracker) == 1  # same pair, canonical order

    def test_retract_entity(self):
        tracker = PairTracker()
        tracker.add_pair("u1", "u2")
        tracker.add_pair("u1", "u3")
        tracker.add_pair("u2", "u3")

        retracted = tracker.retract_entity("u1")
        assert len(retracted) == 2  # u1-u2 and u1-u3
        assert len(tracker) == 1  # only u2-u3 remains
        assert tracker.retraction_count == 2

    def test_retracted_pair_can_be_readded(self):
        tracker = PairTracker()
        tracker.add_pair("u1", "u2")
        tracker.retract_entity("u1")
        assert len(tracker) == 0

        # Re-add after re-evaluation
        tracker.add_pair("u1", "u2")
        assert len(tracker) == 1
        # Should no longer be in retracted set
        assert ("u1", "u2") not in tracker.retracted_pairs

    def test_get_pairs_for_entity(self):
        tracker = PairTracker()
        tracker.add_pair("u1", "u2")
        tracker.add_pair("u1", "u3")
        pairs = tracker.get_pairs_for_entity("u1")
        assert len(pairs) == 2


# =============================================================================
# IncrementalState Tests
# =============================================================================


class TestIncrementalState:
    def test_basic_incremental_processing(self):
        state = IncrementalState()

        # Chunk 0: add 3 entities
        def checker(a, b):
            return a.get("type") == b.get("type")

        stats0 = state.process_chunk(
            chunk_index=0,
            new_entities={
                "u1": {"type": "A"},
                "u2": {"type": "A"},
                "u3": {"type": "B"},
            },
            pair_checker=checker,
        )
        assert stats0["new_entities"] == 3
        assert stats0["pair_checks"] == 3  # C(3,2) = 3
        assert stats0["total_pairs"] == 1  # only u1-u2 match (both type A)

    def test_incremental_chunk(self):
        state = IncrementalState()

        def checker(a, b):
            return a.get("type") == b.get("type")

        # Chunk 0
        state.process_chunk(0, {"u1": {"type": "A"}, "u2": {"type": "B"}}, checker)
        assert len(state.pair_tracker) == 0  # different types

        # Chunk 1: add u3 of type A — should pair with u1
        stats1 = state.process_chunk(1, {"u3": {"type": "A"}}, checker)
        assert stats1["new_entities"] == 1
        assert stats1["pair_checks"] == 2  # u3 vs u1, u3 vs u2
        assert stats1["new_pairs"] == 1  # u3-u1
        assert len(state.pair_tracker) == 1

    def test_retraction_on_update(self):
        """Test non-monotonic discovery: updating an entity invalidates pairs."""
        state = IncrementalState()

        def checker(a, b):
            return a.get("type") == b.get("type")

        # Chunk 0: u1=A, u2=A → pair (u1, u2)
        state.process_chunk(0, {"u1": {"type": "A"}, "u2": {"type": "A"}}, checker)
        assert len(state.pair_tracker) == 1

        # Chunk 1: u1 is updated to type B (reclassification)
        # This should retract (u1, u2) and then re-evaluate
        stats1 = state.process_chunk(1, {"u1": {"type": "B"}}, checker)
        assert stats1["updated_entities"] == 1
        assert stats1["retracted_pairs"] == 1
        assert len(state.pair_tracker) == 0  # u1=B, u2=A → no match

    def test_retraction_with_readd(self):
        """Retraction followed by re-validation (entity still matches some)."""
        state = IncrementalState()

        def checker(a, b):
            return a.get("type") == b.get("type")

        # Chunk 0: u1=A, u2=A, u3=B
        state.process_chunk(
            0, {"u1": {"type": "A"}, "u2": {"type": "A"}, "u3": {"type": "B"}}, checker
        )
        assert len(state.pair_tracker) == 1  # (u1, u2)

        # Chunk 1: u1 updated to B → retract (u1,u2), re-eval → now u1 matches u3
        state.process_chunk(1, {"u1": {"type": "B"}}, checker)
        assert len(state.pair_tracker) == 1  # (u1, u3) — re-evaluated retracted pair
        pairs = state.pair_tracker.get_pairs()
        assert ("u1", "u3") in pairs

    def test_stats_tracking(self):
        state = IncrementalState()

        def always_match(a, b):
            return True

        state.process_chunk(0, {"u1": {}, "u2": {}}, always_match)
        state.process_chunk(1, {"u3": {}}, always_match)

        stats = state.get_stats()
        assert stats["total_entities"] == 3
        assert stats["total_pairs"] == 3  # C(3,2)
        assert stats["chunks_processed"] == 2
        assert stats["total_retractions"] == 0


# =============================================================================
# HistoryManager Tests
# =============================================================================


class TestHistoryManager:
    def _make_history(self, n_iterations: int) -> list[dict]:
        """Build a fake message history with system + N iterations."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "Context metadata..."},
            {"role": "user", "content": "Think step by step..."},
        ]
        for i in range(n_iterations):
            messages.append(
                {"role": "assistant", "content": f"Iteration {i}: I will analyze the data..."}
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Code executed:\n```python\nx_{i} = {i}\n```\n\nREPL output:\n{i}\nREPL variables: ['x_{i}']",
                }
            )
        return messages

    def test_sliding_window_no_prune_needed(self):
        mgr = HistoryManager(strategy="sliding_window", max_recent_iterations=5)
        history = self._make_history(3)  # 3 iters = 6 messages < 10 limit
        pruned = mgr.prune(history)
        assert len(pruned) == len(history)

    def test_sliding_window_prunes(self):
        mgr = HistoryManager(strategy="sliding_window", max_recent_iterations=2)
        history = self._make_history(5)  # 5 iters = 10 messages
        pruned = mgr.prune(history)
        # System (3) + last 2 iters (4 messages) = 7
        assert len(pruned) == 7

    def test_summarize_no_prune_needed(self):
        mgr = HistoryManager(strategy="summarize", max_recent_iterations=5)
        history = self._make_history(3)
        pruned = mgr.prune(history)
        assert len(pruned) == len(history)

    def test_summarize_prunes_with_summary(self):
        mgr = HistoryManager(strategy="summarize", max_recent_iterations=2)
        history = self._make_history(5)
        pruned = mgr.prune(history, turn_number=1)

        # System (3) + summary (1) + ack (1) + last 2 iters (4) = 9
        assert len(pruned) == 9

        # Check that summary message exists
        summary_msg = pruned[3]
        assert summary_msg["role"] == "user"
        assert "PRIOR COMPUTATION SUMMARY" in summary_msg["content"]

    def test_token_budget_prunes(self):
        # Use a very tight budget that forces pruning (each message is ~50 chars = ~12 tokens)
        mgr = HistoryManager(strategy="token_budget", estimated_token_budget=50)
        history = self._make_history(10)  # lots of messages, ~23 total
        pruned = mgr.prune(history)
        # Should be shorter than original
        assert len(pruned) < len(history)
        # System messages always preserved
        assert pruned[0]["role"] == "system"

    def test_turn_summary_recording(self):
        mgr = HistoryManager()
        mgr.add_turn_summary("Turn 1: processed 100 users, found 50 pairs")
        mgr.add_turn_summary("Turn 2: processed 50 new users, found 30 new pairs")
        summaries = mgr.get_turn_summaries()
        assert len(summaries) == 2
        assert "Turn 1" in summaries[0]

    def test_generate_turn_summary(self):
        mgr = HistoryManager()
        history = self._make_history(3)
        summary = mgr.generate_turn_summary(history, final_answer="42 pairs found")
        assert "3 iterations" in summary
        assert "42 pairs found" in summary


# =============================================================================
# Integration Tests: REPL + Incremental Primitives
# =============================================================================


class TestREPLIncrementalIntegration:
    def test_incremental_primitives_available_in_persistent_repl(self):
        """Persistent REPL should have incremental primitives injected."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(persistent=True)
        try:
            assert "EntityCache" in repl.locals
            assert "PairTracker" in repl.locals
            assert "IncrementalState" in repl.locals
            assert "_incremental" in repl.locals

            # Verify they work
            result = repl.execute_code(
                "ec = EntityCache()\nec.add('u1', {'type': 'A'}, 0)\nprint(ec.get('u1'))"
            )
            assert "type" in result.stdout
            assert "A" in result.stdout
        finally:
            repl.cleanup()

    def test_incremental_primitives_not_in_non_persistent_repl(self):
        """Non-persistent REPL should NOT have incremental primitives."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(persistent=False)
        try:
            assert "EntityCache" not in repl.locals
            assert "_incremental" not in repl.locals
        finally:
            repl.cleanup()

    def test_incremental_state_persists_across_executions(self):
        """Incremental state should persist across code executions in same REPL."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(persistent=True)
        try:
            # Execution 1: add entities
            repl.execute_code("""
_incremental.entity_cache.add("u1", {"type": "A"}, 0)
_incremental.entity_cache.add("u2", {"type": "A"}, 0)
print(f"Entities: {len(_incremental.entity_cache)}")
""")

            # Execution 2: check pairs using cached entities
            result = repl.execute_code("""
ids = _incremental.entity_cache.get_ids()
print(f"Cached IDs: {sorted(ids)}")
print(f"u1 type: {_incremental.entity_cache.get('u1')}")
""")
            assert "u1" in result.stdout
            assert "u2" in result.stdout
            assert "type" in result.stdout
        finally:
            repl.cleanup()

    def test_full_incremental_flow(self):
        """End-to-end: 2 chunks, entity caching, pair finding, retraction."""
        from rlm.environments.local_repl import LocalREPL

        repl = LocalREPL(persistent=True, context_payload="chunk 0 data")
        try:
            # Chunk 0: process initial entities
            result = repl.execute_code("""
# Simulate processing chunk 0
entities_0 = {"u1": {"type": "person"}, "u2": {"type": "org"}, "u3": {"type": "person"}}
for eid, attrs in entities_0.items():
    _incremental.entity_cache.add(eid, attrs, 0)

# Find pairs where both are persons
for id1 in sorted(entities_0.keys()):
    for id2 in sorted(entities_0.keys()):
        if id1 < id2:
            a1 = _incremental.entity_cache.get(id1)
            a2 = _incremental.entity_cache.get(id2)
            if a1["type"] == a2["type"]:
                _incremental.pair_tracker.add_pair(id1, id2)

print(f"After chunk 0: {len(_incremental.pair_tracker)} pairs, {len(_incremental.entity_cache)} entities")
print(f"Pairs: {_incremental.pair_tracker.get_pairs()}")
""")
            assert "1 pairs" in result.stdout  # u1-u3 (both person)
            assert "3 entities" in result.stdout

            # Chunk 1: add new context and process incrementally
            repl.add_context("chunk 1 data")
            result = repl.execute_code("""
# Process only NEW entities from chunk 1
new_entities = {"u4": {"type": "person"}, "u5": {"type": "org"}}
existing_ids = _incremental.entity_cache.get_ids()

for eid, attrs in new_entities.items():
    _incremental.entity_cache.add(eid, attrs, 1)

# Check pairs: new × existing ONLY (incremental!)
pair_checks = 0
for new_id in new_entities:
    for existing_id in existing_ids:
        pair_checks += 1
        a1 = _incremental.entity_cache.get(new_id)
        a2 = _incremental.entity_cache.get(existing_id)
        if a1["type"] == a2["type"]:
            _incremental.pair_tracker.add_pair(new_id, existing_id)

# Also check new × new
new_list = sorted(new_entities.keys())
for i, id1 in enumerate(new_list):
    for id2 in new_list[i+1:]:
        pair_checks += 1
        a1 = _incremental.entity_cache.get(id1)
        a2 = _incremental.entity_cache.get(id2)
        if a1["type"] == a2["type"]:
            _incremental.pair_tracker.add_pair(id1, id2)

print(f"Incremental: {pair_checks} pair checks (vs {5*4//2} for full recompute)")
print(f"After chunk 1: {len(_incremental.pair_tracker)} pairs, {len(_incremental.entity_cache)} entities")
print(f"Pairs: {sorted(_incremental.pair_tracker.get_pairs())}")
""")
            # Should have checked 2*3 + 1 = 7 pairs (new × existing + new × new)
            # vs 10 for full C(5,2)
            assert "7 pair checks" in result.stdout
            assert "5 entities" in result.stdout
            # u1-u3, u1-u4, u3-u4 (all persons), u2-u5 (both org) = 4 pairs
            assert "4 pairs" in result.stdout
        finally:
            repl.cleanup()


class TestNonMonotonicRetraction:
    """Test the retraction mechanism for Task 19-style non-monotonic conditions."""

    def test_exactly_one_constraint_retraction(self):
        """Simulate: 'exactly one instance with entity type X'.
        New data adds a second instance, invalidating the previous classification.
        """
        state = IncrementalState()

        # "exactly_one_X" checker: pair valid if both have exactly one X
        def checker(a, b):
            return a.get("x_count") == 1 and b.get("x_count") == 1

        # Chunk 0: u1 has x_count=1, u2 has x_count=1 → valid pair
        state.process_chunk(0, {"u1": {"x_count": 1}, "u2": {"x_count": 1}}, checker)
        assert len(state.pair_tracker) == 1

        # Chunk 1: new data reveals u1 actually has x_count=2
        # This should trigger retraction + re-evaluation → pair invalidated
        stats = state.process_chunk(1, {"u1": {"x_count": 2}}, checker)
        assert stats["updated_entities"] == 1
        assert stats["retracted_pairs"] == 1
        assert len(state.pair_tracker) == 0  # pair invalidated

    def test_non_monotonic_then_recovery(self):
        """A retracted pair can be re-added if conditions become valid again."""
        state = IncrementalState()

        def checker(a, b):
            return a.get("score") > 5 and b.get("score") > 5

        # Chunk 0: both above threshold
        state.process_chunk(0, {"u1": {"score": 10}, "u2": {"score": 8}}, checker)
        assert len(state.pair_tracker) == 1

        # Chunk 1: u1 drops below threshold
        state.process_chunk(1, {"u1": {"score": 3}}, checker)
        assert len(state.pair_tracker) == 0

        # Chunk 2: u1 goes back above threshold
        state.process_chunk(2, {"u1": {"score": 12}}, checker)
        assert len(state.pair_tracker) == 1
