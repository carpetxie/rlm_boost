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
# Monotone Attribute Tests (Iteration 13 — library-level monotone_attrs)
# =============================================================================


class TestMonotoneAttrs:
    """Tests for the monotone_attrs parameter of IncrementalState.process_chunk().

    These tests validate the four behaviors mandated by the Iteration 13 critique:
    (a) Monotone attr preserved on downgrade (truthy old + falsy new → keep old)
    (b) Retraction skipped when ONLY monotone attrs change (and they don't actually change)
    (c) Retraction fires when non-monotone attrs change
    (d) monotone_attrs=None preserves existing behavior exactly
    """

    def _qualifying_checker(self, a, b):
        """Check pair: both must have qualifying=True."""
        return a.get("qualifying", False) and b.get("qualifying", False)

    def test_monotone_attr_preserved_on_downgrade(self):
        """(a) Once an entity qualifies, it stays qualifying even if later chunk
        provides only non-qualifying labels.

        Without monotone_attrs: entity downgraded → pairs permanently lost.
        With monotone_attrs={"qualifying"}: cached True preserved → pairs re-added.
        """
        state = IncrementalState()

        # Chunk 0: u1 qualifies, u2 qualifies → pair (u1, u2) added
        state.process_chunk(
            0,
            {"u1": {"qualifying": True}, "u2": {"qualifying": True}},
            pair_checker=self._qualifying_checker,
            monotone_attrs={"qualifying"},
        )
        assert len(state.pair_tracker) == 1
        assert ("u1", "u2") in state.pair_tracker.get_pairs()

        # Chunk 1: u1 reappears with qualifying=False (non-qualifying labels in this chunk).
        # With monotone merge: u1's qualifying should be preserved as True.
        stats = state.process_chunk(
            1,
            {"u1": {"qualifying": False}},
            pair_checker=self._qualifying_checker,
            monotone_attrs={"qualifying"},
        )

        # u1's effective state is unchanged (qualifying still True after merge)
        # → no retraction should occur, pair (u1, u2) should persist
        assert stats["updated_entities"] == 0, (
            "u1 should be a no-op update (effective state unchanged via monotone merge)"
        )
        assert stats["retracted_pairs"] == 0, "No retraction: u1's qualifying preserved as True"
        assert len(state.pair_tracker) == 1, "Pair (u1, u2) must persist"
        # Verify the cached value is True (monotone merge worked)
        assert state.entity_cache.get("u1")["qualifying"] is True

    def test_retraction_skipped_for_monotone_only_noop_change(self):
        """(b) When ONLY monotone attrs change but they stay truthy, retraction is skipped.

        This is the key optimization: eliminates O(degree × n) retraction cost for
        entities that reappear with only monotone attribute updates that preserve
        their effective classification.
        """
        state = IncrementalState()

        # Chunk 0: 3 qualifying entities → 3 pairs
        state.process_chunk(
            0,
            {"u1": {"qualifying": True}, "u2": {"qualifying": True}, "u3": {"qualifying": True}},
            pair_checker=self._qualifying_checker,
            monotone_attrs={"qualifying"},
        )
        assert len(state.pair_tracker) == 3  # C(3,2) = 3

        # Chunk 1: u1 reappears with qualifying=False → monotone merge preserves True
        # Since qualifying is the only (monotone) attr and it's unchanged → no-op update
        # → u1 NOT in updated_ids → retract_entity(u1) NOT called → 0 retractions
        stats = state.process_chunk(
            1,
            {"u1": {"qualifying": False}},
            pair_checker=self._qualifying_checker,
            monotone_attrs={"qualifying"},
        )

        assert stats["updated_entities"] == 0, "No effective update (monotone merge → no-op)"
        assert stats["retracted_pairs"] == 0, "No retraction for no-op update"
        assert len(state.pair_tracker) == 3, "All 3 pairs intact (no retraction)"
        # All retractions are 0 in cumulative stats too
        cumulative = state.get_stats()
        assert cumulative["total_retractions"] == 0
        assert cumulative["noop_retractions"] == 0
        assert cumulative["permanent_retractions"] == 0

    def test_retraction_fires_when_monotone_attr_genuinely_improves(self):
        """(c) Retraction fires when a monotone attr genuinely IMPROVES (False→True).

        The no-op suppression only applies when cached=truthy AND new=falsy (merge preserves).
        If the entity was NOT qualifying before and NOW qualifies (False→True), this is a
        real state change — the entity may now form new pairs. So u1 IS added to updated_ids
        → retract_entity(u1) fires → new pairs found via updated×all sweep.

        API contract: monotone_attrs controls BOTH (1) the merge logic (preserve truthy old
        values) AND (2) the no-op skip decision (unchanged after merge → skip). Non-monotone
        attrs (e.g. raw labels list) are allowed to change without affecting the no-op test,
        because the pair_checker's correctness only depends on the declared monotone attrs.
        Callers must only declare attrs that the pair_checker uses as monotone_attrs.
        """
        state = IncrementalState()

        # Chunk 0: u1 not qualifying, u2 qualifying → no pair (u1 must also qualify)
        state.process_chunk(
            0,
            {"u1": {"qualifying": False}, "u2": {"qualifying": True}},
            pair_checker=self._qualifying_checker,
            monotone_attrs={"qualifying"},
        )
        assert len(state.pair_tracker) == 0  # u1 doesn't qualify → no pair

        # Chunk 1: u1 reappears with qualifying=True (genuine improvement: False→True)
        # old_val=False, new_val=True → NOT the merge case (merge only fires for True→False)
        # → monotone_all_same=False (bool(True) != bool(False)) → u1 in updated_ids
        # → retract_entity(u1) fires (0 pairs to retract) → updated×all sweep
        # → check (u1, u2): both qualifying=True → pair added
        stats = state.process_chunk(
            1,
            {"u1": {"qualifying": True}},
            pair_checker=self._qualifying_checker,
            monotone_attrs={"qualifying"},
        )

        assert stats["updated_entities"] == 1, (
            "u1 genuinely improved (False→True) → real update, retraction fires"
        )
        assert len(state.pair_tracker) == 1, "Pair (u1, u2) found via updated×all sweep"
        pairs = state.pair_tracker.get_pairs()
        assert ("u1", "u2") in pairs

    def test_none_monotone_attrs_preserves_existing_behavior(self):
        """(d) monotone_attrs=None must produce identical behavior to the original
        process_chunk() signature (backward compatibility).

        Without monotone_attrs: a qualifying entity that reappears with qualifying=False
        is downgraded → pairs permanently retracted (old V2 bug behavior).
        """
        state = IncrementalState()

        # Chunk 0: u1 qualifies, u2 qualifies → pair (u1, u2)
        state.process_chunk(
            0,
            {"u1": {"qualifying": True}, "u2": {"qualifying": True}},
            pair_checker=self._qualifying_checker,
            monotone_attrs=None,  # explicit None — should behave like pre-Iter13 code
        )
        assert len(state.pair_tracker) == 1

        # Chunk 1: u1 reappears with qualifying=False (no monotone merge without monotone_attrs)
        # → u1 downgraded in EntityCache → retract_entity(u1) fires
        # → re-evaluate: u1.qualifying=False → pair NOT re-added (permanent retraction)
        stats = state.process_chunk(
            1,
            {"u1": {"qualifying": False}},
            pair_checker=self._qualifying_checker,
            monotone_attrs=None,
        )

        assert stats["updated_entities"] == 1, "u1 is a real update (no monotone merge)"
        assert stats["retracted_pairs"] == 1, "Pair (u1, u2) retracted"
        assert len(state.pair_tracker) == 0, "Pair not re-added (u1 downgraded)"
        cumulative = state.get_stats()
        assert cumulative["permanent_retractions"] == 1
        assert cumulative["noop_retractions"] == 0


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

        # Check that summary message exists.
        # Role-ordering fix (Iteration 6): summary = assistant (prior computation),
        # ack = user ("Continue..."), producing correct alternating discourse.
        summary_msg = pruned[3]
        assert summary_msg["role"] == "assistant", (
            "Summary message should be role='assistant' (model's prior computation). "
            "A role='user' summary incorrectly frames the model's own computation as user input."
        )
        assert "PRIOR COMPUTATION SUMMARY" in summary_msg["content"]

        # Verify the ack message follows with role='user'
        ack_msg = pruned[4]
        assert ack_msg["role"] == "user"
        assert "Continue" in ack_msg["content"]

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


# =============================================================================
# apply_edits() Tests
# =============================================================================


class TestApplyEdits:
    """Tests for IncrementalState.apply_edits() — the dynamic context API."""

    def _make_qualifying_checker(self):
        """Pair checker: both entities must be qualifying."""
        def checker(a, b):
            return a.get("qualifying", False) and b.get("qualifying", False)
        return checker

    def test_downgrade_removes_pairs(self):
        """Downgrading an entity from qualifying to non-qualifying removes its pairs."""
        state = IncrementalState()
        checker = self._make_qualifying_checker()

        # Add 3 qualifying entities → C(3,2) = 3 pairs
        state.process_chunk(0, {
            "u1": {"qualifying": True},
            "u2": {"qualifying": True},
            "u3": {"qualifying": True},
        }, pair_checker=checker)
        assert len(state.pair_tracker) == 3

        # Downgrade u1 → should lose 2 pairs (u1-u2, u1-u3)
        stats = state.apply_edits(
            {"u1": {"qualifying": False}},
            pair_checker=checker,
        )
        assert stats["entities_edited"] == 1
        assert stats["total_retracted"] == 2  # u1-u2, u1-u3
        assert stats["pairs_readded"] == 0  # u1 no longer qualifying
        assert stats["pairs_after"] == 1  # only u2-u3 remains
        assert len(state.pair_tracker) == 1

    def test_upgrade_adds_pairs(self):
        """Upgrading an entity to qualifying creates new pairs."""
        state = IncrementalState()
        checker = self._make_qualifying_checker()

        # u1, u2 qualifying; u3 not
        state.process_chunk(0, {
            "u1": {"qualifying": True},
            "u2": {"qualifying": True},
            "u3": {"qualifying": False},
        }, pair_checker=checker)
        assert len(state.pair_tracker) == 1  # only u1-u2

        # Upgrade u3 → should gain pairs with u1 and u2
        stats = state.apply_edits(
            {"u3": {"qualifying": True}},
            pair_checker=checker,
        )
        assert stats["entities_edited"] == 1
        assert stats["total_retracted"] == 0  # u3 had no pairs
        assert stats["new_pairs_from_edits"] == 2  # u3-u1, u3-u2
        assert stats["pairs_after"] == 3  # u1-u2, u1-u3, u2-u3

    def test_precision_maintained(self):
        """After edits, all remaining pairs should be valid."""
        state = IncrementalState()
        checker = self._make_qualifying_checker()

        state.process_chunk(0, {
            "u1": {"qualifying": True},
            "u2": {"qualifying": True},
            "u3": {"qualifying": True},
            "u4": {"qualifying": False},
        }, pair_checker=checker)
        assert len(state.pair_tracker) == 3

        # Mixed edit: downgrade u1, upgrade u4
        state.apply_edits({
            "u1": {"qualifying": False},
            "u4": {"qualifying": True},
        }, pair_checker=checker)

        # Verify all remaining pairs are valid
        for p in state.pair_tracker.get_pairs():
            a1 = state.entity_cache.get(p[0])
            a2 = state.entity_cache.get(p[1])
            assert checker(a1, a2), f"Invalid pair {p}: {a1}, {a2}"

        # u2-u3, u2-u4, u3-u4 = 3 pairs (u1 lost, u4 gained)
        assert len(state.pair_tracker) == 3

    def test_telemetry_tracks_edit_retractions(self):
        """apply_edits() updates _total_retractions counter."""
        state = IncrementalState()
        checker = self._make_qualifying_checker()

        state.process_chunk(0, {
            "u1": {"qualifying": True},
            "u2": {"qualifying": True},
        }, pair_checker=checker)
        assert state.get_stats()["total_retractions"] == 0

        state.apply_edits(
            {"u1": {"qualifying": False}},
            pair_checker=checker,
        )
        stats = state.get_stats()
        assert stats["total_retractions"] == 1  # retracted u1-u2
        assert stats["permanent_retractions"] == 1  # not re-added

    def test_noop_edit_preserves_pairs(self):
        """Editing an entity without changing qualifying status keeps pairs intact."""
        state = IncrementalState()
        checker = self._make_qualifying_checker()

        state.process_chunk(0, {
            "u1": {"qualifying": True, "name": "Alice"},
            "u2": {"qualifying": True, "name": "Bob"},
        }, pair_checker=checker)
        assert len(state.pair_tracker) == 1

        # Edit name but keep qualifying=True
        stats = state.apply_edits(
            {"u1": {"qualifying": True, "name": "Alicia"}},
            pair_checker=checker,
        )
        assert stats["total_retracted"] == 1  # still retracted for safety
        assert stats["pairs_readded"] == 1  # re-added after re-check
        assert stats["pairs_after"] == 1  # pair preserved

    def test_multi_entity_edit_shared_pair_deduplicated(self):
        """Two edited entities sharing a pair: retraction count is deduplicated."""
        state = IncrementalState()
        checker = self._make_qualifying_checker()

        # Setup: 3 qualifying entities -> 3 pairs: (u1,u2), (u1,u3), (u2,u3)
        state.process_chunk(0, {
            "u1": {"qualifying": True},
            "u2": {"qualifying": True},
            "u3": {"qualifying": True},
        }, pair_checker=checker)
        assert len(state.pair_tracker) == 3

        # Edit both u1 and u2 (downgrade both).
        # Pair (u1,u2) is shared. Two-phase ensures it's counted once.
        stats = state.apply_edits(
            {"u1": {"qualifying": False}, "u2": {"qualifying": False}},
            pair_checker=checker,
        )
        # u1 has pairs (u1,u2) and (u1,u3) -> 2 retracted
        # u2 has pair (u2,u3) -> 1 retracted (u1,u2 already retracted by u1's pass)
        # Total deduplicated = 3 (all 3 pairs)
        assert stats["total_retracted"] == 3
        # None re-added (both u1 and u2 are now non-qualifying)
        assert stats["pairs_readded"] == 0
        # Only u3 is qualifying, no pairs remain
        assert stats["pairs_after"] == 0

    def test_multi_entity_upgrade_upgrade_discovers_pair(self):
        """Two non-qualifying entities upgraded: their mutual pair is discovered."""
        state = IncrementalState()
        checker = self._make_qualifying_checker()

        # Setup: u1 qualifying, u2 and u3 non-qualifying
        state.process_chunk(0, {
            "u1": {"qualifying": True},
            "u2": {"qualifying": False},
            "u3": {"qualifying": False},
        }, pair_checker=checker)
        assert len(state.pair_tracker) == 0  # no qualifying pairs

        # Upgrade both u2 and u3
        stats = state.apply_edits(
            {"u2": {"qualifying": True}, "u3": {"qualifying": True}},
            pair_checker=checker,
        )
        # Should discover (u1,u2), (u1,u3), and (u2,u3)
        assert stats["pairs_after"] == 3
        assert stats["new_pairs_from_edits"] == 3

    def test_has_pair_method(self):
        """PairTracker.has_pair() works correctly."""
        from rlm.core.incremental import PairTracker
        pt = PairTracker()
        pt.add_pair("a", "b")
        assert pt.has_pair("a", "b") is True
        assert pt.has_pair("b", "a") is True  # order-independent
        assert pt.has_pair("a", "c") is False


class TestProcessChunkDeduplication:
    """Test the idempotency guard added in Iteration 8 (Failure Mode C fix)."""

    def test_double_call_returns_cached_stats(self):
        """Calling process_chunk with the same index twice returns cached stats
        without re-executing the O(u·n) sweep."""
        import warnings

        state = IncrementalState()

        def checker(a, b):
            return a.get("type") == b.get("type")

        # First call: processes normally
        stats_first = state.process_chunk(0, {"u1": {"type": "A"}, "u2": {"type": "A"}}, checker)
        assert stats_first["chunk_index"] == 0

        # Second call with same index: should warn and return cached stats
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            stats_second = state.process_chunk(0, {"u1": {"type": "A"}, "u2": {"type": "A"}}, checker)
            assert len(w) == 1
            assert "called more than once" in str(w[0].message)

        # Cached stats returned unchanged
        assert stats_second == stats_first

    def test_double_call_does_not_double_count_entities(self):
        """Redundant process_chunk calls must not inflate entity or pair counts."""
        state = IncrementalState()

        def checker(a, b):
            return True  # always match

        import warnings
        state.process_chunk(0, {"u1": {}, "u2": {}}, checker)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            state.process_chunk(0, {"u1": {}, "u2": {}}, checker)
            state.process_chunk(0, {"u1": {}, "u2": {}}, checker)

        stats = state.get_stats()
        # Despite 3 calls to process_chunk(0,...), only 1 chunk should be counted
        assert stats["chunks_processed"] == 1
        assert stats["total_entities"] == 2
        assert stats["total_pairs"] == 1  # C(2,2) = 1

    def test_different_chunk_indices_are_independent(self):
        """Each unique chunk_index is processed exactly once."""
        state = IncrementalState()

        def checker(a, b):
            return a.get("type") == b.get("type")

        import warnings
        state.process_chunk(0, {"u1": {"type": "A"}}, checker)
        state.process_chunk(1, {"u2": {"type": "A"}}, checker)

        # Redundant call on chunk 0 — should not re-process
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state.process_chunk(0, {"u3": {"type": "A"}}, checker)
            assert len(w) == 1

        stats = state.get_stats()
        # u3 should NOT appear — chunk 0 was cached
        assert stats["total_entities"] == 2  # only u1, u2
        assert stats["chunks_processed"] == 2


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


class TestConditionDCodeGeneration:
    """Unit tests for the Condition D unrolled code generation.

    The `generate_unrolled_chunk_code()` function dynamically generates Python
    code for the full-recompute RLM (Condition D). It previously had 2 regex bugs
    (fixed on 3rd attempt in Iteration 15). These tests prevent regressions.
    """

    def test_k1_contains_reset_and_one_process_chunk(self):
        """At k=1 (Turn 1), generated code should process exactly chunk 0."""
        from eval.label_aware_v4_experiment import generate_unrolled_chunk_code

        code = generate_unrolled_chunk_code(1)
        assert "_incremental.process_chunk(0," in code
        assert "_incremental.process_chunk(1," not in code
        assert "entities_0" in code

    def test_k3_contains_three_process_chunks(self):
        """At k=3 (Turn 3), generated code should process chunks 0, 1, 2."""
        from eval.label_aware_v4_experiment import generate_unrolled_chunk_code

        code = generate_unrolled_chunk_code(3)
        assert "_incremental.process_chunk(0," in code
        assert "_incremental.process_chunk(1," in code
        assert "_incremental.process_chunk(2," in code
        assert "_incremental.process_chunk(3," not in code
        # Verify independent entity dicts per chunk
        assert "entities_0" in code
        assert "entities_1" in code
        assert "entities_2" in code

    def test_k5_contains_five_process_chunks(self):
        """At k=5, generated code should process chunks 0-4."""
        from eval.label_aware_v4_experiment import generate_unrolled_chunk_code

        code = generate_unrolled_chunk_code(5)
        for ci in range(5):
            assert f"_incremental.process_chunk({ci}," in code
            assert f"entities_{ci}" in code
        assert "_incremental.process_chunk(5," not in code

    def test_regex_matches_pipe_separator(self):
        """Verify the regex correctly matches '||' (pipe separator).

        This is the exact bug that caused 2 failed Condition D runs in Iteration 15:
        over-escaped '\\\\|\\\\|' produced '\\|\\|' in the raw string, matching
        literal backslash-pipe instead of double-pipe.
        """
        import re

        from eval.label_aware_v4_experiment import generate_unrolled_chunk_code

        code = generate_unrolled_chunk_code(1)
        # Extract the regex pattern from the generated code
        regex_match = re.search(r"re\.search\(r'(.+?)',", code)
        assert regex_match, "No re.search pattern found in generated code"
        pattern = regex_match.group(1)

        # The pattern must match actual OOLONG-Pairs format lines
        test_line = "Date: Jan 01, 2025 || User: 12345 || Instance: test text || Label: location"
        m = re.search(pattern, test_line)
        assert m is not None, f"Pattern {pattern!r} failed to match OOLONG line"
        assert m.group(1) == "12345", f"Expected uid=12345, got {m.group(1)}"
        assert m.group(2).strip().lower() == "location"

    def test_monotone_attrs_in_process_chunk_call(self):
        """Verify monotone_attrs={'qualifying'} is in each process_chunk call."""
        from eval.label_aware_v4_experiment import generate_unrolled_chunk_code

        code = generate_unrolled_chunk_code(3)
        for ci in range(3):
            # Check each process_chunk call includes monotone_attrs
            chunk_section = code.split(f"# Process chunk {ci}")[1] if ci < 2 else code.split(f"# Process chunk {ci}")[1]
            assert 'monotone_attrs={"qualifying"}' in chunk_section


# =============================================================================
# verify_lossless() Tests (Iteration 15 — External Reviewer Concern #1)
# =============================================================================


class TestVerifyLossless:
    """Tests for IncrementalState.verify_lossless() — proves cache is lossless."""

    def test_lossless_after_one_chunk(self):
        """After processing 1 chunk, cache contains exactly those entities."""
        state = IncrementalState()
        state.process_chunk(0, {"u1": {"x": 1}, "u2": {"x": 2}}, pair_checker=None)
        result = state.verify_lossless({"u1", "u2"})
        assert result["is_lossless"] is True
        assert result["expected_count"] == 2
        assert result["cached_count"] == 2
        assert result["missing_ids"] == []
        assert result["extra_ids"] == []

    def test_lossless_after_three_chunks(self):
        """After 3 chunks, cache contains union of all entities from chunks 0-2."""
        state = IncrementalState()
        state.process_chunk(0, {"u1": {"x": 1}, "u2": {"x": 2}}, pair_checker=None)
        state.process_chunk(1, {"u3": {"x": 3}}, pair_checker=None)
        state.process_chunk(2, {"u4": {"x": 4}, "u5": {"x": 5}}, pair_checker=None)
        result = state.verify_lossless({"u1", "u2", "u3", "u4", "u5"})
        assert result["is_lossless"] is True
        assert result["expected_count"] == 5
        assert result["cached_count"] == 5

    def test_detects_missing_entity(self):
        """verify_lossless correctly reports missing entities."""
        state = IncrementalState()
        state.process_chunk(0, {"u1": {"x": 1}}, pair_checker=None)
        # Expect u1 and u2, but u2 was never added
        result = state.verify_lossless({"u1", "u2"})
        assert result["is_lossless"] is False
        assert "u2" in result["missing_ids"]
        assert result["extra_ids"] == []

    def test_detects_extra_entity(self):
        """verify_lossless correctly reports extra entities."""
        state = IncrementalState()
        state.process_chunk(0, {"u1": {"x": 1}, "u2": {"x": 2}}, pair_checker=None)
        # Expect only u1, but cache has u2 too
        result = state.verify_lossless({"u1"})
        assert result["is_lossless"] is False
        assert result["missing_ids"] == []
        assert "u2" in result["extra_ids"]

    def test_lossless_after_updates(self):
        """Updates don't drop entities — cache is still lossless."""
        state = IncrementalState()
        state.process_chunk(0, {"u1": {"x": 1}, "u2": {"x": 2}}, pair_checker=None)
        # u1 updated in chunk 1 — should NOT be dropped
        state.process_chunk(1, {"u1": {"x": 99}, "u3": {"x": 3}}, pair_checker=None)
        result = state.verify_lossless({"u1", "u2", "u3"})
        assert result["is_lossless"] is True
        assert result["cached_count"] == 3

    def test_lossless_with_integer_ids(self):
        """Works with integer entity IDs (converted to str internally)."""
        state = IncrementalState()
        state.process_chunk(0, {123: {"x": 1}, 456: {"x": 2}}, pair_checker=None)
        result = state.verify_lossless({123, 456})
        assert result["is_lossless"] is True


# =============================================================================
# memory_usage() Tests (Iteration 15 — External Reviewer Concern #2)
# =============================================================================


class TestMemoryUsage:
    """Tests for IncrementalState.memory_usage() — memory profiling."""

    def test_empty_state_has_minimal_memory(self):
        """Empty state should have very low memory."""
        state = IncrementalState()
        mem = state.memory_usage()
        assert mem["total_bytes"] > 0  # not zero (overhead of empty dicts)
        assert mem["total_kb"] < 1.0  # less than 1 KB for empty state
        assert mem["counts"]["entities"] == 0
        assert mem["counts"]["pairs"] == 0

    def test_memory_increases_with_entities(self):
        """Memory should increase when entities are added."""
        state = IncrementalState()
        mem_before = state.memory_usage()["total_bytes"]

        state.process_chunk(0, {
            f"u{i}": {"name": f"user_{i}", "type": "person", "score": i}
            for i in range(100)
        }, pair_checker=None)

        mem_after = state.memory_usage()
        assert mem_after["total_bytes"] > mem_before
        assert mem_after["counts"]["entities"] == 100

    def test_memory_increases_with_pairs(self):
        """Memory should increase substantially when pairs are added."""
        state = IncrementalState()

        def always_match(a, b):
            return True

        # 50 entities → C(50,2) = 1225 pairs
        state.process_chunk(0, {
            f"u{i}": {"x": 1} for i in range(50)
        }, pair_checker=always_match)

        mem = state.memory_usage()
        assert mem["counts"]["entities"] == 50
        assert mem["counts"]["pairs"] == 1225
        assert mem["pair_tracker_bytes"] > 0
        assert mem["inverted_index_bytes"] > 0

    def test_component_breakdown_sums_to_total(self):
        """Component breakdown should roughly sum to total."""
        state = IncrementalState()
        state.process_chunk(0, {
            f"u{i}": {"qualifying": True} for i in range(20)
        }, pair_checker=lambda a, b: True)

        mem = state.memory_usage()
        component_sum = sum(mem["component_breakdown"].values())
        # Allow small rounding error (1 KB)
        assert abs(component_sum - mem["total_kb"]) < 1.0

    def test_memory_report_has_all_fields(self):
        """Verify all expected fields are present."""
        state = IncrementalState()
        mem = state.memory_usage()
        assert "entity_cache_bytes" in mem
        assert "pair_tracker_bytes" in mem
        assert "inverted_index_bytes" in mem
        assert "total_bytes" in mem
        assert "total_kb" in mem
        assert "total_mb" in mem
        assert "component_breakdown" in mem
        assert "counts" in mem


class TestRebuildPairs:
    """Tests for rebuild_pairs() — two-tier architecture proof."""

    def test_rebuild_matches_original(self):
        """Rebuilt pairs from entity cache should match incremental pairs."""
        state = IncrementalState()

        def score_match(a, b):
            return a.get("score", 0) > 5 and b.get("score", 0) > 5

        state.process_chunk(0, {
            "u1": {"score": 10},
            "u2": {"score": 8},
            "u3": {"score": 3},
        }, pair_checker=score_match)

        # u1-u2 should match (both > 5), u3 doesn't qualify
        assert len(state.pair_tracker) == 1

        result = state.rebuild_pairs(pair_checker=score_match)
        assert result["match"] is True
        assert result["original_count"] == 1
        assert result["rebuilt_count"] == 1
        assert result["missing_pairs"] == 0
        assert result["extra_pairs"] == 0

    def test_rebuild_after_multiple_chunks(self):
        """Rebuild should work correctly after multi-chunk processing."""
        state = IncrementalState()

        def always_match(a, b):
            return True

        state.process_chunk(0, {"u1": {"x": 1}, "u2": {"x": 2}}, pair_checker=always_match)
        state.process_chunk(1, {"u3": {"x": 3}}, pair_checker=always_match)

        # 3 entities → C(3,2) = 3 pairs
        assert len(state.pair_tracker) == 3

        result = state.rebuild_pairs(pair_checker=always_match)
        assert result["match"] is True
        assert result["rebuilt_count"] == 3
        assert result["rebuild_checks"] == 3  # C(3,2) = 3

    def test_rebuild_after_retraction(self):
        """Rebuild should produce correct pairs even after retractions."""
        state = IncrementalState()

        call_count = [0]

        def match_if_both_qualifying(a, b):
            call_count[0] += 1
            return a.get("qualifying", False) and b.get("qualifying", False)

        # Chunk 0: both qualifying → 1 pair
        state.process_chunk(0, {
            "u1": {"qualifying": True},
            "u2": {"qualifying": True},
        }, pair_checker=match_if_both_qualifying)
        assert len(state.pair_tracker) == 1

        # Chunk 1: u1 now non-qualifying → retraction
        state.process_chunk(1, {
            "u1": {"qualifying": False},
        }, pair_checker=match_if_both_qualifying)
        assert len(state.pair_tracker) == 0

        result = state.rebuild_pairs(pair_checker=match_if_both_qualifying)
        assert result["match"] is True
        assert result["rebuilt_count"] == 0

    def test_rebuild_empty_state(self):
        """Rebuild on empty state should return zero pairs."""
        state = IncrementalState()
        result = state.rebuild_pairs(pair_checker=lambda a, b: True)
        assert result["match"] is True
        assert result["original_count"] == 0
        assert result["rebuilt_count"] == 0
        assert result["rebuild_checks"] == 0
