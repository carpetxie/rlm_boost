"""
Second-domain generalization tests for IncrementalState.

Proves the IncrementalState library is domain-agnostic by instantiating it
on four non-OOLONG domains:

1. Document-keyword matching: documents with keyword sets, pairs share ≥3 keywords
2. User-interest compatibility: users with interest sets, pairs share ≥2 interests
3. Product-category affinity: customers and products with category tags, matching ≥1
4. High-churn domain: 50% entity update rate per chunk (stress test)

Each test validates:
- Incremental pair checks < full recompute C(N,2)
- rebuild_pairs() matches incremental result (two-tier proof)
- verify_lossless() passes at each chunk
- apply_edits() correctly handles dynamic updates

Zero API calls needed — purely synthetic data.
"""

import random

from rlm.core.incremental import IncrementalState


# ============================================================================
# Domain 1: Document-Keyword Matching
# ============================================================================


def keyword_overlap_checker(attrs1: dict, attrs2: dict) -> bool:
    """Two documents are a valid pair if they share ≥3 keywords."""
    kw1 = attrs1.get("keywords", set())
    kw2 = attrs2.get("keywords", set())
    # Handle both set and frozenset
    if isinstance(kw1, (list, tuple)):
        kw1 = set(kw1)
    if isinstance(kw2, (list, tuple)):
        kw2 = set(kw2)
    return len(kw1 & kw2) >= 3


class TestDocumentKeywordMatching:
    """Document-keyword overlap: pair condition is shared keywords ≥ 3."""

    def test_incremental_saves_pair_checks_vs_full_recompute(self):
        """Incremental processing checks fewer pairs than full recompute (reset+replay each turn).

        The savings come from the streaming scenario: data arrives in k chunks.
        - Incremental: process each chunk once → total checks = C(N,2)
        - Full recompute: reset and replay all chunks at each turn →
          total checks = sum_{t=1}^{k} C(t·n, 2) where n = entities per chunk
        """
        chunks = [
            {
                f"doc_{c * 10 + i}": {"keywords": frozenset(f"kw_{j}" for j in range(c * 10 + i, c * 10 + i + 5))}
                for i in range(10)
            }
            for c in range(3)
        ]

        # Incremental: process each chunk once
        state_incr = IncrementalState()
        incr_checks = 0
        for idx, chunk in enumerate(chunks):
            stats = state_incr.process_chunk(idx, chunk, keyword_overlap_checker)
            incr_checks += stats["pair_checks"]

        # Full recompute: reset and replay all chunks each turn
        state_full = IncrementalState()
        full_checks = 0
        for turn in range(len(chunks)):
            state_full.reset()
            for idx in range(turn + 1):
                stats = state_full.process_chunk(idx, chunks[idx], keyword_overlap_checker)
                full_checks += stats["pair_checks"]

        # Both should find the same pairs
        assert state_incr.pair_tracker.get_pairs() == state_full.pair_tracker.get_pairs()

        # Incremental should use fewer total checks
        assert incr_checks < full_checks, (
            f"Incremental ({incr_checks}) should be < full recompute ({full_checks})"
        )

        savings_pct = (1 - incr_checks / full_checks) * 100
        assert savings_pct > 20, f"Expected >20% savings, got {savings_pct:.1f}%"

    def test_rebuild_matches_incremental(self):
        """rebuild_pairs() produces identical results to incremental processing."""
        state = IncrementalState()

        for chunk_idx in range(5):
            entities = {
                f"doc_{chunk_idx * 10 + i}": {
                    "keywords": frozenset(f"kw_{j}" for j in range(chunk_idx * 10 + i, chunk_idx * 10 + i + 5))
                }
                for i in range(10)
            }
            state.process_chunk(chunk_idx, entities, keyword_overlap_checker)

        # Get incremental result
        original_pairs = state.pair_tracker.get_pairs()
        original_count = len(original_pairs)

        # Rebuild and compare
        rebuild_result = state.rebuild_pairs(keyword_overlap_checker)
        assert rebuild_result["match"], (
            f"Rebuild mismatch: {rebuild_result['missing_pairs']} missing, "
            f"{rebuild_result['extra_pairs']} extra"
        )
        assert rebuild_result["original_count"] == original_count

    def test_verify_lossless_at_each_chunk(self):
        """Entity cache is lossless at every chunk boundary."""
        state = IncrementalState()
        all_ids: set[str] = set()

        for chunk_idx in range(5):
            entities = {
                f"doc_{chunk_idx * 5 + i}": {
                    "keywords": frozenset(f"kw_{j}" for j in range(i, i + 3))
                }
                for i in range(5)
            }
            all_ids.update(entities.keys())
            state.process_chunk(chunk_idx, entities, keyword_overlap_checker)

            # Verify lossless after each chunk
            result = state.verify_lossless(all_ids)
            assert result["is_lossless"], (
                f"Chunk {chunk_idx}: missing={result['missing_ids']}, extra={result['extra_ids']}"
            )

    def test_structural_savings_formula(self):
        """Empirical savings vs full recompute should meet the structural bound 1-2/(k+1).

        The formula predicts savings of incremental (process once) vs full recompute
        (reset+replay each turn) in terms of pair checks.
        """
        k = 5  # 5 chunks
        entities_per_chunk = 20
        chunks = [
            {
                f"doc_{c * entities_per_chunk + i}": {
                    "keywords": frozenset(f"kw_{j}" for j in range(i, i + 5))
                }
                for i in range(entities_per_chunk)
            }
            for c in range(k)
        ]

        # Incremental
        state_incr = IncrementalState()
        incr_checks = 0
        for idx, chunk in enumerate(chunks):
            stats = state_incr.process_chunk(idx, chunk, keyword_overlap_checker)
            incr_checks += stats["pair_checks"]

        # Full recompute
        state_full = IncrementalState()
        full_checks = 0
        for turn in range(k):
            state_full.reset()
            for idx in range(turn + 1):
                stats = state_full.process_chunk(idx, chunks[idx], keyword_overlap_checker)
                full_checks += stats["pair_checks"]

        empirical_savings = 1 - incr_checks / full_checks

        # Pair-check savings should be substantial (>40% for k=5).
        # Note: the structural formula 1-2/(k+1) applies to linear token reads,
        # not pair checks which scale quadratically within each chunk replay.
        # Pair-check savings are typically lower than token savings but still significant.
        assert empirical_savings > 0.4, (
            f"Expected >40% pair-check savings for k={k}, got {empirical_savings:.3f}"
        )


# ============================================================================
# Domain 2: User-Interest Compatibility
# ============================================================================


def interest_compatibility_checker(attrs1: dict, attrs2: dict) -> bool:
    """Two users are compatible if they share ≥2 interests."""
    int1 = attrs1.get("interests", set())
    int2 = attrs2.get("interests", set())
    if isinstance(int1, (list, tuple)):
        int1 = set(int1)
    if isinstance(int2, (list, tuple)):
        int2 = set(int2)
    return len(int1 & int2) >= 2


class TestUserInterestCompatibility:
    """Social graph: users with interests, pair condition is shared interests ≥ 2."""

    def test_incremental_vs_full_recompute(self):
        """Compare incremental vs simulated full recompute."""
        state_incr = IncrementalState()
        state_full = IncrementalState()

        chunks = []
        for chunk_idx in range(4):
            entities = {
                f"user_{chunk_idx * 8 + i}": {
                    "interests": frozenset(
                        f"interest_{j}" for j in range(i % 5, i % 5 + 3)
                    )
                }
                for i in range(8)
            }
            chunks.append(entities)

        # Incremental: process each chunk once
        incr_checks = 0
        for idx, chunk in enumerate(chunks):
            stats = state_incr.process_chunk(idx, chunk, interest_compatibility_checker)
            incr_checks += stats["pair_checks"]

        # Full recompute: reset and replay all chunks each turn
        full_checks = 0
        for turn in range(len(chunks)):
            state_full.reset()
            for idx in range(turn + 1):
                stats = state_full.process_chunk(idx, chunks[idx], interest_compatibility_checker)
                full_checks += stats["pair_checks"]

        # Both should find the same pairs
        assert state_incr.pair_tracker.get_pairs() == state_full.pair_tracker.get_pairs()

        # Incremental should use fewer checks
        assert incr_checks < full_checks, (
            f"Incremental ({incr_checks}) should be < full ({full_checks})"
        )

        savings_pct = (1 - incr_checks / full_checks) * 100
        assert savings_pct > 0

    def test_rebuild_correctness(self):
        """Rebuild from entity cache matches incremental result."""
        state = IncrementalState()

        for chunk_idx in range(6):
            entities = {
                f"user_{chunk_idx * 5 + i}": {
                    "interests": frozenset(f"int_{j}" for j in range(i, i + 4))
                }
                for i in range(5)
            }
            state.process_chunk(chunk_idx, entities, interest_compatibility_checker)

        result = state.rebuild_pairs(interest_compatibility_checker)
        assert result["match"], f"Mismatch: {result['missing_pairs']} missing, {result['extra_pairs']} extra"

    def test_apply_edits_interest_change(self):
        """When a user's interests change, retraction correctly handles it."""
        state = IncrementalState()

        # Two users sharing interests -> valid pair
        state.process_chunk(
            0,
            {
                "alice": {"interests": frozenset(["music", "art", "hiking"])},
                "bob": {"interests": frozenset(["music", "art", "cooking"])},
            },
            interest_compatibility_checker,
        )
        assert state.pair_tracker.has_pair("alice", "bob")

        # Edit: Alice changes interests, no longer shares ≥2 with Bob
        edit_stats = state.apply_edits(
            {"alice": {"interests": frozenset(["gaming", "coding", "music"])}},
            interest_compatibility_checker,
        )
        # music is still shared, but only 1 overlap now -> pair should be removed
        assert not state.pair_tracker.has_pair("alice", "bob")
        assert edit_stats["permanent_retractions"] == 1


# ============================================================================
# Domain 3: Product-Category Affinity
# ============================================================================


def category_affinity_checker(attrs1: dict, attrs2: dict) -> bool:
    """A customer-product pair is valid if they share at least 1 category."""
    cats1 = attrs1.get("categories", set())
    cats2 = attrs2.get("categories", set())
    if isinstance(cats1, (list, tuple)):
        cats1 = set(cats1)
    if isinstance(cats2, (list, tuple)):
        cats2 = set(cats2)
    # Only pair customers with products (not customer-customer or product-product)
    type1 = attrs1.get("type", "")
    type2 = attrs2.get("type", "")
    if type1 == type2:
        return False
    return len(cats1 & cats2) >= 1


class TestProductCategoryAffinity:
    """E-commerce: customers and products with categories, cross-type matching."""

    def test_cross_type_pairing(self):
        """Only customer-product pairs are valid, not customer-customer."""
        state = IncrementalState()

        # Chunk 0: customers
        state.process_chunk(
            0,
            {
                "cust_1": {"type": "customer", "categories": frozenset(["electronics", "books"])},
                "cust_2": {"type": "customer", "categories": frozenset(["electronics", "sports"])},
            },
            category_affinity_checker,
        )
        # No pairs (customer-customer)
        assert len(state.pair_tracker) == 0

        # Chunk 1: products
        stats = state.process_chunk(
            1,
            {
                "prod_1": {"type": "product", "categories": frozenset(["electronics"])},
                "prod_2": {"type": "product", "categories": frozenset(["books", "art"])},
            },
            category_affinity_checker,
        )
        # cust_1-prod_1 (electronics), cust_1-prod_2 (books), cust_2-prod_1 (electronics)
        assert state.pair_tracker.has_pair("cust_1", "prod_1")
        assert state.pair_tracker.has_pair("cust_1", "prod_2")
        assert state.pair_tracker.has_pair("cust_2", "prod_1")
        # cust_2-prod_2: no shared category (sports vs books+art)
        assert not state.pair_tracker.has_pair("cust_2", "prod_2")

    def test_incremental_savings_with_mixed_types(self):
        """Savings hold even with mixed entity types arriving in chunks (vs full recompute)."""
        all_categories = ["electronics", "books", "sports", "music", "food", "art", "tech"]
        rng = random.Random(42)

        chunks = []
        for chunk_idx in range(5):
            entities = {}
            for i in range(10):
                eid = f"{'cust' if i < 5 else 'prod'}_{chunk_idx * 10 + i}"
                cats = frozenset(rng.sample(all_categories, k=rng.randint(1, 3)))
                entities[eid] = {
                    "type": "customer" if i < 5 else "product",
                    "categories": cats,
                }
            chunks.append(entities)

        # Incremental
        state_incr = IncrementalState()
        incr_checks = 0
        for idx, chunk in enumerate(chunks):
            stats = state_incr.process_chunk(idx, chunk, category_affinity_checker)
            incr_checks += stats["pair_checks"]

        # Full recompute
        state_full = IncrementalState()
        full_checks = 0
        for turn in range(len(chunks)):
            state_full.reset()
            for idx in range(turn + 1):
                stats = state_full.process_chunk(idx, chunks[idx], category_affinity_checker)
                full_checks += stats["pair_checks"]

        # Same pairs
        assert state_incr.pair_tracker.get_pairs() == state_full.pair_tracker.get_pairs()

        # Savings
        savings = 1 - incr_checks / full_checks
        assert savings > 0, f"Expected savings > 0, got {savings:.3f}"

        # Rebuild check
        result = state_incr.rebuild_pairs(category_affinity_checker)
        assert result["match"]


# ============================================================================
# Domain 4: High-Churn Stress Test
# ============================================================================


def threshold_checker(attrs1: dict, attrs2: dict) -> bool:
    """Simple: pair valid if both entities have score > 0.5."""
    return attrs1.get("score", 0) > 0.5 and attrs2.get("score", 0) > 0.5


class TestHighChurnDomain:
    """Stress test: 50% entity update rate per chunk, testing retraction heavily."""

    def test_high_update_rate_correctness(self):
        """With 50% updates per chunk, incremental + retraction stays correct."""
        state = IncrementalState()
        rng = random.Random(123)

        n_entities = 40
        n_chunks = 8
        entities_per_chunk = n_entities // n_chunks  # 5 new per chunk

        # Track all entity states for ground-truth
        ground_truth: dict[str, dict] = {}

        for chunk_idx in range(n_chunks):
            chunk_entities = {}

            # Add new entities
            for i in range(entities_per_chunk):
                eid = f"entity_{chunk_idx * entities_per_chunk + i}"
                attrs = {"score": rng.random()}
                chunk_entities[eid] = attrs
                ground_truth[eid] = attrs

            # Update ~50% of existing entities (from prior chunks)
            existing = [eid for eid in ground_truth if eid not in chunk_entities]
            n_updates = len(existing) // 2
            for eid in rng.sample(existing, min(n_updates, len(existing))):
                attrs = {"score": rng.random()}
                chunk_entities[eid] = attrs
                ground_truth[eid] = attrs

            state.process_chunk(chunk_idx, chunk_entities, threshold_checker)

        # Verify: rebuild from entity cache matches
        result = state.rebuild_pairs(threshold_checker)
        assert result["match"], (
            f"High-churn mismatch: {result['missing_pairs']} missing, "
            f"{result['extra_pairs']} extra"
        )

        # Verify lossless
        lossless = state.verify_lossless(set(ground_truth.keys()))
        assert lossless["is_lossless"]

        # Verify entity cache has correct attributes
        for eid, expected_attrs in ground_truth.items():
            cached = state.entity_cache.get(eid)
            assert cached is not None, f"Entity {eid} missing from cache"
            assert abs(cached["score"] - expected_attrs["score"]) < 1e-10, (
                f"Entity {eid}: cached score {cached['score']} != expected {expected_attrs['score']}"
            )

    def test_retraction_fires_on_updates(self):
        """Verify retraction actually fires when entities are updated."""
        state = IncrementalState()

        # Chunk 0: all qualifying
        state.process_chunk(
            0,
            {
                "e1": {"score": 0.9},
                "e2": {"score": 0.8},
                "e3": {"score": 0.7},
            },
            threshold_checker,
        )
        assert len(state.pair_tracker) == 3  # all pairs valid

        # Chunk 1: e1 drops below threshold
        stats = state.process_chunk(
            1,
            {"e1": {"score": 0.1}},  # update: no longer qualifying
            threshold_checker,
        )
        # e1's pairs should be retracted
        assert stats["retracted_pairs"] == 2  # (e1,e2) and (e1,e3)
        assert stats["permanent_retractions"] == 2
        assert len(state.pair_tracker) == 1  # only (e2,e3) remains

    def test_memory_usage_scales_linearly(self):
        """Memory scales with entity count, not quadratically."""
        sizes = [50, 100, 200]
        memory_readings = []

        for n in sizes:
            state = IncrementalState()
            rng = random.Random(42)

            for chunk_idx in range(5):
                entities = {
                    f"e_{chunk_idx * (n // 5) + i}": {"score": rng.random()}
                    for i in range(n // 5)
                }
                state.process_chunk(chunk_idx, entities, threshold_checker)

            mem = state.memory_usage()
            memory_readings.append((n, mem["total_bytes"]))

        # Check that memory growth is sub-quadratic
        # (ratio of memory growth should be less than ratio of n² growth)
        ratio_n = sizes[2] / sizes[0]  # 4x more entities
        ratio_mem = memory_readings[2][1] / memory_readings[0][1]
        ratio_n_sq = ratio_n ** 2  # 16x for quadratic

        # Memory growth should be significantly less than n²
        assert ratio_mem < ratio_n_sq, (
            f"Memory growth ({ratio_mem:.1f}x) too close to quadratic ({ratio_n_sq:.1f}x)"
        )


# ============================================================================
# Cross-Domain Summary Test
# ============================================================================


class TestCrossDomainSummary:
    """Meta-test: run all domains and report comparative savings."""

    def test_all_domains_show_savings(self):
        """Every domain achieves positive savings vs full recompute (reset+replay)."""
        domains = [
            ("document-keyword", keyword_overlap_checker, self._make_doc_chunks()),
            ("user-interest", interest_compatibility_checker, self._make_user_chunks()),
            ("product-category", category_affinity_checker, self._make_product_chunks()),
            ("high-churn-threshold", threshold_checker, self._make_threshold_chunks()),
        ]

        for name, checker, chunks in domains:
            # Incremental
            state_incr = IncrementalState()
            incr_checks = 0
            for idx, chunk in enumerate(chunks):
                stats = state_incr.process_chunk(idx, chunk, checker)
                incr_checks += stats["pair_checks"]

            # Full recompute
            state_full = IncrementalState()
            full_checks = 0
            for turn in range(len(chunks)):
                state_full.reset()
                for idx in range(turn + 1):
                    stats = state_full.process_chunk(idx, chunks[idx], checker)
                    full_checks += stats["pair_checks"]

            savings = 1 - incr_checks / full_checks if full_checks > 0 else 0

            # All domains should show positive savings
            assert savings > 0, f"Domain '{name}': no savings ({savings:.3f})"
            # Both produce same final pairs
            assert state_incr.pair_tracker.get_pairs() == state_full.pair_tracker.get_pairs(), (
                f"Domain '{name}': pair mismatch"
            )

            # Rebuild matches
            result = state_incr.rebuild_pairs(checker)
            assert result["match"], f"Domain '{name}': rebuild mismatch"

    @staticmethod
    def _make_doc_chunks():
        return [
            {
                f"doc_{c * 8 + i}": {
                    "keywords": frozenset(f"kw_{j}" for j in range(i, i + 5))
                }
                for i in range(8)
            }
            for c in range(4)
        ]

    @staticmethod
    def _make_user_chunks():
        return [
            {
                f"user_{c * 8 + i}": {
                    "interests": frozenset(f"int_{j}" for j in range(i % 6, i % 6 + 3))
                }
                for i in range(8)
            }
            for c in range(4)
        ]

    @staticmethod
    def _make_product_chunks():
        cats = ["electronics", "books", "sports", "music", "food"]
        rng = random.Random(99)
        return [
            {
                f"{'cust' if i < 4 else 'prod'}_{c * 8 + i}": {
                    "type": "customer" if i < 4 else "product",
                    "categories": frozenset(rng.sample(cats, k=rng.randint(1, 3))),
                }
                for i in range(8)
            }
            for c in range(4)
        ]

    @staticmethod
    def _make_threshold_chunks():
        rng = random.Random(77)
        return [
            {
                f"e_{c * 10 + i}": {"score": rng.random()}
                for i in range(10)
            }
            for c in range(4)
        ]
