"""
Tests for incremental state awareness in persistent mode.

Validates that:
1. _get_cached_vars correctly reports REPL state
2. build_user_prompt includes incremental state info when cached_vars provided
3. The persistent mode properly carries forward variables between turns
"""

from rlm.environments.local_repl import LocalREPL
from rlm.utils.prompts import build_user_prompt


class TestGetCachedVars:
    """Test the _get_cached_vars helper."""

    def test_empty_repl_returns_none(self):
        from rlm.core.rlm import RLM

        repl = LocalREPL()
        result = RLM._get_cached_vars(repl)
        assert result is None
        repl.cleanup()

    def test_with_user_variables(self):
        from rlm.core.rlm import RLM

        repl = LocalREPL()
        repl.execute_code("x = 42\ny = 'hello'\nz = [1,2,3]")
        result = RLM._get_cached_vars(repl)
        assert result is not None
        assert "x" in result
        assert result["x"] == "int"
        assert "y" in result
        assert result["y"] == "str"
        assert "z" in result
        assert result["z"] == "list"
        repl.cleanup()

    def test_excludes_context_and_history_vars(self):
        from rlm.core.rlm import RLM

        repl = LocalREPL()
        repl.locals["context_0"] = "some data"
        repl.locals["context"] = "some data"
        repl.locals["history_0"] = []
        repl.locals["user_data"] = {"key": "val"}
        result = RLM._get_cached_vars(repl)
        assert result is not None
        assert "context_0" not in result
        assert "context" not in result
        assert "history_0" not in result
        assert "user_data" in result
        repl.cleanup()

    def test_excludes_private_vars(self):
        from rlm.core.rlm import RLM

        repl = LocalREPL()
        repl.locals["_internal"] = 1
        repl.locals["public"] = 2
        result = RLM._get_cached_vars(repl)
        assert result is not None
        assert "_internal" not in result
        assert "public" in result
        repl.cleanup()


class TestIncrementalUserPrompt:
    """Test that build_user_prompt includes incremental state info."""

    def test_no_cached_vars(self):
        prompt = build_user_prompt(root_prompt="test", iteration=0, cached_vars=None)
        assert "INCREMENTAL STATE" not in prompt["content"]

    def test_with_cached_vars(self):
        cached = {"summaries": "list", "total_count": "int"}
        prompt = build_user_prompt(root_prompt="test", iteration=0, cached_vars=cached)
        assert "INCREMENTAL STATE" in prompt["content"]
        assert "summaries" in prompt["content"]
        assert "total_count" in prompt["content"]
        assert "Build on existing computations" in prompt["content"]

    def test_cached_vars_only_on_first_iteration(self):
        """In practice, cached_vars is only passed on iteration=0 of non-first turns."""
        cached = {"x": "int"}
        # iteration=0 with cached vars should include incremental state
        prompt0 = build_user_prompt(root_prompt="test", iteration=0, cached_vars=cached)
        assert "INCREMENTAL STATE" in prompt0["content"]

        # Without cached vars (later iterations), should not include it
        prompt1 = build_user_prompt(root_prompt="test", iteration=1, cached_vars=None)
        assert "INCREMENTAL STATE" not in prompt1["content"]


class TestPersistentStateCarryForward:
    """Test that REPL state persists across simulated turns."""

    def test_variables_persist_across_executions(self):
        repl = LocalREPL()
        # Turn 1: create variables
        result = repl.execute_code("summaries = ['chunk1_summary', 'chunk2_summary']\ncount = 2")
        assert "summaries" in repl.locals
        assert repl.locals["count"] == 2

        # Turn 2: access and extend
        result = repl.execute_code("summaries.append('chunk3_summary')\ncount += 1\nprint(count)")
        assert repl.locals["count"] == 3
        assert len(repl.locals["summaries"]) == 3
        assert "3" in result.stdout

        repl.cleanup()

    def test_context_versioning_with_persistence(self):
        repl = LocalREPL(context_payload="first chunk of data")
        assert repl.get_context_count() == 1
        assert "context" in repl.locals
        assert "context_0" in repl.locals

        # Add second context
        repl.add_context("second chunk of data")
        assert repl.get_context_count() == 2
        assert "context_1" in repl.locals

        # Both contexts accessible
        result = repl.execute_code("print(len(context_0), len(context_1))")
        assert "19 20" in result.stdout  # "first chunk of data" = 19, "second chunk of data" = 20

        repl.cleanup()
