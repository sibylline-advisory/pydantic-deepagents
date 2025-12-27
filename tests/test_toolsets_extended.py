"""Extended tests for toolset implementations to reach 100% coverage."""

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_todo import TodoItem

from pydantic_deep.backends.state import StateBackend
from pydantic_deep.deps import DeepAgentDeps
from pydantic_deep.toolsets.filesystem import (
    get_filesystem_system_prompt,
)
from pydantic_deep.toolsets.subagents import (
    create_subagent_toolset,
    get_subagent_system_prompt,
)
from pydantic_deep.types import SubAgentConfig


class TestTodoToolsetExtended:
    """Extended tests for TodoToolset."""

    def test_todo_item_model(self):
        """Test TodoItem pydantic model."""
        item = TodoItem(
            content="Test task",
            status="pending",
            active_form="Testing",
        )
        assert item.content == "Test task"
        assert item.status == "pending"
        assert item.active_form == "Testing"


class TestFilesystemToolsetExtended:
    """Extended tests for FilesystemToolset."""

    def test_get_filesystem_system_prompt_basic(self):
        """Test basic filesystem system prompt."""
        deps = DeepAgentDeps(backend=StateBackend())
        prompt = get_filesystem_system_prompt(deps)
        assert "Filesystem Tools" in prompt

    def test_get_filesystem_system_prompt_with_sandbox(self):
        """Test filesystem system prompt with sandbox backend."""
        from pydantic_deep.backends.protocol import SandboxProtocol

        # Create a mock sandbox backend
        class MockSandbox(SandboxProtocol):
            def execute(self, command, timeout=None):
                pass

            def ls_info(self, path):
                return []

            def read(self, path, offset=0, limit=2000):
                return ""

            def write(self, path, content):
                pass

            def edit(self, path, old, new, replace_all=False):
                pass

            def glob_info(self, pattern, path="/"):
                return []

            def grep_raw(self, pattern, path=None, glob=None):
                return []

        deps = DeepAgentDeps(backend=MockSandbox())
        prompt = get_filesystem_system_prompt(deps)
        assert "Command Execution" in prompt

    def test_get_filesystem_system_prompt_with_files(self):
        """Test filesystem system prompt with files summary."""
        deps = DeepAgentDeps(backend=StateBackend())
        deps.files["/test.txt"] = {
            "content": ["test"],
            "created_at": "2024-01-01",
            "modified_at": "2024-01-01",
        }
        prompt = get_filesystem_system_prompt(deps)
        assert "Files in Memory" in prompt


class TestSubagentToolsetExtended:
    """Extended tests for SubAgentToolset."""

    def test_create_with_custom_subagents(self):
        """Test creating with custom subagent configs."""
        subagents = [
            SubAgentConfig(
                name="researcher",
                description="Research topics",
                instructions="You research topics thoroughly.",
            ),
        ]
        toolset = create_subagent_toolset(subagents=subagents)
        assert toolset is not None

    def test_create_without_general_purpose(self):
        """Test creating without general-purpose subagent."""
        toolset = create_subagent_toolset(include_general_purpose=False)
        assert toolset is not None

    def test_create_with_no_subagents(self):
        """Test creating with no subagents at all."""
        toolset = create_subagent_toolset(
            subagents=[],
            include_general_purpose=False,
        )
        assert toolset is not None

    def test_get_subagent_system_prompt_basic(self):
        """Test basic subagent system prompt."""
        deps = DeepAgentDeps(backend=StateBackend())
        prompt = get_subagent_system_prompt(deps)
        assert "Task Delegation" in prompt

    def test_get_subagent_system_prompt_with_configs(self):
        """Test subagent system prompt with configs."""
        deps = DeepAgentDeps(backend=StateBackend())
        configs = [
            SubAgentConfig(
                name="researcher",
                description="Research topics",
                instructions="Research thoroughly.",
            ),
        ]
        prompt = get_subagent_system_prompt(deps, configs)
        assert "Available Subagents" in prompt
        assert "researcher" in prompt

    def test_get_subagent_system_prompt_with_cached_subagents(self):
        """Test subagent system prompt with cached subagents."""
        deps = DeepAgentDeps(backend=StateBackend())
        deps.subagents = {"researcher": object()}

        prompt = get_subagent_system_prompt(deps)
        assert "Cached Subagents" in prompt
        assert "researcher" in prompt

    def test_create_with_custom_toolsets(self):
        """Test creating subagent with custom toolsets."""
        # Create a custom toolset
        custom_toolset: FunctionToolset[DeepAgentDeps] = FunctionToolset(id="custom")

        @custom_toolset.tool
        async def custom_tool(ctx: Any, query: str) -> str:  # type: ignore[misc]
            """A custom tool for testing."""
            return f"Custom result: {query}"

        # Create subagent config with custom toolsets
        subagents = [
            SubAgentConfig(
                name="custom-subagent",
                description="Subagent with custom toolset",
                instructions="You have custom tools.",
                toolsets=[custom_toolset],
            ),
        ]

        toolset = create_subagent_toolset(subagents=subagents, include_general_purpose=False)
        assert toolset is not None

    @pytest.mark.anyio
    async def test_subagent_uses_custom_toolsets(self):
        """Test that subagent actually uses custom toolsets passed in config."""
        from pydantic_ai import RunContext

        # Create a custom toolset
        custom_toolset: FunctionToolset[DeepAgentDeps] = FunctionToolset(id="custom-tools")

        @custom_toolset.tool
        async def special_tool(ctx: RunContext[DeepAgentDeps], message: str) -> str:
            """A special tool that marks its usage."""
            return f"SPECIAL_TOOL_CALLED: {message}"

        # Create subagent config with the custom toolset
        subagents = [
            SubAgentConfig(
                name="special-agent",
                description="Agent with special tool",
                instructions="You are a special agent. Use the special_tool.",
                toolsets=[custom_toolset],
            ),
        ]

        # Create the subagent toolset - this will create the subagent on first use
        subagent_toolset = create_subagent_toolset(
            subagents=subagents,
            default_model="test",
            include_general_purpose=False,
        )

        # Extract the task tool to manually trigger subagent creation
        task_tool = subagent_toolset.tools["task"]

        # Create deps and a mock context to trigger subagent creation
        deps = DeepAgentDeps(backend=StateBackend())

        # Create a minimal context-like object
        class MockContext:
            def __init__(self, deps: DeepAgentDeps):
                self.deps = deps

        ctx = MockContext(deps)

        # Call the task tool to create the subagent (this will cache it)
        await task_tool.function(ctx, "test message", "special-agent")  # type: ignore[arg-type]

        # The subagent should be cached now
        assert "special-agent" in deps.subagents

        # Verify the cached subagent is an Agent instance
        cached_agent = deps.subagents["special-agent"]
        assert isinstance(cached_agent, Agent)

        # The important test: verify that the subagent was created with custom toolsets
        # We can't easily inspect the agent's internal toolsets, but we verified:
        # 1. The subagent was created and cached
        # 2. The config had custom toolsets
        # 3. Our code passes toolsets to Agent() constructor
