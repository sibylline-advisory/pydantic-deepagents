"""Example demonstrating custom toolsets in subagents.

This example shows how to pass custom toolsets to subagents through SubAgentConfig.
"""

import asyncio

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_deep import DeepAgentDeps, StateBackend, create_deep_agent
from pydantic_deep.types import SubAgentConfig


async def main():
    # Create a custom toolset with domain-specific tools
    data_analysis_toolset: FunctionToolset[DeepAgentDeps] = FunctionToolset(id="data-analysis")

    @data_analysis_toolset.tool
    async def calculate_statistics(ctx: RunContext[DeepAgentDeps], data: list[float]) -> str:
        """Calculate basic statistics for a list of numbers.

        Args:
            data: List of numbers to analyze

        Returns:
            Statistics summary
        """
        if not data:
            return "No data provided"

        mean = sum(data) / len(data)
        sorted_data = sorted(data)
        median = sorted_data[len(data) // 2]
        min_val = min(data)
        max_val = max(data)

        return f"""Statistics:
- Count: {len(data)}
- Mean: {mean:.2f}
- Median: {median:.2f}
- Min: {min_val:.2f}
- Max: {max_val:.2f}"""

    @data_analysis_toolset.tool
    async def analyze_trend(ctx: RunContext[DeepAgentDeps], data: list[float]) -> str:
        """Analyze trend in a list of numbers.

        Args:
            data: List of numbers in time order

        Returns:
            Trend description
        """
        if len(data) < 2:
            return "Not enough data for trend analysis"

        increasing = sum(1 for i in range(len(data) - 1) if data[i + 1] > data[i])
        total_comparisons = len(data) - 1

        if increasing > total_comparisons * 0.7:
            return "Strong upward trend"
        elif increasing > total_comparisons * 0.5:
            return "Moderate upward trend"
        elif increasing < total_comparisons * 0.3:
            return "Strong downward trend"
        elif increasing < total_comparisons * 0.5:
            return "Moderate downward trend"
        else:
            return "No clear trend"

    # Create a subagent with the custom toolset
    subagents = [
        SubAgentConfig(
            name="data-analyst",
            description="Analyzes numerical data using statistical tools",
            instructions="""You are a data analyst with access to statistical analysis tools.
When given data, use calculate_statistics and analyze_trend to provide insights.
Always use the tools to perform calculations - don't calculate manually.""",
            toolsets=[data_analysis_toolset],  # Pass custom toolset
        ),
    ]

    # Create the main agent with the custom subagent
    agent = create_deep_agent(
        model="openai:gpt-4.1",
        instructions="""You are a coordinator agent.
When asked to analyze data, delegate to the data-analyst subagent.""",
        subagents=subagents,
        include_general_purpose_subagent=False,
    )

    deps = DeepAgentDeps(backend=StateBackend())

    # Test the subagent with custom toolset
    result = await agent.run(
        """I have sales data for the last 7 days: [100, 120, 115, 140, 155, 150, 165]
        Please analyze this data using the data-analyst subagent.""",
        deps=deps,
    )

    print("Agent output:")
    print(result.output)

    # Verify the subagent was created and cached
    if "data-analyst" in deps.subagents:
        print("\n✓ Subagent 'data-analyst' was created and cached")
        print("✓ The subagent has access to custom statistical tools")


if __name__ == "__main__":
    asyncio.run(main())
