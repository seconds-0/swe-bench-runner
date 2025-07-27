#!/usr/bin/env python3
"""
Example script demonstrating how to dispatch a subagent from the primary agent.
This shows the pattern for having Claude Code use the Task tool to create
a subagent that will trigger the SubagentStop hook upon completion.
"""

# This file demonstrates the PROMPT that the primary agent would use
# when calling the Task tool to dispatch a subagent.

EXAMPLE_TASK_PROMPT = """
I need to dispatch a subagent to handle a specific task. Here's what I'll do:

Task tool parameters:
- description: "Fix type errors in dataset module"
- subagent_type: "general-purpose"
- prompt: '''
Please fix the type errors in the src/swebench_runner/datasets.py file.

Steps:
1. Read the datasets.py file to understand the current implementation
2. Run mypy to identify all type errors
3. Fix each type error one by one
4. Run mypy again to verify all errors are resolved
5. Ensure the code still works correctly

Report back with:
- Number of type errors fixed
- Summary of changes made
- Confirmation that mypy passes
'''
"""

# Example of how the primary agent would describe the task
PRIMARY_AGENT_NARRATIVE = """
When you want to delegate a specific, well-defined task to a subagent:

1. Use the Task tool with clear parameters
2. Specify the subagent_type (usually "general-purpose")
3. Provide a detailed prompt with:
   - Clear objectives
   - Step-by-step instructions
   - Expected deliverables

The SubagentStop hook will automatically run when the subagent completes,
providing an introspection report about:
- Files that were modified
- Any TODO/FIXME markers left behind
- Git status changes
- General task completion assessment

This creates a powerful pattern where:
- Primary agent orchestrates high-level workflow
- Subagents handle specific implementation tasks
- Hooks provide automated quality checks
- Results flow back to primary agent for next steps
"""

if __name__ == "__main__":
    print("=== SUBAGENT DISPATCH PATTERN ===")
    print(EXAMPLE_TASK_PROMPT)
    print("\n" + "="*50 + "\n")
    print(PRIMARY_AGENT_NARRATIVE)
    print("\n=== END EXAMPLE ===")
