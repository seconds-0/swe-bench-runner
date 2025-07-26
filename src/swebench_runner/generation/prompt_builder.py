"""Prompt building for SWE-bench instance data."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TemplateStyle(Enum):
    """Available prompt template styles."""

    DETAILED = "detailed"
    CONCISE = "concise"
    COT = "cot"  # Chain of thought
    MINIMAL = "minimal"


@dataclass
class PromptContext:
    """Complete context for prompt building."""

    system_prompt: str = ""
    user_prompt: str = ""
    code_context: dict[str, str] = field(default_factory=dict)  # filename -> content
    test_context: dict[str, str] = field(default_factory=dict)  # test_name -> content
    problem_statement: str = ""
    instance_id: str = ""
    repo_name: str = ""
    base_commit: str = ""
    hints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_length(self) -> int:
        """Calculate total prompt length."""
        total = (len(self.system_prompt) + len(self.user_prompt) +
                 len(self.problem_statement))
        total += sum(len(content) for content in self.code_context.values())
        total += sum(len(content) for content in self.test_context.values())
        total += sum(len(hint) for hint in self.hints)
        return total


class PromptBuilder:
    """Builds prompts from SWE-bench instances."""

    def __init__(
        self,
        template_style: TemplateStyle = TemplateStyle.DETAILED,
        include_test_info: bool = True,
        max_code_context_lines: int = 500,
        max_test_lines: int = 200,
        prioritize_recent_changes: bool = True,
        include_hints: bool = True,
        system_prompt_override: str | None = None
    ):
        """Initialize the prompt builder.

        Args:
            template_style: The template style to use for prompts
            include_test_info: Whether to include test information in prompts
            max_code_context_lines: Maximum lines of code context to include
            max_test_lines: Maximum lines of test context to include
            prioritize_recent_changes: Whether to prioritize recent code changes
            include_hints: Whether to include hints in the prompt
            system_prompt_override: Optional override for the system prompt
        """
        self.template_style = template_style
        self.include_test_info = include_test_info
        self.max_code_context_lines = max_code_context_lines
        self.max_test_lines = max_test_lines
        self.prioritize_recent_changes = prioritize_recent_changes
        self.include_hints = include_hints
        self.system_prompt_override = system_prompt_override

        # Define system prompts for each template style
        self._system_prompts = {
            TemplateStyle.DETAILED: (
                "You are an expert software engineer fixing bugs in a codebase. "
                "Analyze the problem thoroughly and generate a minimal, focused patch "
                "that fixes the issue without introducing new bugs."
            ),
            TemplateStyle.CONCISE: (
                "You are a code fixing assistant. Generate patches for bugs."
            ),
            TemplateStyle.COT: (
                "You are an expert debugger. Think step-by-step through the problem, "
                "analyze the code, and generate a fix."
            ),
            TemplateStyle.MINIMAL: "Fix bugs. Return patches only."
        }

    def build_context(self, instance: dict[str, Any]) -> PromptContext:
        """Extract and organize information from SWE-bench instance.

        Args:
            instance: SWE-bench instance data

        Returns:
            PromptContext with extracted information
        """
        instance_id = instance.get('instance_id', 'unknown')
        logger.debug(f"Building context for instance {instance_id}")

        context = PromptContext()

        # Extract basic fields
        context.instance_id = instance.get("instance_id", "unknown")
        context.repo_name = instance.get("repo", "")
        context.base_commit = instance.get("base_commit", "")
        context.problem_statement = instance.get("problem_statement", "")

        # Set system prompt
        if self.system_prompt_override:
            context.system_prompt = self.system_prompt_override
        else:
            context.system_prompt = self._system_prompts.get(
                self.template_style,
                self._system_prompts[TemplateStyle.DETAILED]
            )

        # Extract hints
        if self.include_hints:
            hints_text = instance.get("hints_text", "")
            if hints_text:
                context.hints = [hints_text]

            # Also check for hints list
            if isinstance(instance.get("hints"), list):
                context.hints.extend(instance["hints"])

        # Extract test information
        if self.include_test_info:
            test_patch = instance.get("test_patch", "")
            if test_patch:
                context.test_context["test_patch"] = test_patch

            # Check for test_files if available
            test_files = instance.get("test_files", {})
            if isinstance(test_files, dict):
                for filename, content in test_files.items():
                    # Limit number of test files
                    if content and len(context.test_context) < 5:
                        context.test_context[filename] = self._truncate_content(
                            content, self.max_test_lines, preserve_errors=True
                        )

        # Extract code context if available
        code_files = instance.get("code_files", {})
        if isinstance(code_files, dict):
            for filename, content in code_files.items():
                # Limit number of code files
                if content and len(context.code_context) < 10:
                    context.code_context[filename] = self._truncate_content(
                        content, self.max_code_context_lines, preserve_structure=True
                    )

        # Extract additional metadata
        for key in ["created_at", "version", "environment_setup_commit"]:
            if key in instance:
                context.metadata[key] = instance[key]

        logger.debug(
            f"Context built for {context.instance_id}: "
            f"{len(context.code_context)} code files, "
            f"{len(context.test_context)} test files, "
            f"{len(context.hints)} hints"
        )

        return context

    def build_prompt(
        self, context: PromptContext, template: TemplateStyle | None = None
    ) -> str:
        """Convert context to formatted prompt string.

        Args:
            context: The prompt context to format
            template: Optional template style override

        Returns:
            Formatted prompt string
        """
        template_style = template or self.template_style

        if template_style == TemplateStyle.DETAILED:
            return self._build_detailed_prompt(context)
        elif template_style == TemplateStyle.CONCISE:
            return self._build_concise_prompt(context)
        elif template_style == TemplateStyle.COT:
            return self._build_cot_prompt(context)
        elif template_style == TemplateStyle.MINIMAL:
            return self._build_minimal_prompt(context)
        else:
            logger.warning(f"Unknown template style {template_style}, using DETAILED")
            return self._build_detailed_prompt(context)

    def reduce_context(
        self, context: PromptContext, reduction_factor: float = 0.7
    ) -> PromptContext:
        """Reduce context size while preserving important information.

        Args:
            context: The context to reduce
            reduction_factor: Factor by which to reduce context (0.0-1.0)

        Returns:
            Reduced PromptContext
        """
        logger.info(f"Reducing context by factor {reduction_factor}")

        # Create a copy of the context
        reduced = PromptContext(
            system_prompt=context.system_prompt,
            user_prompt=context.user_prompt,
            problem_statement=context.problem_statement,
            instance_id=context.instance_id,
            repo_name=context.repo_name,
            base_commit=context.base_commit,
            hints=context.hints[:max(1, int(len(context.hints) * reduction_factor))],
            metadata=context.metadata.copy()
        )

        # Reduce problem statement if very long
        if len(context.problem_statement) > 2000:
            reduced.problem_statement = self._truncate_content(
                context.problem_statement,
                int(2000 * reduction_factor),
                preserve_errors=True
            )

        # Reduce code context
        if context.code_context:
            # Sort by importance (prefer smaller files, test files, files with errors)
            sorted_files = sorted(
                context.code_context.items(),
                key=lambda x: (
                    "test" not in x[0].lower(),  # Prefer test files
                    len(x[1]),  # Prefer smaller files
                    x[0]  # Alphabetical as tiebreaker
                )
            )

            # Keep only the most important files
            keep_count = max(1, int(len(sorted_files) * reduction_factor))
            for filename, content in sorted_files[:keep_count]:
                reduced.code_context[filename] = self._truncate_content(
                    content,
                    int(self.max_code_context_lines * reduction_factor),
                    preserve_structure=True
                )

        # Reduce test context
        if context.test_context:
            sorted_tests = sorted(
                context.test_context.items(),
                key=lambda x: (len(x[1]), x[0])
            )

            keep_count = max(1, int(len(sorted_tests) * reduction_factor))
            for test_name, content in sorted_tests[:keep_count]:
                reduced.test_context[test_name] = self._truncate_content(
                    content,
                    int(self.max_test_lines * reduction_factor),
                    preserve_errors=True
                )

        original_length = context.total_length()
        reduced_length = reduced.total_length()
        logger.info(
            f"Context reduced from {original_length} to {reduced_length} characters "
            f"({reduced_length/original_length:.1%} of original)"
        )

        return reduced

    def build_retry_prompt(
        self,
        original_prompt: str,
        failed_response: str,
        error_reason: str = "parse_failure"
    ) -> str:
        """Build retry prompt with clarification.

        Args:
            original_prompt: The original prompt that was sent
            failed_response: The response that failed to parse
            error_reason: Reason for the failure

        Returns:
            New prompt for retry
        """
        retry_parts = []

        if error_reason == "parse_failure":
            retry_parts.extend([
                "Your previous response could not be parsed as a valid unified diff "
                "patch.",
                "Please ensure your response:",
                "1. Contains a valid unified diff starting with 'diff --git' or "
                "'--- a/'",
                "2. Has proper diff headers (+++, ---)",
                "3. Uses correct diff syntax with + and - line prefixes",
                "4. Does not include explanatory text outside the patch",
                "",
                "Here was your previous response:",
                "```",
                failed_response[:1000],  # Truncate if very long
                "..." if len(failed_response) > 1000 else "",
                "```",
                "",
                "Please provide ONLY a valid unified diff patch, with no "
                "additional explanation.",
            ])
        elif error_reason == "no_changes":
            retry_parts.extend([
                "Your patch appears to be empty or contains no actual changes.",
                "Please ensure your patch includes actual modifications to fix "
                "the issue.",
                "",
                "Remember to:",
                "1. Include actual changes (lines starting with + or -)",
                "2. Not just show the current code",
                "3. Make the minimal changes needed to fix the problem",
            ])
        else:
            retry_parts.extend([
                f"Your previous response failed validation: {error_reason}",
                "Please provide a valid unified diff patch that fixes the issue.",
            ])

        retry_parts.extend([
            "",
            "Original task:",
            original_prompt
        ])

        return "\n".join(retry_parts)

    def _build_detailed_prompt(self, context: PromptContext) -> str:
        """Build a detailed prompt."""
        parts = [
            f"## Task: Fix Bug in {context.repo_name}",
            "",
            f"**Instance ID**: {context.instance_id}",
            f"**Base Commit**: {context.base_commit}",
            "",
            "### Problem Description",
            context.problem_statement,
        ]

        if context.test_context:
            parts.extend([
                "",
                "### Failing Tests",
                self._format_test_context(context.test_context),
            ])

        if context.code_context:
            parts.extend([
                "",
                "### Relevant Code Context",
                self._format_code_context(context.code_context),
            ])

        if context.hints:
            parts.extend([
                "",
                "### Additional Hints",
                "\n".join(f"- {hint}" for hint in context.hints),
            ])

        parts.extend([
            "",
            "### Instructions",
            "1. Analyze the problem and failing tests",
            "2. Identify the root cause",
            "3. Generate a minimal fix",
            "4. Return the fix as a unified diff patch",
            "",
            "The patch should:",
            "- Fix the failing tests",
            "- Be minimal and focused",
            "- Not introduce new bugs",
            "- Follow the existing code style",
            "",
            "Return only the patch in unified diff format."
        ])

        return "\n".join(parts)

    def _build_concise_prompt(self, context: PromptContext) -> str:
        """Build a concise prompt."""
        parts = [
            f"Fix this bug in {context.repo_name}:",
            (context.problem_statement[:500] + "..."
             if len(context.problem_statement) > 500 else context.problem_statement),
        ]

        if context.test_context:
            test_summary = self._summarize_test_failures(context.test_context)
            if test_summary:
                parts.extend([
                    "",
                    "Tests failing:",
                    test_summary,
                ])

        if context.code_context:
            parts.extend([
                "",
                "Context:",
                self._summarize_code_context(context.code_context),
            ])

        parts.extend([
            "",
            "Return unified diff patch only."
        ])

        return "\n".join(parts)

    def _build_cot_prompt(self, context: PromptContext) -> str:
        """Build a chain-of-thought prompt."""
        parts = [
            "Let's fix this bug step by step.",
            "",
            f"Step 1: Understand the problem in {context.repo_name}",
            context.problem_statement,
            "",
        ]

        if context.test_context:
            parts.extend([
                "Step 2: Analyze failing tests",
                self._format_test_context(context.test_context),
                "What do these tests expect?",
                "",
            ])

        if context.code_context:
            parts.extend([
                "Step 3: Examine the code",
                self._format_code_context(context.code_context),
                "What's causing the failure?",
                "",
            ])

        parts.extend([
            "Step 4: Plan the fix",
            "Think about the minimal change needed.",
            "",
            "Step 5: Generate the patch",
            "Now create a unified diff that fixes the issue.",
        ])

        return "\n".join(parts)

    def _build_minimal_prompt(self, context: PromptContext) -> str:
        """Build a minimal prompt."""
        return f"""Bug: {context.problem_statement[:200]}...
Repo: {context.repo_name}
Fix with unified diff patch."""

    def _format_code_context(self, code_files: dict[str, str]) -> str:
        """Format code files for inclusion in prompt."""
        if not code_files:
            return "No code context available."

        formatted_parts = []
        for filename, content in code_files.items():
            formatted_parts.extend([
                f"#### {filename}",
                "```python",
                content,
                "```",
                ""
            ])

        return "\n".join(formatted_parts)

    def _format_test_context(self, test_files: dict[str, str]) -> str:
        """Format test files for inclusion in prompt."""
        if not test_files:
            return "No test context available."

        formatted_parts = []

        # Special handling for test_patch
        if "test_patch" in test_files:
            formatted_parts.extend([
                "#### Test Patch",
                "```diff",
                test_files["test_patch"],
                "```",
                ""
            ])

        # Format other test files
        for filename, content in test_files.items():
            if filename != "test_patch":
                formatted_parts.extend([
                    f"#### {filename}",
                    "```python",
                    content,
                    "```",
                    ""
                ])

        return "\n".join(formatted_parts)

    def _summarize_test_failures(self, test_context: dict[str, str]) -> str:
        """Create concise summary of test failures."""
        summaries = []

        for test_name, content in test_context.items():
            # Extract key failure information
            lines = content.split('\n')

            # Look for assertion failures or error messages
            error_lines = []
            for line in lines:
                keywords = ['error', 'fail', 'assert', 'exception']
                if any(keyword in line.lower() for keyword in keywords):
                    error_lines.append(line.strip())

            if error_lines:
                summary = f"{test_name}: {error_lines[0]}"
                if len(error_lines) > 1:
                    summary += f" (+{len(error_lines)-1} more)"
                summaries.append(summary)
            else:
                summaries.append(f"{test_name}: Test failures")

        return "\n".join(summaries[:5])  # Limit to 5 test summaries

    def _summarize_code_context(self, code_files: dict[str, str]) -> str:
        """Create concise summary of code context."""
        summaries = []

        for filename, content in list(code_files.items())[:3]:  # Limit to 3 files
            lines = content.split('\n')

            # Extract key elements (classes, functions)
            key_elements = []
            for line in lines:
                if line.strip().startswith(('class ', 'def ')):
                    # Extract just the definition
                    element = line.strip().split('(')[0]
                    key_elements.append(element)

            if key_elements:
                summary = f"{filename}: {', '.join(key_elements[:3])}"
                if len(key_elements) > 3:
                    summary += f" (+{len(key_elements)-3} more)"
                summaries.append(summary)
            else:
                summaries.append(f"{filename}: {len(lines)} lines")

        return "\n".join(summaries)

    def _truncate_content(
        self,
        content: str,
        max_lines: int,
        preserve_errors: bool = False,
        preserve_structure: bool = False
    ) -> str:
        """Intelligently truncate content while preserving important information.

        Args:
            content: The content to truncate
            max_lines: Maximum number of lines to keep
            preserve_errors: Whether to prioritize keeping error messages
            preserve_structure: Whether to preserve code structure (imports, signatures)

        Returns:
            Truncated content
        """
        if not content:
            return content

        lines = content.split('\n')

        if len(lines) <= max_lines:
            return content

        if preserve_errors:
            # Keep lines with error indicators
            error_keywords = ['error', 'exception', 'traceback', 'fail', 'assert']
            important_lines = []
            regular_lines = []

            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in error_keywords):
                    # Include some context around errors
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    important_lines.extend(lines[start:end])
                else:
                    regular_lines.append(line)

            # Deduplicate while preserving order
            seen = set()
            unique_important = []
            for line in important_lines:
                if line not in seen:
                    seen.add(line)
                    unique_important.append(line)

            # Combine important lines with some regular context
            remaining_budget = max_lines - len(unique_important)
            if remaining_budget > 0:
                result = (
                    unique_important[:max_lines//2] +
                    regular_lines[:remaining_budget//2] +
                    unique_important[max_lines//2:]
                )
            else:
                result = unique_important[:max_lines]

            return "\n".join(result[:max_lines]) + "\n... (truncated)"

        elif preserve_structure:
            # Keep imports, class/function definitions
            structure_lines = []
            body_lines = []

            for line in lines:
                stripped = line.strip()
                if (stripped.startswith(('import ', 'from ', 'class ', 'def ', '@')) or
                    stripped.endswith(':') or
                    not stripped):
                    structure_lines.append(line)
                else:
                    body_lines.append(line)

            # Allocate lines between structure and body
            structure_budget = min(len(structure_lines), max_lines // 2)
            body_budget = max_lines - structure_budget

            result = structure_lines[:structure_budget]
            if body_budget > 0:
                # Add some body content from the middle
                middle_start = len(body_lines) // 2 - body_budget // 2
                middle_end = middle_start + body_budget
                result.extend(body_lines[middle_start:middle_end])

            return "\n".join(result) + "\n... (truncated)"

        else:
            # Simple truncation: keep beginning and end
            keep_start = max_lines * 2 // 3
            keep_end = max_lines - keep_start

            result = lines[:keep_start]
            result.append("... (truncated)")
            result.extend(lines[-keep_end:])

            return "\n".join(result)

    def _extract_error_context(self, test_output: str) -> str:
        """Extract key error information from test output."""
        if not test_output:
            return ""

        lines = test_output.split('\n')
        error_context = []

        # Patterns that indicate important error information
        error_patterns = [
            r'AssertionError:',
            r'FAILED',
            r'ERROR',
            r'Traceback',
            r'File ".*", line \d+',
            r'Expected:',
            r'Actual:',
            r'\s+E\s+',  # Pytest error marker
        ]

        in_traceback = False
        for line in lines:
            # Check if we're entering or in a traceback
            if 'Traceback' in line:
                in_traceback = True

            # Check if line matches any error pattern
            if (any(re.search(pattern, line) for pattern in error_patterns) or
                    in_traceback):
                error_context.append(line)

                # Stop traceback at the actual error message
                if in_traceback and line.strip() and not line.startswith(' '):
                    in_traceback = False

        return "\n".join(error_context) if error_context else test_output[:500]
