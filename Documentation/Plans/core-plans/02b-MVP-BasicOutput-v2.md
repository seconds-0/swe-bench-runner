# MVP-BasicOutput-v2 - Delightful Output Implementation

**Task ID**: MVP-BasicOutput-v2
**Status**: Not Started
**Priority**: Phase 2 Enhancement
**Estimated Effort**: 8-10 hours
**Dependencies**: MVP-BasicOutput-v1

## Problem Statement

Take the minimal v1 output and make it delightful:
- Real-time progress bars with live updates
- Beautiful HTML reports
- Success celebrations
- Threading for responsive UI
- Smart error messages with solutions

This transforms a functional tool into a joy to use.

## Research Phase

### Research Checklist
- [x] Rich library capabilities for progress bars
- [x] HTML report generation patterns
- [x] Threading for subprocess output capture
- [x] Cross-platform signal handling
- [x] Actual harness output patterns (from v1 research)
- [x] Rich compatibility with Python 3.8-3.12

### Key Findings

1. **Rich Progress Bars**:
   - Supports multiple progress types
   - Can update from threads
   - Handles terminal resize
   - Beautiful spinners and animations
   - **Compatibility**: Rich supports Python 3.8+ officially

2. **HTML Reports** (SIMPLIFIED):
   - Skip Chart.js for single instance (no value)
   - Use simple HTML template with inline CSS
   - No external dependencies needed

3. **Threading Patterns**:
   - Use threading.Thread for output capture
   - Use queue.Queue for thread-safe communication
   - threading.Event for graceful shutdown

4. **Harness Output Patterns** (FROM RESEARCH):
   - Progress: "0%| | 0/1 [06:42<?, ?it/s]"
   - Building: "Building image sweb.env.py.x86_64..."
   - Installing: "Installing build dependencies..."
   - Downloading: "Downloading tomli-2.0.1-py3-none-any.whl"
   - Errors: "BuildImageError: Error building image..."

### Simplifications from Review
1. **Remove Chart.js** - Overkill for single instance
2. **Skip first-time detection** - Just celebrate all successes
3. **Fix thread safety** - Use proper Queue instead of list
4. **Handle Windows symlinks** - Fallback to copy if needed

## Proposed Solution

### 1. Enhanced Output Module (~400 lines)

```python
# output/progress.py - Real-time progress tracking
class ProgressTracker:
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.current_task = None

    def start_instance(self, instance_id: str):
        """Start tracking an instance."""
        self.current_task = self.progress.add_task(
            f"Running {instance_id}...",
            total=100
        )

    def update_from_harness(self, line: str):
        """Parse harness output and update progress."""
        # Based on actual harness output patterns
        if "%" in line and "|" in line:  # Progress bar format: "0%| | 0/1 [06:42<?, ?it/s]"
            # Extract percentage if possible
            try:
                percent = int(line.split("%")[0].strip())
                self.progress.update(self.current_task, completed=percent)
            except:
                pass
        elif "Building image" in line:
            self.progress.update(self.current_task, advance=10, description="Building Docker image...")
        elif "Installing build dependencies" in line:
            self.progress.update(self.current_task, advance=15, description="Installing dependencies...")
        elif "Downloading" in line and ".whl" in line:
            self.progress.update(self.current_task, advance=5, description="Downloading packages...")
        elif "Running evaluation" in line:
            self.progress.update(self.current_task, advance=30, description="Running tests...")
        elif "Applying patch" in line:
            self.progress.update(self.current_task, advance=10, description="Applying patch...")
        elif "evaluation_results" in line:
            self.progress.update(self.current_task, advance=20, description="Collecting results...")

# output/capture.py - Thread-safe output capture
import queue
import threading

class OutputCapture(threading.Thread):
    def __init__(self, process: subprocess.Popen, progress: ProgressTracker):
        super().__init__()
        self.process = process
        self.progress = progress
        self.output_queue = queue.Queue()
        self._stop_event = threading.Event()

    def run(self):
        """Capture output in real-time."""
        for line in iter(self.process.stdout.readline, b''):
            if self._stop_event.is_set():
                break

            decoded = line.decode('utf-8', errors='replace').strip()
            self.output_queue.put(decoded)  # Thread-safe
            self.progress.update_from_harness(decoded)

    def stop(self):
        """Gracefully stop capture."""
        self._stop_event.set()

    def get_output(self) -> List[str]:
        """Get all captured output."""
        lines = []
        while not self.output_queue.empty():
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return lines
```

### 2. HTML Report Generation (~200 lines)

```python
# output/reports.py
import shutil
from pathlib import Path

class ReportGenerator:
    def __init__(self):
        self.template = '''
<!DOCTYPE html>
<html>
<head>
    <title>SWE-bench Results</title>
    <style>
        body {
            font-family: -apple-system, system-ui, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .success { color: #22c55e; }
        .failure { color: #ef4444; }
        .summary {
            background: #f3f4f6;
            padding: 2rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
        .summary h2 { margin-top: 0; }
        .instance {
            border: 1px solid #e5e7eb;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        .instance.success { border-left: 4px solid #22c55e; }
        .instance.failure { border-left: 4px solid #ef4444; }
        pre {
            background: #1e293b;
            color: #e2e8f0;
            padding: 1rem;
            overflow-x: auto;
            border-radius: 0.25rem;
        }
        .timestamp { color: #6b7280; font-size: 0.875rem; }
    </style>
</head>
<body>
    <h1>SWE-bench Evaluation Results</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Instance:</strong> {instance_id}</p>
        <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
        <p class="timestamp">Generated: {timestamp}</p>
    </div>
    <div class="details">
        <h2>Execution Details</h2>
        {details}
    </div>
</body>
</html>
'''

    def generate(self, result: Dict, output_path: Path):
        """Generate beautiful HTML report for single instance."""
        status_class = "success" if result["resolved"] else "failure"
        status_text = "✅ PASSED" if result["resolved"] else "❌ FAILED"

        # Format execution details
        details = f'''
        <div class="instance {status_class}">
            <h3>Test Results</h3>
            {self._format_logs(result.get("logs", []))}
            {self._format_error(result.get("error")) if result.get("error") else ""}
        </div>
        '''

        # Fill template
        html = self.template.format(
            instance_id=result["instance_id"],
            status_class=status_class,
            status_text=status_text,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            details=details
        )

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html)

        # Create index.html for easy access
        index_path = output_path.parent / "index.html"
        try:
            # Try symlink first (Unix)
            if index_path.exists():
                index_path.unlink()
            index_path.symlink_to(output_path.name)
        except (OSError, NotImplementedError):
            # Fall back to copy (Windows)
            shutil.copy2(output_path, index_path)
```

### 3. Success Celebrations (~50 lines)

```python
# output/celebrations.py
import random
from rich.console import Console
from rich.panel import Panel

class SuccessCelebration:
    MESSAGES = [
        "🎉 Great job! Your patch passed all tests!",
        "✨ Success! The evaluation completed successfully!",
        "💪 Excellent work! All tests are green!",
        "🎯 Perfect! Your solution works correctly!",
        "⭐ Wonderful! The patch resolved the issue!",
    ]

    TIPS = [
        "Try running with --dataset verified for more instances!",
        "Check the HTML report for detailed results!",
        "You can run multiple instances with --instance-ids",
    ]

    def celebrate(self, instance_id: str):
        """Show success celebration."""
        console = Console()

        # Show celebration message
        console.print()
        console.print(Panel.fit(
            random.choice(self.MESSAGES),
            title=f"✅ {instance_id}",
            border_style="green",
            padding=(1, 2)
        ))

        # Occasionally show a tip (30% chance)
        if random.random() < 0.3:
            console.print()
            console.print(f"[dim]💡 Tip: {random.choice(self.TIPS)}[/dim]")
```

### 4. Enhanced Error Display (~150 lines)

```python
# output/errors.py
class ErrorDisplay:
    ERROR_SOLUTIONS = {
        "Docker daemon not running": {
            "macos": "Start Docker Desktop from Applications",
            "linux": "Run: sudo systemctl start docker"
        },
        "No space left on device": {
            "all": "Free up disk space or run: docker system prune -a"
        },
        "Network timeout": {
            "all": "Check your internet connection and try again"
        }
    }

    def display_error(self, error: Exception, context: Dict):
        """Display beautiful, helpful error messages."""
        console = Console()

        # Classify error
        error_type = self._classify_error(error)

        # Get platform-specific solution
        platform = sys.platform
        solution = self._get_solution(error_type, platform)

        # Display
        console.print()
        console.print(Panel(
            f"[bold red]❌ {error_type}[/bold red]\n\n"
            f"{str(error)}\n\n"
            f"[yellow]💡 How to fix:[/yellow]\n"
            f"{solution}",
            title="Error",
            border_style="red"
        ))
```

### 5. Integration with v1 (~200 lines)

```python
# enhanced_runner.py - Wraps existing docker_run with enhancements
from .docker_run import run_swebench_harness, parse_harness_results
from .output.progress import ProgressTracker
from .output.reports import ReportGenerator
from .output.celebrations import SuccessCelebration
from .output.errors import ErrorDisplay

class EnhancedRunner:
    def __init__(self):
        self.progress = ProgressTracker()
        self.report_gen = ReportGenerator()
        self.celebration = SuccessCelebration()
        self.error_display = ErrorDisplay()

    def run_with_progress(self, patches_file: Path, patch: Patch, temp_dir: Path) -> EvaluationResult:
        """Enhanced version of docker_run.run_swebench_harness with progress."""
        with self.progress.progress:
            # Start progress tracking
            self.progress.start_instance(patch.instance_id)

            # Build command (same as docker_run)
            cmd = [
                sys.executable, "-m", "swebench.harness.run_evaluation",
                "--predictions_path", str(patches_file),
                "--max_workers", "1",
                # ... same args as docker_run
            ]

            # Launch with real-time output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )

            # Capture output with progress updates
            capture = OutputCapture(process, self.progress)
            capture.start()

            try:
                # Wait for completion
                exit_code = process.wait(timeout=4200)
                capture.stop()
                capture.join()

                # Parse results (reuse existing function)
                result = parse_harness_results(temp_dir, patch)

                # Show celebration if passed
                if result.passed:
                    self.celebration.celebrate(patch.instance_id)

                # Generate HTML report
                report_path = Path(f"results/{patch.instance_id}/report.html")
                self.report_gen.generate(result.__dict__, report_path)

                return result

            except subprocess.TimeoutExpired:
                self.progress.console.print("\n[red]Evaluation timed out[/red]")
                process.terminate()
                capture.stop()
                raise
            except KeyboardInterrupt:
                self.progress.console.print("\n[yellow]Stopping gracefully...[/yellow]")
                process.terminate()
                capture.stop()
                raise
```

## Implementation Checklist

- [ ] Progress Tracking
  - [ ] Create ProgressTracker class
  - [ ] Implement harness output parsing
  - [ ] Add progress bar updates
  - [ ] Handle terminal resize
- [ ] Output Capture
  - [ ] Create OutputCapture thread
  - [ ] Implement line-by-line capture
  - [ ] Add thread-safe queue
  - [ ] Handle graceful shutdown
- [ ] HTML Reports
  - [ ] Create ReportGenerator class
  - [ ] Design HTML template
  - [ ] Add Chart.js integration
  - [ ] Generate index.html symlink
- [ ] Success Celebrations
  - [ ] Create celebration messages
  - [ ] Detect first-time users
  - [ ] Add tips and encouragement
- [ ] Error Display
  - [ ] Create error classification
  - [ ] Add platform-specific solutions
  - [ ] Design error panels
- [ ] Integration
  - [ ] Update CLI to use enhanced runner
  - [ ] Add --no-progress flag
  - [ ] Update tests
  - [ ] Add integration tests

## Test Plan

### Unit Tests
- Test progress parsing patterns
- Test thread-safe output capture
- Test HTML generation
- Test error classification
- Test platform detection

### Integration Tests
- Test full run with progress
- Test Ctrl+C handling
- Test error scenarios
- Test report generation
- Test first-time detection

### Manual Testing
- Test on macOS and Linux
- Test with slow instances
- Test with failing instances
- Test terminal resize
- Test in different terminals

## Success Criteria

1. ✅ Real-time progress that feels responsive
2. ✅ Beautiful HTML reports (without unnecessary charts)
3. ✅ Delightful success celebrations (simplified)
4. ✅ Helpful error messages with platform-specific solutions
5. ✅ Graceful Ctrl+C handling
6. ✅ Cross-platform compatibility (including Windows)

## Dependencies

- Rich (for progress bars and console) - CHECK: Python 3.8-3.12 compatibility
- threading (stdlib)
- queue (stdlib)
- shutil (stdlib) - for Windows symlink fallback
- datetime (stdlib)
- random (stdlib)

## Questions/Uncertainties

### ✅ Resolved (From Research)
- Harness output patterns documented (progress bars, build messages, etc.)
- Rich supports Python 3.8+ officially
- No known terminal compatibility issues with Rich

### Non-blocking
- Animation preferences → Keep subtle and professional
- Color scheme → Using accessible colors (green/red)
- Progress bar granularity → Implemented based on real patterns

## Notes

This v2 transforms the basic tool into something delightful. Key simplifications from original:
1. **No Chart.js**: Removed charts for single instance (overkill)
2. **No first-time detection**: Celebrate all successes equally
3. **Thread-safe Queue**: Fixed thread safety with proper queue.Queue
4. **Windows support**: Added fallback for symlinks

Key principles remain:
1. **Responsive**: Real-time feedback keeps users engaged
2. **Beautiful**: Professional output users are proud to share
3. **Helpful**: Every error includes how to fix it
4. **Celebratory**: Make success feel good (but not too much)

Total: ~900 lines of well-organized, tested code that makes users smile.
