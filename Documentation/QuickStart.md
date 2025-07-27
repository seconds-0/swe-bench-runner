# SWE-Bench Runner Quick Start Guide

This guide walks you through testing AI models on SWE-bench, a benchmark that measures if AI can fix real bugs in code.

## What is SWE-bench?

SWE-bench contains real GitHub issues from popular Python projects. Each issue has:
- A bug description (what's broken)
- The actual codebase with the bug
- Tests that fail because of the bug

Your AI model's job: Read the bug description and generate a fix that makes the tests pass.

## What are Patches?

A "patch" is a standardized way to describe code changes. It shows:
- Which files to modify
- Which lines to remove (marked with `-`)
- Which lines to add (marked with `+`)

Example patch:
```diff
diff --git a/django/http/response.py b/django/http/response.py
index 123..456 100644
--- a/django/http/response.py
+++ b/django/http/response.py
@@ -100,7 +100,7 @@ class HttpResponse:
     def __init__(self, content='', *args, **kwargs):
-        self.content = content
+        self.content = str(content).encode('utf-8')
         super().__init__(*args, **kwargs)
```

This patch changes line 101 in `django/http/response.py` to fix a Unicode issue.

## Prerequisites

- **Python 3.10+**
- **Docker Desktop** (macOS) or Docker Engine (Linux)
- **16GB+ RAM** allocated to Docker
- **120GB+ free disk space** for Docker images

## Step 1: Install Docker

### macOS
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install and start Docker Desktop
3. In Docker Desktop settings, allocate at least 16GB RAM
4. Verify installation:
   ```bash
   docker --version
   ```

### Linux
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER  # Allow non-root access
# Log out and back in for group changes to take effect
```

## Step 2: Install SWE-Bench Runner

```bash
# Clone the repository
git clone https://github.com/seconds-0/swe-bench-runner.git
cd swe-bench-runner

# Install from source
pip install -e .
```

Verify installation:
```bash
swebench --version
```

## Step 3: Quick Test (No AI Model Needed)

Before integrating your model, let's verify everything works:

```bash
# Run 1 random instance from the lite dataset
# This uses the "gold" patches (correct solutions) just to test the system
swebench run -d lite --count 1
```

You'll see:
- Dataset downloading (first time only, ~4MB)
- Docker pulling images (first time only, ~14GB per image)
- Tests running in Docker
- Pass/Fail result

## Step 4: Test Your AI Model

Now let's generate patches with actual AI models. You need to:
1. Get the bug description from SWE-bench
2. Ask your model to generate a fix
3. Save the fix as a patch
4. Run SWE-bench to test it

### Option A: API-Based Models (Claude, GPT-4, Gemini)

Here's a complete example using Claude:

```python
# test_claude.py
import os
import json
from anthropic import Anthropic
from datasets import load_dataset

# 1. Setup
# First: pip install anthropic datasets
client = Anthropic(api_key="your-api-key-here")  # Get from console.anthropic.com

# 2. Load a real bug from SWE-bench
print("Loading SWE-bench dataset...")
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
issue = dataset[0]  # First issue

print(f"\nTesting on: {issue['instance_id']}")
print(f"Repository: {issue['repo']}")
print(f"Issue preview: {issue['problem_statement'][:200]}...")

# 3. Ask Claude to fix it
prompt = f"""You are an expert programmer. Fix this GitHub issue.

Repository: {issue['repo']}
Issue Description:
{issue['problem_statement']}

Instructions:
- Analyze the issue carefully
- Generate a fix in unified diff format
- Make sure the patch will apply cleanly
- Output ONLY the patch, no explanations

Example format:
diff --git a/path/to/file.py b/path/to/file.py
index 123..456 100644
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,5 +10,5 @@ class Example:
     def method(self):
-        return old_code
+        return new_code
"""

print("\nAsking Claude for a fix...")
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4000,
    temperature=0,  # More deterministic
    messages=[{"role": "user", "content": prompt}]
)

# 4. Save the patch in SWE-bench format
patch = response.content[0].text
patch_data = {
    "instance_id": issue['instance_id'],
    "model": "claude-3.5-sonnet",
    "patch": patch
}

with open("claude_patches.jsonl", "w") as f:
    json.dump(patch_data, f)
    f.write("\n")

print("\nPatch generated! Preview:")
print(patch[:300] + "..." if len(patch) > 300 else patch)
print(f"\nSaved to: claude_patches.jsonl")
print("\nNow run: swebench run --patches claude_patches.jsonl")
```

For GPT-4:
```python
# test_gpt4.py
from openai import OpenAI
# ... similar structure, just change:
client = OpenAI(api_key="your-key")  # From platform.openai.com
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[{"role": "user", "content": prompt}]
)
patch = response.choices[0].message.content
```

For Gemini:
```python
# test_gemini.py
import google.generativeai as genai
# ... similar structure, just change:
genai.configure(api_key="your-key")  # From makersuite.google.com
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content(prompt)
patch = response.text
```

### Option B: Local Models (Ollama, llama.cpp, etc.)

```python
# test_local_model.py
import requests
import json
from datasets import load_dataset

# 1. Make sure Ollama is running
# Install: https://ollama.ai
# Run: ollama serve
# Pull model: ollama pull codellama:13b

# 2. Load issue
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
issue = dataset[0]

# 3. Query local model
prompt = f"Fix this bug (output only a diff patch):\n{issue['problem_statement']}"

response = requests.post('http://localhost:11434/api/generate',
    json={
        "model": "codellama:13b",
        "prompt": prompt,
        "stream": False,
        "temperature": 0
    }
)

patch = response.json()['response']

# 4. Save patch (same format as API models)
with open("local_patches.jsonl", "w") as f:
    json.dump({
        "instance_id": issue['instance_id'],
        "model": "codellama-13b",
        "patch": patch
    }, f)
    f.write("\n")

print("Run: swebench run --patches local_patches.jsonl")
```

### Option C: Multiple Issues at Once

```python
# test_batch.py
import json
from anthropic import Anthropic
from datasets import load_dataset

client = Anthropic(api_key="your-key")
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

# Generate fixes for 5 issues
with open("batch_patches.jsonl", "w") as f:
    for i in range(5):
        issue = dataset[i]
        print(f"Processing {issue['instance_id']}...")

        # Get fix from Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": f"Fix this bug (output only diff):\n{issue['problem_statement']}"
            }]
        )

        # Write each patch as a separate line
        json.dump({
            "instance_id": issue['instance_id'],
            "model": "claude-3.5-sonnet",
            "patch": response.content[0].text
        }, f)
        f.write("\n")

print("Run: swebench run --patches batch_patches.jsonl")
```

## Step 5: Run the Evaluation

After generating patches with your model:

```bash
# Test your model's patches
swebench run --patches your_patches.jsonl

# You'll see:
# - Docker container starting
# - Patch being applied
# - Tests running
# - PASSED or FAILED result
```

## Understanding Results

Results are saved in `results/[timestamp]/`:

```
results/2024-01-20_14-30-00/
├── summary.html          # Open this in a browser
├── summary.json          # Overall statistics
└── django__django-11099/ # Per-instance details
    ├── patch.diff        # The patch that was applied
    ├── test_output.txt   # Why it passed/failed
    └── instance.json     # Detailed metadata
```

Common outcomes:
- **PASSED**: Your patch fixed the bug!
- **FAILED - Tests failed**: Patch applied but didn't fix the issue
- **FAILED - Patch failed**: Patch couldn't be applied (wrong file paths?)
- **TIMEOUT**: Evaluation took too long (60 min limit)

## Tips for Better Results

1. **Patch Format Matters**: The most common failure is incorrect patch format. The model must output exact unified diff format.

2. **Temperature = 0**: Use deterministic generation for more consistent results.

3. **Context Helps**: Some models do better if you include the actual file content:
   ```python
   # If the issue mentions a specific file
   prompt = f"""
   Issue: {issue['problem_statement']}

   Current code in django/http/response.py:
   {get_file_content_somehow()}

   Generate a diff patch to fix this issue.
   """
   ```

4. **Start Small**: Test on 1-2 issues before running hundreds.

5. **Check Your Patches**: Before running evaluation, check the `.jsonl` file:
   ```bash
   cat your_patches.jsonl | jq .
   ```

## Common Problems

### "Docker not running"
- **macOS**: Start Docker Desktop from Applications
- **Linux**: Run `sudo systemctl start docker`

### "Invalid patch format"
- Your model isn't outputting proper diff format
- Add format examples to your prompt
- Check that file paths match the repository structure

### "Timeout"
- Some issues take longer than others
- The default timeout is 60 minutes per issue
- Consider filtering to faster issues first

### "All tests fail"
- Your model might need more context
- Try including relevant code snippets in the prompt
- Some models perform better with step-by-step reasoning

## Advanced Usage

### Filter by Repository
```bash
# Test only Django issues
swebench run -d lite --subset "django__*"
```

### Run Specific Issues
```bash
# Test on known easier issues
swebench run -d lite --instances "astropy__astropy-6938,requests__requests-3362"
```

### Rerun Failed Instances
```bash
# After a run with failures
swebench run -d lite --rerun-failed ./results/[timestamp]
```

## Next Steps

1. **Benchmark Your Model**: Run on the full lite dataset (300 issues):
   ```bash
   swebench run -d lite
   ```

2. **Compare Models**: Test different models/prompts on the same issues

3. **Analyze Failures**: Look at failed patches to improve your prompts

4. **Share Results**: The community tracks model performance on the [SWE-bench leaderboard](https://www.swebench.com)

## Getting Help

- Run `swebench --help` for all options
- Check the [main repository](https://github.com/seconds-0/swe-bench-runner) for updates
- Report issues on GitHub

Remember: The goal isn't just to generate patches, but patches that actually fix the bugs and pass the tests!
