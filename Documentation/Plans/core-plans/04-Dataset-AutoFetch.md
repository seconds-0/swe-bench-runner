# Dataset Auto-Fetch - Automatic SWE-bench Dataset Loading

**Task ID**: FEAT-Dataset-AutoFetch
**Status**: Not Started
**Priority**: High - User Experience Enhancement
**Estimated Effort**: 4-6 hours
**Dependencies**: MVP-BasicOutput-v1 (completed)

## Problem Statement

Currently, users need to:
1. Understand what patches are
2. Create or obtain JSONL files with patches
3. Know the correct format for patches

This is a major barrier to entry. Users should be able to just run:
```bash
# Run specific instances
swebench run --dataset lite --instances django__django-12345,django__django-67890

# Run random 10% of dataset
swebench run --dataset lite --sample 10%

# Run 5 random instances with seed
swebench run --dataset lite --count 5 --random-seed 42

# Run all Django instances using regex
swebench run --dataset lite --subset "django__django-1[0-9]{4}" --regex

# Rerun failed instances from previous run
swebench run --rerun-failed ./results/latest
```

And the tool should automatically fetch and run the requested instances from the SWE-bench dataset.

## Research Phase

### Research Checklist
- [x] How does HuggingFace datasets API work?
- [x] What's the exact structure of SWE-bench datasets on HuggingFace?
- [x] Authentication requirements for HuggingFace API
- [x] Dataset download sizes and caching strategy
- [x] How to extract patches from dataset entries
- [x] Random sampling implementation
- [x] Subset filtering patterns

### Research Findings

#### 1. Dataset Structure (VERIFIED)
- **Available datasets**:
  - `princeton-nlp/SWE-bench_Lite` (300 instances, 1.2MB download)
  - `princeton-nlp/SWE-bench_Verified` (500 instances)
  - `princeton-nlp/SWE-bench` (2294 instances)
- **Key fields**:
  - `instance_id`: e.g., "django__django-12345"
  - `patch`: The actual diff/patch as a string
  - `repo`: e.g., "django/django"
  - `base_commit`: SHA to checkout
  - `problem_statement`: Issue description
  - `test_patch`: Tests added by the PR
- **Split**: All instances are in 'test' split

#### 2. Authentication (VERIFIED)
- **Anonymous access**: ✅ Works for all public SWE-bench datasets
- **No authentication required** for downloading
- **Rate limits**: No explicit limits for dataset downloads
- **Token optional**: Only needed for private datasets

#### 3. Caching (VERIFIED)
- **Default location**: `~/.cache/huggingface/datasets`
- **Size**: SWE-bench_Lite is only 3.6MB on disk
- **Custom cache**: Can set via `HF_DATASETS_CACHE` env var
- **Automatic**: HuggingFace handles download caching

#### 4. Implementation Details (VERIFIED)
- **Loading**: `load_dataset("princeton-nlp/SWE-bench_Lite", split="test")`
- **Filtering**: Works with list comprehension or dataset.filter()
- **Sampling**: Can use dataset.select() with indices
- **Django instances**: 114 out of 300 in Lite dataset

#### 5. Performance Notes
- **All datasets are tiny**: Even full dataset is <20MB
- **No memory concerns**: Load everything, filter in memory
- **Caching works great**: HuggingFace handles it automatically

### Research Tasks (COMPLETED)

#### 1. HuggingFace API Research
```python
# Test script to understand the API
from datasets import load_dataset

# Basic loading
dataset = load_dataset("princeton-nlp/SWE-bench_Lite")
print(f"Dataset info: {dataset}")
print(f"First entry: {dataset['test'][0]}")
print(f"Available fields: {dataset['test'].features}")

# Check download size
print(f"Dataset size: {dataset['test'].info.download_size}")

# Test filtering
django_instances = dataset['test'].filter(lambda x: x['instance_id'].startswith('django__'))
print(f"Django instances: {len(django_instances)}")

# Test random sampling
import random
indices = random.sample(range(len(dataset['test'])), 5)
samples = dataset['test'].select(indices)
```

#### 2. Authentication Research
- HuggingFace token requirements
- Rate limits (anonymous vs authenticated)
- Environment variable conventions (HF_TOKEN)

#### 3. Dataset Structure Research
- Available datasets: SWE-bench_Lite, SWE-bench_Verified, SWE-bench
- Fields in each dataset entry
- How patches are stored (patch field? diff field?)
- Instance ID format and patterns

#### 4. Caching Strategy Research
- HuggingFace datasets default cache location
- How to integrate with our cache directory
- Checksum verification
- Offline mode support

## Proposed Solution

### 1. Core Components

```python
# datasets.py - New module for dataset management
from pathlib import Path
from typing import List, Optional, Dict, Any
import random
from datasets import load_dataset, Dataset
import json

class DatasetManager:
    """Manages SWE-bench dataset downloads and access."""

    DATASET_MAPPING = {
        'lite': 'princeton-nlp/SWE-bench_Lite',
        'verified': 'princeton-nlp/SWE-bench_Verified',
        'full': 'princeton-nlp/SWE-bench'
    }

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "datasets"
        self.cache_dir.mkdir(exist_ok=True)

    def fetch_dataset(self, dataset_name: str, force_download: bool = False) -> Dataset:
        """Fetch dataset from HuggingFace or cache."""
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(self.DATASET_MAPPING.keys())}")

        hf_dataset_name = self.DATASET_MAPPING[dataset_name]

        # Set cache directory for HuggingFace datasets
        import os
        os.environ['HF_DATASETS_CACHE'] = str(self.cache_dir)

        # Load dataset (will download if not cached)
        dataset = load_dataset(
            hf_dataset_name,
            split='test',  # SWE-bench uses 'test' split
            download_mode='force_redownload' if force_download else 'reuse_dataset_if_exists'
        )

        return dataset

    def get_instances(
        self,
        dataset_name: str,
        count: Optional[int] = None,
        subset_pattern: Optional[str] = None,
        random_seed: Optional[int] = None,
        instances: Optional[List[str]] = None,  # NEW: specific instance IDs
        sample_percent: Optional[float] = None,  # NEW: percentage sampling
        use_regex: bool = False  # NEW: regex support
    ) -> List[Dict[str, Any]]:
        """Get instances from dataset with optional filtering."""
        dataset = self.fetch_dataset(dataset_name)

        # Filter by specific instance IDs first
        if instances:
            dataset = dataset.filter(
                lambda x: x['instance_id'] in instances
            )

        # Apply pattern filtering
        if subset_pattern:
            if use_regex:
                import re
                pattern = re.compile(subset_pattern)
                dataset = dataset.filter(
                    lambda x: pattern.match(x['instance_id']) is not None
                )
            else:
                import fnmatch
                dataset = dataset.filter(
                    lambda x: fnmatch.fnmatch(x['instance_id'], subset_pattern)
                )

        # Handle percentage sampling
        if sample_percent:
            count = int(len(dataset) * sample_percent / 100)
            if count == 0:
                count = 1  # At least one instance

        # Apply random sampling if count specified
        if count and count < len(dataset):
            if random_seed is not None:
                random.seed(random_seed)
            indices = random.sample(range(len(dataset)), count)
            dataset = dataset.select(indices)
        elif count and count >= len(dataset):
            print(f"Warning: Requested {count} instances but dataset only has {len(dataset)}")

        # Convert to list of dicts
        instances = []
        for item in dataset:
            instances.append({
                'instance_id': item['instance_id'],
                'patch': item.get('patch', item.get('diff', '')),  # Handle different field names
                # Include other metadata if needed
                'repo': item.get('repo', ''),
                'base_commit': item.get('base_commit', ''),
                'problem_statement': item.get('problem_statement', '')
            })

        return instances

    def save_as_jsonl(self, instances: List[Dict[str, Any]], output_path: Path) -> None:
        """Save instances as JSONL file for compatibility."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for instance in instances:
                # Only save instance_id and patch for compatibility
                json.dump({
                    'instance_id': instance['instance_id'],
                    'patch': instance['patch']
                }, f)
                f.write('\n')

    def cleanup_temp_files(self) -> None:
        """Clean up all temporary JSONL files."""
        temp_dir = self.cache_dir.parent / "temp"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            temp_dir.mkdir()

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset without loading it."""
        from datasets import load_dataset_builder

        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        builder = load_dataset_builder(self.DATASET_MAPPING[dataset_name])
        info = builder.info

        return {
            'name': dataset_name,
            'total_instances': info.splits['test'].num_examples,
            'download_size_mb': info.download_size / 1024 / 1024,
            'dataset_size_mb': info.dataset_size / 1024 / 1024,
            'description': info.description
        }
```

### 2. CLI Integration

```python
# Update cli.py run command
@cli.command()
@click.option(
    '-d', '--dataset',
    type=click.Choice(['lite', 'verified', 'full']),
    help='Use SWE-bench dataset (auto-downloads from HuggingFace)'
)
@click.option(
    '--instances',
    help='Comma-separated list of specific instance IDs to run'
)
@click.option(
    '--count',
    type=int,
    help='Number of instances to run (random sample)'
)
@click.option(
    '--sample',
    help='Random percentage of instances (e.g., "10%" or "random-seed=42")'
)
@click.option(
    '--subset',
    help='Filter instances by pattern (e.g., "django__*", "requests__*")'
)
@click.option(
    '--regex',
    is_flag=True,
    help='Treat --subset as regex instead of glob pattern'
)
@click.option(
    '--patches',
    type=click.Path(exists=True),
    help='Path to JSONL file containing patches (alternative to --dataset)'
)
@click.option(
    '--rerun-failed',
    type=click.Path(exists=True),
    help='Rerun failed instances from a previous run directory'
)
def run(dataset, instances, count, sample, subset, regex, patches, rerun_failed, ...):
    """Run SWE-bench evaluation."""

    # Handle --rerun-failed first
    if rerun_failed:
        failed_instances = load_failed_instances(Path(rerun_failed))
        if not failed_instances:
            click.echo("✅ No failed instances to rerun")
            return
        # Convert to instance list
        instances = ','.join(failed_instances)
        click.echo(f"🔄 Rerunning {len(failed_instances)} failed instances")

    # Priority: explicit patches file > dataset selection
    if not patches and not dataset and not rerun_failed:
        # Try auto-detection
        detected = detect_patches_file()
        if detected:
            patches = detected
            click.echo(f"💡 Using {patches}")
        else:
            click.echo("❌ No patches file found. Use --patches or --dataset")
            ctx.exit(1)

    # Handle dataset fetching
    if dataset and not patches:
        from .datasets import DatasetManager

        click.echo(f"📥 Loading {dataset} dataset from HuggingFace...")

        manager = DatasetManager(get_cache_dir())

        # Parse --sample flag
        sample_percent = None
        random_seed = None
        if sample:
            if sample.endswith('%'):
                sample_percent = float(sample[:-1])
            elif '=' in sample:
                # Handle "random-seed=42" format
                parts = sample.split('=')
                if parts[0] == 'random-seed':
                    random_seed = int(parts[1])

        # Parse --instances flag
        instance_list = None
        if instances:
            instance_list = [i.strip() for i in instances.split(',')]

        instances = manager.get_instances(
            dataset_name=dataset,
            instances=instance_list,
            count=count,
            sample_percent=sample_percent,
            subset_pattern=subset,
            use_regex=regex,
            random_seed=random_seed
        )

        if not instances:
            click.echo("❌ No instances matched your criteria")
            ctx.exit(1)

        # Save to temporary JSONL for compatibility with existing code
        temp_patches = get_cache_dir() / "temp" / f"{dataset}_{uuid.uuid4().hex[:8]}.jsonl"
        manager.save_as_jsonl(instances, temp_patches)
        patches = temp_patches

        # Clean up old temp files
        manager.cleanup_temp_files()

        click.echo(f"✅ Loaded {len(instances)} instances from {dataset} dataset")
        if instance_list:
            click.echo(f"   Specific instances: {', '.join(instance_list[:3])}{'...' if len(instance_list) > 3 else ''}")
        if sample_percent:
            click.echo(f"   Random {sample_percent}% sample")
        if count:
            click.echo(f"   Limited to {count} instances")
        if subset:
            click.echo(f"   Filtered by {'regex' if regex else 'pattern'}: {subset}")
```

### 3. Helper Functions

```python
# In cli.py or utils.py
def load_failed_instances(results_dir: Path) -> List[str]:
    """Load instance IDs that failed from a previous run."""
    failed_instances = []

    # Check for summary.json files
    for summary_file in results_dir.rglob("summary.json"):
        with open(summary_file) as f:
            data = json.load(f)
            if not data.get('passed', True):
                failed_instances.append(data['instance_id'])

    # Alternative: check for FAILED status files
    if not failed_instances:
        for failed_file in results_dir.rglob("FAILED"):
            instance_id = failed_file.parent.name
            failed_instances.append(instance_id)

    return failed_instances

# Add info command for dataset information
@cli.command()
@click.option(
    '-d', '--dataset',
    type=click.Choice(['lite', 'verified', 'full']),
    required=True,
    help='Dataset to get information about'
)
def info(dataset):
    """Get information about a SWE-bench dataset."""
    from .datasets import DatasetManager

    manager = DatasetManager(get_cache_dir())
    info = manager.get_dataset_info(dataset)

    click.echo(f"\n📊 SWE-bench {dataset} dataset:")
    click.echo(f"   Total instances: {info['total_instances']:,}")
    click.echo(f"   Download size: {info['download_size_mb']:.1f} MB")
    click.echo(f"   On-disk size: {info['dataset_size_mb']:.1f} MB")
    if info['description']:
        click.echo(f"   Description: {info['description'][:100]}...")
```

### 4. Environment Variables & Authentication

```python
# In datasets.py
import os

def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment."""
    return os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

def configure_hf_auth():
    """Configure HuggingFace authentication if token available."""
    token = get_hf_token()
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True
    return False
```

### 5. Error Handling

```python
# Simple, clear error messages
try:
    dataset = load_dataset(...)
except Exception as e:
    if "401" in str(e):
        click.echo("❌ Authentication required for this dataset")
        click.echo("   Set HF_TOKEN environment variable with your HuggingFace token")
        click.echo("   Get a token at: https://huggingface.co/settings/tokens")
    elif "Connection" in str(e):
        click.echo("❌ Network error downloading dataset")
        click.echo("   Check your internet connection")
    else:
        click.echo(f"❌ Failed to load dataset: {e}")
        raise
```

## Implementation Checklist

- [x] Research Phase
  - [x] Test HuggingFace datasets API locally
  - [x] Document exact dataset structure for each variant
  - [x] Test authentication flow
  - [x] Measure download sizes and times
  - [x] Verify patch field names
- [ ] Core Implementation
  - [ ] Create `datasets.py` module
  - [ ] Implement `DatasetManager` class with all methods:
    - [ ] `fetch_dataset()` - simple loading
    - [ ] `get_instances()` with all filtering options
    - [ ] `save_as_jsonl()` with directory creation
    - [ ] `cleanup_temp_files()` - simple cleanup
    - [ ] `get_dataset_info()` for dataset information
  - [ ] Implement all filtering options:
    - [ ] Specific instance IDs (--instances)
    - [ ] Percentage sampling (--sample)
    - [ ] Pattern filtering with glob (--subset)
    - [ ] Regex support (--regex)
    - [ ] Count limiting (--count)

- [ ] CLI Integration
  - [ ] Add all new CLI options:
    - [ ] `-d, --dataset` (short flag support)
    - [ ] `--instances` (comma-separated list)
    - [ ] `--count` (random N instances)
    - [ ] `--sample` (percentage with optional seed)
    - [ ] `--subset` (glob/regex pattern)
    - [ ] `--regex` (flag for regex mode)
    - [ ] `--rerun-failed` (rerun from previous)
  - [ ] Implement `load_failed_instances()` helper
  - [ ] Add `info` command for dataset information
  - [ ] Update run command logic with all options
  - [ ] Maintain backward compatibility with --patches

- [ ] Error Handling
  - [ ] Network errors with retry suggestions
  - [ ] Invalid instance IDs with helpful messages
  - [ ] Pattern matching errors with examples
  - [ ] Clear messaging about what's happening

- [ ] Resource Management
  - [ ] Simple temp file cleanup on start
  - [ ] Let HuggingFace handle caching

- [ ] Testing
  - [ ] Unit tests for DatasetManager
  - [ ] Test all filtering combinations
  - [ ] Test percentage sampling accuracy
  - [ ] Test regex vs glob patterns
  - [ ] Test specific instance selection
  - [ ] Test temp file cleanup
  - [ ] Mock HuggingFace API calls
  - [ ] Integration tests with CLI

- [ ] Documentation
  - [ ] Update README with all dataset examples
  - [ ] Document authentication setup
  - [ ] Add examples to --help text
  - [ ] Add troubleshooting section

## Test Plan

### Unit Tests
```python
def test_dataset_manager_init():
    """Test DatasetManager initialization."""
    manager = DatasetManager(Path("/tmp/test"))
    assert manager.cache_dir.exists()

def test_specific_instances():
    """Test filtering by specific instance IDs."""
    instances = manager.get_instances(
        'lite',
        instances=['django__django-123', 'django__django-456']
    )
    assert len(instances) == 2
    assert all(i['instance_id'] in ['django__django-123', 'django__django-456']
               for i in instances)

def test_percentage_sampling():
    """Test percentage-based sampling."""
    # Test 10% sampling
    instances = manager.get_instances('lite', sample_percent=10.0)
    assert len(instances) == 30  # 10% of 300

def test_regex_filtering():
    """Test regex pattern matching."""
    instances = manager.get_instances(
        'lite',
        subset_pattern=r'django__django-1[0-9]{4}',
        use_regex=True
    )
    assert all(re.match(r'django__django-1[0-9]{4}', i['instance_id'])
               for i in instances)

def test_glob_filtering():
    """Test glob pattern matching."""
    instances = manager.get_instances(
        'lite',
        subset_pattern='django__*',
        use_regex=False
    )
    assert all(i['instance_id'].startswith('django__') for i in instances)

def test_combined_filters():
    """Test combining multiple filters."""
    instances = manager.get_instances(
        'lite',
        subset_pattern='django__*',
        count=5,
        random_seed=42
    )
    assert len(instances) == 5
    assert all(i['instance_id'].startswith('django__') for i in instances)

def test_temp_file_cleanup():
    """Test temporary file cleanup."""
    # Create temp files
    temp_dir = manager.cache_dir.parent / "temp"
    old_file = temp_dir / "old_file.jsonl"
    old_file.touch()

    manager.cleanup_temp_files()
    assert not old_file.exists()
```

### Integration Tests
```python
def test_cli_dataset_option(runner):
    """Test CLI with dataset option."""
    result = runner.invoke(cli, ['run', '-d', 'lite', '--count', '1'])
    assert result.exit_code == 0
    assert "Loading lite dataset" in result.output

def test_cli_specific_instances(runner):
    """Test running specific instances."""
    result = runner.invoke(cli, [
        'run', '-d', 'lite',
        '--instances', 'django__django-123,django__django-456'
    ])
    assert "Specific instances:" in result.output

def test_cli_percentage_sample(runner):
    """Test percentage sampling."""
    result = runner.invoke(cli, ['run', '-d', 'lite', '--sample', '10%'])
    assert "Random 10.0% sample" in result.output

def test_cli_regex_subset(runner):
    """Test regex filtering."""
    result = runner.invoke(cli, [
        'run', '-d', 'lite',
        '--subset', 'django__django-1[0-9]{4}',
        '--regex'
    ])
    assert "Filtered by regex:" in result.output

def test_cli_rerun_failed(runner, tmp_path):
    """Test rerunning failed instances."""
    # Create fake failed results
    failed_dir = tmp_path / "results" / "instance1"
    failed_dir.mkdir(parents=True)
    (failed_dir / "FAILED").touch()

    result = runner.invoke(cli, ['run', '--rerun-failed', str(tmp_path)])
    assert "Rerunning 1 failed instances" in result.output

def test_cli_info_command(runner):
    """Test dataset info command."""
    result = runner.invoke(cli, ['info', '-d', 'lite'])
    assert result.exit_code == 0
    assert "Total instances: 300" in result.output
    assert "Download size:" in result.output
```

## Success Criteria

1. ✅ Users can run `swebench run --dataset lite` without any setup
2. ✅ Users can run specific instances: `swebench run -d lite --instances django__django-123`
3. ✅ Users can run percentage samples: `swebench run -d lite --sample 10%`
4. ✅ Users can filter with regex: `swebench run -d lite --subset "django.*1[0-9]{4}" --regex`
5. ✅ Users can rerun failures: `swebench run --rerun-failed ./results/latest`
6. ✅ Dataset downloads are cached and reused
7. ✅ Random sampling works with reproducible seeds
8. ✅ Subset filtering works with both glob and regex patterns
9. ✅ Clear progress indication during download
10. ✅ Temp files are cleaned up automatically
11. ✅ Helpful error messages for common failures
12. ✅ Works offline if dataset is cached
13. ✅ Maintains compatibility with existing --patches workflow
14. ✅ Short flag `-d` works for dataset selection
15. ✅ Dataset info available via `swebench info -d lite`

## Dependencies

- `datasets` library from HuggingFace
- `huggingface_hub` for authentication
- Existing cache infrastructure
- Internet connection for first download

## Questions/Uncertainties

### Blocking
None - all critical questions have been answered through research.

### Non-blocking
1. **Progress bar integration**: HuggingFace shows its own progress, may need to capture/redirect
2. **Full dataset memory**: Need to verify actual memory usage for 2294 instances
3. **Offline mode detection**: How to gracefully handle when user is offline
4. **Dataset versioning**: Should we pin to specific dataset versions for reproducibility?
5. **Multimodal support**: SWE-bench_Multimodal exists - should we support it?

## Acceptable Tradeoffs

1. **Temporary JSONL files**: Creating temp files maintains compatibility with existing code
2. **Full dataset download**: Can't download individual instances, must get whole dataset
3. **Memory usage**: Loading full dataset into memory is OK for lite/verified

## Notes

- HuggingFace datasets library handles caching automatically
- This feature delivers the "magic" - zero setup evaluation
- Aligns with PRD vision: "Run any subset of SWE-bench with one clear command"
- Future enhancements:
  - `--offline` flag for airplane mode
  - Streaming mode for very large datasets
  - Support for SWE-bench_Multimodal dataset
  - Dataset version pinning for reproducibility
- The `info` command helps users understand what they're downloading before they commit
