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
swebench run --dataset lite --count 5
```

And the tool should automatically fetch and run 5 instances from the SWE-bench lite dataset.

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
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get instances from dataset with optional filtering."""
        dataset = self.fetch_dataset(dataset_name)

        # Apply subset filtering if specified
        if subset_pattern:
            import fnmatch
            dataset = dataset.filter(
                lambda x: fnmatch.fnmatch(x['instance_id'], subset_pattern)
            )

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
        with open(output_path, 'w') as f:
            for instance in instances:
                # Only save instance_id and patch for compatibility
                json.dump({
                    'instance_id': instance['instance_id'],
                    'patch': instance['patch']
                }, f)
                f.write('\n')
```

### 2. CLI Integration

```python
# Update cli.py run command
@cli.command()
@click.option(
    '--dataset',
    type=click.Choice(['lite', 'verified', 'full']),
    help='Use SWE-bench dataset (auto-downloads from HuggingFace)'
)
@click.option(
    '--count',
    type=int,
    help='Number of instances to run (random sample)'
)
@click.option(
    '--subset',
    help='Filter instances by pattern (e.g., "django__*", "requests__*")'
)
@click.option(
    '--random-seed',
    type=int,
    help='Random seed for reproducible sampling'
)
@click.option(
    '--patches',
    type=click.Path(exists=True),
    help='Path to JSONL file containing patches (alternative to --dataset)'
)
def run(dataset, count, subset, random_seed, patches, ...):
    """Run SWE-bench evaluation."""

    # Priority: explicit patches file > dataset selection
    if not patches and not dataset:
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
        instances = manager.get_instances(
            dataset_name=dataset,
            count=count,
            subset_pattern=subset,
            random_seed=random_seed
        )

        # Save to temporary JSONL for compatibility with existing code
        temp_patches = get_cache_dir() / "temp" / f"{dataset}_{uuid.uuid4().hex[:8]}.jsonl"
        temp_patches.parent.mkdir(exist_ok=True)

        manager.save_as_jsonl(instances, temp_patches)
        patches = temp_patches

        click.echo(f"✅ Loaded {len(instances)} instances from {dataset} dataset")
        if count:
            click.echo(f"   Random sample with seed: {random_seed or 'auto'}")
        if subset:
            click.echo(f"   Filtered by pattern: {subset}")
```

### 3. Environment Variables & Authentication

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

### 4. Error Handling

```python
# Enhanced error messages
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
        click.echo("   Use --offline if you have cached data")
    else:
        raise
```

## Implementation Checklist

- [ ] Research Phase
  - [ ] Test HuggingFace datasets API locally
  - [ ] Document exact dataset structure for each variant
  - [ ] Test authentication flow
  - [ ] Measure download sizes and times
  - [ ] Verify patch field names

- [ ] Core Implementation
  - [ ] Create `datasets.py` module
  - [ ] Implement `DatasetManager` class
  - [ ] Add dataset fetching logic
  - [ ] Implement filtering by pattern
  - [ ] Implement random sampling
  - [ ] Add progress bars for downloads

- [ ] CLI Integration
  - [ ] Add new CLI options (--dataset, --count, --subset, --random-seed)
  - [ ] Update run command logic
  - [ ] Maintain backward compatibility with --patches
  - [ ] Add helpful error messages

- [ ] Authentication
  - [ ] Support HF_TOKEN environment variable
  - [ ] Handle authenticated vs anonymous access
  - [ ] Clear error messages for auth failures

- [ ] Caching
  - [ ] Integrate with existing cache directory
  - [ ] Support offline mode
  - [ ] Add cache info to clean command

- [ ] Testing
  - [ ] Unit tests for DatasetManager
  - [ ] Mock HuggingFace API calls
  - [ ] Test filtering and sampling
  - [ ] Test error scenarios
  - [ ] Integration tests with CLI

- [ ] Documentation
  - [ ] Update README with dataset examples
  - [ ] Document authentication setup
  - [ ] Add examples to --help text

## Test Plan

### Unit Tests
```python
def test_dataset_manager_init():
    """Test DatasetManager initialization."""
    manager = DatasetManager(Path("/tmp/test"))
    assert manager.cache_dir.exists()

def test_get_instances_with_filter():
    """Test filtering instances by pattern."""
    # Mock dataset
    mock_dataset = [
        {'instance_id': 'django__django-123', 'patch': '...'},
        {'instance_id': 'requests__requests-456', 'patch': '...'}
    ]
    instances = manager.get_instances('lite', subset_pattern='django__*')
    assert len(instances) == 1
    assert instances[0]['instance_id'].startswith('django__')

def test_random_sampling_deterministic():
    """Test random sampling with seed is deterministic."""
    instances1 = manager.get_instances('lite', count=5, random_seed=42)
    instances2 = manager.get_instances('lite', count=5, random_seed=42)
    assert instances1 == instances2
```

### Integration Tests
```python
def test_cli_dataset_option(runner):
    """Test CLI with dataset option."""
    result = runner.invoke(cli, ['run', '--dataset', 'lite', '--count', '1'])
    assert result.exit_code == 0
    assert "Loading lite dataset" in result.output
```

## Success Criteria

1. ✅ Users can run `swebench run --dataset lite` without any setup
2. ✅ Dataset downloads are cached and reused
3. ✅ Random sampling works with reproducible seeds
4. ✅ Subset filtering works with glob patterns
5. ✅ Clear progress indication during download
6. ✅ Helpful error messages for common failures
7. ✅ Works offline if dataset is cached
8. ✅ Maintains compatibility with existing --patches workflow

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
2. **Large dataset handling**: Full SWE-bench dataset size unknown (need to verify)
3. **Offline mode detection**: How to gracefully handle when user is offline

## Acceptable Tradeoffs

1. **Temporary JSONL files**: Creating temp files maintains compatibility with existing code
2. **Full dataset download**: Can't download individual instances, must get whole dataset
3. **Memory usage**: Loading full dataset into memory is OK for lite/verified

## Notes

- HuggingFace datasets library handles caching automatically
- Consider adding --offline flag for airplane mode
- Future enhancement: streaming mode for very large datasets
- Consider adding dataset info command: `swebench info --dataset lite`
