"""Dataset management for SWE-bench datasets from HuggingFace."""

from __future__ import annotations

import fnmatch
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    def fetch_dataset(self, dataset_name: str, force_download: bool = False) -> Any:
        """Fetch dataset from HuggingFace or cache."""
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(self.DATASET_MAPPING.keys())}")

        hf_dataset_name = self.DATASET_MAPPING[dataset_name]

        # Set cache directory for HuggingFace datasets
        os.environ['HF_DATASETS_CACHE'] = str(self.cache_dir)

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required for dataset auto-fetch. "
                "Install with: pip install datasets"
            )

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
        count: int | None = None,
        subset_pattern: str | None = None,
        random_seed: int | None = None,
        instances: list[str] | None = None,
        sample_percent: float | None = None,
        use_regex: bool = False
    ) -> list[dict[str, Any]]:
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
                pattern = re.compile(subset_pattern)
                dataset = dataset.filter(
                    lambda x: pattern.match(x['instance_id']) is not None
                )
            else:
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
        result_instances = []
        for item in dataset:
            result_instances.append({
                'instance_id': item['instance_id'],
                'patch': item.get('patch', item.get('diff', '')),  # Handle different field names
                # Include other metadata if needed
                'repo': item.get('repo', ''),
                'base_commit': item.get('base_commit', ''),
                'problem_statement': item.get('problem_statement', '')
            })

        return result_instances

    def save_as_jsonl(self, instances: list[dict[str, Any]], output_path: Path) -> None:
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
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)

    def get_dataset_info(self, dataset_name: str) -> dict[str, Any]:
        """Get information about a dataset without loading it."""
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        try:
            from datasets import load_dataset_builder
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required for dataset info. "
                "Install with: pip install datasets"
            )

        builder = load_dataset_builder(self.DATASET_MAPPING[dataset_name])
        info = builder.info

        return {
            'name': dataset_name,
            'total_instances': info.splits['test'].num_examples,
            'download_size_mb': info.download_size / 1024 / 1024,
            'dataset_size_mb': info.dataset_size / 1024 / 1024,
            'description': info.description
        }


def get_hf_token() -> str | None:
    """Get HuggingFace token from environment."""
    return os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')


def configure_hf_auth() -> bool:
    """Configure HuggingFace authentication if token available."""
    token = get_hf_token()
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            return True
        except ImportError:
            # huggingface_hub not available, skip auth
            pass
    return False
