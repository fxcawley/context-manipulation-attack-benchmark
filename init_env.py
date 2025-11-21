#!/usr/bin/env python3
"""
Environment initialization for context manipulation attack experiments.
Sets up paths and ensures reproducibility.
"""

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch


def set_source_root():
    """
    Find and return the project root directory.
    Adds src/ to Python path for imports.
    """
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    
    # Add project root and src to path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    return project_root


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_environment(seed=42):
    """
    Complete environment setup: paths and seeds.
    
    Args:
        seed (int): Random seed value
        
    Returns:
        Path: Project root directory
    """
    project_root = set_source_root()
    set_random_seeds(seed)
    return project_root


if __name__ == "__main__":
    root = setup_environment()
    print(f"Project root: {root}")
    print(f"Python path: {sys.path[:3]}")

