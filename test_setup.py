#!/usr/bin/env python3
"""Quick test to verify setup is working."""

print("Testing imports...")

try:
    from init_env import setup_environment
    print("[OK] init_env imported")
    
    from src.models import load_model
    print("[OK] src.models imported")
    
    from src.attack import FalseConversationInjection, GaslightingAttack, IterativeContextPoisoning
    print("[OK] src.attack imported")
    
    from src.eval import evaluate_response, breakdown_detection
    print("[OK] src.eval imported")
    
    print("\n[OK] All imports successful!")
    print("\nInitializing environment...")
    
    root = setup_environment(seed=42)
    print(f"[OK] Environment initialized at: {root}")
    
    print("\n[SUCCESS] Setup test PASSED!")
    
except Exception as e:
    print(f"\n[FAIL] Setup test FAILED: {e}")
    import traceback
    traceback.print_exc()

