#!/usr/bin/env python3
"""
Master script to run all experiments and generate all outputs.

This script runs the complete analysis pipeline:
1. Attack pattern analysis
2. Statistical tests
3. Defense evaluation
4. Parameter sweep
5. Paper artifacts

Everything works WITHOUT requiring HuggingFace models!
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script(script_name, description):
    """Run a Python script and report results."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"[OK] {description} completed successfully")
            # Show last few lines of output
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 10:
                print("\n...Last 10 lines:")
                for line in output_lines[-10:]:
                    print(line)
            else:
                print(result.stdout)
            return True
        else:
            print(f"[FAIL] {description} failed")
            print(f"Error output:\n{result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description} exceeded 5 minutes")
        return False
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        return False


def main():
    """Run all experiments."""
    print("\n" + "="*70)
    print(" "*15 + "COMPLETE EXPERIMENTAL PIPELINE")
    print(" "*15 + "Context Manipulation Attacks")
    print("="*70)
    
    start_time = datetime.now()
    
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {Path.cwd()}")
    print("\nThis will run all experiments and generate all outputs.")
    print("Estimated time: 3-5 minutes")
    print("\nPress Ctrl+C to cancel...")
    
    try:
        input("\nPress Enter to begin...")
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return
    
    # Track results
    results = {}
    
    # 1. Attack pattern analysis
    results['attack_patterns'] = run_script(
        'analyze_attack_patterns.py',
        'Attack Pattern Analysis & Visualization'
    )
    
    # 2. Statistical analysis
    results['statistics'] = run_script(
        'statistical_analysis.py',
        'Statistical Hypothesis Testing'
    )
    
    # 3. Defense evaluation
    results['defense'] = run_script(
        'defense_evaluation.py',
        'Defense Mechanism Evaluation'
    )
    
    # 4. Parameter sweep
    results['param_sweep'] = run_script(
        'parameter_sweep.py',
        'Parameter Sweep Analysis'
    )
    
    # 5. Paper artifacts
    results['paper_artifacts'] = run_script(
        'generate_paper_artifacts.py',
        'Paper-Ready Artifact Generation'
    )
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE COMPLETE!")
    print("="*70)
    
    print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration.total_seconds():.1f} seconds")
    
    print("\n" + "-"*70)
    print("RESULTS SUMMARY")
    print("-"*70)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name.replace('_', ' ').title()}")
    
    print(f"\nSuccess rate: {success_count}/{total_count} " +
          f"({success_count/total_count*100:.0f}%)")
    
    # List generated files
    print("\n" + "-"*70)
    print("GENERATED FILES")
    print("-"*70)
    
    output_dir = Path("outputs")
    if output_dir.exists():
        print("\nMain outputs/:")
        for file in sorted(output_dir.iterdir()):
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"  - {file.name:<40} ({size_kb:>6.1f} KB)")
        
        paper_dir = output_dir / "paper_figures"
        if paper_dir.exists():
            print(f"\nPaper figures (outputs/paper_figures/):")
            for file in sorted(paper_dir.iterdir()):
                if file.is_file():
                    size_kb = file.stat().st_size / 1024
                    print(f"  - {file.name:<40} ({size_kb:>6.1f} KB)")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
1. Review visualizations:
   - outputs/single_attack_analysis.png
   - outputs/multi_attack_comparison.png
   - outputs/statistical_distributions.png
   - outputs/parameter_sweep.png

2. Review reports:
   - outputs/statistical_report.txt
   - outputs/defense_evaluation.txt
   - outputs/paper_results_summary.txt

3. For paper writing:
   - outputs/paper_figures/ (3 PDF figures)
   - outputs/paper_results_table.tex
   - outputs/paper_statistics_table.tex

4. Interactive exploration:
   python interactive_demo.py

5. When HuggingFace access is available:
   python run_experiment.py --model gpt2 --attack all
""")
    
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
    except Exception as e:
        print(f"\n\nPipeline error: {e}")
        import traceback
        traceback.print_exc()

