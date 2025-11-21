#!/usr/bin/env python3
"""
Statistical analysis of context manipulation attacks.

Performs rigorous statistical testing and generates comprehensive reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import List, Dict, Tuple

from init_env import setup_environment
from src.attack.attack_simulator import generate_synthetic_dataset
from src.eval import evaluate_response


def calculate_effect_sizes(coherence_baseline: float, coherence_attacked: List[float]) -> Dict:
    """
    Calculate statistical effect sizes for attack impact.
    
    Args:
        coherence_baseline: Baseline coherence score
        coherence_attacked: List of coherence scores under attack
        
    Returns:
        Dictionary of effect size metrics
    """
    coherence_attacked = np.array(coherence_attacked)
    
    # Cohen's d
    pooled_std = np.std(coherence_attacked)
    cohens_d = (coherence_baseline - np.mean(coherence_attacked)) / pooled_std if pooled_std > 0 else 0
    
    # Percentage change
    pct_change = ((coherence_baseline - np.mean(coherence_attacked)) / coherence_baseline * 100) if coherence_baseline > 0 else 0
    
    return {
        'cohens_d': cohens_d,
        'percent_change': pct_change,
        'mean_attacked': np.mean(coherence_attacked),
        'std_attacked': np.std(coherence_attacked),
        'min_attacked': np.min(coherence_attacked),
        'max_attacked': np.max(coherence_attacked)
    }


def analyze_degradation_trajectory(results) -> Dict:
    """
    Analyze the trajectory of coherence degradation.
    
    Fits exponential decay model: y = a * exp(-b * x) + c
    
    Args:
        results: List of AttackResult objects
        
    Returns:
        Dictionary with trajectory analysis
    """
    iterations = []
    coherence_scores = []
    
    for result in results:
        m = evaluate_response(result.model_response)
        iterations.append(result.iteration)
        coherence_scores.append(m.coherence_score)
    
    iterations = np.array(iterations)
    coherence_scores = np.array(coherence_scores)
    
    # Fit linear model for rate of change
    if len(iterations) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(iterations, coherence_scores)
    else:
        slope, intercept, r_value, p_value, std_err = 0, 0, 0, 1, 0
    
    # Calculate time to breakdown (coherence < 0.5)
    breakdown_iter = None
    for i, score in enumerate(coherence_scores):
        if score < 0.5:
            breakdown_iter = i + 1
            break
    
    return {
        'degradation_rate': slope,  # Per iteration
        'initial_coherence': coherence_scores[0] if len(coherence_scores) > 0 else 0,
        'final_coherence': coherence_scores[-1] if len(coherence_scores) > 0 else 0,
        'total_degradation': coherence_scores[0] - coherence_scores[-1] if len(coherence_scores) > 0 else 0,
        'r_squared': r_value**2,
        'p_value': p_value,
        'iterations_to_breakdown': breakdown_iter,
        'trajectory': list(zip(iterations, coherence_scores))
    }


def compare_attack_variants(dataset: List[Dict]) -> pd.DataFrame:
    """
    Statistical comparison of different attack variants.
    
    Args:
        dataset: List of attack results
        
    Returns:
        DataFrame with comparative statistics
    """
    comparisons = []
    
    for item in dataset:
        results = item['results']
        trajectory = analyze_degradation_trajectory(results)
        
        # Calculate breakdown metrics
        breakdown_count = 0
        for result in results:
            m = evaluate_response(result.model_response)
            if m.breakdown_detected:
                breakdown_count += 1
        
        comparisons.append({
            'prompt': item['prompt'][:40] + '...',
            'attack_type': item.get('attack_type', 'iterative'),
            'iterations': len(results),
            'initial_coherence': trajectory['initial_coherence'],
            'final_coherence': trajectory['final_coherence'],
            'degradation_rate': trajectory['degradation_rate'],
            'total_degradation': trajectory['total_degradation'],
            'breakdown_rate': breakdown_count / len(results),
            'iter_to_breakdown': trajectory['iterations_to_breakdown'],
            'r_squared': trajectory['r_squared']
        })
    
    return pd.DataFrame(comparisons)


def hypothesis_testing(dataset: List[Dict]) -> Dict:
    """
    Perform hypothesis tests on attack effectiveness.
    
    H0: Attacks do not significantly degrade coherence
    H1: Attacks significantly degrade coherence (p < 0.05)
    
    Args:
        dataset: List of attack results
        
    Returns:
        Dictionary with test results
    """
    initial_coherence = []
    final_coherence = []
    
    for item in dataset:
        results = item['results']
        if len(results) > 0:
            m_initial = evaluate_response(results[0].model_response)
            m_final = evaluate_response(results[-1].model_response)
            initial_coherence.append(m_initial.coherence_score)
            final_coherence.append(m_final.coherence_score)
    
    initial_coherence = np.array(initial_coherence)
    final_coherence = np.array(final_coherence)
    
    # Paired t-test (same prompts, before vs after)
    t_stat, p_value_paired = stats.ttest_rel(initial_coherence, final_coherence)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, p_value_wilcoxon = stats.wilcoxon(initial_coherence, final_coherence)
    
    # Effect size
    effect_size = np.mean(initial_coherence - final_coherence) / np.std(initial_coherence - final_coherence)
    
    return {
        'paired_t_test': {
            't_statistic': t_stat,
            'p_value': p_value_paired,
            'significant': p_value_paired < 0.05
        },
        'wilcoxon_test': {
            'w_statistic': w_stat,
            'p_value': p_value_wilcoxon,
            'significant': p_value_wilcoxon < 0.05
        },
        'effect_size': effect_size,
        'mean_initial': np.mean(initial_coherence),
        'mean_final': np.mean(final_coherence),
        'mean_difference': np.mean(initial_coherence - final_coherence)
    }


def create_statistical_report(dataset: List[Dict], output_path: str = 'outputs/statistical_report.txt'):
    """Generate comprehensive statistical report."""
    
    # Comparative analysis
    comparison_df = compare_attack_variants(dataset)
    
    # Hypothesis testing
    hypothesis_results = hypothesis_testing(dataset)
    
    # Generate report
    report = []
    report.append("="*70)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("Context Manipulation Attack Effectiveness")
    report.append("="*70)
    report.append("")
    
    # Dataset overview
    report.append("1. DATASET OVERVIEW")
    report.append("-"*70)
    report.append(f"Total attack scenarios: {len(dataset)}")
    report.append(f"Total iterations: {comparison_df['iterations'].sum()}")
    report.append(f"Attack types: {comparison_df['attack_type'].unique()}")
    report.append("")
    
    # Descriptive statistics
    report.append("2. DESCRIPTIVE STATISTICS")
    report.append("-"*70)
    report.append("\nCoherence Metrics:")
    report.append(f"  Initial coherence (mean ± std): {comparison_df['initial_coherence'].mean():.3f} ± {comparison_df['initial_coherence'].std():.3f}")
    report.append(f"  Final coherence (mean ± std): {comparison_df['final_coherence'].mean():.3f} ± {comparison_df['final_coherence'].std():.3f}")
    report.append(f"  Total degradation (mean): {comparison_df['total_degradation'].mean():.3f}")
    report.append(f"  Degradation rate (mean): {comparison_df['degradation_rate'].mean():.4f} per iteration")
    report.append("")
    report.append("Breakdown Metrics:")
    report.append(f"  Average breakdown rate: {comparison_df['breakdown_rate'].mean()*100:.1f}%")
    report.append(f"  Scenarios with breakdown: {(comparison_df['breakdown_rate'] > 0).sum()}/{len(dataset)}")
    breakdown_iters = comparison_df['iter_to_breakdown'].dropna()
    if len(breakdown_iters) > 0:
        report.append(f"  Mean iterations to breakdown: {breakdown_iters.mean():.1f}")
    report.append("")
    
    # Hypothesis testing
    report.append("3. HYPOTHESIS TESTING")
    report.append("-"*70)
    report.append("\nH0: Attacks do not significantly degrade coherence")
    report.append("H1: Attacks significantly degrade coherence (alpha = 0.05)")
    report.append("")
    report.append("Paired t-test:")
    report.append(f"  t-statistic: {hypothesis_results['paired_t_test']['t_statistic']:.4f}")
    report.append(f"  p-value: {hypothesis_results['paired_t_test']['p_value']:.6f}")
    report.append(f"  Result: {'REJECT H0' if hypothesis_results['paired_t_test']['significant'] else 'FAIL TO REJECT H0'}")
    report.append("")
    report.append("Wilcoxon signed-rank test:")
    report.append(f"  W-statistic: {hypothesis_results['wilcoxon_test']['w_statistic']:.4f}")
    report.append(f"  p-value: {hypothesis_results['wilcoxon_test']['p_value']:.6f}")
    report.append(f"  Result: {'REJECT H0' if hypothesis_results['wilcoxon_test']['significant'] else 'FAIL TO REJECT H0'}")
    report.append("")
    report.append("Effect Size:")
    report.append(f"  Cohen's d: {hypothesis_results['effect_size']:.4f}")
    interpretation = "large" if abs(hypothesis_results['effect_size']) > 0.8 else "medium" if abs(hypothesis_results['effect_size']) > 0.5 else "small"
    report.append(f"  Interpretation: {interpretation} effect")
    report.append("")
    
    # Trajectory analysis
    report.append("4. DEGRADATION TRAJECTORY ANALYSIS")
    report.append("-"*70)
    report.append(f"\nLinear model fit (R² values):")
    for idx, row in comparison_df.iterrows():
        report.append(f"  Prompt {idx+1}: R² = {row['r_squared']:.3f}")
    report.append(f"\nMean R²: {comparison_df['r_squared'].mean():.3f}")
    report.append("")
    
    # Detailed results table
    report.append("5. DETAILED RESULTS BY PROMPT")
    report.append("-"*70)
    report.append("")
    report.append(comparison_df.to_string(index=False))
    report.append("")
    
    # Conclusions
    report.append("6. CONCLUSIONS")
    report.append("-"*70)
    if hypothesis_results['paired_t_test']['significant']:
        report.append("✓ Attacks SIGNIFICANTLY degrade model coherence (p < 0.05)")
    else:
        report.append("✗ No significant degradation detected")
    
    if comparison_df['breakdown_rate'].mean() > 0.2:
        report.append(f"✓ High breakdown rate ({comparison_df['breakdown_rate'].mean()*100:.1f}%) indicates effective attacks")
    
    if hypothesis_results['effect_size'] > 0.5:
        report.append(f"✓ {interpretation.capitalize()} effect size indicates substantial impact")
    
    report.append("")
    report.append("="*70)
    
    # Write report
    report_text = "\n".join(report)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n[OK] Report saved to: {output_path}")
    
    return report_text


def create_distribution_plots(dataset: List[Dict], output_path: str = 'outputs/statistical_distributions.png'):
    """Create statistical distribution visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Collect all metrics
    all_initial = []
    all_final = []
    all_degradation = []
    all_breakdown_iter = []
    all_rates = []
    
    for item in dataset:
        trajectory = analyze_degradation_trajectory(item['results'])
        all_initial.append(trajectory['initial_coherence'])
        all_final.append(trajectory['final_coherence'])
        all_degradation.append(trajectory['total_degradation'])
        if trajectory['iterations_to_breakdown']:
            all_breakdown_iter.append(trajectory['iterations_to_breakdown'])
        all_rates.append(abs(trajectory['degradation_rate']))
    
    # 1. Initial vs Final Coherence
    ax = axes[0, 0]
    ax.scatter(all_initial, all_final, alpha=0.6, s=100)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='No change')
    ax.axhline(0.5, color='orange', linestyle=':', alpha=0.5, label='Breakdown threshold')
    ax.set_xlabel('Initial Coherence')
    ax.set_ylabel('Final Coherence')
    ax.set_title('Initial vs Final Coherence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Degradation distribution
    ax = axes[0, 1]
    ax.hist(all_degradation, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(all_degradation), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_degradation):.3f}')
    ax.set_xlabel('Total Degradation')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Coherence Degradation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Iterations to breakdown
    ax = axes[0, 2]
    if all_breakdown_iter:
        ax.hist(all_breakdown_iter, bins=range(1, max(all_breakdown_iter)+2), color='red', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(all_breakdown_iter), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_breakdown_iter):.1f}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Frequency')
        ax.set_title('Iterations Until Breakdown', fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No breakdowns\ndetected', ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Degradation rate distribution
    ax = axes[1, 0]
    ax.hist(all_rates, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(all_rates), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_rates):.4f}')
    ax.set_xlabel('Degradation Rate (per iteration)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Degradation Rates', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Box plot comparison
    ax = axes[1, 1]
    bp = ax.boxplot([all_initial, all_final], labels=['Initial', 'Final'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.5, label='Breakdown threshold')
    ax.set_ylabel('Coherence Score')
    ax.set_title('Initial vs Final Coherence Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. QQ plot for normality check
    ax = axes[1, 2]
    stats.probplot(all_degradation, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot: Degradation Normality', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Statistical Distributions: Attack Effectiveness', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Distribution plots saved to: {output_path}")


def main():
    """Run complete statistical analysis."""
    print("="*70)
    print("Statistical Analysis of Context Manipulation Attacks")
    print("="*70)
    
    # Setup
    setup_environment(seed=42)
    
    # Generate data
    print("\n[1/3] Generating synthetic dataset...")
    dataset = generate_synthetic_dataset(num_prompts=10, iterations=7)
    print(f"[OK] Generated {len(dataset)} attack scenarios")
    
    # Statistical analysis
    print("\n[2/3] Performing statistical analysis...")
    create_statistical_report(dataset)
    
    # Visualizations
    print("\n[3/3] Creating distribution plots...")
    create_distribution_plots(dataset)
    
    print("\n" + "="*70)
    print("Statistical analysis complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - outputs/statistical_report.txt")
    print("  - outputs/statistical_distributions.png")


if __name__ == "__main__":
    main()

