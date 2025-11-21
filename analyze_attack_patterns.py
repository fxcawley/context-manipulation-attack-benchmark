#!/usr/bin/env python3
"""
Analyze attack patterns and generate comprehensive visualizations.

This script analyzes simulated or real attack results and generates
publication-quality visualizations and statistical reports.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from init_env import setup_environment
from src.attack.attack_simulator import AttackSimulator, generate_synthetic_dataset
from src.eval import evaluate_response

# Setup
setup_environment(seed=42)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


def analyze_attack_progression(results):
    """Analyze how attack effectiveness progresses over iterations."""
    metrics = []
    
    for result in results:
        m = evaluate_response(result.model_response)
        metrics.append({
            'iteration': result.iteration + 1,
            'coherence': m.coherence_score,
            'breakdown': m.breakdown_detected,
            'token_diversity': m.token_diversity,
            'response_length': m.response_length,
            'repetition': m.repetition_score,
            'non_ascii_ratio': m.non_ascii_ratio,
            'perplexity': result.perplexity if result.perplexity else 0
        })
    
    return pd.DataFrame(metrics)


def create_comprehensive_visualization(df, output_path='outputs/attack_analysis.png'):
    """Create multi-panel visualization of attack progression."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    iterations = df['iteration'].values
    
    # 1. Coherence over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iterations, df['coherence'], 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Coherence Score', fontsize=11)
    ax1.set_title('Coherence Degradation', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Token diversity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, df['token_diversity'], 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Token Diversity', fontsize=11)
    ax2.set_title('Token Diversity Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Response length
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iterations, df['response_length'], 'o-', linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Response Length (chars)', fontsize=11)
    ax3.set_title('Response Length Over Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Breakdown detection
    ax4 = fig.add_subplot(gs[1, 0])
    colors = ['red' if b else 'green' for b in df['breakdown']]
    ax4.bar(iterations, df['breakdown'].astype(int), color=colors, alpha=0.7)
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Breakdown', fontsize=11)
    ax4.set_title('Breakdown Detection', fontsize=12, fontweight='bold')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Repetition score
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(iterations, df['repetition'], 'o-', linewidth=2, markersize=8, color='purple')
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('Repetition Score', fontsize=11)
    ax5.set_title('Text Repetition', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Non-ASCII ratio
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(iterations, df['non_ascii_ratio'], 'o-', linewidth=2, markersize=8, color='brown')
    ax6.set_xlabel('Iteration', fontsize=11)
    ax6.set_ylabel('Non-ASCII Ratio', fontsize=11)
    ax6.set_title('Foreign Character Frequency', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Perplexity (if available)
    ax7 = fig.add_subplot(gs[2, 0])
    if df['perplexity'].sum() > 0:
        ax7.plot(iterations, df['perplexity'], 'o-', linewidth=2, markersize=8, color='red')
        ax7.set_xlabel('Iteration', fontsize=11)
        ax7.set_ylabel('Perplexity', fontsize=11)
        ax7.set_title('Model Perplexity', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Perplexity\nNot Available', 
                ha='center', va='center', fontsize=12, transform=ax7.transAxes)
        ax7.axis('off')
    
    # 8. Correlation heatmap
    ax8 = fig.add_subplot(gs[2, 1])
    corr_metrics = df[['coherence', 'token_diversity', 'repetition', 'response_length']].corr()
    sns.heatmap(corr_metrics, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=ax8, cbar_kws={'shrink': 0.8})
    ax8.set_title('Metric Correlations', fontsize=12, fontweight='bold')
    
    # 9. Summary statistics
    ax9 = fig.add_subplot(gs[2, 2])
    summary_text = f"""
Attack Summary
{'='*25}

Total Iterations: {len(df)}
Breakdown Rate: {df['breakdown'].sum()}/{len(df)}
                ({df['breakdown'].mean()*100:.1f}%)

Coherence:
  Start: {df['coherence'].iloc[0]:.3f}
  End: {df['coherence'].iloc[-1]:.3f}
  Change: {df['coherence'].iloc[-1] - df['coherence'].iloc[0]:.3f}

Token Diversity:
  Mean: {df['token_diversity'].mean():.3f}
  Min: {df['token_diversity'].min():.3f}

First Breakdown:
  Iteration {df[df['breakdown']].index[0]+1 if any(df['breakdown']) else 'None'}
"""
    
    ax9.text(0.05, 0.95, summary_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax9.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax9.axis('off')
    
    plt.suptitle('Context Manipulation Attack Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Visualization saved to: {output_path}")
    
    return fig


def compare_multiple_attacks(dataset, output_path='outputs/attack_comparison.png'):
    """Compare attack effectiveness across multiple prompts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    all_coherence = []
    all_breakdown_rates = []
    prompt_labels = []
    
    for item in dataset:
        df = analyze_attack_progression(item['results'])
        all_coherence.append(df['coherence'].values)
        all_breakdown_rates.append(df['breakdown'].mean())
        prompt_labels.append(item['prompt'][:30] + '...')
    
    # 1. Coherence trajectories
    ax = axes[0, 0]
    for i, coherence in enumerate(all_coherence):
        ax.plot(range(1, len(coherence)+1), coherence, 'o-', label=f'Prompt {i+1}', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Coherence Score')
    ax.set_title('Coherence Degradation Across Prompts', fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 2. Breakdown rates
    ax = axes[0, 1]
    ax.barh(range(len(all_breakdown_rates)), all_breakdown_rates, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(prompt_labels)))
    ax.set_yticklabels(prompt_labels, fontsize=8)
    ax.set_xlabel('Breakdown Rate')
    ax.set_title('Breakdown Rate by Prompt', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Final coherence distribution
    ax = axes[1, 0]
    final_coherence = [c[-1] for c in all_coherence]
    ax.hist(final_coherence, bins=10, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Breakdown threshold')
    ax.set_xlabel('Final Coherence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Final Coherence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Attack success summary
    ax = axes[1, 1]
    success_count = sum(1 for c in final_coherence if c < 0.5)
    partial_count = sum(1 for c in final_coherence if 0.5 <= c < 0.7)
    fail_count = len(final_coherence) - success_count - partial_count
    
    ax.pie([success_count, partial_count, fail_count],
           labels=['Full Success\n(coherence < 0.5)', 
                   'Partial Success\n(0.5 ≤ coherence < 0.7)',
                   'Failed\n(coherence ≥ 0.7)'],
           autopct='%1.1f%%',
           colors=['red', 'orange', 'green'],
           startangle=90)
    ax.set_title('Attack Success Rate', fontweight='bold')
    
    plt.suptitle('Multi-Prompt Attack Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison saved to: {output_path}")
    
    return fig


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("Context Manipulation Attack Pattern Analysis")
    print("="*60)
    
    # Generate synthetic dataset
    print("\n[1/4] Generating synthetic attack data...")
    dataset = generate_synthetic_dataset(num_prompts=8, iterations=7)
    print(f"[OK] Generated {len(dataset)} attack scenarios")
    
    # Analyze first attack in detail
    print("\n[2/4] Analyzing detailed attack progression...")
    first_attack = dataset[0]
    df = analyze_attack_progression(first_attack['results'])
    print(f"[OK] Analyzed {len(df)} iterations")
    print(f"\nMetrics Summary:")
    print(df.describe().round(3))
    
    # Create visualizations
    print("\n[3/4] Creating visualizations...")
    create_comprehensive_visualization(df, 'outputs/single_attack_analysis.png')
    
    # Compare multiple attacks
    print("\n[4/4] Comparing across multiple prompts...")
    compare_multiple_attacks(dataset, 'outputs/multi_attack_comparison.png')
    
    # Generate report
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - outputs/single_attack_analysis.png")
    print("  - outputs/multi_attack_comparison.png")
    print("\nKey Findings:")
    
    # Calculate aggregate statistics
    all_final_coherence = []
    all_breakdown_rates = []
    for item in dataset:
        df_temp = analyze_attack_progression(item['results'])
        all_final_coherence.append(df_temp['coherence'].iloc[-1])
        all_breakdown_rates.append(df_temp['breakdown'].mean())
    
    print(f"  - Average final coherence: {np.mean(all_final_coherence):.3f}")
    print(f"  - Average breakdown rate: {np.mean(all_breakdown_rates)*100:.1f}%")
    print(f"  - Attacks causing breakdown: {sum(1 for r in all_breakdown_rates if r > 0)}/{len(dataset)}")
    print(f"  - Attacks achieving coherence < 0.5: {sum(1 for c in all_final_coherence if c < 0.5)}/{len(dataset)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

