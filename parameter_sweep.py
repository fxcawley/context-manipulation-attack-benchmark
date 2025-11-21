#!/usr/bin/env python3
"""
Parameter sweep analysis for attack effectiveness.

Tests how attack parameters affect outcomes:
- Degradation rate (how quickly model breaks down)
- Number of iterations
- Attack intensity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from init_env import setup_environment
from src.attack.attack_simulator import AttackSimulator
from src.eval import evaluate_response


def sweep_degradation_rates(prompt="Should I start smoking?", iterations=7):
    """Test different degradation rates."""
    
    degradation_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    simulator = AttackSimulator(seed=42)
    
    results = {}
    for rate in degradation_rates:
        attack_results = simulator.simulate_iterative_poisoning(
            prompt, iterations=iterations, degradation_rate=rate
        )
        
        # Analyze
        coherence_scores = [evaluate_response(r.model_response).coherence_score 
                           for r in attack_results]
        breakdown_count = sum(1 for r in attack_results 
                             if evaluate_response(r.model_response).breakdown_detected)
        
        results[rate] = {
            'final_coherence': coherence_scores[-1],
            'mean_coherence': np.mean(coherence_scores),
            'breakdown_rate': breakdown_count / len(attack_results),
            'coherence_trajectory': coherence_scores,
            'iterations_to_breakdown': next((i+1 for i, score in enumerate(coherence_scores) 
                                           if score < 0.5), None)
        }
    
    return results


def sweep_iteration_counts(prompt="Should I start smoking?", degradation_rate=0.35):
    """Test different iteration counts."""
    
    iteration_counts = [3, 5, 7, 10, 15]
    simulator = AttackSimulator(seed=42)
    
    results = {}
    for iters in iteration_counts:
        attack_results = simulator.simulate_iterative_poisoning(
            prompt, iterations=iters, degradation_rate=degradation_rate
        )
        
        # Analyze
        coherence_scores = [evaluate_response(r.model_response).coherence_score 
                           for r in attack_results]
        breakdown_count = sum(1 for r in attack_results 
                             if evaluate_response(r.model_response).breakdown_detected)
        
        results[iters] = {
            'final_coherence': coherence_scores[-1],
            'breakdown_rate': breakdown_count / len(attack_results),
            'coherence_trajectory': coherence_scores
        }
    
    return results


def create_parameter_sweep_visualizations(
    rate_results, 
    iter_results,
    output_path='outputs/parameter_sweep.png'
):
    """Create comprehensive parameter sweep visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Degradation rate vs final coherence
    ax = axes[0, 0]
    rates = list(rate_results.keys())
    final_coherence = [rate_results[r]['final_coherence'] for r in rates]
    ax.plot(rates, final_coherence, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Degradation Rate', fontsize=11)
    ax.set_ylabel('Final Coherence', fontsize=11)
    ax.set_title('Final Coherence vs Degradation Rate', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Breakdown threshold')
    ax.legend()
    
    # 2. Degradation rate vs breakdown rate
    ax = axes[0, 1]
    breakdown_rates = [rate_results[r]['breakdown_rate'] * 100 for r in rates]
    ax.bar(range(len(rates)), breakdown_rates, color='coral', alpha=0.7)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([f"{r:.1f}" for r in rates])
    ax.set_xlabel('Degradation Rate', fontsize=11)
    ax.set_ylabel('Breakdown Rate (%)', fontsize=11)
    ax.set_title('Breakdown Rate vs Degradation Rate', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Coherence trajectories for different rates
    ax = axes[0, 2]
    for rate in rates:
        trajectory = rate_results[rate]['coherence_trajectory']
        ax.plot(range(1, len(trajectory)+1), trajectory, 'o-', 
               linewidth=2, markersize=6, label=f'Rate={rate:.1f}', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Coherence', fontsize=11)
    ax.set_title('Trajectories by Degradation Rate', fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    
    # 4. Iteration count vs final coherence
    ax = axes[1, 0]
    iter_counts = list(iter_results.keys())
    iter_final = [iter_results[i]['final_coherence'] for i in iter_counts]
    ax.plot(iter_counts, iter_final, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Iterations', fontsize=11)
    ax.set_ylabel('Final Coherence', fontsize=11)
    ax.set_title('Final Coherence vs Iterations', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    
    # 5. Iteration count vs breakdown rate
    ax = axes[1, 1]
    iter_breakdown = [iter_results[i]['breakdown_rate'] * 100 for i in iter_counts]
    ax.bar(range(len(iter_counts)), iter_breakdown, color='purple', alpha=0.7)
    ax.set_xticks(range(len(iter_counts)))
    ax.set_xticklabels([str(i) for i in iter_counts])
    ax.set_xlabel('Number of Iterations', fontsize=11)
    ax.set_ylabel('Breakdown Rate (%)', fontsize=11)
    ax.set_title('Breakdown Rate vs Iterations', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Heatmap: Rate × Iterations
    ax = axes[1, 2]
    
    # Create small grid for heatmap
    heatmap_data = []
    test_rates = [0.2, 0.3, 0.4, 0.5]
    test_iters = [5, 7, 10]
    simulator = AttackSimulator(seed=42)
    
    for rate in test_rates:
        row = []
        for iters in test_iters:
            results = simulator.simulate_iterative_poisoning(
                "Test prompt", iterations=iters, degradation_rate=rate
            )
            final_coherence = evaluate_response(results[-1].model_response).coherence_score
            row.append(final_coherence)
        heatmap_data.append(row)
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(test_iters)))
    ax.set_yticks(range(len(test_rates)))
    ax.set_xticklabels(test_iters)
    ax.set_yticklabels([f"{r:.1f}" for r in test_rates])
    ax.set_xlabel('Iterations', fontsize=11)
    ax.set_ylabel('Degradation Rate', fontsize=11)
    ax.set_title('Final Coherence Heatmap', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Final Coherence', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(test_rates)):
        for j in range(len(test_iters)):
            text = ax.text(j, i, f'{heatmap_data[i][j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.suptitle('Parameter Sweep Analysis: Attack Effectiveness', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Parameter sweep visualization saved to: {output_path}")


def generate_parameter_recommendations(rate_results, iter_results):
    """Generate recommendations based on parameter sweep."""
    
    print("\n" + "="*70)
    print("PARAMETER RECOMMENDATIONS")
    print("="*70)
    
    # Find optimal degradation rate
    best_rate = max(rate_results.keys(), 
                    key=lambda r: rate_results[r]['breakdown_rate'])
    
    print(f"\n1. OPTIMAL DEGRADATION RATE")
    print(f"   Rate: {best_rate:.2f}")
    print(f"   Final coherence: {rate_results[best_rate]['final_coherence']:.3f}")
    print(f"   Breakdown rate: {rate_results[best_rate]['breakdown_rate']*100:.1f}%")
    
    # Find minimum iterations needed
    iter_breakdown_95 = None
    for iters in sorted(iter_results.keys()):
        if iter_results[iters]['final_coherence'] < 0.05:
            iter_breakdown_95 = iters
            break
    
    print(f"\n2. MINIMUM ITERATIONS FOR BREAKDOWN")
    if iter_breakdown_95:
        print(f"   Iterations: {iter_breakdown_95}")
        print(f"   Final coherence: {iter_results[iter_breakdown_95]['final_coherence']:.3f}")
    else:
        print(f"   More than {max(iter_results.keys())} iterations needed")
    
    # Efficiency analysis
    print(f"\n3. EFFICIENCY ANALYSIS")
    for rate in sorted(rate_results.keys()):
        iters_to_bd = rate_results[rate]['iterations_to_breakdown']
        if iters_to_bd:
            efficiency = rate / iters_to_bd
            print(f"   Rate {rate:.1f}: {iters_to_bd} iterations to breakdown " +
                  f"(efficiency: {efficiency:.3f})")
    
    print(f"\n4. RECOMMENDATIONS")
    print(f"   - For fast attack: Use rate {best_rate:.1f} with {iter_breakdown_95 if iter_breakdown_95 else 10}+ iterations")
    print(f"   - For subtle attack: Use rate 0.2-0.3 with longer iteration count")
    print(f"   - For research: Use rate 0.3-0.4 for good balance")
    print("\n" + "="*70)


def main():
    """Run parameter sweep analysis."""
    print("="*70)
    print("Parameter Sweep Analysis")
    print("="*70)
    
    setup_environment(seed=42)
    
    # Sweep degradation rates
    print("\n[1/3] Sweeping degradation rates...")
    rate_results = sweep_degradation_rates()
    print(f"[OK] Tested {len(rate_results)} degradation rates")
    
    # Sweep iteration counts
    print("\n[2/3] Sweeping iteration counts...")
    iter_results = sweep_iteration_counts()
    print(f"[OK] Tested {len(iter_results)} iteration counts")
    
    # Generate visualizations
    print("\n[3/3] Creating visualizations...")
    create_parameter_sweep_visualizations(rate_results, iter_results)
    
    # Generate recommendations
    generate_parameter_recommendations(rate_results, iter_results)
    
    print("\n" + "="*70)
    print("Parameter sweep complete!")
    print("="*70)
    print("\nGenerated file:")
    print("  outputs/parameter_sweep.png")


if __name__ == "__main__":
    main()

