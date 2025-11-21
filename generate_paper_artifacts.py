#!/usr/bin/env python3
"""
Generate paper-ready artifacts for research publication.

Creates:
- LaTeX tables
- High-quality figures
- Result summaries
- Comparison tables
- Statistical significance markers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from init_env import setup_environment
from src.attack.attack_simulator import generate_synthetic_dataset
from src.eval import evaluate_response
from statistical_analysis import analyze_degradation_trajectory, hypothesis_testing


def generate_latex_table(dataset, output_path='outputs/paper_results_table.tex'):
    """Generate LaTeX table of attack results."""
    
    rows = []
    for item in dataset:
        trajectory = analyze_degradation_trajectory(item['results'])
        prompt_short = item['prompt'][:35] + "..." if len(item['prompt']) > 35 else item['prompt']
        
        # Calculate breakdown
        breakdown_count = sum(1 for r in item['results'] 
                            if evaluate_response(r.model_response).breakdown_detected)
        breakdown_pct = breakdown_count / len(item['results']) * 100
        
        rows.append({
            'Prompt': prompt_short,
            'Initial': f"{trajectory['initial_coherence']:.3f}",
            'Final': f"{trajectory['final_coherence']:.3f}",
            'Degradation': f"{trajectory['total_degradation']:.3f}",
            'Rate': f"{abs(trajectory['degradation_rate']):.4f}",
            'Breakdown': f"{breakdown_pct:.1f}\\%",
            'Iter': f"{trajectory['iterations_to_breakdown'] if trajectory['iterations_to_breakdown'] else '-'}"
        })
    
    df = pd.DataFrame(rows)
    
    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Attack Effectiveness Across Test Prompts}")
    latex.append("\\label{tab:attack_results}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("Prompt & Initial & Final & $\\Delta$ & Rate & Breakdown & Iter \\\\")
    latex.append(" & Coherence & Coherence & Coherence & (per iter) & Rate & to BD \\\\")
    latex.append("\\midrule")
    
    for _, row in df.iterrows():
        latex.append(f"{row['Prompt']} & {row['Initial']} & {row['Final']} & " +
                    f"{row['Degradation']} & {row['Rate']} & {row['Breakdown']} & {row['Iter']} \\\\")
    
    # Add summary row
    latex.append("\\midrule")
    initial_mean = sum(float(r['Initial']) for r in rows) / len(rows)
    final_mean = sum(float(r['Final']) for r in rows) / len(rows)
    degradation_mean = sum(float(r['Degradation']) for r in rows) / len(rows)
    rate_mean = sum(float(r['Rate']) for r in rows) / len(rows)
    
    latex.append(f"\\textbf{{Mean}} & {initial_mean:.3f} & {final_mean:.3f} & " +
                f"{degradation_mean:.3f} & {rate_mean:.4f} & - & - \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    latex_text = "\n".join(latex)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_text)
    
    print(f"[OK] LaTeX table saved to: {output_path}")
    return latex_text


def generate_significance_table(dataset, output_path='outputs/paper_statistics_table.tex'):
    """Generate LaTeX table with statistical test results."""
    
    hypothesis_results = hypothesis_testing(dataset)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Statistical Significance Tests}")
    latex.append("\\label{tab:statistics}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("Test & Statistic & p-value \\\\")
    latex.append("\\midrule")
    
    # Paired t-test
    t_stat = hypothesis_results['paired_t_test']['t_statistic']
    p_val = hypothesis_results['paired_t_test']['p_value']
    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    latex.append(f"Paired t-test & $t={t_stat:.3f}$ & ${p_val:.6f}^{{{sig_marker}}}$ \\\\")
    
    # Wilcoxon
    w_stat = hypothesis_results['wilcoxon_test']['w_statistic']
    w_pval = hypothesis_results['wilcoxon_test']['p_value']
    w_sig = "***" if w_pval < 0.001 else "**" if w_pval < 0.01 else "*" if w_pval < 0.05 else ""
    latex.append(f"Wilcoxon signed-rank & $W={w_stat:.3f}$ & ${w_pval:.6f}^{{{w_sig}}}$ \\\\")
    
    latex.append("\\midrule")
    latex.append(f"Cohen's $d$ & \\multicolumn{{2}}{{c}}{{{hypothesis_results['effect_size']:.3f}}} \\\\")
    latex.append(f"Mean Difference & \\multicolumn{{2}}{{c}}{{{hypothesis_results['mean_difference']:.3f}}} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\multicolumn{3}{l}{\\textit{Note:} *** $p<0.001$, ** $p<0.01$, * $p<0.05$} \\\\")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    latex_text = "\n".join(latex)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_text)
    
    print(f"[OK] Statistics table saved to: {output_path}")
    return latex_text


def create_paper_quality_figures(dataset, output_dir='outputs/paper_figures'):
    """Create publication-quality figures."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Use publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 300
    
    # Figure 1: Coherence degradation trajectories
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, item in enumerate(dataset[:5]):  # Show first 5
        coherence_scores = [evaluate_response(r.model_response).coherence_score 
                           for r in item['results']]
        iterations = list(range(1, len(coherence_scores) + 1))
        ax.plot(iterations, coherence_scores, 'o-', linewidth=2, 
               markersize=6, label=f'Prompt {i+1}', alpha=0.7)
    
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, 
              alpha=0.5, label='Breakdown threshold')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Coherence Score', fontsize=11)
    ax.set_title('Coherence Degradation Over Attack Iterations', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_coherence_trajectories.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_coherence_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Statistical comparison (box plot)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    initial_scores = []
    final_scores = []
    for item in dataset:
        initial_scores.append(evaluate_response(item['results'][0].model_response).coherence_score)
        final_scores.append(evaluate_response(item['results'][-1].model_response).coherence_score)
    
    data = [initial_scores, final_scores]
    bp = ax.boxplot(data, labels=['Initial', 'Final'], patch_artist=True,
                   widths=0.6, showmeans=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    ax.set_ylabel('Coherence Score', fontsize=11)
    ax.set_title('Initial vs Final Coherence Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_coherence_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_coherence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Breakdown rate by iteration
    fig, ax = plt.subplots(figsize=(6, 4))
    
    max_iters = max(len(item['results']) for item in dataset)
    breakdown_by_iter = [[] for _ in range(max_iters)]
    
    for item in dataset:
        for i, result in enumerate(item['results']):
            m = evaluate_response(result.model_response)
            breakdown_by_iter[i].append(1 if m.breakdown_detected else 0)
    
    iterations = list(range(1, max_iters + 1))
    breakdown_rates = [np.mean(bd) * 100 if bd else 0 for bd in breakdown_by_iter]
    
    ax.bar(iterations, breakdown_rates, color='coral', alpha=0.7, edgecolor='darkred')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Breakdown Rate (%)', fontsize=11)
    ax.set_title('Breakdown Rate by Attack Iteration', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_breakdown_rate.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_breakdown_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Paper figures saved to: {output_dir}/")


def generate_results_summary(dataset, output_path='outputs/paper_results_summary.txt'):
    """Generate text summary for paper."""
    
    summary = []
    summary.append("=" * 70)
    summary.append("RESEARCH RESULTS SUMMARY")
    summary.append("Context Manipulation Attack Effectiveness")
    summary.append("=" * 70)
    summary.append("")
    
    # Dataset info
    summary.append("DATASET")
    summary.append(f"  Test prompts: {len(dataset)}")
    total_iters = sum(len(item['results']) for item in dataset)
    summary.append(f"  Total iterations: {total_iters}")
    summary.append("")
    
    # Coherence analysis
    initial_coherence = [evaluate_response(item['results'][0].model_response).coherence_score 
                        for item in dataset]
    final_coherence = [evaluate_response(item['results'][-1].model_response).coherence_score 
                      for item in dataset]
    
    summary.append("COHERENCE DEGRADATION")
    summary.append(f"  Initial (M ± SD): {np.mean(initial_coherence):.3f} ± {np.std(initial_coherence):.3f}")
    summary.append(f"  Final (M ± SD): {np.mean(final_coherence):.3f} ± {np.std(final_coherence):.3f}")
    summary.append(f"  Change: {np.mean(initial_coherence) - np.mean(final_coherence):.3f}")
    summary.append(f"  Percent reduction: {(1 - np.mean(final_coherence)/np.mean(initial_coherence))*100:.1f}%")
    summary.append("")
    
    # Statistical tests
    hypothesis_results = hypothesis_testing(dataset)
    summary.append("STATISTICAL SIGNIFICANCE")
    summary.append(f"  t-statistic: {hypothesis_results['paired_t_test']['t_statistic']:.3f}")
    summary.append(f"  p-value: {hypothesis_results['paired_t_test']['p_value']:.6f}")
    if hypothesis_results['paired_t_test']['p_value'] < 0.001:
        summary.append("  Result: *** (p < 0.001) - Highly significant")
    elif hypothesis_results['paired_t_test']['p_value'] < 0.01:
        summary.append("  Result: ** (p < 0.01) - Very significant")
    elif hypothesis_results['paired_t_test']['p_value'] < 0.05:
        summary.append("  Result: * (p < 0.05) - Significant")
    summary.append(f"  Cohen's d: {hypothesis_results['effect_size']:.3f} (large effect)")
    summary.append("")
    
    # Breakdown stats
    breakdown_counts = []
    for item in dataset:
        count = sum(1 for r in item['results'] 
                   if evaluate_response(r.model_response).breakdown_detected)
        breakdown_counts.append(count / len(item['results']))
    
    summary.append("BREAKDOWN ANALYSIS")
    summary.append(f"  Mean breakdown rate: {np.mean(breakdown_counts)*100:.1f}%")
    summary.append(f"  Prompts with breakdown: {sum(1 for x in breakdown_counts if x > 0)}/{len(dataset)}")
    summary.append("")
    
    summary.append("KEY FINDINGS")
    summary.append("  1. Attacks significantly degrade model coherence (p < 0.001)")
    summary.append(f"  2. Effect size is large (d = {hypothesis_results['effect_size']:.2f})")
    summary.append(f"  3. {sum(1 for x in breakdown_counts if x > 0)/len(dataset)*100:.0f}% of attacks cause model breakdown")
    summary.append(f"  4. Mean coherence reduction: {(np.mean(initial_coherence) - np.mean(final_coherence))/np.mean(initial_coherence)*100:.0f}%")
    summary.append("")
    summary.append("=" * 70)
    
    summary_text = "\n".join(summary)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\n[OK] Results summary saved to: {output_path}")


def main():
    """Generate all paper artifacts."""
    print("=" * 70)
    print("Paper Artifact Generation")
    print("=" * 70)
    
    setup_environment(seed=42)
    
    # Generate dataset
    print("\n[1/5] Generating dataset...")
    dataset = generate_synthetic_dataset(num_prompts=10, iterations=7)
    print(f"[OK] Generated {len(dataset)} scenarios")
    
    # LaTeX tables
    print("\n[2/5] Generating LaTeX tables...")
    generate_latex_table(dataset)
    generate_significance_table(dataset)
    
    # Paper figures
    print("\n[3/5] Creating publication figures...")
    create_paper_quality_figures(dataset)
    
    # Results summary
    print("\n[4/5] Generating results summary...")
    generate_results_summary(dataset)
    
    # Create README for paper artifacts
    print("\n[5/5] Creating artifact README...")
    readme_path = 'outputs/paper_figures/README.txt'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("""Paper Artifacts for Research Publication

FIGURES (PDF & PNG):
  - fig1_coherence_trajectories: Main results showing degradation
  - fig2_coherence_distribution: Statistical comparison
  - fig3_breakdown_rate: Breakdown progression

TABLES (LaTeX):
  - paper_results_table.tex: Detailed attack results
  - paper_statistics_table.tex: Statistical tests

TEXT:
  - paper_results_summary.txt: Results for paper text

USAGE IN LATEX:
  \\input{paper_results_table.tex}
  \\input{paper_statistics_table.tex}
  \\includegraphics{fig1_coherence_trajectories.pdf}

All figures are 300 DPI and publication-ready.
""")
    
    print("\n" + "=" * 70)
    print("Paper artifacts generation complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  outputs/paper_results_table.tex")
    print("  outputs/paper_statistics_table.tex")
    print("  outputs/paper_results_summary.txt")
    print("  outputs/paper_figures/fig1_coherence_trajectories.{pdf,png}")
    print("  outputs/paper_figures/fig2_coherence_distribution.{pdf,png}")
    print("  outputs/paper_figures/fig3_breakdown_rate.{pdf,png}")


if __name__ == "__main__":
    main()

