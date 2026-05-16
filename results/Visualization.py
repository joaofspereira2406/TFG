"""
FIGURES
Author: Joao Pereira

These 5 figures directly answer the research questions:
- RQ1: Small size → measurable bias? (Figure 1)
- RQ2: Which groups are targeted? (Figures 2, 4, 6)
- RQ3: Direction and magnitude? (Figures 3, 5)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

# Set publication-ready style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (10, 6),
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_thesis_data(df_bias, df_model_sizes):
    """
    Prepare all necessary data structures for thesis figures.

    Creates two separate model aggregates:
    - model_avg_absolute: For Figure 1 (bias magnitude, always positive)
    - model_avg_signed: For Figure 7 (preserves direction, can be negative)
    """

    # Create category from dataset (remove 'BBQ-' prefix)
    df_bias = df_bias.copy()
    df_bias['category'] = df_bias['dataset'].str.replace('BBQ-', '').str.replace('_', ' ')

    # Merge size information
    df_merged = df_bias.merge(
        df_model_sizes,
        left_on='model',
        right_on='model_clean',
        how='left'
    )

    # For Figure 1: Absolute values (bias magnitude only)
    model_avg_absolute = df_merged.groupby(['model', 'parameters']).agg({
        'soft_disambig_bias': lambda x: x.abs().mean(),
        'soft_ambig_bias': lambda x: x.abs().mean(),
        'soft_ambig_accuracy': 'mean'
    }).reset_index()

    # For Figure 7: Signed values (preserve direction!)
    model_avg_signed = df_merged.groupby(['model', 'parameters']).agg({
        'soft_ambig_bias': 'mean',  # NO absolute value - keep the sign!
        'soft_ambig_accuracy': 'mean'
    }).reset_index()

    # Calculate per-category statistics (for RQ2 & RQ3)
    category_stats = df_merged.groupby('category').agg({
        'soft_disambig_bias': ['mean', 'std', 'sem', 'count'],
        'soft_ambig_bias': ['mean', 'std', 'sem']
    }).round(4)

    # Calculate directional bias (signed) per category
    directional = df_merged.groupby('category')['soft_disambig_bias'].agg([
        ('mean_signed', 'mean'),
        ('mean_absolute', lambda x: x.abs().mean()),
        ('std', 'std'),
        ('sem', lambda x: x.std() / np.sqrt(len(x)))
    ]).round(4)

    # Prepare heatmap data (model × category)
    heatmap_data = df_merged.pivot_table(
        index='model',
        columns='category',
        values='soft_disambig_bias',
        aggfunc='mean'
    )

    # Rename for readability
    def clean_model_name(name):
        mapping = {
            'facebook_bart_large_mnli': 'BART-L',
            'roberta_large_mnli': 'RoBERTa-L',
            'valhalla_distilbart_mnli_12_3': 'DistilBART',
            'MoritzLaurer_deberta_v3_large_zeroshot_v2.0': 'DeBERTa-L',
            'MoritzLaurer_mDeBERTa_v3_base_mnli_xnli': 'mDeBERTa-B',
            'Jiva_xlm_roberta_large_it_mnli': 'XLM-R-L',
            'NbAiLab_nb_bert_base_mnli': 'NB-BERT',
            'Recognai_zeroshot_selectra_small': 'Selectra-S',
            'cross_encoder_nli_MiniLM2_L6_H768': 'MiniLM-L6',
            'cross_encoder_nli_deberta_base': 'DeBERTa-B'
        }
        return mapping.get(name, name.split('_')[0])

    def clean_category_name(name):
        mapping = {
            'AGE': 'Age',
            'DISABILITY': 'Disability',
            'GENDER': 'Gender',
            'NATIONALITY': 'Nationality',
            'PHYSICAL APPEARANCE': 'Appearance',
            'RACE ETHNICITY': 'Race/Ethnicity',
            'RACE GENDER': 'Race × Gender',
            'RACE SES': 'Race × SES',
            'RELIGION': 'Religion',
            'SES': 'SES',
            'SEXUAL ORIENTATION': 'Sexual Orientation'
        }
        return mapping.get(name.upper(), name)

    # Apply clean names
    model_avg_absolute['model_clean'] = model_avg_absolute['model'].apply(clean_model_name)
    model_avg_signed['model_clean'] = model_avg_signed['model'].apply(clean_model_name)
    heatmap_data.index = heatmap_data.index.map(clean_model_name)
    heatmap_data.columns = heatmap_data.columns.map(clean_category_name)
    directional.index = directional.index.map(clean_category_name)

    # Get model order for consistent plotting
    model_order = ['Selectra-S', 'DistilBART', 'NB-BERT', 'MiniLM-L6',
                   'mDeBERTa-B', 'DeBERTa-B', 'RoBERTa-L', 'BART-L',
                   'DeBERTa-L', 'XLM-R-L']

    # Category order for consistent plotting
    category_order = ['Age', 'Gender', 'Nationality', 'Religion', 'SES',
                      'Sexual Orientation', 'Appearance', 'Race/Ethnicity',
                      'Race × Gender', 'Disability', 'Race × SES']

    return {
        'model_avg_absolute': model_avg_absolute,  # For Figure 1 (bias magnitude)
        'model_avg_signed': model_avg_signed,  # For Figure 7 (preserves direction)
        'category_stats': category_stats,
        'directional': directional,
        'heatmap_data': heatmap_data,
        'df_merged': df_merged,
        'clean_names': {'model': clean_model_name, 'category': clean_category_name},
        'model_order': model_order,
        'category_order': category_order
    }

# ============================================================================
# CSV EXPORT FUNCTIONS
# ============================================================================

def export_figure_1_csv(data, output_path):
    """Export Figure 1 data (size vs bias) as CSV"""

    model_avg = data['model_avg_absolute']
    plot_data = model_avg.dropna(subset=['parameters', 'soft_disambig_bias'])

    # Sort by model size
    plot_data = plot_data.sort_values('parameters')

    # Create DataFrame for export
    export_df = pd.DataFrame({
        'model': plot_data['model_clean'].tolist(),
        'parameters_millions': (plot_data['parameters'] / 1e6).tolist(),
        'parameters_raw': plot_data['parameters'].tolist(),
        'mean_absolute_bias': plot_data['soft_disambig_bias'].tolist(),
        'mean_ambiguous_bias': plot_data['soft_ambig_bias'].tolist(),
        'mean_ambiguous_accuracy': plot_data['soft_ambig_accuracy'].tolist()
    })

    # Convert Path object to string before using replace
    output_path_str = str(output_path)
    export_df.to_csv(output_path_str, index=False)
    print(f"Exported CSV to {output_path_str}")

    # Also save correlation info
    corr, p_value = spearmanr(plot_data['parameters'], plot_data['soft_disambig_bias'])
    corr_path = output_path_str.replace('.csv', '_correlation.csv')
    pd.DataFrame({
        'metric': ['spearman_rho', 'p_value', 'n_models'],
        'value': [corr, p_value, len(plot_data)]
    }).to_csv(corr_path, index=False)
    print(f"Exported correlation to {corr_path}")


def export_figure_2_csv(data, output_path):
    """Export Figure 2 data (directional heatmap) as CSV"""

    heatmap_data = data['heatmap_data']

    # Ensure consistent ordering
    models_present = [m for m in data['model_order'] if m in heatmap_data.index]
    categories_present = [c for c in data['category_order'] if c in heatmap_data.columns]

    # Reorder
    heatmap_data = heatmap_data.reindex(index=models_present, columns=categories_present)

    # Reset index to make model a column
    export_df = heatmap_data.reset_index()
    export_df = export_df.rename(columns={'index': 'model'})

    # Convert Path object to string
    output_path_str = str(output_path)
    export_df.to_csv(output_path_str, index=False)
    print(f"Exported CSV to {output_path_str}")

    # Also save metadata
    meta_path = output_path_str.replace('.csv', '_metadata.csv')
    pd.DataFrame({
        'attribute': ['shape_rows', 'shape_cols', 'max_bias', 'min_bias', 'color_scale'],
        'value': [heatmap_data.shape[0], heatmap_data.shape[1],
                  heatmap_data.abs().max().max(), heatmap_data.min().min(), 'RdBu_r']
    }).to_csv(meta_path, index=False)
    print(f"Exported metadata to {meta_path}")


def export_figure_3_csv(data, output_path):
    """Export Figure 3 data (signed bias by category) as CSV"""

    directional = data['directional'].copy()

    # Ensure consistent ordering
    categories_present = [c for c in data['category_order'] if c in directional.index]
    directional = directional.reindex(categories_present)

    # Create export DataFrame
    export_df = pd.DataFrame({
        'category': categories_present,
        'mean_signed_bias': directional['mean_signed'].values,
        'mean_absolute_bias': directional['mean_absolute'].values,
        'standard_deviation': directional['std'].values,
        'standard_error': directional['sem'].values,
        'ci_lower_95': (directional['mean_signed'] - 1.96 * directional['sem']).values,
        'ci_upper_95': (directional['mean_signed'] + 1.96 * directional['sem']).values
    })

    output_path_str = str(output_path)
    export_df.to_csv(output_path_str, index=False)
    print(f"Exported CSV to {output_path_str}")


def export_figure_4_csv(data, output_path):
    """Export Figure 4 data (bias distribution by group) as CSV"""

    df_merged = data['df_merged']
    clean_cat = data['clean_names']['category']

    # Prepare data with clean category names
    plot_data = df_merged[['category', 'soft_disambig_bias']].copy()
    plot_data['category_clean'] = plot_data['category'].apply(clean_cat)

    # Create long format DataFrame with all bias values
    export_rows = []
    for cat in data['category_order']:
        cat_values = plot_data[plot_data['category_clean'] == cat]['soft_disambig_bias'].dropna()
        for value in cat_values:
            export_rows.append({'category': cat, 'bias_score': value})

    export_df = pd.DataFrame(export_rows)
    output_path_str = str(output_path)
    export_df.to_csv(output_path_str, index=False)
    print(f"Exported CSV to {output_path_str}")

    # Also save summary statistics
    stats_path = output_path_str.replace('.csv', '_statistics.csv')
    stats_rows = []
    for cat in data['category_order']:
        cat_values = plot_data[plot_data['category_clean'] == cat]['soft_disambig_bias'].dropna()
        if len(cat_values) > 0:
            stats_rows.append({
                'category': cat,
                'mean': cat_values.mean(),
                'median': cat_values.median(),
                'std': cat_values.std(),
                'q1': cat_values.quantile(0.25),
                'q3': cat_values.quantile(0.75),
                'min': cat_values.min(),
                'max': cat_values.max(),
                'n': len(cat_values)
            })

    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    print(f"Exported statistics to {stats_path}")


def export_figure_5_csv(data, output_path):
    """Export Figure 5 data (bias magnitude by category) as CSV"""

    directional = data['directional'].copy()

    # Sort by magnitude
    directional_sorted = directional.sort_values('mean_absolute', ascending=True)

    export_df = pd.DataFrame({
        'category': directional_sorted.index,
        'mean_absolute_bias': directional_sorted['mean_absolute'].values,
        'standard_error': directional_sorted['sem'].values,
        'ci_lower_95': (directional_sorted['mean_absolute'] - 1.96 * directional_sorted['sem']).values,
        'ci_upper_95': (directional_sorted['mean_absolute'] + 1.96 * directional_sorted['sem']).values
    })

    output_path_str = str(output_path)
    export_df.to_csv(output_path_str, index=False)
    print(f"Exported CSV to {output_path_str}")


def export_figure_6_csv(data, df_group, output_path):
    """Export Figure 6 data (Disability groups heatmap) as CSV"""

    # Filter for disability data and disambig_bias metric
    disability_data = df_group[
        (df_group['category'] == 'BBQ-DISABILITY') &
        (df_group['metric'] == 'disambig_bias')
        ].copy()

    # Get unique disability groups dynamically
    disability_groups = disability_data['group'].unique().tolist()
    print(f"Found disability groups: {disability_groups}")

    # Clean model names
    def clean_model_name(name):
        mapping = {
            'facebook_bart_large_mnli': 'BART-L',
            'roberta_large_mnli': 'RoBERTa-L',
            'valhalla_distilbart_mnli_12_3': 'DistilBART',
            'MoritzLaurer_deberta_v3_large_zeroshot_v2.0': 'DeBERTa-L',
            'MoritzLaurer_mDeBERTa_v3_base_mnli_xnli': 'mDeBERTa-B',
            'Jiva_xlm_roberta_large_it_mnli': 'XLM-R-L',
            'NbAiLab_nb_bert_base_mnli': 'NB-BERT',
            'Recognai_zeroshot_selectra_small': 'Selectra-S',
            'cross_encoder_nli_MiniLM2_L6_H768': 'MiniLM-L6',
            'cross_encoder_nli_deberta_base': 'DeBERTa-B'
        }
        return mapping.get(name, name.split('_')[0])

    disability_data['model_clean'] = disability_data['model'].apply(clean_model_name)

    # Use 'value' column for bias scores
    # Create wide format for export
    export_df = disability_data.pivot_table(
        index='model_clean',
        columns='group',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Rename model column for clarity
    export_df = export_df.rename(columns={'model_clean': 'model'})

    output_path_str = str(output_path)
    export_df.to_csv(output_path_str, index=False)
    print(f"Exported Figure 6 CSV to {output_path_str}")

    # Also export long format
    long_df = disability_data[['model_clean', 'group', 'value']].copy()
    long_df = long_df.rename(columns={'model_clean': 'model', 'value': 'bias_score'})
    long_path = output_path_str.replace('.csv', '_long.csv')
    long_df.to_csv(long_path, index=False)
    print(f"Exported long format to {long_path}")


def export_figure_7_csv(data, output_path):
    """Export Figure 7 data (ambiguous bias vs accuracy) as CSV"""

    model_avg = data['model_avg_signed']
    plot_data = model_avg.dropna(subset=['soft_ambig_bias', 'soft_ambig_accuracy'])

    # Calculate mean accuracy across all models
    mean_accuracy = plot_data['soft_ambig_accuracy'].mean()

    # Clean model names
    def clean_model_name(name):
        mapping = {
            'facebook_bart_large_mnli': 'BART-L',
            'roberta_large_mnli': 'RoBERTa-L',
            'valhalla_distilbart_mnli_12_3': 'DistilBART',
            'MoritzLaurer_deberta_v3_large_zeroshot_v2.0': 'DeBERTa-L',
            'MoritzLaurer_mDeBERTa_v3_base_mnli_xnli': 'mDeBERTa-B',
            'Jiva_xlm_roberta_large_it_mnli': 'XLM-R-L',
            'NbAiLab_nb_bert_base_mnli': 'NB-BERT',
            'Recognai_zeroshot_selectra_small': 'Selectra-S',
            'cross_encoder_nli_MiniLM2_L6_H768': 'MiniLM-L6',
            'cross_encoder_nli_deberta_base': 'DeBERTa-B'
        }
        return mapping.get(name, name.split('_')[0])

    plot_data['model_clean'] = plot_data['model'].apply(clean_model_name)

    # Determine quadrant for each model
    quadrants = []
    for _, row in plot_data.iterrows():
        bias = row['soft_ambig_bias']
        acc = row['soft_ambig_accuracy']

        if bias >= 0 and acc >= mean_accuracy:
            quadrant = 'High Accuracy, Stereotypical'
        elif bias >= 0 and acc < mean_accuracy:
            quadrant = 'Low Accuracy, Stereotypical'
        elif bias < 0 and acc >= mean_accuracy:
            quadrant = 'High Accuracy, Counter-Stereotypical'
        else:
            quadrant = 'Low Accuracy, Counter-Stereotypical'
        quadrants.append(quadrant)

    # Create export DataFrame
    export_df = pd.DataFrame({
        'model': plot_data['model_clean'].tolist(),
        'ambiguous_bias': plot_data['soft_ambig_bias'].tolist(),
        'ambiguous_accuracy': plot_data['soft_ambig_accuracy'].tolist(),
        'quadrant': quadrants,
        'parameters_millions': (
                    plot_data['parameters'] / 1e6).tolist() if 'parameters' in plot_data.columns else [None] * len(
            plot_data)
    })

    # Sort by accuracy (highest to lowest) for better readability
    export_df = export_df.sort_values('ambiguous_accuracy', ascending=False)

    # Calculate correlation
    corr, p_value = spearmanr(plot_data['soft_ambig_bias'], plot_data['soft_ambig_accuracy'])

    output_path_str = str(output_path)
    export_df.to_csv(output_path_str, index=False)
    print(f"Exported Figure 7 CSV to {output_path_str}")

    # Save correlation and summary stats
    stats_path = output_path_str.replace('.csv', '_statistics.csv')
    pd.DataFrame({
        'metric': [
            'spearman_rho',
            'p_value',
            'n_models',
            'mean_accuracy',
            'std_accuracy',
            'min_accuracy',
            'max_accuracy',
            'mean_bias',
            'std_bias',
            'min_bias',
            'max_bias'
        ],
        'value': [
            corr,
            p_value,
            len(plot_data),
            mean_accuracy,
            plot_data['soft_ambig_accuracy'].std(),
            plot_data['soft_ambig_accuracy'].min(),
            plot_data['soft_ambig_accuracy'].max(),
            plot_data['soft_ambig_bias'].mean(),
            plot_data['soft_ambig_bias'].std(),
            plot_data['soft_ambig_bias'].min(),
            plot_data['soft_ambig_bias'].max()
        ]
    }).to_csv(stats_path, index=False)
    print(f"Exported statistics to {stats_path}")

    # Save quadrant summary
    quadrant_path = output_path_str.replace('.csv', '_quadrant_summary.csv')
    quadrant_counts = export_df['quadrant'].value_counts().reset_index()
    quadrant_counts.columns = ['quadrant', 'count']
    quadrant_counts['percentage'] = (quadrant_counts['count'] / len(export_df)) * 100
    quadrant_counts.to_csv(quadrant_path, index=False)
    print(f"Exported quadrant summary to {quadrant_path}")


def export_all_csv_data(data, output_dir="./thesis_figures"):
    """Export all figure data as CSV files"""

    # Create CSV directory
    csv_dir = Path(output_dir) / "csv_data"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("EXPORTING CSV DATA")
    print("=" * 60)

    # Export each figure's data
    export_figure_1_csv(data, csv_dir / "figure_1_size_vs_bias_data.csv")
    export_figure_2_csv(data, csv_dir / "figure_2_directional_heatmap_data.csv")
    export_figure_3_csv(data, csv_dir / "figure_3_signed_bias_data.csv")
    export_figure_4_csv(data, csv_dir / "figure_4_bias_distribution_data.csv")
    export_figure_5_csv(data, csv_dir / "figure_5_bias_magnitude_data.csv")

    # Export combined data (long format, similar to your original CSV)
    combined_long = data['df_merged'][['model', 'category', 'soft_disambig_bias', 'soft_ambig_bias', 'soft_ambig_accuracy']].copy()
    combined_long['model_clean'] = combined_long['model'].apply(data['clean_names']['model'])
    combined_long['category_clean'] = combined_long['category'].apply(data['clean_names']['category'])
    combined_long = combined_long.drop(columns=['model', 'category'])
    combined_long = combined_long.rename(columns={'model_clean': 'model', 'category_clean': 'category'})

    combined_long.to_csv(csv_dir / "figure_combined_bias_data.csv", index=False)
    print(f"Exported combined CSV to {csv_dir / 'figure_combined_bias_data.csv'}")

    # Export wide format (models as rows, categories as columns)
    wide_format = combined_long.pivot_table(
        index='model',
        columns='category',
        values='soft_disambig_bias',
        aggfunc='first'
    )
    wide_format.to_csv(csv_dir / "figure_wide_format_bias_matrix.csv")
    print(f"Exported wide format CSV to {csv_dir / 'figure_wide_format_bias_matrix.csv'}")


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def figure_1_size_vs_bias(data, save_path=None):
    """FIGURE 1: Does model size predict measurable bias?"""

    model_avg = data['model_avg_absolute']
    plot_data = model_avg.dropna(subset=['parameters', 'soft_disambig_bias'])

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        plot_data['parameters'] / 1e6,
        plot_data['soft_disambig_bias'],
        s=150,
        c=range(len(plot_data)),
        cmap='viridis',
        alpha=0.8,
        edgecolors='black',
        linewidth=1.5
    )

    for _, row in plot_data.iterrows():
        ax.annotate(
            row['model_clean'],
            (row['parameters'] / 1e6, row['soft_disambig_bias']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8
        )

    if len(plot_data) >= 3:
        z = np.polyfit(np.log10(plot_data['parameters']), plot_data['soft_disambig_bias'], 1)
        p = np.poly1d(z)
        x_trend = np.logspace(np.log10(plot_data['parameters'].min()),
                              np.log10(plot_data['parameters'].max()), 100)
        ax.plot(x_trend / 1e6, p(np.log10(x_trend)),
                'r--', alpha=0.5, label=f'Trend: slope={z[0]:.3f}')

    corr, p_value = spearmanr(plot_data['parameters'], plot_data['soft_disambig_bias'])

    ax.set_xlabel('Model Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel('Mean Absolute Bias Score |bias|', fontsize=12, fontweight='bold')
    ax.set_title(
        'Figure 1: Model Size vs. Bias Magnitude\n(Higher values indicate the model is less unbiased, showing stronger stereotypical or anti-stereotypical associations)',
        fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {p_value:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 1 to {save_path}")

    return fig, {'correlation': corr, 'p_value': p_value, 'n_models': len(plot_data)}



def figure_2_directional_heatmap(data, save_path=None):
    """FIGURE 2: Directional bias heatmap"""

    heatmap_data = data['heatmap_data']

    # Reorder for better visualization
    models_present = [m for m in data['model_order'] if m in heatmap_data.index]
    categories_present = [c for c in data['category_order'] if c in heatmap_data.columns]
    heatmap_data = heatmap_data.reindex(index=models_present, columns=categories_present)

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(heatmap_data.values, cmap='RdBu_r', aspect='auto',
                   vmin=-0.25, vmax=0.25, interpolation='nearest')

    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(heatmap_data.index, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Bias Score\n(+1 = Favors stereotype, -1 = Favors counter-stereotype)',
                   fontsize=10, fontweight='bold')

    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if abs(value) > 0.15 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=text_color, fontsize=8, fontweight='bold')

    ax.set_xlabel('Demographic Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(
        'Figure 2: Stereotypical Bias Direction by Model and Category\n(Red = Stereotype-consistent predictions, Blue = Counter-stereotypical predictions)',
        fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 2 to {save_path}")

    return fig, {'shape': heatmap_data.shape, 'max_bias': heatmap_data.abs().max().max()}


def figure_3_signed_bias_by_category(data, save_path=None):
    """FIGURE 3: Signed bias by category"""

    directional = data['directional'].copy()

    # Reorder
    categories_present = [c for c in data['category_order'] if c in directional.index]
    directional = directional.reindex(categories_present)
    directional = directional.sort_values('mean_absolute', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    categories = directional.index
    means = directional['mean_signed'].values
    errors = directional['sem'].values * 1.96

    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, means, xerr=errors, capsize=3,
                   color=['red' if x > 0 else 'blue' for x in means],
                   alpha=0.7, edgecolor='black', linewidth=1)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)

    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        if mean > 0:
            ax.text(mean + err + 0.01, i, f'{mean:.3f}',
                    va='center', fontsize=9)
        else:
            ax.text(mean - err - 0.03, i, f'{mean:.3f}',
                    va='center', ha='right', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel('Mean Signed Bias Score\n(+ = Stereotypical bias, - = Counter-stereotypical bias)',
                  fontsize=12, fontweight='bold')
    ax.set_title(
        'Figure 3: Direction of Stereotypical Bias by Demographic Category\n(Positive = Model favors stereotypes, Negative = Model favors counter-stereotypes)',
        fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Stereotype-consistent bias'),
                       Patch(facecolor='blue', alpha=0.7, label='Counter-stereotypical bias')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 3 to {save_path}")

    return fig, directional[['mean_signed', 'mean_absolute']].to_dict()


def figure_4_bias_distribution_by_group(data, save_path=None):
    """FIGURE 4: Bias distribution per group"""

    df_merged = data['df_merged']
    clean_cat = data['clean_names']['category']

    plot_data = df_merged[['category', 'soft_disambig_bias']].copy()
    plot_data['category_clean'] = plot_data['category'].apply(clean_cat)

    # Reorder categories
    categories_present = [c for c in data['category_order'] if c in plot_data['category_clean'].unique()]

    fig, ax = plt.subplots(figsize=(12, 7))

    bp = ax.boxplot([plot_data[plot_data['category_clean'] == cat]['soft_disambig_bias'].dropna()
                     for cat in categories_present],
                    labels=categories_present,
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    widths=0.7)

    for i, cat in enumerate(categories_present):
        median_val = plot_data[plot_data['category_clean'] == cat]['soft_disambig_bias'].median()
        if median_val > 0.05:
            color = 'lightcoral'
        elif median_val < -0.05:
            color = 'lightblue'
        else:
            color = 'lightgray'
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Demographic Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bias Score\n(+ = Stereotypical, - = Counter-stereotypical)',
                  fontsize=12, fontweight='bold')
    ax.set_title(
        'Figure 4: Bias Variability Across Models by Category\n(Red = Stereotype-consistent, Blue = Counter-stereotypical, Gray = Neutral)',
        fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax.text(0.02, 0.98,
            'Narrow boxes = consistent bias direction across models\nWide boxes = model-dependent bias direction',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 4 to {save_path}")

    return fig, {'categories': categories_present, 'n_models': len(df_merged['model'].unique())}


def figure_5_bias_magnitude_by_category(data, save_path=None):
    """FIGURE 5: Absolute bias magnitude by category"""

    directional = data['directional'].copy()

    # Sort by magnitude
    directional = directional.sort_values('mean_absolute', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    categories = directional.index
    magnitudes = directional['mean_absolute'].values
    errors = directional['sem'].values * 1.96

    bars = ax.barh(categories, magnitudes, xerr=errors, capsize=3,
                   color='coral', alpha=0.7, edgecolor='black', linewidth=1)

    for i, (bar, mag, err) in enumerate(zip(bars, magnitudes, errors)):
        ax.text(mag + err + 0.005, i, f'{mag:.3f}',
                va='center', fontsize=9)

    ax.set_xlabel('Mean Absolute Bias Score |bias| (0 = No bias, Higher = Stronger stereotypical associations)',
                  fontsize=12, fontweight='bold')
    ax.set_title(
        'Figure 5: Stereotypical Bias Magnitude by Demographic Category\n(Higher values indicate categories where models show stronger stereotypical associations)',
        fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(magnitudes + errors) * 1.1)

    ax.text(0.95, 0.02,
            'Note: Magnitude ignores direction\n(both stereotype-consistent and counter-stereotypical bias contribute equally)',
            transform=ax.transAxes, fontsize=9, horizontalalignment='right',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 5 to {save_path}")

    return fig, directional['mean_absolute'].to_dict()


def figure_6_disability_groups_heatmap(data, df_group, save_path=None):
    """
    FIGURE 6: Directional bias heatmap focusing on Disability groups.
    """

    # Filter for disability data and disambig_bias metric
    disability_data = df_group[
        (df_group['category'] == 'BBQ-DISABILITY') &
        (df_group['metric'] == 'disambig_bias')
        ].copy()

    # Get unique disability groups dynamically
    disability_groups = disability_data['group'].unique().tolist()
    print(f"Plotting disability groups: {disability_groups}")

    # Clean model names
    def clean_model_name(name):
        mapping = {
            'facebook_bart_large_mnli': 'BART-L',
            'roberta_large_mnli': 'RoBERTa-L',
            'valhalla_distilbart_mnli_12_3': 'DistilBART',
            'MoritzLaurer_deberta_v3_large_zeroshot_v2.0': 'DeBERTa-L',
            'MoritzLaurer_mDeBERTa_v3_base_mnli_xnli': 'mDeBERTa-B',
            'Jiva_xlm_roberta_large_it_mnli': 'XLM-R-L',
            'NbAiLab_nb_bert_base_mnli': 'NB-BERT',
            'Recognai_zeroshot_selectra_small': 'Selectra-S',
            'cross_encoder_nli_MiniLM2_L6_H768': 'MiniLM-L6',
            'cross_encoder_nli_deberta_base': 'DeBERTa-B'
        }
        return mapping.get(name, name.split('_')[0])

    disability_data['model_clean'] = disability_data['model'].apply(clean_model_name)

    # Pivot to create heatmap matrix (models × groups)
    heatmap_matrix = disability_data.pivot_table(
        index='model_clean',
        columns='group',
        values='value',
        aggfunc='first'
    )

    # Reorder models for consistent presentation
    model_order = ['Selectra-S', 'DistilBART', 'NB-BERT', 'MiniLM-L6',
                   'mDeBERTa-B', 'DeBERTa-B', 'RoBERTa-L', 'BART-L',
                   'DeBERTa-L', 'XLM-R-L']

    # Reindex to ensure consistent ordering
    models_present = [m for m in model_order if m in heatmap_matrix.index]
    heatmap_matrix = heatmap_matrix.reindex(index=models_present)

    # Create figure - adjust width based on number of groups
    fig_width = max(8, len(disability_groups) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # Create heatmap
    im = ax.imshow(heatmap_matrix.values, cmap='RdBu_r', aspect='auto',
                   vmin=-0.3, vmax=0.3, interpolation='nearest')

    # Configure axes
    ax.set_xticks(np.arange(len(heatmap_matrix.columns)))
    ax.set_yticks(np.arange(len(heatmap_matrix.index)))

    # Format group labels (capitalize, replace underscores with spaces)
    group_labels = [g.replace('_', ' ').capitalize() for g in heatmap_matrix.columns]
    ax.set_xticklabels(group_labels, fontsize=11, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(heatmap_matrix.index, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Bias Score\n(+1 = Favors stereotype, -1 = Favors counter-stereotype)',
                   fontsize=10, fontweight='bold')

    # Add value annotations in each cell
    for i in range(len(heatmap_matrix.index)):
        for j in range(len(heatmap_matrix.columns)):
            value = heatmap_matrix.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if abs(value) > 0.15 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                        color=text_color, fontsize=9, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Disability Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(
        'Figure 6: Stereotypical Bias Toward Disability Groups Across Models\n(Red = Stereotype-consistent bias, Blue = Counter-stereotypical bias)',
        fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 6 to {save_path}")

    return fig, heatmap_matrix


def figure_7_ambig_bias_vs_accuracy(data, save_path=None):
    """
    FIGURE 7: Relationship between ambiguous bias and ambiguous accuracy.

    This figure shows how models perform on ambiguous examples:
    - Higher ambiguous bias = more stereotypical predictions on ambiguous items
    - Higher ambiguous accuracy = better performance on ambiguous examples

    The plot is divided into 4 quadrants by:
    - Vertical line at x=0 (separating stereotypical vs counter-stereotypical bias)
    - Horizontal line at mean accuracy (separating above/below average performance)
    X-axis ranges from -0.1 to 0.1, centered at 0.
    """

    # Use the signed version (preserves direction!)
    plot_data = data['model_avg_signed'].copy()
    plot_data = plot_data.dropna(subset=['soft_ambig_bias', 'soft_ambig_accuracy'])

    # Calculate mean accuracy across all models
    mean_accuracy = plot_data['soft_ambig_accuracy'].mean()

    # Clean model names if not already done
    if 'model_clean' not in plot_data.columns:
        def clean_model_name(name):
            mapping = {
                'facebook_bart_large_mnli': 'BART-L',
                'roberta_large_mnli': 'RoBERTa-L',
                'valhalla_distilbart_mnli_12_3': 'DistilBART',
                'MoritzLaurer_deberta_v3_large_zeroshot_v2.0': 'DeBERTa-L',
                'MoritzLaurer_mDeBERTa_v3_base_mnli_xnli': 'mDeBERTa-B',
                'Jiva_xlm_roberta_large_it_mnli': 'XLM-R-L',
                'NbAiLab_nb_bert_base_mnli': 'NB-BERT',
                'Recognai_zeroshot_selectra_small': 'Selectra-S',
                'cross_encoder_nli_MiniLM2_L6_H768': 'MiniLM-L6',
                'cross_encoder_nli_deberta_base': 'DeBERTa-B'
            }
            return mapping.get(name, name.split('_')[0])

        plot_data['model_clean'] = plot_data['model'].apply(clean_model_name)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color points by model size if available
    if 'parameters' in plot_data.columns:
        scatter = ax.scatter(
            plot_data['soft_ambig_bias'],
            plot_data['soft_ambig_accuracy'],
            s=200,
            c=plot_data['parameters'] / 1e6,
            cmap='viridis',
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5,
            vmin=0,
            vmax=600
        )
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Model Size (Millions of parameters)', fontsize=11, fontweight='bold')
    else:
        ax.scatter(
            plot_data['soft_ambig_bias'],
            plot_data['soft_ambig_accuracy'],
            s=200,
            color='steelblue',
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5
        )

    # Add model labels
    for _, row in plot_data.iterrows():
        ax.annotate(
            row['model_clean'],
            (row['soft_ambig_bias'], row['soft_ambig_accuracy']),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

    # Add quadrant lines
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    ax.axhline(y=mean_accuracy, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    # Set x-axis limits centered at 0
    ax.set_xlim(-0.1, 0.1)

    # Add quadrant background shading
    # Quadrant 1 (Top-Right): High accuracy, Stereotypical bias
    ax.axvspan(0, 0.1, alpha=0.05, color='red',
               ymin=(mean_accuracy - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]), ymax=1)
    # Quadrant 2 (Top-Left): High accuracy, Counter-stereotypical bias
    ax.axvspan(-0.1, 0, alpha=0.05, color='blue',
               ymin=(mean_accuracy - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]), ymax=1)
    # Quadrant 3 (Bottom-Left): Low accuracy, Counter-stereotypical bias
    ax.axvspan(-0.1, 0, alpha=0.05, color='blue',
               ymin=0, ymax=(mean_accuracy - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))
    # Quadrant 4 (Bottom-Right): Low accuracy, Stereotypical bias
    ax.axvspan(0, 0.1, alpha=0.05, color='red',
               ymin=0, ymax=(mean_accuracy - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))

    # Add quadrant labels
    ax.text(-0.09, 0.47, 'High Accuracy\nCounter-Stereotypical',
            fontsize=10, ha='left', va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(-0.09, 0.05, 'Low Accuracy\nCounter-Stereotypical',
            fontsize=10, ha='left', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(0.07, 0.47, 'High Accuracy\nStereotypical',
            fontsize=10, ha='left', va='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.text(0.01, 0.05, 'Low Accuracy\nStereotypical',
            fontsize=10, ha='left', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # Calculate correlations
    corr_bias_acc, p_bias_acc = spearmanr(plot_data['soft_ambig_bias'], plot_data['soft_ambig_accuracy'])

    # Labels and title
    ax.set_xlabel('Ambiguous Bias Score\n(+ = Stereotypical predictions, - = Counter-stereotypical predictions)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Ambiguous Accuracy (0-1 scale)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Figure 7: Ambiguous Bias vs. Ambiguous Accuracy Across Models\n(Horizontal line at mean accuracy = {mean_accuracy:.3f})',
        fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set y-axis limits
    ax.set_ylim(-0.02, 1.02)

    # Add correlation annotation
    ax.text(0.95, 0.05, f'Spearman ρ = {corr_bias_acc:.3f}\np = {p_bias_acc:.4f}',
            transform=ax.transAxes, fontsize=11,
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add note about quadrants
    ax.text(0.02, 0.98,
            'Quadrants:\n• Top = Above avg accuracy\n• Bottom = Below avg accuracy\n• Right = Stereotypical bias\n• Left = Counter-stereotypical bias',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 7 to {save_path}")

    return fig, {
        'correlation_bias_accuracy': corr_bias_acc,
        'p_value_bias_accuracy': p_bias_acc,
        'mean_accuracy': mean_accuracy,
        'n_models': len(plot_data)
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_thesis_figures(df_bias, df_model_sizes, df_group, output_dir="./thesis_figures"):
    """Generate all thesis figures including Figure 6 for disability groups."""

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("Preparing data for thesis figures...")
    data = prepare_thesis_data(df_bias, df_model_sizes)
    print(f"Data prepared: {len(data['model_avg_signed'])} models, {len(data['heatmap_data'].columns)} categories")

    # Export CSV data
    export_all_csv_data(data, output_dir)

    # Export Figure 6 CSV
    csv_dir = Path(output_dir) / "csv_data"
    csv_dir.mkdir(parents=True, exist_ok=True)
    export_figure_6_csv(data, df_group, csv_dir / "figure_6_disability_groups_data.csv")
    export_figure_7_csv(data, csv_dir / "figure_7_ambig_bias_vs_accuracy_data.csv")

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING THESIS FIGURES")
    print("=" * 60)

    # Figure 1
    print("\nFigure 1: Model Size vs. Bias (RQ1)")
    fig1, stats1 = figure_1_size_vs_bias(data, save_path=f"{output_dir}/figure_1_size_vs_bias.jpg")
    plt.close(fig1)

    # Figure 2
    print("\nFigure 2: Directional Bias Heatmap (RQ2 & RQ3)")
    fig2, stats2 = figure_2_directional_heatmap(data, save_path=f"{output_dir}/figure_2_directional_heatmap.jpg")
    plt.close(fig2)

    # Figure 3
    print("\nFigure 3: Signed Bias by Category (RQ3 - Direction)")
    fig3, stats3 = figure_3_signed_bias_by_category(data, save_path=f"{output_dir}/figure_3_signed_bias.jpg")
    plt.close(fig3)

    # Figure 4
    print("\nFigure 4: Bias Distribution by Group (RQ2)")
    fig4, stats4 = figure_4_bias_distribution_by_group(data, save_path=f"{output_dir}/figure_4_bias_distribution.jpg")
    plt.close(fig4)

    # Figure 5
    print("\nFigure 5: Absolute Bias Magnitude (RQ3 - Magnitude)")
    fig5, stats5 = figure_5_bias_magnitude_by_category(data, save_path=f"{output_dir}/figure_5_bias_magnitude.jpg")
    plt.close(fig5)

    # Figure 6
    print("\nFigure 6: Disability Groups Bias Heatmap")
    fig6, matrix6 = figure_6_disability_groups_heatmap(
        data, df_group,
        save_path=f"{output_dir}/figure_6_disability_groups_heatmap.jpg"
    )
    plt.close(fig6)

    # Figure 7
    print("\nFigure 7: Ambiguous Bias vs. Ambiguous Accuracy")
    fig7, stats7 = figure_7_ambig_bias_vs_accuracy(
        data,
        save_path=f"{output_dir}/figure_7_ambig_bias_vs_accuracy.jpg"
    )
    plt.close(fig7)

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nFigures saved to: {output_dir}/")
    print(f"CSV data saved to: {output_dir}/csv_data/")

    print("\nFigure Reference for Thesis:")
    print("  • Figure 1: RQ1 - Model size predicts bias?")
    print("  • Figure 2: RQ2 & RQ3 - Which groups, which direction?")
    print("  • Figure 3: RQ3 - Direction of bias per group")
    print("  • Figure 4: RQ2 - Bias consistency across models")
    print("  • Figure 5: RQ3 - Magnitude of bias per group")

    return {
        'fig1': (fig1, stats1),
        'fig2': (fig2, stats2),
        'fig3': (fig3, stats3),
        'fig4': (fig4, stats4),
        'fig5': (fig5, stats5),
        'fig6': (fig6, matrix6),
        'data': data
    }


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Load your data
    df_bias = pd.read_csv('summary.csv')
    df_model_sizes = pd.read_csv('model_sizes.csv')
    df_group = pd.read_csv('group_df.csv')

    # Generate all figures and CSV exports
    results = generate_all_thesis_figures(df_bias, df_model_sizes,df_group, output_dir="./thesis_figures")

