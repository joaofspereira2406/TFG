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
from fontTools.ttLib.tables.S_V_G_ import doc_index_entry_format_0Size
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

def prepare_thesis_data(df_bias, df_model_sizes,df_group,cat_df):
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


    print("df_bias columns:", df_bias.columns.tolist())
    print("df_model_sizes columns:", df_model_sizes.columns.tolist())
    print("df_model_sizes head:\n", df_model_sizes.head())

    # Create category from dataset (remove 'BBQ-' prefix)
    df_bias = df_bias.copy()
    df_bias['category'] = df_bias['dataset'].str.replace('_', '-').str.replace('BBQ-', '').str.replace('-', ' ')
    df_bias['model'] = df_bias['model'].apply(clean_model_name)
    df_bias['category'] = df_bias['category'].apply(clean_category_name)
    df_group['category'] = df_group['dataset'].str.replace('_', '-').str.replace('BBQ-', '').str.replace('-', ' ')
    df_group['model'] = df_group['model'].apply(clean_model_name)
    df_group['category'] = df_group['category'].apply(clean_category_name)
    cat_df['category'] = cat_df['dataset'].str.replace('_', '-').str.replace('BBQ-', '').str.replace('-', ' ')
    cat_df['model'] = cat_df['model'].apply(clean_model_name)
    cat_df['category'] = cat_df['category'].apply(clean_category_name)
    df_model_sizes['model'] = df_model_sizes['model'].apply(clean_model_name    )


    def data_f1(df_bias, df_model_sizes):
        """Create data for Figure 1 (Size vs Bias)"""

        df_merged = df_bias.merge(df_model_sizes, on='model', how='inner')

        # Check required columns exist for aggregation
        required_cols = ['soft_disambig_bias', 'soft_ambig_bias', 'soft_ambig_accuracy']
        for col in required_cols:
            if col not in df_merged.columns:
                print(f"[WARNING] Column '{col}' missing from merged data!")

        # Perform aggregation
        try:
            data_f1 = df_merged.groupby(['model', 'parameters']).agg({
                'soft_disambig_bias': lambda x: x.abs().mean(),
                'soft_ambig_bias': lambda x: x.abs().mean(),
                'soft_ambig_accuracy': 'mean'
            }).reset_index()
            # Select only needed columns
            data_f1 = data_f1[['model', 'parameters', 'soft_ambig_bias']]
            return data_f1

        except Exception as e:
            print(f"[ERROR] During aggregation: {e}")
            return pd.DataFrame(columns=['model', 'parameters', 'soft_ambig_bias'])

    def data_f2_3_4_5(cat_df):
        df_ambig_bias = cat_df[cat_df['metric'] == 'soft_ambig_bias'][['model', 'category', 'value']].rename(
            columns={'value': 'soft_ambig_bias'})

        return df_ambig_bias

    def data_f6(df_group, category):
        df_grouped = df_group[df_group['category'] == category].copy()

        df_grouped = df_grouped[
            ['model', 'group', 'metric', 'value']
        ]

        # if you only want one metric, filter it properly
        df_grouped = df_grouped[df_grouped['metric'] == 'ambig_bias']

        return df_grouped


    # For Figure 7: Signed values (preserve direction!)
    def data_f7(df_bias):
        data_f7 = df_bias.groupby(['model']).agg({
            'soft_disambig_bias': lambda x: x.mean(),  # abs first, then mean
            'soft_ambig_bias': lambda x: x.mean(),  # abs first, then mean
            'soft_ambig_accuracy': 'mean'
        }).reset_index()
        data_f7=data_f7[['model','soft_ambig_bias','soft_ambig_accuracy']]
        return data_f7




    return {
        'data_f1':data_f1(df_bias,df_model_sizes),
        'data_f2_f3_f4_f5': data_f2_3_4_5(cat_df),
        'data_f6':data_f6(df_group, "Disability"),
        'data_f7':data_f7(df_bias)
    }

def getorder(arg):
    models=[
        "Recognai_zeroshot_selectra_small",  # 22,409,216
        "cross_encoder_nli_MiniLM2_L6_H768",  # 82,118,400
        "cross_encoder_nli_deberta_base",  # 138,601,728
        "NbAiLab_nb_bert_base_mnli",  # 177,853,440
        "valhalla_distilbart_mnli_12_3",  # 255,121,408
        "MoritzLaurer_mDeBERTa_v3_base_mnli_xnli",  # 278,218,752
        "roberta_large_mnli",  # 355,359,744
        "facebook_bart_large_mnli",  # 406,291,456
        "MoritzLaurer_deberta_v3_large_zeroshot_v2.0",  # 434,012,160
        "Jiva_xlm_roberta_large_it_mnli"  # 559,890,432
    ]
    categories=[
        "Age",
        "Disability",
        "Gender",
        "Nationality",
        "Appearance",
        "Race/Ethnicity",
        "Race × Gender",
        "SES",
        "Religion",
        "Sexual Orientation"
    ]

    if arg == "model":
        return models
    elif arg == "category":
        return categories

# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def figure_1_size_vs_bias(data, save_path=None):
    """FIGURE 1: Does model size predict measurable bias?"""

    # Check for null values before dropping

    # Drop NA
    plot_data = data.dropna(subset=['parameters', 'soft_ambig_bias'])

    if len(plot_data) == 0:
        print("[ERROR] No valid data points after dropping NA!")
        return None, None


    if (plot_data['parameters'] <= 0).any():
        print(
            f"[WARNING] Non-positive parameters found: {plot_data[plot_data['parameters'] <= 0]['parameters'].tolist()}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate scaled parameters
    params_millions = plot_data['parameters'] / 1e6

    # Create scatter plot
    try:
        scatter = ax.scatter(
            params_millions,
            plot_data['soft_ambig_bias'],
            s=150,
            c=range(len(plot_data)),
            cmap='viridis',
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5
        )
    except Exception as e:
        print(f"[ERROR] Failed to create scatter plot: {e}")
        raise

    # Add annotations
    for idx, row in plot_data.iterrows():
        try:
            ax.annotate(
                row['model'],
                (row['parameters'] / 1e6, row['soft_ambig_bias']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
        except Exception as e:
            print(f"[ERROR] Failed to annotate point {idx}: {e}")

    # Add trend line
    if len(plot_data) >= 3:

        try:
            log_params = np.log10(plot_data['parameters'])

            z = np.polyfit(log_params, plot_data['soft_ambig_bias'], 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(plot_data['parameters'].min()),
                                  np.log10(plot_data['parameters'].max()), 100)
            ax.plot(x_trend / 1e6, p(np.log10(x_trend)),
                    'r--', alpha=0.5, label=f'Trend: slope={z[0]:.3f}')
        except Exception as e:
            print(f"[ERROR] Failed to add trend line: {e}")
    # Calculate correlation
    try:
        corr, p_value = spearmanr(plot_data['parameters'], plot_data['soft_ambig_bias'])
    except Exception as e:
        print(f"[ERROR] Failed to calculate correlation: {e}")
        corr, p_value = None, None

    # Set labels and title
    ax.set_xlabel('Model Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel('Mean Absolute Bias Score |bias|', fontsize=12, fontweight='bold')
    ax.set_title(
        'Model Size vs. Bias Magnitude\n(Higher values indicate the model is less unbiased, showing stronger stereotypical or anti-stereotypical associations)',
        fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add correlation text
    if corr is not None:
        ax.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {p_value:.4f}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()

    # Save if requested
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Figure 1 to {save_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save figure: {e}")

    # FIXED: Use plot_data instead of data
    result_dict = {
        'model': plot_data['model'].tolist() if 'model' in plot_data.columns else None,
        'parameters': plot_data['parameters'].tolist(),
        'bias': plot_data['soft_ambig_bias'].tolist(),
        'correlation': corr,
        'p_value': p_value,
        'n_models': len(plot_data)
    }

    return fig, result_dict


def figure_2_directional_heatmap(data, save_path=None):
    """FIGURE 2: Directional bias heatmap"""

    try:
        heatmap_data = data.pivot(
            index='model',
            columns='category',
            values='soft_ambig_bias'  # or 'value'
        )

    except Exception as e:
        print(f"[ERROR] Pivot failed: {e}")
        raise

    # Check for missing data after pivot

    # Get expected orders



    # Filter to present values
    model_order_local = ['Selectra-S', 'MiniLM-L6', 'DeBERTa-B', 'NB-BERT', 'DistilBART',
                         'mDeBERTa-B', 'RoBERTa-L', 'BART-L', 'DeBERTa-L', 'XLM-R-L']

    category_order_local = ['Age', 'Disability', 'Gender', 'Nationality', 'Appearance',
                            'Race/Ethnicity', 'Race × Gender', 'Race × SES', 'SES',
                            'Religion', 'Sexual Orientation']
    models_present = [m for m in model_order_local if m in heatmap_data.index]
    categories_present = [c for c in category_order_local if c in heatmap_data.columns]



    # Check for categories that might have been dropped
    all_categories_in_data = set(data['category'].unique()) if 'category' in data.columns else set()
    categories_in_pivot = set(heatmap_data.columns)
    missing_categories = all_categories_in_data - categories_in_pivot
    if missing_categories:
        print(f"[WARNING] Categories present in original data but missing after pivot: {missing_categories}")
        for cat in missing_categories:
            cat_data = data[data['category'] == cat]
            print(f"  - {cat}: {len(cat_data)} rows, all NaN? {cat_data['soft_ambig_bias'].isna().all()}")

    # Reindex

    heatmap_data = heatmap_data.reindex(index=models_present, columns=categories_present)


    # Check if any rows/columns are all NaN
    rows_all_nan = heatmap_data.isna().all(axis=1)
    cols_all_nan = heatmap_data.isna().all(axis=0)

    if rows_all_nan.any():
        print(f"[WARNING] Rows with all NaN values: {heatmap_data.index[rows_all_nan].tolist()}")
    if cols_all_nan.any():
        print(f"[WARNING] Columns with all NaN values: {heatmap_data.columns[cols_all_nan].tolist()}")


    # Check value range for colorbar
    if not heatmap_data.empty and not heatmap_data.values.size == 0:
        valid_values = heatmap_data.values[~np.isnan(heatmap_data.values)]


    # Original plotting code
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

    # Count how many values are actually plotted
    plotted_values = 0
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                plotted_values += 1
                text_color = 'white' if abs(value) > 0.15 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=text_color, fontsize=8, fontweight='bold')



    ax.set_xlabel('Demographic Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(
        'Stereotypical Bias Direction by Model and Category\n(Red = Stereotype-consistent predictions, Blue = Counter-stereotypical predictions)',
        fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 2 to {save_path}")

    print("=" * 60 + "\n")

    return fig, {
        'shape': heatmap_data.shape,
        'max_bias': heatmap_data.abs().max().max(),
        'heatmap_matrix': heatmap_data  # Add this line
    }


def figure_3_signed_bias_by_category(data, save_path=None):
    """FIGURE 3: Signed bias by category"""

    # 1. ORDER FIRST (safe, no indexing hacks)
    order_categories = getorder("category")

    # 2. AGGREGATE FIRST (removes duplicates properly)
    grouped = data.groupby('category').agg(
        mean_signed=('soft_ambig_bias', 'mean'),
        std=('soft_ambig_bias', 'std'),
        count=('soft_ambig_bias', 'count')
    ).reset_index()

    # 3. SEM
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])

    # 4. ABS for sorting (safe after aggregation)
    grouped['abs_signed'] = grouped['mean_signed'].abs()

    # 5. ORDER categories safely
    grouped = grouped.set_index('category').reindex(
        [c for c in order_categories if c in grouped['category'].values]
    ).reset_index()

    # 6. SORT (optional but stable now)
    grouped = grouped.sort_values('abs_signed', ascending=True)

    # 7. PLOT VALUES
    categories = grouped['category'].values
    means = grouped['mean_signed'].values
    errors = grouped['sem'].values * 1.96

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(categories))

    ax.barh(
        y_pos,
        means,
        xerr=errors,
        capsize=3,
        color=['red' if x > 0 else 'blue' for x in means],
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=11)

    ax.set_xlabel(
        'Mean Signed Bias Score\n(+ stereotype, - counter-stereotype)',
        fontsize=12, fontweight='bold'
    )

    ax.set_title(
        'Direction of Stereotypical Bias by Demographic Category',
        fontsize=14, fontweight='bold'
    )

    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 3 to {save_path}")

    result_dict = {
        'categories': categories.tolist(),
        'means': means.tolist(),
        'errors': errors.tolist(),
        'counts': grouped['count'].tolist(),
        'std': grouped['std'].tolist(),
        'abs_signed': grouped['abs_signed'].tolist()
    }

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 3 to {save_path}")

    return fig, result_dict


def figure_4_bias_distribution_by_group(data, save_path=None):
    """FIGURE 4: Bias distribution per group"""

    # CORRECTED: Use the original bias scores directly, no aggregation needed
    plot_data = data[['model', 'category', 'soft_ambig_bias']].copy()
    plot_data = plot_data.rename(columns={'soft_ambig_bias': 'bias_score'})

    # Reorder categories
    category_order = getorder('category')

    # Get categories present in data
    actual_categories = plot_data['category'].unique().tolist()

    # Check for missing categories
    categories_present = [c for c in category_order if c in plot_data['category'].values]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for boxplot - use raw bias scores per category
    boxplot_data = []
    for cat in categories_present:
        cat_values = plot_data[plot_data['category'] == cat]['bias_score'].dropna()
        boxplot_data.append(cat_values)

    # Create boxplot
    bp = ax.boxplot(boxplot_data,
                    labels=categories_present,
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    widths=0.7)

    # Color boxes based on median bias
    for i, cat in enumerate(categories_present):
        median_val = plot_data[plot_data['category'] == cat]['bias_score'].median()
        if median_val > 0.05:
            color = 'lightcoral'
        elif median_val < -0.05:
            color = 'lightblue'
        else:
            color = 'lightgray'
        bp['boxes'][i].set_facecolor(color)
        bp['boxes'][i].set_alpha(0.7)

    # Rest of plotting code
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Demographic Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bias Score\n(+ = Stereotypical, - = Counter-stereotypical)',
                  fontsize=12, fontweight='bold')
    ax.set_title(
        'Bias Variability Across Models by Category\n(Red = Stereotype-consistent, Blue = Counter-stereotypical, Gray = Neutral)',
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

    # Calculate boxplot statistics for each category
    boxplot_stats = []
    for cat in categories_present:
        cat_data = plot_data[plot_data['category'] == cat]['bias_score'].dropna()
        if len(cat_data) > 0:
            stats = {
                'category': cat,
                'median': cat_data.median(),
                'mean': cat_data.mean(),
                'q1': cat_data.quantile(0.25),
                'q3': cat_data.quantile(0.75),
                'iqr': cat_data.quantile(0.75) - cat_data.quantile(0.25),
                'whisker_low': max(cat_data.min(),
                                   cat_data.quantile(0.25) - 1.5 * (cat_data.quantile(0.75) - cat_data.quantile(0.25))),
                'whisker_high': min(cat_data.max(),
                                    cat_data.quantile(0.75) + 1.5 * (
                                            cat_data.quantile(0.75) - cat_data.quantile(0.25))),
                'outliers': cat_data[
                    (cat_data < (cat_data.quantile(0.25) - 1.5 * (cat_data.quantile(0.75) - cat_data.quantile(0.25)))) |
                    (cat_data > (cat_data.quantile(0.75) + 1.5 * (
                            cat_data.quantile(0.75) - cat_data.quantile(0.25))))].tolist(),
                'n_models': len(cat_data)
            }
            boxplot_stats.append(stats)

    return fig, {
        'boxplot_stats': pd.DataFrame(boxplot_stats),
        'categories': categories_present,
        'n_models': len(plot_data['model'].unique())
    }


def figure_5_bias_magnitude_by_category(data, save_path=None):
    """FIGURE 5: Absolute bias magnitude by category"""

    data = data[['soft_ambig_bias', 'category', 'model']].copy()

    # absolute bias
    data['abs_bias'] = data['soft_ambig_bias'].abs()

    # compute mean + SEM per category
    summary = data.groupby('category')['abs_bias'].agg(
        mean_absolute='mean',
        std='std',
        sem=lambda x: x.std(ddof=1) / np.sqrt(x.count()) if x.count() > 1 else 0
    )

    # sort
    summary = summary.sort_values('mean_absolute', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    categories = summary.index
    magnitudes = summary['mean_absolute'].values
    errors = summary['sem'].values * 1.96  # 95% CI

    bars = ax.barh(
        categories,
        magnitudes,
        xerr=errors,
        capsize=3,
        color='coral',
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )

    for i, (mag, err) in enumerate(zip(magnitudes, errors)):
        ax.text(mag + err + 0.005, i, f'{mag:.3f}',
                va='center', fontsize=9)

    ax.set_xlabel(
        'Mean Absolute Bias Score |bias| (0 = No bias, Higher = Stronger stereotypical associations)',
        fontsize=12, fontweight='bold'
    )

    ax.set_title(
        'Stereotypical Bias Magnitude by Demographic Category\n'
        '(Higher values indicate categories where models show stronger stereotypical associations)',
        fontsize=14, fontweight='bold'
    )

    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(magnitudes + errors) * 1.1)

    ax.text(
        0.95, 0.02,
        'Note: Magnitude ignores direction\n(both stereotype-consistent and counter-stereotypical bias contribute equally)',
        transform=ax.transAxes,
        fontsize=9,
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 5 to {save_path}")

    return fig, {
        'mean_absolute': summary['mean_absolute'].to_dict(),
        'sem': summary['sem'].to_dict(),
        'std': summary['std'].to_dict(),
        'categories': summary.index.tolist()
    }


def figure_6_disability_groups_heatmap(data, save_path=None):
    """FIGURE 6: Directional bias heatmap focusing on Disability groups."""

    # Check if 'metric' column exists and drop it if present
    if 'metric' in data.columns:
        data = data[['model', 'group', 'value']].copy()

    if len(data) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No disability group data available',
                ha='center', va='center', fontsize=14)
        ax.set_title('Figure 6: Disability Groups Heatmap (No Data)')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, None

    # Create pivot table
    heatmap_matrix = data.pivot_table(
        index='model',
        columns='group',
        values='value',
        aggfunc='mean'
    )

    # Use LOCAL model order for Figure 6 (your cleaned model names)
    model_order_local = ['Selectra-S', 'MiniLM-L6', 'DeBERTa-B', 'NB-BERT', 'DistilBART',
                         'mDeBERTa-B', 'RoBERTa-L', 'BART-L', 'DeBERTa-L', 'XLM-R-L']

    # Reindex with local order
    models_present = [m for m in model_order_local if m in heatmap_matrix.index]
    heatmap_matrix = heatmap_matrix.reindex(index=models_present)

    # Create plot
    fig_width = max(8, len(heatmap_matrix.columns) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # Use appropriate vmin/vmax based on data range
    vmin = -0.3
    vmax = 0.3

    im = ax.imshow(
        heatmap_matrix.values,
        cmap='RdBu_r',
        aspect='auto',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )

    ax.set_xticks(np.arange(len(heatmap_matrix.columns)))
    ax.set_yticks(np.arange(len(heatmap_matrix.index)))

    # Clean up group names for display
    group_labels = [g.replace('_', ' ').capitalize() for g in heatmap_matrix.columns]

    ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax.set_yticklabels(heatmap_matrix.index, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Bias Score\n(+ = Stereotypical, - = Counter-stereotypical)', fontsize=10)

    # Add value annotations
    for i in range(len(heatmap_matrix.index)):
        for j in range(len(heatmap_matrix.columns)):
            value = heatmap_matrix.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if abs(value) > 0.15 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                        color=text_color, fontsize=9, fontweight='bold')

    ax.set_xlabel('Disability Group', fontweight='bold', fontsize=12)
    ax.set_ylabel('Model', fontweight='bold', fontsize=12)
    ax.set_title(
        'Stereotypical Bias Toward Disability Groups Across Models',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

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
    plot_data = data[['model','soft_ambig_bias', 'soft_ambig_accuracy']].copy()
    plot_data = plot_data.dropna()

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
        print(plot_data.columns.tolist())
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
        'n_models': len(plot_data),
        'data_points': plot_data[['model_clean', 'soft_ambig_bias', 'soft_ambig_accuracy']].copy()
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_thesis_figures(df_bias, df_model_sizes,df_group,cat_df, output_dir="./thesis_figures"):
    """Generate all thesis figures including Figure 6 for disability groups."""

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("Preparing data for thesis figures...")
    data = prepare_thesis_data(df_bias, df_model_sizes,df_group,cat_df)

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING THESIS FIGURES")
    print("=" * 60)

    # Figure 1
    print("\nFigure 1: Model Size vs. Bias (RQ1)")
    fig1, stats1 = figure_1_size_vs_bias(data['data_f1'], save_path=f"{output_dir}/figure_1_size_vs_bias.png")
    ###plt.close(fig1)

    # Figure 2
    print("\nFigure 2: Directional Bias Heatmap (RQ2 & RQ3)")
    fig2, stats2 = figure_2_directional_heatmap(data['data_f2_f3_f4_f5'], save_path=f"{output_dir}/figure_2_directional_heatmap.png")
    plt.close(fig2)

    # Figure 3
    print("\nFigure 3: Signed Bias by Category (RQ3 - Direction)")
    fig3, stats3 = figure_3_signed_bias_by_category(data['data_f2_f3_f4_f5'], save_path=f"{output_dir}/figure_3_signed_bias.png")
    plt.close(fig3)

    # Figure 4
    print("\nFigure 4: Bias Distribution by Group (RQ2)")
    fig4, stats4 = figure_4_bias_distribution_by_group(data['data_f2_f3_f4_f5'], save_path=f"{output_dir}/figure_4_bias_distribution.png")
    plt.close(fig4)

    # Figure 5
    print("\nFigure 5: Absolute Bias Magnitude (RQ3 - Magnitude)")
    fig5, stats5 = figure_5_bias_magnitude_by_category(data['data_f2_f3_f4_f5'], save_path=f"{output_dir}/figure_5_bias_magnitude.png")
    plt.close(fig5)

    # Figure 6
    print("\nFigure 6: Disability Groups Bias Heatmap")
    fig6, matrix6 = figure_6_disability_groups_heatmap(data['data_f6'], save_path=f"{output_dir}/figure_6_disability_groups_heatmap.png")
    plt.close(fig6)

    # Figure 7
    print("\nFigure 7: Ambiguous Bias vs. Ambiguous Accuracy")
    fig7, stats7 = figure_7_ambig_bias_vs_accuracy(data['data_f7'], save_path=f"{output_dir}/figure_7_ambig_bias_vs_accuracy.png")
    plt.close(fig7)

    # Summary
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nFigures saved to: {output_dir}")
    print(f"CSV data saved to: {output_dir}/csv_data2/")

    print("\nFigure Reference for Thesis:")
    print("  • Figure 1: RQ1 - Model size predicts bias?")
    print("  • Figure 2: RQ2 & RQ3 - Which groups, which direction?")
    print("  • Figure 3: RQ3 - Direction of bias per group")
    print("  • Figure 4: RQ2 - Bias consistency across models")
    print("  • Figure 5: RQ3 - Magnitude of bias per group")
    print("  • Figure 6: Disability Groups Bias Heatmap")

    export_figure_stats({
        'fig1': stats1,
        'fig2': stats2,
        'fig3': stats3,
        'fig4': stats4,
        'fig5': stats5,
        'fig6': matrix6,
        'fig7': stats7,  # Note: changed from 'fig7': fig7 to use stats7
    }, output_dir)

    return {
        #'fig1': (fig1, stats1),
        'fig2': (fig2, stats2),
        'fig3': (fig3, stats3),
        'fig4': (fig4, stats4),
        'fig5': (fig5, stats5),
        'fig6': (fig6, matrix6),
        #'data': data
    }


import os
import pandas as pd


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

def export_figure_stats(figures_dict, output_dir='./'):
    """
    Export stats from figure functions to CSV files.
    figures_dict should contain just the stats objects, not (fig, stats) tuples
    """

    print(f"\n🔍 DEBUG: Starting export_figure_stats")
    print(f"🔍 DEBUG: output_dir = {output_dir}")
    print(f"🔍 DEBUG: figures_dict keys = {list(figures_dict.keys())}")

    # Create the csv_data2 directory if it doesn't exist
    csv_dir = f"{output_dir}/csv_data2/"
    print(f"🔍 DEBUG: Attempting to create directory: {csv_dir}")
    os.makedirs(csv_dir, exist_ok=True)
    print(f"🔍 DEBUG: Directory ready: {os.path.exists(csv_dir)}")
    if 'fig1' in figures_dict:
        stats1 = figures_dict['fig1']
        if isinstance(stats1, dict) and 'model' in stats1:
            # Create DataFrame from the dict
            df1 = pd.DataFrame({
                'model': stats1['model'],
                'parameters': stats1['parameters'],
                'soft_bias_complete': stats1['bias']
            })
            df1['model_clean'] = df1['model'].apply(clean_model_name)
            df1 = df1.sort_values('parameters').reset_index(drop=True)
            df1[['model_clean', 'parameters', 'soft_bias_complete']].to_csv(f"{csv_dir}fig1_model_bias_table.csv", index=False)
            print(f"✅ Exported fig1 to {csv_dir}fig1_model_bias_table.csv")
        else:
            print(f"❌ ERROR: stats1 is not a dict, it's {type(stats1)}")
    else:
        print(f"⚠️ DEBUG: 'fig1' not found in figures_dict")
    # Figure 2 - shape and max_bias
    if 'fig2' in figures_dict:
        stats2 = figures_dict['fig2']
        if 'heatmap_matrix' in stats2:
            stats2['heatmap_matrix'].to_csv(f"{csv_dir}fig2_heatmap_matrix.csv")
            print(f"✅ Exported fig2 heatmap matrix to {csv_dir}fig2_heatmap_matrix.csv")
        else:
            print(f"❌ ERROR: stats2 is not a dict, it's {type(stats2)}")
    else:
        print(f"⚠️ DEBUG: 'fig2' not found in figures_dict")

    # Figure 3 - signed and absolute bias per category
    if 'fig3' in figures_dict:
        print(f"🔍 DEBUG: Processing fig3")
        stats3 = figures_dict['fig3']
        print(f"🔍 DEBUG: stats3 type = {type(stats3)}")

        if isinstance(stats3, dict):
            df3 = pd.DataFrame(stats3)
            save_path = f"{csv_dir}fig3_stats.csv"
            df3.to_csv(save_path)
            print(f"✅ Exported fig3 stats to {save_path}")
        else:
            print(f"❌ ERROR: stats3 is not a dict, it's {type(stats3)}")
    else:
        print(f"⚠️ DEBUG: 'fig3' not found in figures_dict")

    # Figure 4 - categories and number of models
    if 'fig4' in figures_dict:
        stats4 = figures_dict['fig4']

        if isinstance(stats4, dict):
            # Save boxplot statistics (median, Q1, Q3, IQR, whiskers)
            if 'boxplot_stats' in stats4:
                stats4['boxplot_stats'].to_csv(f"{csv_dir}fig4_boxplot_stats.csv", index=False)
                print(f"✅ Exported fig4 boxplot stats to {csv_dir}fig4_boxplot_stats.csv")

            # Save metadata
            metadata = pd.DataFrame([{
                'n_models': stats4['n_models'],
                'n_categories': len(stats4['categories'])
            }])
            metadata.to_csv(f"{csv_dir}fig4_metadata.csv", index=False)
            print(f"✅ Exported fig4 metadata to {csv_dir}fig4_metadata.csv")
        else:
            print(f"❌ ERROR: stats4 is not a dict, it's {type(stats4)}")
    else:
        print(f"⚠️ DEBUG: 'fig4' not found in figures_dict")

    # Figure 5 - absolute bias per category
    if 'fig5' in figures_dict:
        stats5 = figures_dict['fig5']

        if isinstance(stats5, dict):
            # Create DataFrame with all info
            df5 = pd.DataFrame({
                'category': stats5['categories'],
                'mean_absolute_bias': [stats5['mean_absolute'][cat] for cat in stats5['categories']],
                'sem': [stats5['sem'][cat] for cat in stats5['categories']],
                'std':[stats5['std'][cat] for cat in stats5['categories']],
                'ci_95': [stats5['sem'][cat] * 1.96 for cat in stats5['categories']]
            })
            df5 = df5.sort_values('mean_absolute_bias', ascending=True)
            df5.to_csv(f"{csv_dir}fig5_bias_magnitude.csv", index=False)
            print(f"✅ Exported fig5 bias magnitude to {csv_dir}fig5_bias_magnitude.csv")
        else:
            print(f"❌ ERROR: stats5 is not a dict, it's {type(stats5)}")
    else:
        print(f"⚠️ DEBUG: 'fig5' not found in figures_dict")

    # Figure 6 - heatmap matrix
    if 'fig6' in figures_dict:
        print(f"🔍 DEBUG: Processing fig6")
        matrix6 = figures_dict['fig6']
        print(f"🔍 DEBUG: matrix6 type = {type(matrix6)}")

        save_path = f"{csv_dir}fig6_heatmap_matrix.csv"
        if isinstance(matrix6, pd.DataFrame):
            matrix6.to_csv(save_path)
        elif isinstance(matrix6, (list, dict, np.ndarray)):
            pd.DataFrame(matrix6).to_csv(save_path)
        else:
            print(f"❌ ERROR: Unexpected matrix6 type: {type(matrix6)}")
        print(f"✅ Exported fig6 heatmap matrix to {save_path}")
    else:
        print(f"⚠️ DEBUG: 'fig6' not found in figures_dict")

    # Figure 7 - correlation and accuracy stats
    if 'fig7' in figures_dict:
        stats7 = figures_dict['fig7']

        if isinstance(stats7, dict):
            # Save correlation summary
            df7_summary = pd.DataFrame([{
                'correlation_bias_accuracy': stats7['correlation_bias_accuracy'],
                'p_value_bias_accuracy': stats7['p_value_bias_accuracy'],
                'mean_accuracy': stats7['mean_accuracy'],
                'n_models': stats7['n_models']
            }])
            df7_summary.to_csv(f"{csv_dir}fig7_correlation_stats.csv", index=False)
            print(f"✅ Exported fig7 correlation stats to {csv_dir}fig7_correlation_stats.csv")

            # Save data points (coordinates for each model)
            if 'data_points' in stats7:
                stats7['data_points'].to_csv(f"{csv_dir}fig7_data_points.csv", index=False)
                print(f"✅ Exported fig7 data points to {csv_dir}fig7_data_points.csv")
        else:
            print(f"❌ ERROR: stats7 is not a dict, it's {type(stats7)}")
    else:
        print(f"⚠️ DEBUG: 'fig7' not found in figures_dict")

    print("\n📊 Export process complete!")


# Usage example:
# figures = {
#     'fig2': (fig2, stats2),
#     'fig3': (fig3, stats3),
#     'fig4': (fig4, stats4),
#     'fig5': (fig5, stats5),
#     'fig6': (fig6, matrix6),
#     'fig7': (fig7, stats7),
#     'data': data
# }
# 
# export_figure_stats(figures, output_dir='./metrics/')
# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Load your data
    df_bias = pd.read_csv('./metrics/summary.csv')
    cat_df= pd.read_csv('./metrics/cat_df.csv')
    df_model_sizes = pd.read_csv('./metrics/model_sizes.csv')
    df_group = pd.read_csv('./metrics/group_df.csv')

    # Generate all figures and CSV exports
    #results = generate_all_thesis_figures(df_bias, df_model_sizes,df_group, output_dir="./metrics/thesis_figures")
    #data = prepare_thesis_data(df_bias,df_model_sizes, df_group,cat_df)
    #print(type(data))
    generate_all_thesis_figures(df_bias, df_model_sizes, df_group, cat_df, output_dir="./thesis_figures2")


