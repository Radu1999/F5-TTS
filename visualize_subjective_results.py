import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os
import re
import math

STIMULUS_NAME_MAP = {
    'C1': 'Ro-F5TTS',
    'C2': 'F5TTS',
    'C3': 'MMS-TTS-RON',
    'C4': 'Real Speaker',
}

def map_stimulus_name(code):
    return STIMULUS_NAME_MAP.get(str(code), str(code))

# Fixed colors for stimuli across all plots
STIMULUS_COLOR_MAP = {
    'C1': '#1f77b4',  # blue
    'C2': '#ff7f0e',  # orange
    'C3': '#2ca02c',  # green
    'C4': '#d62728',  # red
}

def get_stimulus_color(code: str) -> str:
    code_str = str(code)
    return STIMULUS_COLOR_MAP.get(code_str, '#7f7f7f')  # fallback gray

# Global font scaling for readability
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

def format_trial_title(trial_id: str) -> str:
    """Convert raw trial ids to human-friendly titles.
    Examples: trial1 -> Trial 1, pron_trial2 -> Pron Trial 2, code_trial3 -> Code Trial 3
    """
    s = str(trial_id)
    m = re.search(r"(\d+)$", s)
    suffix = m.group(1) if m else ""
    if s.startswith('pron_trial') and suffix:
        return f"Trial {suffix}"
    if s.startswith('code_trial') and suffix:
        return f"Trial {suffix}"
    if s.startswith('trial') and suffix:
        return f"Trial {suffix}"
    return s.replace('_', ' ').title()

def load_and_analyze_data(csv_file):
    """Load the CSV file and analyze the data structure"""
    df = pd.read_csv(csv_file)
    
    print("Data shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nUnique session_test_ids:", df['session_test_id'].unique())
    print("\nUnique trial_ids:", sorted(df['trial_id'].unique()))
    print("\nUnique rating_stimulus:", sorted(df['rating_stimulus'].unique()))
    print("\nRating score range:", df['rating_score'].min(), "to", df['rating_score'].max())
    
    return df

def extract_task_from_trial(trial_id):
    """Extract task type from trial_id"""
    if trial_id.startswith('pron_'):
        return 'pronunciation'
    elif trial_id.startswith('code_'):
        return 'code_switching'
    elif trial_id.startswith('trial'):
        return 'voice_cloning'
    else:
        return 'other'

def calculate_statistics(df):
    """Calculate statistics for candle chart visualization"""
    # Add task column
    df['task'] = df['trial_id'].apply(extract_task_from_trial)
    # Group by task, trial, and stimulus to calculate statistics
    stats = df.groupby(['task', 'trial_id', 'rating_stimulus'])['rating_score'].agg([
        'count', 'mean', 'std', 'min', 'max',
        lambda x: np.percentile(x, 25),  # Q1
        lambda x: np.percentile(x, 50),  # Median
        lambda x: np.percentile(x, 75)   # Q3
    ]).round(2)
    
    stats.columns = ['count', 'mean', 'std', 'min', 'max', 'q1', 'median', 'q3']
    stats = stats.reset_index()
    
    return stats

def create_boxplots_per_trial(df, task_name, output_dir):
    """Create one figure per trial with all stimuli side-by-side (simplified)."""
    df_task = df.copy()
    df_task['task'] = df_task['trial_id'].apply(extract_task_from_trial)
    df_task = df_task[df_task['task'] == task_name]
    if df_task.empty:
        print(f"No data found for task: {task_name}")
        return

    trials = sorted(df_task['trial_id'].unique())
    stimuli = sorted(df_task['rating_stimulus'].unique())

    for trial in trials:
        trial_df = df_task[df_task['trial_id'] == trial]
        if trial_df.empty:
            continue

        data = []
        labels = []
        codes_in_plot = []
        for stim in stimuli:
            arr = trial_df.loc[trial_df['rating_stimulus'] == stim, 'rating_score'].values
            if arr.size > 0:
                data.append(arr)
                labels.append(map_stimulus_name(stim))
                codes_in_plot.append(stim)

        if not data:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        ax.set_title(f"{task_name.title()} - {format_trial_title(trial)}", fontsize=24, fontweight='bold')

        bp = ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            patch_artist=True,
            widths=0.6
        )

        # Color boxes consistently by stimulus code
        colors_fixed = [get_stimulus_color(code) for code in codes_in_plot]
        for box, color in zip(bp['boxes'], colors_fixed):
            box.set(facecolor=color, edgecolor='black', linewidth=1.5)
        for median in bp['medians']:
            median.set(color='black', linewidth=2.5)
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1.5)
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1.5)

        # Overlay means
        means = [np.mean(arr) for arr in data]
        ax.scatter(np.arange(1, len(means) + 1), means, c='red', s=60, zorder=3, label='Mean')

        # Axis styling
        ax.set_ylim(-5, 105)
        ax.set_ylabel('Rating Score', fontsize=20)
        ax.set_xlabel('Stimulus', fontsize=20)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')

        # Figure-level concise summary
        overall_mean = trial_df['rating_score'].mean()
        overall_std = trial_df['rating_score'].std()
        total_n = len(trial_df)
        ax.text(
            0.01,
            0.98,
            f"Overall: {overall_mean:.1f} ± {overall_std:.1f} (n={total_n})",
            transform=ax.transAxes,
            va='top',
            fontsize=16,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

        out_path = os.path.join(output_dir, f"boxplot_{task_name}_{trial}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_overall_boxplot(df, task_name, save_path):
    """Create a single overall figure per task aggregating across trials (simplified)."""
    df_task = df.copy()
    df_task['task'] = df_task['trial_id'].apply(extract_task_from_trial)
    df_task = df_task[df_task['task'] == task_name]
    if df_task.empty:
        print(f"No data found for task: {task_name}")
        return

    stimuli = sorted(df_task['rating_stimulus'].unique())
    data = []
    labels = []
    codes_in_plot = []
    for stim in stimuli:
        arr = df_task.loc[df_task['rating_stimulus'] == stim, 'rating_score'].values
        if arr.size > 0:
            data.append(arr)
            labels.append(map_stimulus_name(stim))
            codes_in_plot.append(stim)

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    ax.set_title(f"{task_name.title()}", fontsize=26, fontweight='bold')

    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.7
    )

    colors_fixed = [get_stimulus_color(code) for code in codes_in_plot]
    for box, color in zip(bp['boxes'], colors_fixed):
        box.set(facecolor=color, edgecolor='black', linewidth=1.8)
    for median in bp['medians']:
        median.set(color='black', linewidth=3)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.8)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.8)

    means = [np.mean(arr) for arr in data]
    ax.scatter(np.arange(1, len(means) + 1), means, c='red', s=70, zorder=3, label='Mean')

    ax.set_ylim(-5, 105)
    ax.set_ylabel('Rating Score', fontsize=20)
    ax.set_xlabel('Model', fontsize=20)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')

    overall_mean = df_task['rating_score'].mean()
    overall_std = df_task['rating_score'].std()
    total_n = len(df_task)
    ax.text(
        0.01,
        0.98,
        f"Overall: {overall_mean:.1f} ± {overall_std:.1f} (n={total_n})",
        transform=ax.transAxes,
        va='top',
        fontsize=16,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_grouped_trial_panels(df, trial_prefix, save_path):
    """Create one figure containing subplots for each trial id with the given prefix.
    Each subplot shows all stimuli side-by-side as boxplots. Minimal annotations.
    """
    df_local = df.copy()
    # Filter trials by prefix, e.g., 'trial', 'pron_trial', 'code_trial'
    mask = df_local['trial_id'].astype(str).str.startswith(trial_prefix)
    df_local = df_local[mask]
    if df_local.empty:
        print(f"No data found for prefix: {trial_prefix}")
        return

    def _trial_sort_key(t):
        m = re.search(r"(\d+)$", str(t))
        return int(m.group(1)) if m else float('inf')

    trials = sorted(df_local['trial_id'].unique(), key=_trial_sort_key)
    stimuli = sorted(df_local['rating_stimulus'].unique())

    # Layout: up to 5 columns; for 10 trials -> 2x5 grid
    n = len(trials)
    cols = max(1, min(5, n))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.4, rows * 5.0), sharey=True)
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Consistent colors across stimuli (fixed mapping)
    stim_to_color = {stim: get_stimulus_color(stim) for stim in stimuli}

    for ax_idx, trial in enumerate(trials):
        ax = axes_flat[ax_idx]
        trial_df = df_local[df_local['trial_id'] == trial]

        data = []
        labels = []
        colors = []
        for stim in stimuli:
            values = trial_df.loc[trial_df['rating_stimulus'] == stim, 'rating_score'].values
            if values.size > 0:
                data.append(values)
                labels.append(map_stimulus_name(stim))
                colors.append(stim_to_color[stim])

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            patch_artist=True,
            widths=0.7
        )
        for box, color in zip(bp['boxes'], colors):
            box.set(facecolor=color, edgecolor='black', linewidth=1.2)
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1.2)
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1.2)

        ax.set_title(format_trial_title(trial), fontsize=22, fontweight='bold', pad=10)
        ax.set_ylim(-5, 105)
        ax.grid(True, axis='y', alpha=0.25, linestyle='--')
        # Hide x-axis labels for trial subplots within the grouped figure
        ax.set_xticklabels([])
        ax.set_xlabel('')

    # Hide any leftover empty axes
    for j in range(len(trials), len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Figure-level legend (one entry per stimulus) placed below to avoid overlap
    handles = [plt.Line2D([0], [0], color='black', lw=0, marker='s', markersize=16,
                          markerfacecolor=stim_to_color[s], label=map_stimulus_name(s)) for s in stimuli]
    legend = fig.legend(handles=handles, loc='lower center', ncol=len(stimuli), bbox_to_anchor=(0.5, -0.06))


    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_candle_chart(stats, task_name, save_path):
    """Create a candle chart for a specific task with all trials grouped together"""
    task_data = stats[stats['task'] == task_name].copy()
    
    if task_data.empty:
        print(f"No data found for task: {task_name}")
        return
    
    # Get unique stimuli for this task and aggregate across all trials
    stimuli = sorted(task_data['rating_stimulus'].unique())
    
    # Aggregate statistics across all trials for each stimulus
    aggregated_stats = task_data.groupby('rating_stimulus').agg({
        'mean': 'mean',
        'std': 'mean', 
        'min': 'min',
        'max': 'max',
        'q1': 'mean',
        'median': 'mean',
        'q3': 'mean',
        'count': 'sum'
    }).round(2)
    
    # Create single figure for the task
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    fig.suptitle(f'Overall Performance - {task_name.title()} Task', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Use fixed colors
    
    x_positions = []
    labels = []
    
    for j, stimulus in enumerate(stimuli):
        if stimulus in aggregated_stats.index:
            row = aggregated_stats.loc[stimulus]
            x_pos = j * 2.0  # Increase spacing between stimuli
            x_positions.append(x_pos)
            labels.append(stimulus)
            
            # Draw the candle with larger boxes
            box_width = 1.2
            box_height = row['q3'] - row['q1']
            box = Rectangle((x_pos - box_width/2, row['q1']), box_width, box_height, 
                          facecolor=get_stimulus_color(stimulus), alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(box)
            
            # Median line (thicker)
            ax.plot([x_pos - box_width/2, x_pos + box_width/2], [row['median'], row['median']], 
                   'k-', linewidth=4)
            
            # Whiskers (min to max)
            ax.plot([x_pos, x_pos], [row['min'], row['q1']], 'k-', linewidth=2)
            ax.plot([x_pos, x_pos], [row['q3'], row['max']], 'k-', linewidth=2)
            
            # Min and max caps (wider)
            cap_width = 0.4
            ax.plot([x_pos - cap_width, x_pos + cap_width], [row['min'], row['min']], 'k-', linewidth=2)
            ax.plot([x_pos - cap_width, x_pos + cap_width], [row['max'], row['max']], 'k-', linewidth=2)
            
            # Mean point (larger)
            ax.plot(x_pos, row['mean'], 'ro', markersize=12, markerfacecolor='red', 
                   markeredgecolor='darkred', markeredgewidth=2)
    
    # Set axis properties with more space
    ax.set_xlim(-1, (len(stimuli)-1)*2.0 + 1)
    ax.set_ylim(-5, 105)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    ax.set_ylabel('Rating Score', fontsize=16, fontweight='bold')
    ax.set_xlabel('Stimulus', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add overall task statistics as text annotation
    overall_mean = task_data['mean'].mean()
    overall_std = task_data['std'].mean()
    total_samples = task_data['count'].sum()
    best_stimulus = aggregated_stats.loc[aggregated_stats['mean'].idxmax()]
    worst_stimulus = aggregated_stats.loc[aggregated_stats['mean'].idxmin()]
    
    stats_text = f"""OVERALL TASK STATISTICS
Average Rating: {overall_mean:.1f} ± {overall_std:.1f}
Total Samples: {int(total_samples)}
Best Stimulus: {aggregated_stats['mean'].idxmax()} ({best_stimulus['mean']:.1f})
Worst Stimulus: {aggregated_stats['mean'].idxmin()} ({worst_stimulus['mean']:.1f})"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
           facecolor='lightgray', alpha=0.8))
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, edgecolor='black', label='Q1-Q3 Range'),
        plt.Line2D([0], [0], color='black', linewidth=4, label='Median'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Min-Max Range'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markeredgecolor='darkred', markersize=12, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return None

def create_summary_statistics_table(stats, save_path):
    """Create a summary statistics table"""
    # Create a comprehensive summary table
    summary = stats.pivot_table(
        index=['task', 'trial_id'], 
        columns='rating_stimulus', 
        values=['mean', 'std', 'count'], 
        aggfunc='first'
    )
    
    # Save to CSV
    summary.to_csv(save_path)
    print(f"Summary statistics saved to: {save_path}")
    
    return summary

def create_overall_comparison_chart(stats, save_path):
    """Create separate comparison charts to avoid crowding"""
    
    # Create separate charts for better readability
    charts = [
        ('mean', 'Average Mean Scores by Task and Stimulus', 'RdYlBu_r'),
        ('std', 'Average Standard Deviation by Task and Stimulus', 'Reds'),
        ('count', 'Total Sample Counts by Task and Stimulus', 'Blues'),
    ]
    
    for metric, title, cmap in charts:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if metric == 'count':
            pivot_data = stats.pivot_table(index='task', columns='rating_stimulus', values=metric, aggfunc='sum')
        else:
            pivot_data = stats.pivot_table(index='task', columns='rating_stimulus', values=metric, aggfunc='mean')
        
        # Create bar chart instead of heatmap for better readability
        x_pos = np.arange(len(pivot_data.columns))
        width = 0.25
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(pivot_data.index)))
        
        for i, (task, row) in enumerate(pivot_data.iterrows()):
            offset = (i - len(pivot_data.index)/2 + 0.5) * width
            bars = ax.bar(x_pos + offset, row.values, width, label=task.title(), 
                         color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, row.values):
                height = bar.get_height()
                if not np.isnan(value):
                    if metric == 'count':
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{int(value)}', ha='center', va='bottom', fontsize=9)
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Stimulus', fontsize=12)
        ax.set_ylabel(metric.title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pivot_data.columns, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        chart_save_path = save_path.replace('.png', f'_{metric}.png')
        plt.savefig(chart_save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    # Create a summary table chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a summary showing best performing stimulus per task
    summary_data = []
    for task in stats['task'].unique():
        task_stats = stats[stats['task'] == task]
        best_mean = task_stats.loc[task_stats['mean'].idxmax()]
        most_consistent = task_stats.loc[task_stats['std'].idxmin()]
        
        summary_data.append({
            'Task': task.title(),
            'Best Mean Stimulus': f"{best_mean['rating_stimulus']} ({best_mean['mean']:.1f})",
            'Most Consistent Stimulus': f"{most_consistent['rating_stimulus']} ({most_consistent['std']:.1f})",
            'Total Samples': int(task_stats['count'].sum())
        })
    
    # Create table
    table_data = []
    headers = ['Task', 'Best Mean Stimulus', 'Most Consistent Stimulus', 'Total Samples']
    
    for row in summary_data:
        table_data.append([row[header] for header in headers])
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Summary Statistics by Task', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    summary_save_path = save_path.replace('.png', '_summary_table.png')
    plt.savefig(summary_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return None

def _ensure_user_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable user identifier column named 'user_id'.
    Prefers 'email', falls back to 'session_uuid', then 'session_test_id'.
    """
    df = df.copy()
    candidates = []
    if 'email' in df.columns:
        candidates.append(df['email'])
    if 'session_uuid' in df.columns:
        candidates.append(df['session_uuid'])
    if 'session_test_id' in df.columns:
        candidates.append(df['session_test_id'])
    if candidates:
        # Build user_id by first non-null among candidates
        stacked = pd.concat(candidates, axis=1)
        df['user_id'] = stacked.bfill(axis=1).iloc[:, 0]
    else:
        df['user_id'] = 'unknown_user'
    return df

def compute_per_user_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user average rating per task and stimulus.
    Returns a DataFrame with columns: task, rating_stimulus, user_id, user_mean.
    """
    df_local = df.copy()
    df_local['task'] = df_local['trial_id'].apply(extract_task_from_trial)
    df_local = _ensure_user_id_column(df_local)
    grouped = (
        df_local
        .groupby(['task', 'rating_stimulus', 'user_id'], dropna=False)['rating_score']
        .mean()
        .reset_index()
        .rename(columns={'rating_score': 'user_mean'})
    )
    return grouped

def create_per_user_boxplot(per_user_means: pd.DataFrame, task_name: str, save_path: str) -> None:
    """Create boxplots of per-user mean scores by stimulus for a given task."""
    task_df = per_user_means[per_user_means['task'] == task_name]
    if task_df.empty:
        print(f"No per-user data found for task: {task_name}")
        return

    stimuli = sorted(task_df['rating_stimulus'].unique())
    data = []
    labels = []
    colors = []
    for stim in stimuli:
        vals = task_df.loc[task_df['rating_stimulus'] == stim, 'user_mean'].values
        if vals.size > 0:
            data.append(vals)
            labels.append(map_stimulus_name(stim))
            colors.append(get_stimulus_color(stim))

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.7
    )
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color, edgecolor='black', linewidth=1.8)
    for median in bp['medians']:
        median.set(color='black', linewidth=3)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.4)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.4)

    means = [np.mean(arr) for arr in data]
    ax.scatter(np.arange(1, len(means) + 1), means, c='red', s=70, zorder=3, label='Mean')

    ax.set_ylim(-5, 105)
    # ax.set_ylabel('Per-user Mean Rating', fontsize=20)
    # ax.set_xlabel('Model', fontsize=20)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')

    overall_mean = np.mean(np.concatenate(data)) if data else float('nan')
    overall_std = np.std(np.concatenate(data)) if data else float('nan')
    total_users = len(task_df['user_id'].unique())

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    # Load and analyze data
    csv_file = 'subjective_results.csv'
    df = load_and_analyze_data(csv_file)
    
    # Calculate statistics
    stats = calculate_statistics(df)
    print("\nStatistics calculated for", len(stats), "combinations of task/trial/stimulus")
    
    # Create output directory
    output_dir = 'subjective_results_visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique tasks
    tasks = stats['task'].unique()
    print(f"\nFound tasks: {tasks}")
    
    # Create simplified per-trial and overall figures for each task
    # for task in tasks:
    #     print(f"\nCreating per-trial boxplots for task: {task}")
    #     create_boxplots_per_trial(df, task, output_dir)
    #     print(f"Creating overall boxplot for task: {task}")
    #     overall_path = os.path.join(output_dir, f'overall_boxplot_{task}.png')
    #     create_overall_boxplot(df, task, overall_path)

    # Create grouped panels by trial id prefix so all related trials are in one figure
    print("\nCreating grouped panels per trial family...")
    create_grouped_trial_panels(
        df,
        trial_prefix='trial',
        save_path=os.path.join(output_dir, 'grouped_trials_general.png')
    )
    create_grouped_trial_panels(
        df,
        trial_prefix='pron_trial',
        save_path=os.path.join(output_dir, 'grouped_trials_pronunciation.png')
    )
    create_grouped_trial_panels(
        df,
        trial_prefix='code_trial',
        save_path=os.path.join(output_dir, 'grouped_trials_codeswitch.png')
    )
    
    # Compute per-user means and visualize per task
    per_user_means = compute_per_user_means(df)
    per_user_csv = os.path.join(output_dir, 'per_user_means.csv')
    per_user_means.to_csv(per_user_csv, index=False)

    print("\nCreating per-user mean boxplots by stimulus per task...")
    for t in sorted(per_user_means['task'].unique()):
        out = os.path.join(output_dir, f'per_user_boxplot_{t}.png')
        create_per_user_boxplot(per_user_means, t, out)

    # Create summary statistics table
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary = create_summary_statistics_table(stats, summary_path)
    
    # (Optional) skip crowded multi-metric comparison charts to keep visuals simple
    
    # Print some key insights
    print("\n" + "="*50)
    print("KEY INSIGHTS:")
    print("="*50)
    
    # Best performing stimulus per task
    for task in tasks:
        task_stats = stats[stats['task'] == task]
        best_stimulus = task_stats.loc[task_stats['mean'].idxmax()]
        print(f"\n{task.upper()} Task:")
        print(f"  Best performing stimulus: {map_stimulus_name(best_stimulus['rating_stimulus'])} "
              f"(mean: {best_stimulus['mean']:.1f})")
        most_consistent_row = task_stats.loc[task_stats['std'].idxmin()]
        print(f"  Most consistent stimulus: {map_stimulus_name(most_consistent_row['rating_stimulus'])} "
              f"(std: {most_consistent_row['std']:.1f})")
    
    print(f"\nAll visualizations saved to: {output_dir}/")

if __name__ == "__main__":
    main()
