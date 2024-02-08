#!/usr/bin/python3

import os
import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter, MultipleLocator, StrMethodFormatter, FuncFormatter

FILE_BASE = "task.csv"

df = pd.read_csv(FILE_BASE, sep=";")

def task_frequency(df):
    process_counts = df['Comm'].value_counts()

    # Sort the counts in ascending order and select the top 10
    top_10_process_counts = process_counts.sort_values(ascending=False).head(20)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Create a sorted horizontal bar chart (barh) for the top 10 processes
    bar_positions = range(len(top_10_process_counts))
    ax.barh(bar_positions, top_10_process_counts.values, color='skyblue')

    ax.tick_params(axis='y', length=0)

    ax.invert_yaxis()

    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.8', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.8', color='#bbbbbb')

    ax.set_ylabel('Task')
    ax.set_xlabel('Number of Task Runs')
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(top_10_process_counts.index)

    plt.tight_layout()

    outbase = "task-schedules"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')

def task_cumulative_runtime(df):
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.barh(cumulative_runtimes.index[::-1], cumulative_runtimes.values[::-1], color='skyblue')

    ax.set_ylabel('Task')
    ax.set_xlabel('Cumulative Runtime [ms]')

    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.8', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.8', color='#bbbbbb')

    plt.tight_layout()
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000000:.0f}'))

    outbase = "task-cumulative-runtime"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')


def task_violin_distribution(df):
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()
    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index
    filtered_df = df[df['Comm'].isin(top_20_tasks)]
    data = [filtered_df[filtered_df['Comm'] == task]['Runtime'] for task in top_20_tasks]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    violin_parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, pc in enumerate(violin_parts['bodies']):
        #pc.set_edgecolor('black')
        pc.set_alpha(0.4)

    # Calculate and overlay quartiles
    for i, d in enumerate(data):
        q1, q3 = np.percentile(d, [25, 75])
        # Overlay quartiles as horizontal lines or scatter points
        ax.scatter([i + 1, i + 1], [q1, q3], color='blue', marker="d", lw=1, zorder=4, alpha=0.3, edgecolor=None, label='Quartiles (first and third)' if i == 0 else "")

    # Overlaying means
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(means) + 1), means, zorder=3, s=30, color='blue', alpha=0.3, edgecolor=None, label='Mean')

    # Adjustments to improve readability and aesthetics
    ax.set_yscale('log')
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))
    ax.set_xticks(range(1, len(top_20_tasks) + 1))
    ax.set_xticklabels(top_20_tasks, rotation=45, ha='right')
    ax.set_xlabel('Task')
    ax.set_ylabel('Task Runtime [μs]')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)
    plt.tight_layout()

    # Handle legend for quartiles only once
    handles, labels = ax.get_legend_handles_labels()
    if 'Quartiles (first and third)' in labels:
        legend_labels = dict(zip(labels, handles))
        ax.legend(legend_labels.values(), legend_labels.keys())

    # Legend for median, quartiles, and mean
    mean_marker = plt.Line2D([], [], color='blue', marker='o', linestyle='None', alpha=0.3, label='Mean')
    quartiles_marker = plt.Line2D([], [], color='blue', marker='d', linestyle='None', alpha=0.3, label='Quartiles (first and third)')
    median_line = plt.Line2D([], [], color='mediumpurple', marker='_', linestyle='None', label='Median')
    ax.legend(handles=[mean_marker, quartiles_marker, median_line])

    outbase = "task-violin-distribution"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')


def task_scatter_distribution(df):

    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()

    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index

    filtered_df = df[df['Comm'].isin(top_20_tasks)]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a scatter plot of runtime against task with a logarithmic y-axis scale
    for task in top_20_tasks:
        x = np.random.normal(size=len(filtered_df[filtered_df['Comm'] == task]['Runtime']), loc=top_20_tasks.get_loc(task) + 1, scale=0.1)
        y = filtered_df[filtered_df['Comm'] == task]['Runtime']
        ax.scatter(x, y, alpha=0.1, s=25)

    ax.set_yscale('log')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))

    # Add labels and title
    ax.set_xticks(range(1, len(top_20_tasks) + 1))
    ax.set_xticklabels(top_20_tasks, rotation=45, ha='right')
    ax.set_xlabel('Task')
    ax.set_ylabel('Task Runtime [μs]')

    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    # Show the scatter plot
    plt.tight_layout()


    outbase = "task-scatter-distribution"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')

def task_runtime_distribution_boxplot(df):
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()
    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index
    filtered_df = df[df['Comm'].isin(top_20_tasks)]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for boxplot
    data = [filtered_df[filtered_df['Comm'] == task]['Runtime'] for task in top_20_tasks]

    box = ax.boxplot(data, vert=True, patch_artist=True)
    for i, pc in enumerate(box['boxes']):
        #pc.set_edgecolor('black')
        pc.set_facecolor("lightblue")

    ax.set_yscale('log')

    ax.set_ylabel('Task Runtime [μs]')
    ax.set_xlabel('Task')

    ax.set_xticklabels(top_20_tasks, rotation=45, ha='right')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))

    plt.tight_layout()

    outbase = "task-runtime-distribution-boxplot"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')

def calculate_spread_rating(task_runtimes):
    if len(task_runtimes) <= 1:
        return 0.0  # No spread with one or no value
    
    q1, q3 = np.percentile(task_runtimes, [25, 75])
    iqr = q3 - q1
    median = np.median(task_runtimes)
    
    if median == 0 and iqr == 0:
        # Handle edge case where median and IQR are 0 to avoid division by zero
        return 0.0
    
    # Normalize IQR by the median (or another meaningful value) and scale to get a percentage
    spread_rating = (iqr / median) * 100 if median != 0 else iqr * 100  # Adjusted for cases with median=0
    
    return spread_rating

def task_runtime_cv_distribution(df):
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()
    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index

    spread_ratings = [calculate_spread_rating(df[df['Comm'] == task]['Runtime']) for task in top_20_tasks]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(top_20_tasks, spread_ratings, color="lightblue")

    for bar, rating in zip(bars, spread_ratings):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{rating:.1f}%', ha='center', va='bottom')

    ax.set_ylabel('Runtime Spread Rating [%]')
    ax.set_xlabel('Task')
    ax.set_xticks(range(len(top_20_tasks)))
    ax.set_xticklabels(top_20_tasks, rotation=45, ha='right')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    plt.tight_layout()

    outbase = "task-uniformity-score-distribution"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')


def task_spread_rating_distribution_grouped(df):
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()
    top_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    bar_width = 0.35  
    index = np.arange(len(top_tasks))
    
    # Lists for runtime and sleeptime spread ratings
    runtime_spreads = []
    sleeptime_spreads = []
    
    # Calculate spread ratings for runtime and sleeptime
    for task in top_tasks:
        runtime_spreads.append(calculate_spread_rating(df[df['Comm'] == task]['Runtime']))
        sleeptime_spreads.append(calculate_spread_rating(df[df['Comm'] == task]['Time Out-In']))
    
    # Plotting
    bars1 = ax.bar(index - bar_width/2, runtime_spreads, bar_width, label='Runtime Spread')
    bars2 = ax.bar(index + bar_width/2, sleeptime_spreads, bar_width, label='Sleeptime Spread')

    # Adding text for labels, title, and axes ticks
    ax.set_xlabel('Task')
    ax.set_ylabel('Spread Rating [%]')
    ax.set_xticks(index)
    ax.set_xticklabels(top_tasks, rotation=45, ha='right')
    ax.legend()

    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    ax.set_yscale('log')

    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Use a function to ensure no scientific notation, with customization for percentages if necessary
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))


    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=5)

    plt.tight_layout()

    outbase = "task-run-sleeptime-grouped"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')



task_frequency(df)
task_cumulative_runtime(df)
task_violin_distribution(df)
task_scatter_distribution(df)
task_runtime_distribution_boxplot(df)
task_runtime_cv_distribution(df)

task_spread_rating_distribution_grouped(df)
