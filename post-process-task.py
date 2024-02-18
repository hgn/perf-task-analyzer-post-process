#!/usr/bin/python3

import csv
import inspect
import itertools
import math
import os
import random
import shutil
import string
import subprocess
import sys
import time
import concurrent.futures
from functools import partial
from typing import Dict, List, Union
from types import FrameType
from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (FuncFormatter, MultipleLocator,
                               ScalarFormatter, StrMethodFormatter)


FILE_BASE = "reference.csv"

df = pd.read_csv(FILE_BASE, sep=";")


def calculate_spread_rating(task_runtimes: List[Union[int, float]]) -> float:
    if len(task_runtimes) <= 1:
        return 0.0

    q1, q3 = np.percentile(task_runtimes, [25, 75])
    iqr = q3 - q1

    median = np.median(task_runtimes)
    if median == 0 and iqr == 0:
        return 0.0
    
    spread_rating = (iqr / median) * 100 if median != 0 else iqr * 100
    return spread_rating


def generate_random_filename(length: int = 12, suffix: str = "") -> str:
    chars = string.ascii_letters + string.digits
    filename = ''.join(random.choice(chars) for _ in range(length))
    return f"/tmp/{filename}{suffix}"


def create_images(frame: FrameType) -> None:
    plt.tight_layout()
    outbase = frame.f_code.co_name.replace("_", "-")
    for ext in ['png', 'pdf']:
        outfile = f"{outbase}.{ext}"
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        #print(f"Generated {outfile}")
    plt.close()
    cleanup_directory()


def create_graphviz(in_filename, frame) -> None:
    outbase = frame.f_code.co_name.replace("_", "-")
    out_fdp_png = f"{outbase}-fdp.png"
    out_dot_png = f"{outbase}-dot.png"
    #print(f'generate {out_fdp_png}')
    subprocess.run(['fdp', in_filename, '-Tpng', '-Gdpi=300', '-o', out_fdp_png])
    #print(f'generate {out_dot_png}')
    subprocess.run(['dot', in_filename, '-Tpng', '-Gdpi=600', '-o', out_dot_png])
    cleanup_directory()


def categorize_files(target_dir: str) -> Dict[str, List[str]]:
    """Categorize files into comm, pid, and tid based on their names or directories."""
    categories = {'comm': [], 'pid': [], 'tid': []}
    
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.png'):
                if 'comm' in file.lower():
                    categories['comm'].append(os.path.join(root, file))
                elif 'pid' in file.lower():
                    categories['pid'].append(os.path.join(root, file))
                elif 'tid' in file.lower():
                    categories['tid'].append(os.path.join(root, file))
    
    for category in categories:
        categories[category].sort()
    
    return categories


def generate_readme(readme_path: str) -> None:
    """Generate a README file with categorized images."""
    categories = categorize_files(".")
    
    with open(readme_path, 'w') as readme:
        readme.write(f"# Linux Perf Task Analyzer - Post Processing\n\n")
        for category, files in categories.items():
            readme.write(f"## {category.capitalize()} Analysis\n\n")
            for file in files:
                filename_no_ext = os.path.splitext(os.path.basename(file))[0]
                title = filename_no_ext.replace('-', ' ').title()
                readme.write(f"### {title}\n\n")
                relative_path = os.path.relpath(file, os.path.dirname(readme_path))
                readme.write(f"![title]({relative_path})\n\n")
                

def cleanup_directory(directory: str = ".") -> None:
    """
    Organizes files in the specified directory by moving them into subdirectories
    based on their prefixes ('comm_', 'pid_', 'tid_').

    :param directory: The path to the directory to clean up. Defaults to the current directory.
    """
    # Define the prefixes and their corresponding new directories
    prefixes = {
        'comm-': 'comm',
        'pid-': 'pid',
        'tid-': 'tid',
    }

    # Ensure the target directories exist
    for prefix, target_dir in prefixes.items():
        target_path = os.path.join(directory, target_dir)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
    
    # Move files to their respective directories
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            continue
        
        for prefix, target_dir in prefixes.items():
            if filename.startswith(prefix):
                source_path = os.path.join(directory, filename)
                target_path = os.path.join(directory, target_dir, filename)
                shutil.move(source_path, target_path)
                #print(f"move '{filename}' to '{target_dir}/{filename}'")
                break


def pid_runs(df: pd.DataFrame):
    df['Label'] = df.apply(lambda x: f"{x['PID']} {x['Comm']}", axis=1)

    # Step 2: Count occurrences of each unique Label (previously TID, now PID and Comm)
    process_counts = df['Label'].value_counts()

    # Step 3: Get the top 20 processes based on their counts
    top_20_process_counts = process_counts.head(20)

    # Step 4: Plotting with updated aesthetics
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Choose a color palette or continue with the existing one
    colors = ['blueviolet', 'mediumorchid']
    bar_positions = np.arange(len(top_20_process_counts))

    # Create horizontal bars
    for i, value in enumerate(top_20_process_counts.values):
        ax.barh(bar_positions[i], value, color=colors[i % len(colors)], edgecolor='none')

    # Add text labels to the bars
    for i, value in enumerate(top_20_process_counts.values):
        label_x_pos = value + max(top_20_process_counts.values) * 0.01  # Slight offset for readability
        ax.text(label_x_pos, i, f'{value}', va='center', fontsize=9)

    # Adjust y-axis to display the new label format and invert to have the highest value on top
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.8', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.8', color='#bbbbbb')

    # Update axis labels to reflect the change
    ax.set_ylabel('PID (Comm)')
    ax.set_xlabel('Number of Task Runs')
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(top_20_process_counts.index, fontsize=9)

    create_images(inspect.currentframe())


def tid_runs(df: pd.DataFrame) -> None:
    df['Label'] = df.apply(lambda x: f"{x['TID']} ({x['Comm']}, {x['PID']})", axis=1)

    # Step 2: Count occurrences of each unique TID (now using the Label column for clarity)
    process_counts = df['Label'].value_counts()

    # Step 3: Get the top 20 TIDs based on their counts
    top_20_process_counts = process_counts.head(20)

    # Step 4: Plotting
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    colors = ['blueviolet', 'mediumorchid']
    bar_positions = range(len(top_20_process_counts))
    for i, value in enumerate(top_20_process_counts.values):
        ax.barh(bar_positions[i], value, color=colors[i % len(colors)], edgecolor='none')

    # Add text labels to the bars
    for i, value in enumerate(top_20_process_counts.values):
        label_x_pos = value + max(top_20_process_counts.values) * 0.01
        ax.text(label_x_pos, i, f'{value}', va='center', fontsize=9)

    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.8', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.8', color='#bbbbbb')

    ax.set_ylabel('TID (Comm, PID)')
    ax.set_xlabel('Number of Task Runs')
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(top_20_process_counts.index, fontsize=9)

    create_images(inspect.currentframe())


def comm_runs(df: pd.DataFrame) -> None:
    process_counts = df['Comm'].value_counts()

    top_10_process_counts = process_counts.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    colors = ['blueviolet', 'mediumorchid']
    bar_positions = range(len(top_10_process_counts))
    for i, value in enumerate(top_10_process_counts.values):
        ax.barh(bar_positions[i], value, color=colors[i % len(colors)])

    for container in ax.containers:
        for i, bar in enumerate(container):
            width = bar.get_width()
            label_x_pos = width + max(top_10_process_counts.values) * 0.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width}', va='center', fontsize=9)

    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.8', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.8', color='#bbbbbb')

    ax.set_ylabel('Comm')
    ax.set_xlabel('Number of Comm Runs')
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(top_10_process_counts.index)

    create_images(inspect.currentframe())

def tid_runtime_cumulative(df: pd.DataFrame) -> None:
    df['Label'] = df.apply(lambda x: f"{x['TID']} ({x['Comm']}, {x['PID']})", axis=1)
    cumulative_runtimes = df.groupby('TID')['Runtime'].sum().nlargest(20) / 1e6
    tid_to_label = df.drop_duplicates('TID').set_index('TID')['Label']
    cumulative_runtimes.index = cumulative_runtimes.index.map(tid_to_label)

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    colors = ['royalblue', 'dodgerblue']

    for i, (index, value) in enumerate(zip(cumulative_runtimes.index[::-1], cumulative_runtimes.values[::-1])):
        bar = ax.barh(index, value, color=colors[i % len(colors)])
        width = bar[0].get_width()
        label_x_pos = width + ax.get_xlim()[1] * 0.01
        ax.text(label_x_pos, bar[0].get_y() + bar[0].get_height() / 2, f' {width:.2f}', va='center', fontsize=9)

    ax.set_ylabel('TID (Comm, PID)')
    ax.set_xlabel('Cumulative Runtime [ms]')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.8', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.8', color='#bbbbbb')

    create_images(inspect.currentframe())

def pid_runtime_cumulative(df: pd.DataFrame) -> None:
    df['Runtime'] /= 1e9
    df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')

    cumulative_runtimes = df.groupby('PID')['Runtime'].sum().sort_values(ascending=False).head(20)
    pid_comm_map = df.drop_duplicates('PID').set_index('PID')['Comm']
    cumulative_runtimes = cumulative_runtimes.to_frame().join(pid_comm_map).head(20)

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    colors = ['royalblue', 'dodgerblue']

    cumulative_runtimes.sort_values(by='Runtime', inplace=True)

    for i, (pid, row) in enumerate(cumulative_runtimes.iterrows()):
        label = f"{pid} ({row['Comm']})"
        value = row['Runtime']
        bar = ax.barh(label, value, color=colors[i % len(colors)])
        width = bar[0].get_width()
        label_x_pos = width + ax.get_xlim()[1] * 0.01
        ax.text(label_x_pos, bar[0].get_y() + bar[0].get_height() / 2, f'{width:.2f}', va='center', fontsize=9)

    ax.set_ylabel('PID (Comm)')
    ax.set_xlabel('Cumulative Runtime [s]')

    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.8', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.8', color='#bbbbbb')

    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3f}'))

    create_images(inspect.currentframe())


def comm_runtime_cumulative(df: pd.DataFrame) -> None:
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    colors = ['royalblue', 'dodgerblue']  # Alternating colors

    for i, (index, value) in enumerate(zip(cumulative_runtimes.index[::-1], cumulative_runtimes.values[::-1])):
        bar = ax.barh(index, value, color=colors[i % len(colors)])
        width = bar[0].get_width()
        label_x_pos = width + ax.get_xlim()[1] * 0.01
        ax.text(label_x_pos, bar[0].get_y() + bar[0].get_height() / 2, f'{width:.2f}', va='center', fontsize=9)

    ax.set_ylabel('Task')
    ax.set_xlabel('Cumulative Runtime [ms]')

    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.8', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.8', color='#bbbbbb')

    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000000:.0f}'))

    create_images(inspect.currentframe())


def tid_runtime_distribution_violin(df: pd.DataFrame) -> None:
    # Calculate cumulative runtimes and select top 20 TIDs
    cumulative_runtimes = df.groupby('TID')['Runtime'].sum()
    top_20_tids = cumulative_runtimes.sort_values(ascending=False).head(20).index
    filtered_df = df[df['TID'].isin(top_20_tids)]

    # Prepare figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Prepare data for the violin plot
    data = [filtered_df[filtered_df['TID'] == tid]['Runtime'] for tid in top_20_tids]

    # Create violin plot
    violin_parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('dodgerblue')
        pc.set_alpha(0.4)

    # Calculate and overlay quartiles
    for i, d in enumerate(data):
        q1, q3 = np.percentile(d, [25, 75])
        ax.scatter([i + 1, i + 1], [q1, q3], color='blue', marker="d", lw=1,
                   zorder=4, alpha=0.3, edgecolor=None,
                   label='Quartiles (first and third)' if i == 0 else "")

    # Overlay means
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(means) + 1), means, zorder=3, s=30, color='blue',
               alpha=0.3, edgecolor=None, label='Mean')

    ax.set_yscale('log')
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))

    # Generate labels with TID, Comm, and PID
    labels = []
    for tid in top_20_tids:
        # Fetch the first occurrence of Comm and PID for each TID
        representative_record = filtered_df[filtered_df['TID'] == tid].iloc[0]
        labels.append(f"{tid} ({representative_record['Comm']}, {representative_record['PID']})")

    ax.set_xticks(range(1, len(top_20_tids) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Task')
    ax.set_ylabel('Task Runtime [μs]')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)
    plt.tight_layout()

    # Update legend to reflect additional data points
    handles, labels = ax.get_legend_handles_labels()
    if 'Quartiles (first and third)' in labels:
        legend_labels = dict(zip(labels, handles))
        ax.legend(legend_labels.values(), legend_labels.keys(), loc='upper right')

    # Adjust legend for clarity
    mean_marker = plt.Line2D([], [], color='blue', marker='o',
                             linestyle='None', alpha=0.3, label='Mean')
    quartiles_marker = plt.Line2D([], [], color='blue', marker='d',
                                  linestyle='None', alpha=0.3, label='Quartiles (first and third)')
    median_line = plt.Line2D([], [], color='mediumpurple', marker='_',
                             linestyle='None', label='Median')
    ax.legend(handles=[mean_marker, quartiles_marker, median_line], loc='upper right')


    create_images(inspect.currentframe())



def comm_runtime_distribution_violin(df: pd.DataFrame) -> None:
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()
    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index
    filtered_df = df[df['Comm'].isin(top_20_tasks)]
    data = [filtered_df[filtered_df['Comm'] == task]['Runtime'] for task in top_20_tasks]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    violin_parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor('dodgerblue')
        pc.set_alpha(0.4)

    # Calculate and overlay quartiles
    for i, d in enumerate(data):
        q1, q3 = np.percentile(d, [25, 75])
        # Overlay quartiles as horizontal lines or scatter points
        ax.scatter([i + 1, i + 1], [q1, q3], color='blue', marker="d", lw=1,
                zorder=4, alpha=0.3, edgecolor=None,
                label='Quartiles (first and third)' if i == 0 else "")

    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(means) + 1), means, zorder=3, s=30, color='blue',
            alpha=0.3, edgecolor=None, label='Mean')

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

    handles, labels = ax.get_legend_handles_labels()
    if 'Quartiles (first and third)' in labels:
        legend_labels = dict(zip(labels, handles))
        ax.legend(legend_labels.values(), legend_labels.keys())

    # Legend for median, quartiles, and mean
    mean_marker = plt.Line2D([], [], color='blue', marker='o',
            linestyle='None', alpha=0.3, label='Mean')
    quartiles_marker = plt.Line2D([], [], color='blue', marker='d',
            linestyle='None', alpha=0.3, label='Quartiles (first and third)')
    median_line = plt.Line2D([], [], color='mediumpurple', marker='_',
            linestyle='None', label='Median')
    ax.legend(handles=[mean_marker, quartiles_marker, median_line])

    create_images(inspect.currentframe())


def tid_runtime_distribution_scatter(df: pd.DataFrame) -> None:
    df['TID'] = df['TID'].astype(str)
    
    cumulative_runtimes_by_tid = df.groupby('TID')['Runtime'].sum().sort_values(ascending=False).head(20)
    top_20_tids = cumulative_runtimes_by_tid.index
    filtered_df = df[df['TID'].isin(top_20_tids)]

    fig, ax = plt.subplots(figsize=(12, 8))
    for tid in top_20_tids:
        task_df = filtered_df[filtered_df['TID'] == tid]
        x = np.random.normal(size=len(task_df), loc=np.where(top_20_tids == tid)[0][0] + 1, scale=0.1)
        y = task_df['Runtime']
        ax.scatter(x, y, alpha=0.1, s=25, color="royalblue")

    tid_labels = [f'{tid} ({filtered_df.loc[filtered_df["TID"] == tid, "Comm"].iloc[0]}, {filtered_df.loc[filtered_df["TID"] == tid, "PID"].iloc[0]})' for tid in top_20_tids]
    
    ax.set_yscale('log')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))
    ax.set_xticks(range(1, len(top_20_tids) + 1))
    ax.set_xticklabels(tid_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('TID (Comm, PID)')
    ax.set_ylabel('Task Runtime [μs]')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    create_images(inspect.currentframe())


def comm_runtime_distribution_scatter(df: pd.DataFrame) -> None:
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()
    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index
    filtered_df = df[df['Comm'].isin(top_20_tasks)]

    fig, ax = plt.subplots(figsize=(12, 8))

    for task in top_20_tasks:
        x = np.random.normal(size=len(filtered_df[filtered_df['Comm'] ==
            task]['Runtime']), loc=top_20_tasks.get_loc(task) + 1, scale=0.1)
        y = filtered_df[filtered_df['Comm'] == task]['Runtime']
        ax.scatter(x, y, alpha=0.1, s=25, color="royalblue")

    ax.set_yscale('log')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))

    ax.set_xticks(range(1, len(top_20_tasks) + 1))
    ax.set_xticklabels(top_20_tasks, rotation=45, ha='right')
    ax.set_xlabel('Task')
    ax.set_ylabel('Task Runtime [μs]')

    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    create_images(inspect.currentframe())

def tid_runtime_distribution_boxplot(df: pd.DataFrame) -> None:
    # Calculate cumulative runtimes and select top 20 tasks
    cumulative_runtimes = df.groupby('TID')['Runtime'].sum()
    top_20_tids = cumulative_runtimes.sort_values(ascending=False).head(20).index
    filtered_df = df[df['TID'].isin(top_20_tids)]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for the boxplot
    data = [filtered_df[filtered_df['TID'] == tid]['Runtime'] for tid in top_20_tids]

    # Create boxplot
    box = ax.boxplot(data, vert=True, patch_artist=True)
    for pc in box['boxes']:
        pc.set_facecolor("dodgerblue")
        pc.set_alpha(0.5)

    ax.set_yscale('log')
    ax.set_ylabel('Task Runtime [μs]')
    ax.set_xlabel('TID (Comm, PID)')

    # Generate labels with TID, Comm, and PID
    labels = []
    for tid in top_20_tids:
        # Find a representative record for each TID to extract Comm and PID
        representative_record = filtered_df[filtered_df['TID'] == tid].iloc[0]
        labels.append(f"{tid} ({representative_record['Comm']}, {representative_record['PID']})")

    ax.set_xticks(range(1, len(top_20_tids) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))


    create_images(inspect.currentframe())


def comm_runtime_distribution_boxplot(df: pd.DataFrame) -> None:
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()
    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index
    filtered_df = df[df['Comm'].isin(top_20_tasks)]

    fig, ax = plt.subplots(figsize=(12, 8))

    data = [filtered_df[filtered_df['Comm'] == task]['Runtime'] for task in top_20_tasks]

    box = ax.boxplot(data, vert=True, patch_artist=True)
    for i, pc in enumerate(box['boxes']):
        pc.set_facecolor("dodgerblue")
        pc.set_alpha(0.5)

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

    create_images(inspect.currentframe())


def tid_runtime_spread(df: pd.DataFrame) -> None:
    df['TaskLabel'] = df.apply(lambda x: f"{x['TID']} ({x['Comm']}, {x['PID']})", axis=1)
    cumulative_runtimes = df.groupby(['TID', 'TaskLabel'])['Runtime'].sum().reset_index()
    top_20_tasks = cumulative_runtimes.sort_values(by='Runtime', ascending=False).head(20)
    spread_ratings_df = pd.DataFrame({
        'TaskLabel': top_20_tasks['TaskLabel'],
        'SpreadRating': [calculate_spread_rating(df[df['TID'] ==
            row['TID']]['Runtime']) for index, row in top_20_tasks.iterrows()]
    })

    spread_ratings_df_sorted = spread_ratings_df.sort_values(by='SpreadRating', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bars = ax.bar(spread_ratings_df_sorted['TaskLabel'],
            spread_ratings_df_sorted['SpreadRating'], color="dodgerblue")
    for bar, rating in zip(bars, spread_ratings_df_sorted['SpreadRating']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{rating:.0f}%',
                ha='center', va='bottom')

    ax.set_ylabel('Runtime Spread Rating [%]')
    ax.set_xlabel('TID (Comm, PID)')
    ax.set_xticks(np.arange(len(spread_ratings_df_sorted)))
    ax.set_xticklabels(spread_ratings_df_sorted['TaskLabel'], rotation=45, ha='right')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='gray')
    ax.set_axisbelow(True)

    create_images(inspect.currentframe())


def comm_runtime_spread(df: pd.DataFrame) -> None:
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()
    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index

    spread_ratings = [calculate_spread_rating(df[df['Comm'] == task]['Runtime']) for task in top_20_tasks]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    bars = ax.bar(top_20_tasks, spread_ratings, color="dodgerblue")

    for bar, rating in zip(bars, spread_ratings):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{rating:.0f}%', ha='center', va='bottom')

    ax.set_ylabel('Runtime Spread Rating [%]')
    ax.set_xlabel('Task')
    ax.set_xticks(range(len(top_20_tasks)))
    ax.set_xticklabels(top_20_tasks, rotation=45, ha='right')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    create_images(inspect.currentframe())

def tid_runtime_sleeptime_spread(df: pd.DataFrame) -> None:
    cumulative_runtimes = df.groupby('TID')['Runtime'].sum()
    top_tids = cumulative_runtimes.sort_values(ascending=False).head(20).index

    # Prepare labels "<TID> (<Comm>, <PID>)" for the top TIDs
    # Assuming 'Comm' and 'PID' don't vary within each 'TID', otherwise adjust this logic
    tid_labels = df[df['TID'].isin(top_tids)].drop_duplicates('TID').set_index('TID')
    tid_labels['Label'] = tid_labels.apply(lambda x: f'{x.name} ({x["Comm"]}, {x["PID"]})', axis=1)
    labels_for_plot = tid_labels.loc[top_tids]['Label']

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bar_width = 0.35
    index = np.arange(len(top_tids))

    # Lists for runtime and sleeptime spread ratings
    runtime_spreads = []
    sleeptime_spreads = []

    # Calculate spread ratings for runtime and sleeptime for the top TIDs
    for tid in top_tids:
        runtime_spreads.append(calculate_spread_rating(df[df['TID'] == tid]['Runtime']))
        sleeptime_spreads.append(calculate_spread_rating(df[df['TID'] == tid]['Time Out-In']))

    # Plotting
    bars1 = ax.bar(index - bar_width/2, runtime_spreads, bar_width, label='Runtime Spread', color="dodgerblue")
    bars2 = ax.bar(index + bar_width/2, sleeptime_spreads, bar_width, label='Sleeptime Spread', color="darkorange")

    # Adding text for labels, title, and axes ticks
    ax.set_xlabel('Task')
    ax.set_ylabel('Spread Rating [%]')
    ax.set_xticks(index)
    ax.set_xticklabels(labels_for_plot, rotation=45, ha='right')
    ax.legend()

    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ScalarFormatter())
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

    create_images(inspect.currentframe())


def comm_runtime_sleeptime_spread(df: pd.DataFrame) -> None:
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
    bars1 = ax.bar(index - bar_width/2, runtime_spreads, bar_width, label='Runtime Spread', color="dodgerblue")
    bars2 = ax.bar(index + bar_width/2, sleeptime_spreads, bar_width, label='Sleeptime Spread', color="darkorange")

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

    create_images(inspect.currentframe())

def tid_runtime_linechart(df: pd.DataFrame) -> None:
    tid1, tid2, tid3 = 3959, 4230, 3902

    p1 = df[df['TID'] == tid1]
    p2 = df[df['TID'] == tid2]
    p3 = df[df['TID'] == tid3]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p1_line, = ax.plot(p1['Switched-In'], p1['Runtime'], marker=None, linestyle='-', color='royalblue')
    p2_line, = ax.plot(p2['Switched-In'], p2['Runtime'], marker=None, linestyle='-', color='darkorchid')
    p3_line, = ax.plot(p3['Switched-In'], p3['Runtime'], marker=None, linestyle='-', color='darkorange')

    label_offset = 0.1

    label1 = f'{tid1} ({p1["Comm"].iloc[0]}, {p1["PID"].iloc[0]})'
    label2 = f'{tid2} ({p2["Comm"].iloc[0]}, {p2["PID"].iloc[0]})'
    label3 = f'{tid3} ({p3["Comm"].iloc[0]}, {p3["PID"].iloc[0]})'

    ax.text(p1['Switched-In'].iloc[-1] + label_offset, p1['Runtime'].iloc[-1], label1, va='center', fontsize=12)
    ax.text(p2['Switched-In'].iloc[-1] + label_offset, p2['Runtime'].iloc[-1], label2, va='center', fontsize=12)
    ax.text(p3['Switched-In'].iloc[-1] + label_offset, p3['Runtime'].iloc[-1], label3, va='center', fontsize=12)

    ax.scatter(p1['Switched-In'], p1['Runtime'], color='royalblue',  alpha=0.4, edgecolor='none')
    ax.scatter(p2['Switched-In'], p2['Runtime'], color='darkorchid', alpha=0.4, edgecolor='none')
    ax.scatter(p3['Switched-In'], p3['Runtime'], color='darkorange', alpha=0.4, edgecolor='none')

    ax.set_xlabel('Time')
    ax.set_ylabel('Task Runtime [μs]')
    ax.set_yscale('log')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))


    create_images(inspect.currentframe())



def comm_runtime_linechart(df: pd.DataFrame) -> None:
    p1 = df[df['Comm'] == 'rcu_preempt']
    p2 = df[df['Comm'] == 'perf']
    p3 = df[df['Comm'] == 'kitty']

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p1_line, = ax.plot(p1['Switched-In'], p1['Runtime'],
            marker=None, linestyle='-', color='royalblue')
    p2_line, = ax.plot(p2['Switched-In'], p2['Runtime'],
            marker=None, linestyle='-', color='darkorchid')
    p3_line, = ax.plot(p3['Switched-In'], p3['Runtime'],
            marker=None, linestyle='-', color='darkorange')

    label_offset = 0.1
    ax.text(p1['Switched-In'].iloc[-1] + label_offset,
            p1['Runtime'].iloc[-1], 'rcu_preempt',
            va='center', fontsize=12)
    ax.text(p2['Switched-In'].iloc[-1] + label_offset,
            p2['Runtime'].iloc[-1], 'perf',
            va='center', fontsize=12)
    ax.text(p3['Switched-In'].iloc[-1] + label_offset,
            p3['Runtime'].iloc[-1], 'kitty',
            va='center', fontsize=12)

    ax.scatter(p1['Switched-In'], p1['Runtime'], color='royalblue',  alpha=0.4, edgecolor='none')
    ax.scatter(p2['Switched-In'], p2['Runtime'], color='darkorchid', alpha=0.4, edgecolor='none')
    ax.scatter(p3['Switched-In'], p3['Runtime'], color='darkorange', alpha=0.4, edgecolor='none')

    ax.set_xlabel('Time')
    ax.set_ylabel('Task Runtime [μs]')

    ax.set_yscale('log')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))

    create_images(inspect.currentframe())


def tid_sleeptime_linechart(df: pd.DataFrame) -> None:
    tid_values = [3959, 4230, 3957]

    columns_to_check = ['Runtime', 'Time Out-In', 'Time Out-Out', 'Time In-In', 'Time In-Out']
    df = df.loc[~(df[columns_to_check] == -1).any(axis=1)]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    colors = ['royalblue', 'darkorchid', 'darkorange']
    for i, tid in enumerate(tid_values):
        p = df[df['TID'] == tid]
        if not p.empty:
            unique_comm = p['Comm'].iloc[0]
            unique_pid = p['PID'].iloc[0]
            label = f"{tid} ({unique_comm}, {unique_pid})"
            ax.plot(p['Switched-In'], p['Time Out-In'], linestyle='-', color=colors[i], label=label)
            ax.scatter(p['Switched-In'], p['Time Out-In'], color=colors[i], alpha=0.4, edgecolor='none')
            try:
                label_offset = 0.1
                ax.text(p['Switched-In'].iloc[-1] + label_offset, p['Time Out-In'].iloc[-1],
                        label, va='center', fontsize=12)
            except IndexError:
                pass

    ax.set_xlabel('Time')
    ax.set_ylabel('Task Sleeptime [μs]')
    ax.set_yscale('log')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)
    ax.legend()  # Add legend to distinguish multiple lines
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))

    create_images(inspect.currentframe())


def comm_sleeptime_linechart(df: pd.DataFrame) -> None:
    key = "Time Out-In"

    # remove -1 lines
    columns_to_check = ['Runtime', 'Time Out-In', 'Time Out-Out', 'Time In-In', 'Time In-Out']
    df = df.loc[~(df[columns_to_check] == -1).any(axis=1)]

    p2 = df[df['Comm'] == 'perf']
    p3 = df[df['Comm'] == 'kitty']
    p1 = df[df['Comm'] == 'rcu_preempt']

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p1_line, = ax.plot(p1['Switched-In'], p1[key],
            linestyle='-', color='royalblue')
    p2_line, = ax.plot(p2['Switched-In'], p2[key],
            linestyle='-', color='darkorchid')
    p3_line, = ax.plot(p3['Switched-In'], p3[key],
            linestyle='-', color='darkorange')

    ax.scatter(p1['Switched-In'], p1['Time Out-In'], color='royalblue',  alpha=0.4, edgecolor='none')
    ax.scatter(p2['Switched-In'], p2['Time Out-In'], color='darkorchid', alpha=0.4, edgecolor='none')
    ax.scatter(p3['Switched-In'], p3['Time Out-In'], color='darkorange', alpha=0.4, edgecolor='none')


    label_offset = 0.1
    ax.text(p1['Switched-In'].iloc[-1] + label_offset,
            p1[key].iloc[-1], 'rcu_preempt',
            va='center', fontsize=12)
    ax.text(p2['Switched-In'].iloc[-1] + label_offset,
            p2[key].iloc[-1], 'perf',
            va='center', fontsize=12)
    ax.text(p3['Switched-In'].iloc[-1] + label_offset,
            p3[key].iloc[-1], 'kitty',
            va='center', fontsize=12)

    ax.set_xlabel('Time')
    ax.set_ylabel('Task Sleeptime [μs]')

    ax.set_yscale('log')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000:.0f}'))

    create_images(inspect.currentframe())


def tid_runs_vs_runtime_quadrant(df: pd.DataFrame) -> None:
    total_runtime = df.groupby('TID')['Runtime'].sum().nlargest(40)

    execution_count = df.groupby('TID').size().reindex(total_runtime.index)

    labels_df = df.drop_duplicates('TID').set_index('TID')[['Comm', 'PID']]
    labels_df['Label'] = labels_df.apply(lambda x: f"{x.name} ({x['Comm']}, {x['PID']})", axis=1)

    task_summary = pd.DataFrame({
        'Total_Runtime': total_runtime,
        'Count': execution_count
    }).join(labels_df)

    fig, ax = plt.subplots(figsize=(12, 12))

    sizes = task_summary['Total_Runtime'] / task_summary['Total_Runtime'].max() * 1000  # Normalize sizes for visibility
    scatter = ax.scatter(task_summary['Total_Runtime'], task_summary['Count'], s=sizes, alpha=0.6)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Total Runtime [ms] (low ← → high)')
    ax.set_ylabel('Execution Count [#] (low ← → high)')

    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    x_geo_mean = (task_summary['Total_Runtime'].max() - task_summary['Total_Runtime'].min()) // 2
    y_geo_mean = (task_summary['Count'].max() - task_summary['Count'].min()) // 2
    ax.axvline(x=x_geo_mean, color='grey', linestyle='--')
    ax.axhline(y=y_geo_mean, color='grey', linestyle='--')

    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):d}'))
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000000:.0f}'))

    for index, row in task_summary.iterrows():
        ax.text(row['Total_Runtime'], row['Count'], ' ' + row['Label'], va='center', fontsize=8)

    create_images(inspect.currentframe())


def comm_runs_vs_runtime_quadrant(df: pd.DataFrame) -> None:
    total_runtime = df.groupby('Comm')['Runtime'].sum().nlargest(40)
    execution_count = df.groupby('Comm').size().reindex(total_runtime.index)

    task_summary = pd.DataFrame({
        'Total_Runtime': total_runtime,
        'Count': execution_count
    })

    fig, ax = plt.subplots(figsize=(12, 12))

    sizes = task_summary['Total_Runtime'] / task_summary['Total_Runtime'].max() * 1000
    scatter = ax.scatter(task_summary['Total_Runtime'], task_summary['Count'], s=sizes, alpha=0.6)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Total Runtime [ms] (low ← → high)')
    ax.set_ylabel('Execution Count [#] (low ← → high)')

    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.xaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    x_median = np.sqrt(task_summary['Total_Runtime'].min() * task_summary['Total_Runtime'].max())
    y_median = np.sqrt(task_summary['Count'].min() * task_summary['Count'].max())
    x_geo_mean = (task_summary['Total_Runtime'].max() - task_summary['Total_Runtime'].min()) // 2
    y_geo_mean = (task_summary['Count'].max() - task_summary['Count'].min()) // 2
    ax.axvline(x=x_geo_mean, color='grey', linestyle='--')
    ax.axhline(y=y_geo_mean, color='grey', linestyle='--')

    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):d}'))
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1000000:.0f}'))

    for task_name, row in task_summary.iterrows():
        ax.text(row['Total_Runtime'], row['Count'], ' ' + task_name, va='center')

    create_images(inspect.currentframe())


def tid_runtime_utilization(df: pd.DataFrame) -> None:
    # FIXME, buggy.
    # THe overall value is above 100%, this is not doable by one TID
    return
    runtime_sum = df.groupby('TID')['Runtime'].sum().reset_index()
    top_runtime_sum = runtime_sum.nlargest(10, 'Runtime')

    # Join to get Comm and PID for labeling
    top_tasks = top_runtime_sum.merge(df[['TID', 'Comm', 'PID']].drop_duplicates(), on='TID', how='left')

    # Creating a label column "<TID> (<Comm>, <PID>)"
    top_tasks['Label'] = top_tasks.apply(lambda x: f'{x["TID"]} ({x["Comm"]}, {x["PID"]})', axis=1)

    # Initialize a DataFrame to track CPU utilization per second for each of the top 10 tasks
    time_min = math.floor(df['Switched-In'].min())
    time_max = math.ceil(df['Switched-In'].max())
    cpu_utilization = pd.DataFrame(index=np.arange(time_min, time_max + 1))

    # Initialize columns for each top task
    for label in top_tasks['Label']:
        cpu_utilization[label] = 0

    # Accumulate runtime to the nearest second for each task in the top 10
    for _, row in df.iterrows():
        if row['TID'] in top_tasks['TID'].values:
            second = math.floor(row['Switched-In'])
            label = top_tasks[top_tasks['TID'] == row['TID']]['Label'].values[0]  # Find the label for the TID
            cpu_utilization.at[second, label] += row['Runtime']

    # Convert accumulated runtime in nanoseconds to CPU utilization percentage per second
    cpu_utilization = cpu_utilization / 1e6  # Adjusted division for correct percentage calculation

    # Plotting
    fig, axes = plt.subplots(nrows=10, figsize=(20, 10), sharex=True)

    # Find the highest utilization to set y_lim
    max_utilization = cpu_utilization.max().max()

    for i, label in enumerate(top_tasks['Label']):
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

        axes[i].yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='gray')
        axes[i].xaxis.grid(which='major', linestyle=':', linewidth='0.5', color='gray')

        axes[i].fill_between(cpu_utilization.index, 0, cpu_utilization[label], step='mid', alpha=0.5)
        axes[i].axhline(y=100, color='r', linestyle='--', linewidth=0.5)
        axes[i].set_ylim(0, max(max_utilization, 100))
        axes[i].set_xlim(time_min, time_max)

        axes[i].set_ylabel(f'{label}', fontsize=8)

    axes[-1].set_xlabel('Time (s)')


    create_images(inspect.currentframe())

def comm_runtime_utilization(df: pd.DataFrame) -> None:
    # Calculate total runtime for each task and select the top 10 tasks
    total_runtime = df.groupby('Comm')['Runtime'].sum().nlargest(10)

    # Initialize a DataFrame to track CPU utilization per second for each of the top 10 tasks
    time_min = math.floor(df['Switched-In'].min())
    time_max = math.ceil(df['Switched-In'].max())
    cpu_utilization = pd.DataFrame(index=np.arange(time_min, time_max + 1))

    for task in total_runtime.index:
        cpu_utilization[task] = 0  # Initialize column for each task

    # Accumulate runtime to the nearest second for each task in the top 10
    for _, row in df.iterrows():
        if row['Comm'] in total_runtime.index:
            second = math.floor(row['Switched-In'])
            cpu_utilization.at[second, row['Comm']] += row['Runtime']

    # Convert accumulated runtime in milliseconds to CPU utilization percentage per second
    cpu_utilization = cpu_utilization / 1000000  # Adjusted division for correct percentage calculation

    # Plotting
    fig, axes = plt.subplots(nrows=10, figsize=(20, 10), sharex=True)

    # Find the highest utilization to set y_lim
    max_utilization = cpu_utilization.max().max()

    for i, task in enumerate(total_runtime.index):

        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

        axes[i].yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
        axes[i].yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
        axes[i].xaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
        axes[i].xaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')

        axes[i].fill_between(cpu_utilization.index, 0, cpu_utilization[task], step='mid', alpha=0.5)
        # Print the name of the task on the right axis
        #axes[i].text(cpu_utilization.index[-1], cpu_utilization[task].median(),
        #        f'{task}', horizontalalignment='right',
        #        verticalalignment='center')

        axes[i].axhline(y=100, color='r', linestyle='--', linewidth=.5)
        axes[i].set_ylim(0, max(max_utilization, 100))
        axes[i].set_xlim(time_min, time_max)

#        axes[i].set_ylabel(f'{task}', fontsize=8)

    axes[-1].set_xlabel('Time (s)')

    create_images(inspect.currentframe())


def tid_runtime_overall_pie(df: pd.DataFrame) -> None:
    df['Runtime'] /= 1e9
    df['Label'] = df['TID'].astype(str) + " (" + df['Comm'] + ", " + df['PID'].astype(str) + ")"

    total_runtime_per_label = df.groupby('Label')['Runtime'].sum()
    total_runtime = total_runtime_per_label.sum()
    runtime_percentage = (total_runtime_per_label / total_runtime) * 100

    # Threshold for grouping into "Other"
    remain_threshold = 3
    remain_runtime = total_runtime_per_label[runtime_percentage < remain_threshold].sum()
    filtered_runtime_per_label = total_runtime_per_label[runtime_percentage >= remain_threshold]
    remain_label = 'Other' if remain_runtime > 0 else ''

    if remain_runtime > 0:
        filtered_runtime_per_label[remain_label] = remain_runtime

    explode = [0.1 if label == remain_label else 0 for label in filtered_runtime_per_label.index]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.get_cmap('tab10').colors 
    ax.pie(filtered_runtime_per_label, labels=filtered_runtime_per_label.index, autopct='%1.1f%%',
           startangle=90, counterclock=False, colors=colors[:len(filtered_runtime_per_label)], explode=explode)

    create_images(inspect.currentframe())


def comm_runtime_overall_pie(df: pd.DataFrame) -> None:
    df['Runtime'] /= 1e9

    total_runtime_per_comm = df.groupby('Comm')['Runtime'].sum()
    total_runtime = total_runtime_per_comm.sum()
    runtime_percentage = (total_runtime_per_comm / total_runtime) * 100

    remain_threshold = 3
    remain_runtime = total_runtime_per_comm[runtime_percentage < remain_threshold].sum()
    filtered_runtime_per_comm = total_runtime_per_comm[runtime_percentage >= remain_threshold]
    remain_label = 'Remain' if remain_runtime > 0 else ''

    if remain_runtime > 0:
        filtered_runtime_per_comm[remain_label] = remain_runtime

    explode = [0.1 if label == remain_label else 0 for label in filtered_runtime_per_comm.index]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.get_cmap('tab10').colors
    ax.pie(filtered_runtime_per_comm, labels=filtered_runtime_per_comm.index, autopct='%1.1f%%',
           startangle=90, counterclock=False, colors=colors[:len(filtered_runtime_per_comm)], explode=explode)

    create_images(inspect.currentframe())


def pid_tid_accumulated_runtime_stacked(df: pd.DataFrame) -> None:
    cumulative_runtimes_seconds = df.groupby('PID')['Runtime'].sum() / 1e9
    top_20_tasks = cumulative_runtimes_seconds.sort_values(ascending=False).head(20).index

    filtered_df = df[df['PID'].isin(top_20_tasks)]
    comm_names = filtered_df.groupby('PID')['Comm'].agg(lambda x: x.mode()[0])

    thread_counts = filtered_df.groupby('PID')['TID'].nunique()

    pivot_df = filtered_df.groupby(['PID', 'TID'])['Runtime'].sum().unstack(fill_value=0) / 1e9
    pivot_df = pivot_df.loc[top_20_tasks]

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, pid in enumerate(pivot_df.index):
        comm_label = f"{comm_names[pid]} ({pid})"
        runtimes = pivot_df.loc[pid]
        bottom = 0
        for tid, runtime in runtimes.items():
            ax.bar(i, runtime, bottom=bottom, label=tid if bottom == 0 else "")
            bottom += runtime
        cumulative_runtime = cumulative_runtimes_seconds[pid]
        threads = thread_counts[pid]
        ax.text(i, bottom, f'{cumulative_runtime:.2f}s\n{threads} TIDs', ha='center', va='bottom')

    ax.set_xticks(range(len(pivot_df)))
    ax.set_xticklabels([f"{comm_names[pid]} ({pid})" for pid in pivot_df.index], rotation=45, ha="right")

    ax.set_xlabel('Comm (PID)')
    ax.set_ylabel('Runtime [Seconds]')
    ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
    ax.set_axisbelow(True)

    create_images(inspect.currentframe())


def numpy_stats(df: pd.DataFrame) -> None:
    print(df.groupby('Comm')['Runtime'].sum().sort_values(ascending=False))
    print(df.groupby('PID')['Runtime'].sum().sort_values(ascending=False))

def comm_pid_tid_hierarchy_dot(runtimes, filename, filter) -> None:
    with open(filename, 'w') as f:
        f.write('digraph TaskHierarchy {\n')
        f.write('  node [shape=box];\n')
        
        # Node styles
        comm_style = 'style=filled, fillcolor="#c4b47d", color=none'
        pid_style = 'style=filled, fillcolor="#2a9df4", color=none'
        tid_style = 'style=filled, fillcolor="#d0efff", color=none'

        for comm, pids in runtimes.items():
            if filter:
                if not any(comm.lower() in string.lower() or comm.lower() == string.lower() for string in filter):
                    continue
            comm_runtime = runtimes[comm]['_total']['_total']
            no_pids = len(pids) - 1
            comm_label = f"{comm}\\nTotal Runtime: {comm_runtime:.2f}s\\nProcesses: {no_pids}"
            f.write(f'  "{comm}" [{comm_style}, label="{comm_label}"];\n')
            
            for pid, tid_dict in pids.items():
                len_tids = len(tid_dict) - 1
                if pid == '_total': continue
                pid_runtime = tid_dict['_total']
                # Include comm name in PID label
                pid_label = f"{comm}\\nPID: {pid}\\nRuntime: {pid_runtime:.2f}s\\nThreads: {len_tids}"
                f.write(f'  "{comm}_{pid}" [{pid_style}, label="{pid_label}"];\n')
                f.write(f'  "{comm}" -> "{comm}_{pid}" [penwidth=4,style=dotted];\n')
                
                for tid, runtime in tid_dict.items():
                    if tid == '_total': continue
                    # Include comm name in TID label
                    tid_label = f"{comm}\\nTID: {tid}\\nRuntime: {runtime:.2f}s"
                    f.write(f'  "{comm}_{pid}_{tid}" [{tid_style}, label="{tid_label}"];\n')
                    f.write(f'  "{comm}_{pid}" -> "{comm}_{pid}_{tid}" [penwidth=2];\n')
        f.write('}\n')


def comm_pid_tid_hierarchy(filter=None) -> None:
    runtimes = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    with open(FILE_BASE, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)[0].split(";")
        comm_index = header.index('Comm')
        pid_index = header.index('PID')
        tid_index = header.index('TID')
        runtime_index = header.index('Runtime')
        
        for rowline in csvreader:
            row = rowline[0].split(";")
            comm = row[comm_index]
            pid = row[pid_index]
            tid = row[tid_index]
            runtime = float(row[runtime_index]) / 1e9

            runtimes[comm][pid][tid] += runtime
            runtimes[comm][pid]['_total'] += runtime
            runtimes[comm]['_total']['_total'] += runtime
    dot_filename = generate_random_filename(suffix=".dot")
    comm_pid_tid_hierarchy_dot(runtimes, dot_filename, filter=filter)
    create_graphviz(dot_filename, inspect.currentframe())
    os.remove(dot_filename)


def pid_tid_hierarchy_dot(runtimes, filename, filter) -> None:
    with open(filename, 'w') as f:
        f.write('digraph PIDHierarchy {\n')
        f.write('  node [shape=box];\n')
        
        pid_style = 'style=filled, fillcolor="#2a9df4", color=none'
        tid_style = 'style=filled, fillcolor="#d0efff", color=none'

        for pid, data in runtimes.items():
            comm = data["_comm"]
            if filter:
                if not any(comm.lower() in string.lower() or comm.lower() == string.lower() for string in filter):
                    continue
            pid_total_runtime = data["_total"]
            no_threads = len(data["tids"])
            pid_label = f"{pid} ({comm})\\nTotal Runtime: {pid_total_runtime:.2f}s\\nThreads: {no_threads}"
            # Apply style for PID nodes
            f.write(f'  "{pid}" [{pid_style}, label="{pid_label}"];\n')
            
            for tid, info in data["tids"].items():
                runtime = info["runtime"]
                tid_label = f"TID: {tid} ({info['comm']})\\nRuntime: {runtime:.2f}s"
                # Apply style for TID nodes (no background color)
                f.write(f'  "{pid}_{tid}" [{tid_style}, label="{tid_label}"];\n')
                f.write(f'  "{pid}" -> "{pid}_{tid}" [penwidth=2];\n')
        f.write('}\n')


def pid_tid_hierarchy(filter=None) -> None:
    runtimes = defaultdict(lambda: {"_total": 0, "_comm": "", "tids":
        defaultdict(lambda: {"runtime": 0, "comm": ""})})

    with open(FILE_BASE, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)[0].split(";")
        comm_index = header.index('Comm')
        pid_index = header.index('PID')
        tid_index = header.index('TID')
        runtime_index = header.index('Runtime')
        
        for rowline in csvreader:
            row = rowline[0].split(";")
            comm = row[comm_index]
            pid = row[pid_index]
            tid = row[tid_index]
            runtime = float(row[runtime_index]) / 1e9
            
            runtimes[pid]["tids"][tid]["runtime"] += runtime
            runtimes[pid]["_total"] += runtime
            if pid == tid:
                # we only update the PID comm if PID == TID
                # If not equal, the thraad may have a different Comm
                runtimes[pid]["_comm"] = comm
            runtimes[pid]["tids"][tid]["comm"] = comm

    dot_filename = generate_random_filename(suffix=".dot")
    pid_tid_hierarchy_dot(runtimes, dot_filename, filter=filter)
    create_graphviz(dot_filename, inspect.currentframe())
    os.remove(dot_filename)

def execute_task(task) -> None:
    start_time = time.time()
    task()
    end_time = time.time()
    print(f"{task.func.__name__:>40s}() - {end_time - start_time:4.1f}s")

tasks = [
    # TID
    partial(tid_runs, df.copy(deep=True)),
    partial(tid_runtime_cumulative, df.copy(deep=True)),

    partial(tid_runtime_distribution_violin, df.copy()),
    partial(tid_runtime_distribution_boxplot, df.copy()),
    partial(tid_runtime_distribution_scatter, df.copy()),

    partial(tid_runtime_spread, df.copy()),

    partial(tid_runtime_overall_pie, df.copy()),
    partial(tid_runtime_utilization, df.copy()),
    partial(tid_runtime_linechart, df.copy()),

    partial(tid_sleeptime_linechart, df.copy()),

    partial(tid_runtime_sleeptime_spread, df.copy()),
    partial(tid_runs_vs_runtime_quadrant, df.copy()),

    # PID
    partial(pid_runs, df.copy(deep=True)),
    partial(pid_runtime_cumulative, df.copy()),

    partial(pid_tid_accumulated_runtime_stacked, df.copy()),

    # COmm
    partial(comm_runs, df.copy()),
    partial(comm_runtime_cumulative, df.copy()),

    partial(comm_runtime_distribution_violin, df.copy()),
    partial(comm_runtime_distribution_scatter, df.copy()),
    partial(comm_runtime_distribution_boxplot, df.copy()),

    partial(comm_runtime_spread, df.copy()),

    partial(comm_runtime_overall_pie, df.copy()),
    partial(comm_runtime_utilization, df.copy()),
    partial(comm_runtime_linechart, df.copy()),

    partial(comm_sleeptime_linechart, df.copy()),

    partial(comm_runtime_sleeptime_spread, df.copy()),
    partial(comm_runs_vs_runtime_quadrant, df.copy()),

    # Misc
    partial(comm_pid_tid_hierarchy, df.copy(), filter=["chromium", "ThreadPoolForeg"]),
    partial(pid_tid_hierarchy, df.copy(), filter=["chromium", "ThreadPoolForeg"]),
]

print(f"Task analysis starting, now executing {len(tasks)} analysis in parallel")
start_time = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(execute_task, tasks)
processing_time = time.time() - start_time
print(f"Post-processed all analysis in whooping {processing_time:.1f} seconds - bye")
print(f"All files are generated into comm/, pid/ and /tid")

print(f"Updating README.md")
generate_readme('README.md')
