#!/usr/bin/python3

import os
import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator, StrMethodFormatter

FILE_BASE = "task.csv"

df = pd.read_csv(FILE_BASE, sep=";")

def task_frequency(df):
    process_counts = df['Comm'].value_counts()

    # Sort the counts in ascending order and select the top 10
    top_10_process_counts = process_counts.sort_values(ascending=False).head(20)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Create a sorted horizontal bar chart (barh) for the top 10 processes
    bar_positions = range(len(top_10_process_counts))
    ax.barh(bar_positions, top_10_process_counts.values, color='skyblue')

    # Reverse the order of the y-axis to display the most frequent at the top
    ax.invert_yaxis()

    # Add labels and title
    ax.set_xlabel('Frequency')
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(top_10_process_counts.index)

    # Show the horizontal bar chart
    plt.tight_layout()  # Ensure labels fit within the figure

    outbase = "task-schedules"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')

def task_cumulative_runtime(df):

    # Group the DataFrame by the 'Comm' column and calculate the cumulative runtime
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum().sort_values(ascending=False).head(20)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Create a horizontal bar chart for the cumulative runtime of each task
    ax.barh(cumulative_runtimes.index[::-1], cumulative_runtimes.values[::-1], color='skyblue')


    # Add labels and title
    ax.set_xlabel('Task')
    ax.set_ylabel('Cumulative Runtime')

    # Show the bar chart
    plt.tight_layout()  # Ensure labels fit within the figure
    plt.xticks(rotation=90)

    outbase = "task-cumulative-runtime"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')

def task_violin_distribution(df):

        # Group the DataFrame by the 'Comm' column and calculate the cumulative runtime
    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()

    # Sort the cumulative runtimes in descending order and select the top 20 tasks
    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index

    # Filter the DataFrame to include only the top 20 tasks
    filtered_df = df[df['Comm'].isin(top_20_tasks)]

    # Create a figure and axis object for the violin plots
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create violin plots for the distribution of runtimes for each task
    data = [filtered_df[filtered_df['Comm'] == task]['Runtime'] for task in top_20_tasks]
    violin_parts = ax.violinplot(data, showmedians=True)


    # Customize the appearance of the violin plots
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    ax.set_yscale('log')

    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))

    # Add labels and title
    ax.set_xticks(range(1, len(top_20_tasks) + 1))
    ax.set_xticklabels(top_20_tasks, rotation=45, ha='right')
    ax.set_xlabel('Task')
    ax.set_ylabel('Runtime')

    # Show the violin plot
    plt.tight_layout()

    outbase = "task-violin-distribution"
    out_png = f"{outbase}.png"
    print(f'generate {out_png}')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    out_pdf = f"{outbase}.pdf"
    print(f'generate {out_pdf}')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')

def task_scatter_distribution(df):

    cumulative_runtimes = df.groupby('Comm')['Runtime'].sum()

    # Sort the cumulative runtimes in descending order and select the top 20 tasks
    top_20_tasks = cumulative_runtimes.sort_values(ascending=False).head(20).index

    # Filter the DataFrame to include only the top 20 tasks
    filtered_df = df[df['Comm'].isin(top_20_tasks)]

    # Create a figure and axis object for the scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a scatter plot of runtime against task with a logarithmic y-axis scale
    for task in top_20_tasks:
        x = np.random.normal(size=len(filtered_df[filtered_df['Comm'] == task]['Runtime']), loc=top_20_tasks.get_loc(task) + 1, scale=0.1)
        y = filtered_df[filtered_df['Comm'] == task]['Runtime']
        ax.scatter(x, y, alpha=0.7, s=50)

    # Set the y-axis scale to logarithmic
    ax.set_yscale('log')

    # Customize the y-axis formatting
    ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:.0f}'))

    # Add labels and title
    ax.set_xticks(range(1, len(top_20_tasks) + 1))
    ax.set_xticklabels(top_20_tasks, rotation=45, ha='right')
    ax.set_xlabel('Task')
    ax.set_ylabel('Runtime (Log Scale)')
    ax.set_title('Scatter Plot of Runtimes for Top 20 Most Frequent Tasks (Log Scale)')

    # Show the scatter plot
    plt.tight_layout()


    outbase = "task-scatter-distribution"
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




sys.exit(0)

df['RoundedTime'] = df['Time'].round()
grouped = df.groupby(['RoundedTime', 'PID', 'Comm']).size().reset_index(name='EventCount')

# filter out task with less events
min_events_per_task = 500
total_events_per_pid = grouped.groupby(['PID', 'Comm'])['EventCount'].sum()
pids_to_include = total_events_per_pid[total_events_per_pid > min_events_per_task].index
filtered_grouped = grouped[grouped.set_index(['PID', 'Comm']).index.isin(pids_to_include)]

pivot_df = filtered_grouped.pivot(index='RoundedTime', columns=['PID', 'Comm'], values='EventCount')

# fill missing time values with 0
complete_time_index = pd.RangeIndex(start=df['RoundedTime'].min(), stop=df['RoundedTime'].max() + 1)
pivot_df = pivot_df.reindex(complete_time_index, fill_value=0)

plt.rcParams.update({'font.size': 6})
markersize = 2
linewidth = .7

fig, ax = plt.subplots(figsize=(10, 6))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
ax.xaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
ax.yaxis.grid(which='major', linestyle=':', linewidth='0.5', color='#bbbbbb')
ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='#bbbbbb')
ax.set_axisbelow(True)
ax.ticklabel_format(style='plain')
marker = itertools.cycle(('o', 's', '^', 'v', '*', '+', 'x', 'D')) 

for (pid, comm) in pivot_df:
    ax.plot(pivot_df.index, pivot_df[(pid, comm)], label=f'{comm} ({pid})',
            marker=next(marker), markersize=markersize, linewidth=linewidth)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Timer Expires [Hz]')
ax.legend()
#ax.set_yscale('log')

