#!/usr/bin/env python3
"""
Quick Start Script for Alibaba Trace Analysis
Loads and explores the CallGraph data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Setup
sns.set_style('darkgrid')
data_path = Path('data/raw/data/CallGraph/CallGraph_0.csv')

print("=" * 60)
print("Alibaba Microservices Trace Analysis - Quick Start")
print("=" * 60)

# Load first hour of data
print(f"\n📂 Loading data from {data_path}...")
print("⚠️  Note: This may take a moment due to large file size...")
df = pd.read_csv(data_path, on_bad_lines='skip', low_memory=False)

print(f"\n✅ Loaded {len(df):,} trace records")
print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Data overview
print("\n" + "=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# Basic statistics
print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)
print(f"\nTotal trace records: {len(df):,}")
print(f"Unique trace IDs: {df['traceid'].nunique():,}")
print(f"Unique services: {df['service'].nunique():,}")
print(f"Unique upstream microservices (UM): {df['um'].nunique():,}")
print(f"Unique downstream microservices (DM): {df['dm'].nunique():,}")
print(f"Unique interfaces: {df['interface'].nunique():,}")
print(f"RPC types: {df['rpctype'].unique()}")

# Response time analysis
print("\n" + "=" * 60)
print("RESPONSE TIME ANALYSIS")
print("=" * 60)
rt_stats = df['rt'].describe(percentiles=[.5, .90, .95, .99, .999])
print(f"\n{rt_stats}")

# Top microservices by call frequency
print("\n" + "=" * 60)
print("TOP 10 UPSTREAM MICROSERVICES (Most Calls)")
print("=" * 60)
top_um = df['um'].value_counts().head(10)
for i, (ms, count) in enumerate(top_um.items(), 1):
    print(f"{i:2d}. {ms:30s}: {count:,} calls")

print("\n" + "=" * 60)
print("TOP 10 DOWNSTREAM MICROSERVICES (Most Called)")
print("=" * 60)
top_dm = df['dm'].value_counts().head(10)
for i, (ms, count) in enumerate(top_dm.items(), 1):
    print(f"{i:2d}. {ms:30s}: {count:,} calls")

# Service dependencies
print("\n" + "=" * 60)
print("SERVICE DEPENDENCIES")
print("=" * 60)
service_pairs = df.groupby(['um', 'dm']).agg({
    'rt': ['count', 'mean', 'median', 'std'],
    'traceid': 'nunique'
}).reset_index()
service_pairs.columns = ['upstream', 'downstream', 'call_count', 'avg_rt', 'median_rt', 'std_rt', 'unique_traces']
service_pairs = service_pairs.sort_values('call_count', ascending=False)

print(f"\nTotal unique service pairs: {len(service_pairs):,}")
print(f"\nTop 10 service pairs by call count:")
print(service_pairs.head(10).to_string(index=False))

# Visualization
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Response time distribution
axes[0, 0].hist(df['rt'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Response Time (ms)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Response Time Distribution (Hour 0)')
axes[0, 0].set_xlim(0, df['rt'].quantile(0.99))

# 2. Top 10 upstream services
top_um.plot(kind='barh', ax=axes[0, 1])
axes[0, 1].set_xlabel('Call Count')
axes[0, 1].set_ylabel('Upstream Service')
axes[0, 1].set_title('Top 10 Upstream Microservices')

# 3. Top 10 downstream services
top_dm.plot(kind='barh', ax=axes[1, 0])
axes[1, 0].set_xlabel('Call Count')
axes[1, 0].set_ylabel('Downstream Service')
axes[1, 0].set_title('Top 10 Downstream Microservices')

# 4. Response time over time
time_bins = pd.cut(df['timestamp'], bins=60)
rt_over_time = df.groupby(time_bins)['rt'].mean()
axes[1, 1].plot(range(len(rt_over_time)), rt_over_time.values)
axes[1, 1].set_xlabel('Time Bin (1 minute intervals)')
axes[1, 1].set_ylabel('Avg Response Time (ms)')
axes[1, 1].set_title('Response Time Trend (Hour 0)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = 'results/quick_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved visualization to {output_file}")

# Save processed service pairs
output_csv = 'data/processed/service_pairs_hour0.csv'
service_pairs.to_csv(output_csv, index=False)
print(f"✅ Saved service pairs to {output_csv}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Run Jupyter notebooks for deeper analysis:")
print("   jupyter notebook notebooks/")
print("\n2. Build the full service dependency graph:")
print("   python src/graph_builder.py --input data/processed/service_pairs_hour0.csv --output models/graph.pkl --visualize")
print("\n3. Analyze more data by processing additional hours")
print("=" * 60)
