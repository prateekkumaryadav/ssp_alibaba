#!/usr/bin/env python3
"""
Full Project Build Script
Processes multiple hours of data and builds complete analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ALIBABA MICROSERVICES TRACE - FULL PROJECT BUILD")
print("=" * 70)

# Configuration
data_dir = Path('data/raw/data/CallGraph')
output_dir = Path('data/processed')
results_dir = Path('results')
models_dir = Path('models')

output_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

# Step 1: Load and merge multiple hours of data
print("\n[1/5] Loading and processing trace data...")
csv_files = sorted(data_dir.glob('CallGraph_[0-4].csv'))
print(f"Found {len(csv_files)} data files")

all_data = []
for i, csv_file in enumerate(csv_files):
    print(f"  Loading {csv_file.name}... ", end='', flush=True)
    df = pd.read_csv(csv_file, on_bad_lines='skip', low_memory=False)
    print(f"✓ {len(df):,} records")
    all_data.append(df)

# Merge all data
print("\nMerging all data...")
df_full = pd.concat(all_data, ignore_index=True)
print(f"✓ Total records: {len(df_full):,}")
print(f"✓ Memory: {df_full.memory_usage(deep=True).sum() / 1024**3:.2f} GB")

# Step 2: Build service dependency graph
print("\n[2/5] Building service dependency graph...")

# Extract service pairs with metrics
service_pairs = df_full.groupby(['um', 'dm']).agg({
    'rt': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'traceid': 'nunique',
    'rpctype': lambda x: x.mode()[0] if len(x) > 0 else 'unknown'
}).reset_index()

service_pairs.columns = ['upstream', 'downstream', 'call_count', 'avg_rt', 
                         'median_rt', 'std_rt', 'min_rt', 'max_rt', 
                         'unique_traces', 'primary_rpc_type']

# Calculate percentiles
print("  Computing latency percentiles...")
percentiles = df_full.groupby(['um', 'dm'])['rt'].quantile([0.95, 0.99]).unstack()
percentiles.columns = ['p95_rt', 'p99_rt']
service_pairs = service_pairs.merge(
    percentiles.reset_index(), 
    left_on=['upstream', 'downstream'], 
    right_on=['um', 'dm'],
    how='left'
)
service_pairs.drop(['um', 'dm'], axis=1, errors='ignore', inplace=True)

# Save service pairs
pairs_file = output_dir / 'service_pairs_full.csv'
service_pairs.to_csv(pairs_file, index=False)
print(f"✓ Saved {len(service_pairs):,} service pairs to {pairs_file}")

# Build NetworkX graph
print("\n  Building dependency graph...")
G = nx.DiGraph()

for _, row in service_pairs.iterrows():
    G.add_edge(
        row['upstream'], 
        row['downstream'],
        weight=row['call_count'],
        avg_latency=row['avg_rt'],
        p95_latency=row['p95_rt'],
        success_rate=100.0  # Can compute from actual data if available
    )

print(f"✓ Graph: {G.number_of_nodes()} services, {G.number_of_edges()} dependencies")

# Compute graph metrics
print("\n  Computing graph metrics...")
metrics = {
    'num_services': G.number_of_nodes(),
    'num_dependencies': G.number_of_edges(),
    'density': nx.density(G),
    'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
}

# Centrality
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())
metrics['most_called'] = max(in_degrees, key=in_degrees.get)
metrics['most_calling'] = max(out_degrees, key=out_degrees.get)

print(f"✓ Metrics computed")
for key, value in metrics.items():
    print(f"    {key}: {value}")

# Save graph
graph_file = models_dir / 'service_graph.pkl'
with open(graph_file, 'wb') as f:
    pickle.dump(G, f)
print(f"✓ Saved graph to {graph_file}")

# Step 3: Performance Analysis
print("\n[3/5] Running performance analysis...")

# Overall statistics
stats = {
    'total_traces': df_full['traceid'].nunique(),
    'total_calls': len(df_full),
    'unique_services': df_full['service'].nunique(),
    'unique_upstream': df_full['um'].nunique(),
    'unique_downstream': df_full['dm'].nunique(),
    'rpc_types': df_full['rpctype'].unique().tolist(),
}

print(f"✓ Total traces: {stats['total_traces']:,}")
print(f"✓ Total calls: {stats['total_calls']:,}")
print(f"✓ Unique services: {stats['unique_services']:,}")

# Latency analysis
rt_stats = df_full['rt'].describe(percentiles=[.5, .9, .95, .99, .999])
print(f"\n  Response Time Statistics:")
print(f"    Mean: {rt_stats['mean']:.2f} ms")
print(f"    Median: {rt_stats['50%']:.2f} ms")
print(f"    P95: {rt_stats['95%']:.2f} ms")
print(f"    P99: {rt_stats['99%']:.2f} ms")

# Identify hotspots
print("\n  Identifying performance hotspots...")
hotspots = service_pairs.nlargest(10, 'avg_rt')[
    ['upstream', 'downstream', 'call_count', 'avg_rt', 'p95_rt', 'p99_rt']
]
print(f"✓ Top 10 slowest service pairs:")
print(hotspots.to_string(index=False))

# Step 4: Generate Visualizations
print("\n[4/5] Generating visualizations...")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Response time distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df_full['rt'].clip(upper=df_full['rt'].quantile(0.99)), bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Response Time (ms)')
ax1.set_ylabel('Frequency')
ax1.set_title('Response Time Distribution (5 Hours)')

# 2. Top upstream services
ax2 = fig.add_subplot(gs[0, 1])
top_um = df_full['um'].value_counts().head(10)
ax2.barh(range(len(top_um)), top_um.values)
ax2.set_yticks(range(len(top_um)))
ax2.set_yticklabels(top_um.index)
ax2.set_xlabel('Call Count')
ax2.set_title('Top 10 Upstream Services')

# 3. Top downstream services
ax3 = fig.add_subplot(gs[0, 2])
top_dm = df_full['dm'].value_counts().head(10)
ax3.barh(range(len(top_dm)), top_dm.values)
ax3.set_yticks(range(len(top_dm)))
ax3.set_yticklabels(top_dm.index)
ax3.set_xlabel('Call Count')
ax3.set_title('Top 10 Downstream Services')

# 4. RPC type distribution
ax4 = fig.add_subplot(gs[1, 0])
rpc_counts = df_full['rpctype'].value_counts()
ax4.pie(rpc_counts.values, labels=rpc_counts.index, autopct='%1.1f%%', startangle=90)
ax4.set_title('RPC Type Distribution')

# 5. Call count heatmap (top services)
ax5 = fig.add_subplot(gs[1, 1])
top_pairs = service_pairs.nlargest(20, 'call_count')
pivot_data = top_pairs.pivot_table(
    values='call_count', 
    index='upstream', 
    columns='downstream', 
    fill_value=0
)
sns.heatmap(pivot_data, cmap='YlOrRd', annot=False, fmt='g', ax=ax5, cbar_kws={'label': 'Calls'})
ax5.set_title('Service Call Heatmap (Top 20)')
ax5.set_xlabel('Downstream')
ax5.set_ylabel('Upstream')

# 6. Latency percentiles
ax6 = fig.add_subplot(gs[1, 2])
percentiles = [50, 90, 95, 99, 99.9]
values = [np.percentile(df_full['rt'], p) for p in percentiles]
ax6.bar([f'P{p}' for p in percentiles], values, color='steelblue', edgecolor='black')
ax6.set_ylabel('Latency (ms)')
ax6.set_title('Latency Percentiles')
ax6.grid(axis='y', alpha=0.3)

# 7. Service dependency graph (small sample)
ax7 = fig.add_subplot(gs[2, :])
# Create subgraph of most active services
top_services = list(set(
    list(top_um.head(5).index) + 
    list(top_dm.head(5).index)
))
subG = G.subgraph([n for n in top_services if n in G])

pos = nx.spring_layout(subG, k=2, iterations=50)
nx.draw_networkx_nodes(subG, pos, node_color='lightblue', node_size=2000, alpha=0.9, ax=ax7)
nx.draw_networkx_edges(subG, pos, alpha=0.5, arrows=True, arrowsize=20, ax=ax7)
nx.draw_networkx_labels(subG, pos, font_size=8, font_weight='bold', ax=ax7)
ax7.set_title('Service Dependency Graph (Top Services)')
ax7.axis('off')

plt.suptitle('Alibaba Microservices Trace Analysis - Complete Report', fontsize=16, y=0.995)
viz_file = results_dir / 'complete_analysis.png'
plt.savefig(viz_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved complete visualization to {viz_file}")
plt.close()

# Additional: Service degree distribution
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

in_deg_vals = list(in_degrees.values())
out_deg_vals = list(out_degrees.values())

ax[0].hist(in_deg_vals, bins=30, edgecolor='black', alpha=0.7)
ax[0].set_xlabel('In-Degree (Number of Callers)')
ax[0].set_ylabel('Number of Services')
ax[0].set_title('In-Degree Distribution')
ax[0].set_yscale('log')

ax[1].hist(out_deg_vals, bins=30, edgecolor='black', alpha=0.7, color='coral')
ax[1].set_xlabel('Out-Degree (Number of Callees)')
ax[1].set_ylabel('Number of Services')
ax[1].set_title('Out-Degree Distribution')
ax[1].set_yscale('log')

plt.tight_layout()
degree_file = results_dir / 'degree_distribution.png'
plt.savefig(degree_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved degree distribution to {degree_file}")
plt.close()

# Step 5: Generate Final Report
print("\n[5/5] Generating final report...")

report_file = results_dir / 'ANALYSIS_REPORT.md'
with open(report_file, 'w') as f:
    f.write("# Alibaba Microservices Trace Analysis Report\n\n")
    f.write("## Executive Summary\n\n")
    f.write(f"This analysis examines {len(csv_files)} hours of production microservice trace data ")
    f.write(f"from Alibaba's cloud infrastructure, containing {stats['total_calls']:,} individual ")
    f.write(f"service calls across {stats['unique_services']:,} unique services.\n\n")
    
    f.write("## Dataset Overview\n\n")
    f.write(f"- **Total Trace Records**: {stats['total_calls']:,}\n")
    f.write(f"- **Unique Traces**: {stats['total_traces']:,}\n")
    f.write(f"- **Unique Services**: {stats['unique_services']:,}\n")
    f.write(f"- **Unique Service Pairs**: {len(service_pairs):,}\n")
    f.write(f"- **RPC Types**: {', '.join(stats['rpc_types'])}\n\n")
    
    f.write("## Service Dependency Graph\n\n")
    f.write(f"- **Nodes (Services)**: {metrics['num_services']}\n")
    f.write(f"- **Edges (Dependencies)**: {metrics['num_dependencies']}\n")
    f.write(f"- **Graph Density**: {metrics['density']:.4f}\n")
    f.write(f"- **Average Degree**: {metrics['avg_degree']:.2f}\n")
    f.write(f"- **Most Called Service**: {metrics['most_called']}\n")
    f.write(f"- **Most Calling Service**: {metrics['most_calling']}\n\n")
    
    f.write("## Performance Metrics\n\n")
    f.write("### Response Time Statistics\n\n")
    f.write(f"- **Mean**: {rt_stats['mean']:.2f} ms\n")
    f.write(f"- **Median (P50)**: {rt_stats['50%']:.2f} ms\n")
    f.write(f"- **P90**: {rt_stats['90%']:.2f} ms\n")
    f.write(f"- **P95**: {rt_stats['95%']:.2f} ms\n")
    f.write(f"- **P99**: {rt_stats['99%']:.2f} ms\n")
    f.write(f"- **P99.9**: {rt_stats['99.9%']:.2f} ms\n")
    f.write(f"- **Max**: {rt_stats['max']:.2f} ms\n\n")
    
    f.write("### Top 10 Performance Hotspots\n\n")
    f.write("| Upstream | Downstream | Calls | Avg RT (ms) | P95 RT (ms) | P99 RT (ms) |\n")
    f.write("|----------|------------|-------|-------------|-------------|-------------|\n")
    for _, row in hotspots.iterrows():
        f.write(f"| {row['upstream']} | {row['downstream']} | {row['call_count']:,} | ")
        f.write(f"{row['avg_rt']:.2f} | {row['p95_rt']:.2f} | {row['p99_rt']:.2f} |\n")
    
    f.write("\n## Key Findings\n\n")
    f.write("1. **Service Complexity**: The system exhibits a complex microservice architecture with ")
    f.write(f"{metrics['num_services']} distinct services and {metrics['num_dependencies']} dependencies.\n\n")
    
    f.write("2. **Performance Characteristics**: The median response time is relatively low ")
    f.write(f"({rt_stats['50%']:.2f}ms), but there is significant tail latency with P99 at {rt_stats['99%']:.2f}ms.\n\n")
    
    f.write("3. **Critical Services**: Certain services act as major hubs in the dependency graph, ")
    f.write("indicating potential single points of failure.\n\n")
    
    f.write("## Visualizations\n\n")
    f.write("- [Complete Analysis](complete_analysis.png)\n")
    f.write("- [Degree Distribution](degree_distribution.png)\n\n")
    
    f.write("## Data Files\n\n")
    f.write("- Service pairs: `data/processed/service_pairs_full.csv`\n")
    f.write("- Dependency graph: `models/service_graph.pkl`\n\n")
    
    f.write(f"*Report generated: {pd.Timestamp.now()}*\n")

print(f"✓ Saved analysis report to {report_file}")

# Summary
print("\n" + "=" * 70)
print("BUILD COMPLETE!")
print("=" * 70)
print(f"\n📊 Processed: {stats['total_calls']:,} calls across {len(csv_files)} hours")
print(f"🕸️  Graph: {metrics['num_services']} services, {metrics['num_dependencies']} dependencies")
print(f"⚡ Performance: P50={rt_stats['50%']:.1f}ms, P95={rt_stats['95%']:.1f}ms, P99={rt_stats['99%']:.1f}ms")
print(f"\n📁 Output Files:")
print(f"   - {pairs_file}")
print(f"   - {graph_file}")
print(f"   - {viz_file}")
print(f"   - {degree_file}")
print(f"   - {report_file}")
print("\n🎯 Next Steps:")
print("   - Open Jupyter notebooks for interactive analysis")
print("   - Review visualizations in results/ folder")
print("   - Read complete report in results/ANALYSIS_REPORT.md")
print("=" * 70)
