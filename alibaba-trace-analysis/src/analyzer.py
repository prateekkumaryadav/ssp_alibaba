"""
Performance Analyzer
Analyzes microservice performance metrics and patterns
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import argparse
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceAnalyzer:
    """Analyze microservice performance characteristics"""
    
    def __init__(self, graph_path: str = None):
        self.graph = None
        if graph_path:
            self.load_graph(graph_path)
        self.analysis_results = {}
        
    def load_graph(self, path: str):
        """Load service dependency graph"""
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"Loaded graph with {self.graph.number_of_nodes()} services")
    
    def analyze_latency_distribution(self, data: pd.DataFrame) -> Dict:
        """
        Analyze latency distribution across all services
        
        Args:
            data: DataFrame with latency column
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = data['latency'].values
        
        stats = {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p90': np.percentile(latencies, 90),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'p999': np.percentile(latencies, 99.9)
        }
        
        self.analysis_results['latency_stats'] = stats
        return stats
    
    def identify_hotspots(self, data: pd.DataFrame, 
                         top_n: int = 10) -> pd.DataFrame:
        """
        Identify performance hotspots (slow service pairs)
        
        Args:
            data: DataFrame with caller, callee, latency
            top_n: Number of top hotspots to return
            
        Returns:
            DataFrame with top hotspots
        """
        if 'caller' not in data.columns or 'callee' not in data.columns:
            return pd.DataFrame()
        
        # Group by service pair and compute statistics
        hotspots = data.groupby(['caller', 'callee']).agg({
            'latency': ['mean', 'median', 'std', 'max', 'count']
        }).reset_index()
        
        hotspots.columns = ['caller', 'callee', 'avg_latency', 
                           'median_latency', 'std_latency', 
                           'max_latency', 'call_count']
        
        # Sort by average latency
        hotspots = hotspots.sort_values('avg_latency', ascending=False)
        
        self.analysis_results['hotspots'] = hotspots.head(top_n)
        return hotspots.head(top_n)
    
    def analyze_temporal_patterns(self, data: pd.DataFrame,
                                  window: str = '5min') -> pd.DataFrame:
        """
        Analyze temporal patterns in performance
        
        Args:
            data: DataFrame with timestamp and latency
            window: Time window for aggregation
            
        Returns:
            DataFrame with temporal statistics
        """
        if 'timestamp' not in data.columns:
            return pd.DataFrame()
        
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        
        # Aggregate by time window
        temporal = data.resample(window).agg({
            'latency': ['mean', 'median', 'std', 'count']
        })
        
        temporal.columns = ['avg_latency', 'median_latency', 
                           'std_latency', 'request_count']
        
        return temporal
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        method: str = 'zscore',
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalous latency patterns
        
        Args:
            data: DataFrame with latency data
            method: 'zscore' or 'iqr'
            threshold: Anomaly threshold
            
        Returns:
            DataFrame with anomalies
        """
        if method == 'zscore':
            z_scores = np.abs((data['latency'] - data['latency'].mean()) / 
                             data['latency'].std())
            anomalies = data[z_scores > threshold].copy()
            anomalies['anomaly_score'] = z_scores[z_scores > threshold]
            
        elif method == 'iqr':
            Q1 = data['latency'].quantile(0.25)
            Q3 = data['latency'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            anomalies = data[(data['latency'] < lower) | 
                           (data['latency'] > upper)].copy()
            anomalies['anomaly_score'] = np.abs(anomalies['latency'] - 
                                               data['latency'].median())
        
        self.analysis_results['anomalies'] = anomalies
        return anomalies
    
    def compute_service_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-service performance metrics
        
        Args:
            data: DataFrame with service and performance data
            
        Returns:
            DataFrame with service-level metrics
        """
        if 'service' not in data.columns:
            return pd.DataFrame()
        
        metrics = data.groupby('service').agg({
            'latency': ['mean', 'median', 'std', 'min', 'max', 'count'],
            'status': lambda x: (x == 'success').sum() / len(x) * 100 
                      if 'status' in data.columns else 100
        }).reset_index()
        
        metrics.columns = ['service', 'avg_latency', 'median_latency',
                          'std_latency', 'min_latency', 'max_latency',
                          'request_count', 'success_rate']
        
        return metrics
    
    def analyze_critical_path(self, source: str, target: str) -> Dict:
        """
        Analyze critical path between two services
        
        Args:
            source: Source service
            target: Target service
            
        Returns:
            Dictionary with path analysis
        """
        if not self.graph:
            return {}
        
        try:
            # Find shortest path
            shortest_path = nx.shortest_path(self.graph, source, target)
            
            # Compute total latency along path
            total_latency = 0
            for i in range(len(shortest_path) - 1):
                u, v = shortest_path[i], shortest_path[i + 1]
                if self.graph.has_edge(u, v):
                    total_latency += self.graph[u][v].get('avg_latency', 0)
            
            analysis = {
                'path': shortest_path,
                'path_length': len(shortest_path) - 1,
                'total_latency': total_latency
            }
            
            return analysis
            
        except nx.NetworkXNoPath:
            return {'error': f'No path from {source} to {target}'}
    
    def visualize_latency_distribution(self, data: pd.DataFrame,
                                       output_path: str = None):
        """Visualize latency distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(data['latency'], bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Latency (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Latency Distribution')
        
        # Box plot
        axes[0, 1].boxplot(data['latency'])
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title('Latency Box Plot')
        
        # CDF
        sorted_latencies = np.sort(data['latency'])
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        axes[1, 0].plot(sorted_latencies, cdf)
        axes[1, 0].set_xlabel('Latency (ms)')
        axes[1, 0].set_ylabel('CDF')
        axes[1, 0].set_title('Cumulative Distribution')
        axes[1, 0].grid(True)
        
        # Percentiles
        percentiles = [50, 90, 95, 99, 99.9]
        values = [np.percentile(data['latency'], p) for p in percentiles]
        axes[1, 1].bar([str(p) for p in percentiles], values)
        axes[1, 1].set_xlabel('Percentile')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_title('Latency Percentiles')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, output_path: str):
        """Generate analysis report"""
        with open(output_path, 'w') as f:
            f.write("# Microservice Performance Analysis Report\n\n")
            
            # Latency statistics
            if 'latency_stats' in self.analysis_results:
                stats = self.analysis_results['latency_stats']
                f.write("## Latency Statistics\n\n")
                for key, value in stats.items():
                    f.write(f"- {key}: {value:.2f} ms\n")
                f.write("\n")
            
            # Hotspots
            if 'hotspots' in self.analysis_results:
                hotspots = self.analysis_results['hotspots']
                f.write("## Performance Hotspots\n\n")
                f.write(hotspots.to_markdown(index=False))
                f.write("\n\n")
            
            # Anomalies
            if 'anomalies' in self.analysis_results:
                anomalies = self.analysis_results['anomalies']
                f.write(f"## Anomalies Detected: {len(anomalies)}\n\n")
        
        print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze microservice performance')
    parser.add_argument('--graph', help='Path to graph pickle file')
    parser.add_argument('--data', required=True, help='Path to trace data CSV')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}")
    data = pd.read_csv(args.data)
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(args.graph)
    
    # Run analyses
    print("\nAnalyzing latency distribution...")
    latency_stats = analyzer.analyze_latency_distribution(data)
    print("\nLatency Statistics:")
    for key, value in latency_stats.items():
        print(f"  {key}: {value:.2f} ms")
    
    print("\nIdentifying hotspots...")
    hotspots = analyzer.identify_hotspots(data)
    print(f"\nTop 5 Hotspots:")
    print(hotspots.head())
    
    print("\nDetecting anomalies...")
    anomalies = analyzer.detect_anomalies(data)
    print(f"Found {len(anomalies)} anomalies")
    
    # Generate visualizations
    os.makedirs(args.output, exist_ok=True)
    viz_path = os.path.join(args.output, 'latency_distribution.png')
    analyzer.visualize_latency_distribution(data, viz_path)
    
    # Generate report
    report_path = os.path.join(args.output, 'analysis_report.md')
    analyzer.generate_report(report_path)


if __name__ == '__main__':
    main()
