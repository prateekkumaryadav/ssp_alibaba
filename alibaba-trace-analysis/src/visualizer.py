"""
Visualization utilities for trace analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Tuple


class TraceVisualizer:
    """Visualization utilities for microservice traces"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize visualizer with style"""
        try:
            plt.style.use(style)
        except:
            pass
        sns.set_palette("husl")
    
    def plot_service_heatmap(self, data: pd.DataFrame,
                            output_path: str = None):
        """
        Plot heatmap of service-to-service latencies
        
        Args:
            data: DataFrame with caller, callee, latency
            output_path: Path to save figure
        """
        if 'caller' not in data.columns or 'callee' not in data.columns:
            print("Missing caller/callee columns")
            return
        
        # Create pivot table
        pivot = data.pivot_table(
            values='latency',
            index='caller',
            columns='callee',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'Avg Latency (ms)'})
        plt.title('Service-to-Service Latency Heatmap')
        plt.xlabel('Callee Service')
        plt.ylabel('Caller Service')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_temporal_trends(self, data: pd.DataFrame,
                            metric: str = 'latency',
                            output_path: str = None):
        """
        Plot temporal trends of metrics
        
        Args:
            data: DataFrame with timestamp and metric columns
            metric: Metric to plot
            output_path: Path to save figure
        """
        if 'timestamp' not in data.columns:
            print("Missing timestamp column")
            return
        
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        plt.figure(figsize=(15, 6))
        plt.plot(data['timestamp'], data[metric], alpha=0.6)
        plt.xlabel('Time')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_service_comparison(self, data: pd.DataFrame,
                               services: List[str] = None,
                               output_path: str = None):
        """
        Compare latency distributions across services
        
        Args:
            data: DataFrame with service and latency columns
            services: List of services to compare (None = all)
            output_path: Path to save figure
        """
        if 'service' not in data.columns:
            print("Missing service column")
            return
        
        if services:
            data = data[data['service'].isin(services)]
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, x='service', y='latency')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Service')
        plt.ylabel('Latency (ms)')
        plt.title('Latency Distribution by Service')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_call_volume(self, data: pd.DataFrame,
                        top_n: int = 10,
                        output_path: str = None):
        """
        Plot top service pairs by call volume
        
        Args:
            data: DataFrame with caller and callee columns
            top_n: Number of top pairs to show
            output_path: Path to save figure
        """
        if 'caller' not in data.columns or 'callee' not in data.columns:
            print("Missing caller/callee columns")
            return
        
        # Count calls per pair
        call_counts = data.groupby(['caller', 'callee']).size().reset_index(name='count')
        call_counts['pair'] = call_counts['caller'] + ' → ' + call_counts['callee']
        call_counts = call_counts.sort_values('count', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(call_counts['pair'], call_counts['count'])
        plt.xlabel('Call Count')
        plt.ylabel('Service Pair')
        plt.title(f'Top {top_n} Service Pairs by Call Volume')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
