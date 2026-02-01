"""
Service Dependency Graph Builder
Constructs interaction graphs from trace data
"""

import pandas as pd
import networkx as nx
import pickle
import argparse
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class ServiceGraphBuilder:
    """Build service dependency graphs from trace data"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def build_from_pairs(self, service_pairs: pd.DataFrame) -> nx.DiGraph:
        """
        Build directed graph from service pair data
        
        Args:
            service_pairs: DataFrame with caller, callee, and metrics
            
        Returns:
            NetworkX DiGraph of service dependencies
        """
        print("Building service dependency graph...")
        
        for _, row in service_pairs.iterrows():
            caller = row['caller']
            callee = row['callee']
            
            # Add nodes if they don't exist
            if caller not in self.graph:
                self.graph.add_node(caller)
            if callee not in self.graph:
                self.graph.add_node(callee)
            
            # Add edge with attributes
            self.graph.add_edge(
                caller, callee,
                weight=row.get('call_count', 1),
                avg_latency=row.get('avg_latency', 0),
                success_rate=row.get('success_rate', 100)
            )
        
        print(f"Graph created with {self.graph.number_of_nodes()} nodes "
              f"and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def compute_metrics(self) -> Dict:
        """
        Compute graph metrics
        
        Returns:
            Dictionary of graph metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['num_services'] = self.graph.number_of_nodes()
        metrics['num_dependencies'] = self.graph.number_of_edges()
        metrics['density'] = nx.density(self.graph)
        
        # Degree metrics
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        metrics['avg_in_degree'] = sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0
        metrics['avg_out_degree'] = sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0
        
        # Find most connected services
        metrics['most_called_service'] = max(in_degrees, key=in_degrees.get) if in_degrees else None
        metrics['most_calling_service'] = max(out_degrees, key=out_degrees.get) if out_degrees else None
        
        # Centrality metrics
        try:
            pagerank = nx.pagerank(self.graph)
            metrics['most_central_service'] = max(pagerank, key=pagerank.get)
        except:
            metrics['most_central_service'] = None
        
        # Check for cycles (circular dependencies)
        try:
            cycles = list(nx.simple_cycles(self.graph))
            metrics['num_cycles'] = len(cycles)
            metrics['has_cycles'] = len(cycles) > 0
        except:
            metrics['num_cycles'] = 0
            metrics['has_cycles'] = False
        
        return metrics
    
    def find_critical_paths(self, source: str = None, 
                           target: str = None) -> List[List[str]]:
        """
        Find critical paths in the service graph
        
        Args:
            source: Source service (if None, find all paths)
            target: Target service (if None, find all paths)
            
        Returns:
            List of paths
        """
        if source and target:
            try:
                paths = list(nx.all_simple_paths(self.graph, source, target))
                return paths
            except:
                return []
        return []
    
    def identify_bottlenecks(self, latency_threshold: float = 100) -> List[Tuple]:
        """
        Identify potential bottleneck services based on latency
        
        Args:
            latency_threshold: Latency threshold in ms
            
        Returns:
            List of (caller, callee, latency) tuples
        """
        bottlenecks = []
        
        for u, v, data in self.graph.edges(data=True):
            avg_latency = data.get('avg_latency', 0)
            if avg_latency > latency_threshold:
                bottlenecks.append((u, v, avg_latency))
        
        # Sort by latency
        bottlenecks.sort(key=lambda x: x[2], reverse=True)
        
        return bottlenecks
    
    def visualize(self, output_path: str = None, 
                 layout: str = 'spring',
                 figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize the service dependency graph
        
        Args:
            output_path: Path to save visualization
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color='lightblue',
                              node_size=1000,
                              alpha=0.9)
        
        # Draw edges with varying thickness based on call count
        edges = self.graph.edges()
        weights = [self.graph[u][v].get('weight', 1) for u, v in edges]
        max_weight = max(weights) if weights else 1
        normalized_weights = [w / max_weight * 5 for w in weights]
        
        nx.draw_networkx_edges(self.graph, pos,
                              width=normalized_weights,
                              alpha=0.5,
                              arrows=True,
                              arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, 
                               font_size=10,
                               font_weight='bold')
        
        plt.title("Microservice Dependency Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_graph(self, output_path: str):
        """Save graph to file"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to {output_path}")
    
    def load_graph(self, input_path: str):
        """Load graph from file"""
        with open(input_path, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"Graph loaded from {input_path}")
        return self.graph


def main():
    parser = argparse.ArgumentParser(description='Build service dependency graph')
    parser.add_argument('--input', required=True, 
                       help='Input CSV with service pairs')
    parser.add_argument('--output', required=True,
                       help='Output path for graph pickle file')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization')
    args = parser.parse_args()
    
    # Load service pairs
    print(f"Loading service pairs from {args.input}")
    pairs = pd.read_csv(args.input)
    
    # Build graph
    builder = ServiceGraphBuilder()
    graph = builder.build_from_pairs(pairs)
    
    # Compute metrics
    metrics = builder.compute_metrics()
    print("\nGraph Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Find bottlenecks
    bottlenecks = builder.identify_bottlenecks()
    if bottlenecks:
        print(f"\nTop 5 Bottlenecks:")
        for caller, callee, latency in bottlenecks[:5]:
            print(f"  {caller} -> {callee}: {latency:.2f}ms")
    
    # Save graph
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    builder.save_graph(args.output)
    
    # Visualize if requested
    if args.visualize:
        viz_path = args.output.replace('.pkl', '_graph.png')
        builder.visualize(output_path=viz_path)


if __name__ == '__main__':
    main()
