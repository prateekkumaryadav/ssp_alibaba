"""
Updated Data Loader for Alibaba Cluster Trace (actual schema)
Loads and parses trace data files based on real Alibaba dataset structure
"""

import pandas as pd
import os
from typing import Dict, List
import argparse
from tqdm import tqdm
import tarfile


class TraceDataLoader:
    """Load and parse Alibaba cluster trace data"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to raw trace data directory
        """
        self.data_path = data_path
        self.traces = None
        
    def load_callgraph_data(self, limit: int = 3) -> pd.DataFrame:
        """
        Load CallGraph CSV files
        
        CallGraph schema:
        - timestamp: microsecond timestamp
        - traceid: trace identifier
        - service: service identifier
        - rpc_id: RPC call identifier
        - rpctype: type of RPC (rpc, http, etc)
        - um: upstream microservice
        - uminstanceid: upstream instance ID
        - interface: interface name
        - dm: downstream microservice
        - dminstanceid: downstream instance ID
        - rt: response time (latency)
        
        Returns:
            DataFrame containing CallGraph records
        """
        callgraph_path = os.path.join(self.data_path, 'CallGraph')
        
        if not os.path.exists(callgraph_path):
            raise FileNotFoundError(f"CallGraph directory not found: {callgraph_path}")
        
        csv_files = [f for f in os.listdir(callgraph_path) if f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found. Extracting from tar.gz archives...")
            self._extract_archives(callgraph_path, limit=limit)
            csv_files = [f for f in os.listdir(callgraph_path) if f.endswith('.csv')]
        
        print(and enrich microservice call data
        
        Args:
            df: DataFrame to parse (uses self.traces if None)
            
        Returns:
            Parsed DataFrame with microservice calls
        """
        if df is None:
            df = self.traces
            
        if df is None:
            raise ValueError("No data loaded. Call load_callgraph_data() first")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
        
        # Add caller and callee columns for easier analysis
        df['caller'] = df['um']
        df['callee'] = df['dm']
        df['latency'] = df['rt']
        
        # Add status column (all calls in this dataset are successful)
        df['status'] = 'success')
            print(f"Extracting {archive}...")
            try:
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(path=path)
            except Exception as e:
                print(f"Error extracting {archive}: {e}")
    
    def load_trace_files(self, file_pattern: str = "*.csv") -> pd.DataFrame:
        """Legacy method - use load_callgraph_data instead"""
        return self.load_callgraph_data()
    
    def parse_microservice_calls(self, df: pd with aggregated metrics
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with caller-callee pairs and metrics
        """
        if df is None:
            df = self.traces
        
        # Ensure we have the right columns
        if 'caller' not in df.columns:
            df = self.parse_microservice_calls(df)
        
        # Group by service pairs and aggregate metrics
        service_pairs = df.groupby(['caller', 'callee']).agg({
            'latency': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'rpctype': lambda x: x.mode()[0] if len(x) > 0 else 'unknown'
        }).reset_index()
        
        service_pairs.columns = ['caller', 'callee', 'call_count', 
                                'avg_latency', 'median_latency', 'std_latency',
                                'min_latency', 'max_latency', 'rpc_type']
        
        # Add success rate (all calls are successful in this dataset)
        service_pairs['success_rate'] = 100.0
        if df is None:
            raise ValueError("No data loaded. Call load_trace_files() first")
        
        # Parse timestamp if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
        
        # Extract caller and callee services from RPC calls
        # This depends on the actual trace format
        # Adjust based on actual Alibaba trace schema
        
        return df
    
    def get_service_pairs(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract service-to-service call pairs
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with caller-callee pairs and metrics
        """
        if df is None:
            df = self.traces
            
        # Group by service pairs and aggregate metrics
        # This is a simplified version - adjust based on actual data
        
        service_pairs = df.groupby(['caller_service', 'callee_service']).agg({
            'latency': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'status': lambda x: (x == 'success').sum() / len(x) * 100  # success rate
        }).reset_index()
        
        service_pairs.columns = ['caller', 'callee', 'call_count', 
                                'avg_latency', 'median_latency', 'std_latency',
                                'min_latency', 'max_latency', 'success_rate']
        
        return service_pairs


def main():
    parser = argparse.ArgumentParser(description='Load Alibaba trace data')
    parser.add_argument('--input', required=True, help='Input directory with trace files')
    parser.add_argument('--output', required=True, help='Output directory for procedata')
    parser.add_argument('--output', required=True, help='Output directory for processed data')
    parser.add_argument('--sample', type=int, help='Sample N records (optional)')
    parser.add_argument('--limit', type=int, default=3, help='Limit number of tar files to extract')
    args = parser.parse_args()
    
    # Load data
    loader = TraceDataLoader(args.input)
    traces = loader.load_callgraph_data(limit=args.limit)
    
    # Sample if requested
    if args.sample and args.sample < len(traces):
        print(f"Sampling {args.sample} records...")
        traces = traces.sample(n=args.sample, random_state=42)
    
    # Parse microservice calls
    parsed = loader.parse_microservice_calls(traces)
    
    # Save processed data
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, 'parsed_traces.csv')
    parsed.to_csv(output_file, index=False)
    print(f"Saved processed traces to {output_file}")
    
    # Extract and save service pairs
    pairs = loader.get_service_pairs(parsed)
    pairs_file = os.path.join(args.output, 'service_pairs.csv')
    pairs.to_csv(pairs_file, index=False)
    print(f"Saved service pairs to {pairs_file}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total traces: {len(parsed):,}")
    print(f"Unique services: {parsed['service'].nunique()}")
    print(f"Unique upstream services: {parsed['caller'].nunique()}")
    print(f"Unique downstream services: {parsed['callee'].nunique()}")
    print(f"Service pairs: {len(pairs)}")
    print(f"\nLatency statistics:")
    print(f"  Mean: {parsed['latency'].mean():.2f} ms")
    print(f"  Median: {parsed['latency'].median():.2f} ms")
    print(f"  P95: {parsed['latency'].quantile(0.95):.2f} ms")
    print(f"  P99: {parsed['latency'].quantile(0.99):.2f} ms

if __name__ == '__main__':
    main()
