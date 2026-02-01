"""
Updated Data Loader for Alibaba Cluster Trace
Adapted for actual Alibaba dataset schema
"""

import pandas as pd
import os
import argparse
from tqdm import tqdm
import tarfile


class TraceDataLoader:
    """Load and parse Alibaba cluster trace data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.traces = None
        
    def load_callgraph_data(self, limit: int = 3) -> pd.DataFrame:
        """Load CallGraph CSV files from Alibaba trace"""
        callgraph_path = os.path.join(self.data_path, 'CallGraph')
        
        if not os.path.exists(callgraph_path):
            raise FileNotFoundError(f"CallGraph directory not found: {callgraph_path}")
        
        csv_files = [f for f in os.listdir(callgraph_path) if f.endswith('.csv')]
        
        if not csv_files:
            print("Extracting tar.gz archives...")
            self._extract_archives(callgraph_path, limit)
            csv_files = [f for f in os.listdir(callgraph_path) if f.endswith('.csv')]
        
        print(f"Loading {len(csv_files)} CSV files...")
        dfs = []
        for file in tqdm(csv_files):
            try:
                df = pd.read_csv(os.path.join(callgraph_path, file), on_bad_lines='skip')
                dfs.append(df)
                print(f"Loaded {len(df):,} records from {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        self.traces = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.traces):,} records")
        return self.traces
    
    def _extract_archives(self, path: str, limit: int):
        """Extract tar.gz files"""
        archives = sorted([f for f in os.listdir(path) if f.endswith('.tar.gz')])[:limit]
        for archive in archives:
            print(f"Extracting {archive}...")
            with tarfile.open(os.path.join(path, archive), 'r:gz') as tar:
                tar.extractall(path=path)
    
    def parse_microservice_calls(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Parse microservice calls"""
        if df is None:
            df = self.traces
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
        df['caller'] = df['um']
        df['callee'] = df['dm']
        df['latency'] = df['rt']
        df['status'] = 'success'
        return df
    
    def get_service_pairs(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Get service pair statistics"""
        if df is None:
            df = self.traces
        
        if 'caller' not in df.columns:
            df = self.parse_microservice_calls(df)
        
        pairs = df.groupby(['caller', 'callee']).agg({
            'latency': ['count', 'mean', 'median', 'std', 'min', 'max'],
        }).reset_index()
        
        pairs.columns = ['caller', 'callee', 'call_count', 'avg_latency', 
                        'median_latency', 'std_latency', 'min_latency', 'max_latency']
        pairs['success_rate'] = 100.0
        return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    
    loader = TraceDataLoader(args.input)
    traces = loader.load_callgraph_data(limit=args.limit)
    parsed = loader.parse_microservice_calls(traces)
    
    os.makedirs(args.output, exist_ok=True)
    parsed.to_csv(os.path.join(args.output, 'parsed_traces.csv'), index=False)
    print(f"Saved to {args.output}/parsed_traces.csv")
    
    pairs = loader.get_service_pairs(parsed)
    pairs.to_csv(os.path.join(args.output, 'service_pairs.csv'), index=False)
    
    print(f"\n=== Summary ===")
    print(f"Total traces: {len(parsed):,}")
    print(f"Unique services: {parsed['service'].nunique()}")
    print(f"Service pairs: {len(pairs)}")
    print(f"Mean latency: {parsed['latency'].mean():.2f} ms")
    print(f"P95: {parsed['latency'].quantile(0.95):.2f} ms")


if __name__ == '__main__':
    main()
