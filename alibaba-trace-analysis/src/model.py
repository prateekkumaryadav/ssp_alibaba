"""
Microservice Interaction Models
Statistical and probabilistic models of service behavior
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
import pickle
import argparse
import os


class MicroserviceInteractionModel:
    """Model microservice interactions and performance"""
    
    def __init__(self):
        self.latency_models = {}
        self.throughput_models = {}
        self.transition_probabilities = {}
        
    def fit_latency_distribution(self, data: pd.DataFrame, 
                                 service_pair: tuple = None) -> dict:
        """
        Fit latency distribution for service calls
        
        Args:
            data: DataFrame with latency data
            service_pair: (caller, callee) tuple to filter data
            
        Returns:
            Dictionary with distribution parameters
        """
        if service_pair:
            caller, callee = service_pair
            mask = (data['caller'] == caller) & (data['callee'] == callee)
            latencies = data[mask]['latency'].values
        else:
            latencies = data['latency'].values
        
        # Try fitting different distributions
        distributions = ['norm', 'lognorm', 'expon', 'gamma']
        best_dist = None
        best_params = None
        best_ks = float('inf')
        
        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(latencies)
                
                # Kolmogorov-Smirnov test
                ks_stat, _ = stats.kstest(latencies, dist_name, args=params)
                
                if ks_stat < best_ks:
                    best_ks = ks_stat
                    best_dist = dist_name
                    best_params = params
            except:
                continue
        
        model = {
            'distribution': best_dist,
            'parameters': best_params,
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'median': np.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'ks_statistic': best_ks
        }
        
        if service_pair:
            self.latency_models[service_pair] = model
        
        return model
    
    def build_markov_chain(self, traces: pd.DataFrame) -> dict:
        """
        Build Markov chain model of service transitions
        
        Args:
            traces: DataFrame with trace_id and service columns
            
        Returns:
            Transition probability matrix
        """
        # Group by trace to get service sequences
        service_sequences = traces.groupby('trace_id')['service'].apply(list)
        
        # Count transitions
        transitions = {}
        for sequence in service_sequences:
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_service = sequence[i + 1]
                
                if current not in transitions:
                    transitions[current] = {}
                
                if next_service not in transitions[current]:
                    transitions[current][next_service] = 0
                
                transitions[current][next_service] += 1
        
        # Convert to probabilities
        transition_probs = {}
        for current, nexts in transitions.items():
            total = sum(nexts.values())
            transition_probs[current] = {
                next_s: count / total 
                for next_s, count in nexts.items()
            }
        
        self.transition_probabilities = transition_probs
        return transition_probs
    
    def cluster_latency_patterns(self, data: pd.DataFrame, 
                                n_clusters: int = 3) -> GaussianMixture:
        """
        Cluster latency patterns using Gaussian Mixture Model
        
        Args:
            data: DataFrame with latency and other features
            n_clusters: Number of clusters
            
        Returns:
            Fitted GMM model
        """
        features = data[['latency']].values
        
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(features)
        
        # Assign cluster labels
        data['latency_cluster'] = gmm.predict(features)
        
        print(f"Identified {n_clusters} latency patterns:")
        for i in range(n_clusters):
            cluster_data = data[data['latency_cluster'] == i]
            print(f"  Cluster {i}: mean={cluster_data['latency'].mean():.2f}ms, "
                  f"count={len(cluster_data)}")
        
        return gmm
    
    def predict_latency(self, service_pair: tuple, 
                       percentile: int = 50) -> float:
        """
        Predict latency for a service call
        
        Args:
            service_pair: (caller, callee) tuple
            percentile: Percentile to return (50=median, 95=p95, etc.)
            
        Returns:
            Predicted latency in ms
        """
        if service_pair not in self.latency_models:
            return None
        
        model = self.latency_models[service_pair]
        
        if percentile == 50:
            return model['median']
        elif percentile == 95:
            return model['p95']
        elif percentile == 99:
            return model['p99']
        else:
            # Use distribution to compute percentile
            dist_name = model['distribution']
            params = model['parameters']
            dist = getattr(stats, dist_name)
            return dist.ppf(percentile / 100, *params)
    
    def predict_service_path(self, start_service: str, 
                           max_steps: int = 10) -> list:
        """
        Predict likely service call path using Markov chain
        
        Args:
            start_service: Starting service
            max_steps: Maximum path length
            
        Returns:
            List of services in predicted path
        """
        if not self.transition_probabilities:
            return [start_service]
        
        path = [start_service]
        current = start_service
        
        for _ in range(max_steps):
            if current not in self.transition_probabilities:
                break
            
            # Choose next service based on transition probabilities
            transitions = self.transition_probabilities[current]
            next_services = list(transitions.keys())
            probs = list(transitions.values())
            
            next_service = np.random.choice(next_services, p=probs)
            path.append(next_service)
            current = next_service
        
        return path
    
    def save_models(self, output_path: str):
        """Save all models to file"""
        models = {
            'latency_models': self.latency_models,
            'throughput_models': self.throughput_models,
            'transition_probabilities': self.transition_probabilities
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"Models saved to {output_path}")
    
    def load_models(self, input_path: str):
        """Load models from file"""
        with open(input_path, 'rb') as f:
            models = pickle.load(f)
        
        self.latency_models = models.get('latency_models', {})
        self.throughput_models = models.get('throughput_models', {})
        self.transition_probabilities = models.get('transition_probabilities', {})
        
        print(f"Models loaded from {input_path}")


def main():
    parser = argparse.ArgumentParser(description='Build microservice interaction models')
    parser.add_argument('--input', required=True, help='Input CSV with trace data')
    parser.add_argument('--output', required=True, help='Output path for models')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}")
    data = pd.read_csv(args.input)
    
    # Build models
    model = MicroserviceInteractionModel()
    
    # Fit latency models for each service pair
    if 'caller' in data.columns and 'callee' in data.columns:
        service_pairs = data[['caller', 'callee']].drop_duplicates()
        
        for _, row in service_pairs.iterrows():
            pair = (row['caller'], row['callee'])
            latency_model = model.fit_latency_distribution(data, pair)
            print(f"\nLatency model for {pair[0]} -> {pair[1]}:")
            print(f"  Distribution: {latency_model['distribution']}")
            print(f"  Mean: {latency_model['mean']:.2f}ms")
            print(f"  P95: {latency_model['p95']:.2f}ms")
    
    # Build Markov chain if trace data available
    if 'trace_id' in data.columns and 'service' in data.columns:
        transition_probs = model.build_markov_chain(data)
        print(f"\nBuilt Markov chain with {len(transition_probs)} states")
    
    # Save models
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save_models(args.output)


if __name__ == '__main__':
    main()
