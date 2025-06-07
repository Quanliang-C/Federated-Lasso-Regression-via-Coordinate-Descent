"""
Lasso Regression Analysis Runner
This is the runner function for the porject.
The functions are used in the Jupyter notebooks.
You can run the Jupyter notebooks to see the results.
Or you could run the code directly. I set a main function same as the Jupyter notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import classes and functions
from federated_lasso import (
    LassoCoordinateDescent, 
    load_and_prepare_data, 
    train_validation_split, 
    tune_lambda
)
from trusted_aggregator import TrustedAggregator


# Single Node Analysis Functions
def analyze_single_node(node_idx, data_file, test_file="test_data.csv", 
                        lambda_candidates=None, model_tolerance=1e-6,
                        verbose=True):
    """
    Analyze Lasso regression performance for a single data node
    
    Parameters:
    - node_idx: Node index (0, 1, 2)
    - data_file: Node data file path
    - test_file: Test data file path
    - lambda_candidates: List of λ candidate values
    - model_tolerance: Model convergence tolerance
    - verbose: Verbosity (includes text output for λ tuning)
    
    Returns:
    - Node analysis results dictionary
    """
    if lambda_candidates is None:
        lambda_candidates = np.logspace(-4, 4, 50)
    
    if verbose:
        print(f"Analyzing Node {node_idx + 1}: {data_file}")
    
    X, y = load_and_prepare_data(data_file)
    X_test, y_test = load_and_prepare_data(test_file)
    
    X_train, X_val, y_train, y_val = train_validation_split(X, y)
    
    if verbose:
        print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
        print("Starting λ parameter tuning:")
        
    best_lambda, val_losses = tune_lambda(
        X_train, y_train, X_val, y_val, 
        verbose=False    
    )
    
    if verbose:
        print(f"Best λ value: {best_lambda:.6f} (Validation loss: {val_losses[np.argmin(val_losses)]:.6f})")
        print(f"\nTraining final model with best λ={best_lambda:.6f}")
    final_model = LassoCoordinateDescent(lambda_reg=best_lambda, tol=model_tolerance)
    final_model.fit(X_train, y_train)
    
    nonzero_indices = final_model.get_nonzero_indices()
    n_nonzero_beta = len(nonzero_indices)
    if verbose:
        print(f"Number of non-zero coefficients: {n_nonzero_beta}")
        print(f" Non-zero coefficient indices:\n {list(map(int, nonzero_indices))}")
    
    y_test_pred = final_model.predict(X_test)
    test_loss = mean_squared_error(y_test, y_test_pred)
    if verbose:
        print(f"Test loss: {test_loss:.6f}")
        print(f"\nNode {node_idx + 1} analysis completed!")

    return {
        'node_idx': node_idx,
        'data_file': data_file,
        'best_lambda': best_lambda,
        'nonzero_indices': nonzero_indices,
        'n_nonzero_beta': n_nonzero_beta,
        'test_loss': test_loss,
        'validation_losses': val_losses,
        'training_losses': final_model.training_losses
    }

def create_summary_visualizations(node_results_list):
    """
    Generates two plots (shown, not saved):
    1. A 1x3 plot: λ tuning curves, non-zero counts, test losses.
    2. A 1xN plot: Training loss convergence for each node (N = num_nodes).

    Parameters:
    - node_results_list: List of node analysis results.
    """
    if not node_results_list:
        print("No results to visualize.")
        return

    lambda_candidates = np.logspace(-4, 4, 50)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    
    num_nodes = len(node_results_list)
    node_labels = [f'Node {res.get("node_idx", i) + 1}' for i, res in enumerate(node_results_list)]
    node_colors = colors[:num_nodes]

    # Plot 1: 1x3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: λ Parameter Tuning
    ax1 = axes[0]
    for i, result in enumerate(node_results_list):
        if 'validation_losses' in result and result['validation_losses'] is not None:
            ax1.semilogx(lambda_candidates, result['validation_losses'], 'o-', color=node_colors[i % len(node_colors)], 
                         label=node_labels[i], alpha=0.7, markersize=4)
    ax1.set_xlabel('λ (log scale)')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('λ Parameter Tuning (All Nodes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Non-zero Coefficients Count Comparison
    ax2 = axes[1]
    n_nonzero_betas = [res.get('n_nonzero_beta', 0) for res in node_results_list]
    ax2.bar(node_labels, n_nonzero_betas, color=node_colors, alpha=0.7)
    ax2.set_ylabel('Number of Non-zero Coefficients')
    ax2.set_title('Non-zero Coefficients Count Comparison')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Test Loss Comparison
    ax3 = axes[2]
    test_losses = [res.get('test_loss', np.nan) for res in node_results_list]
    ax3.bar(node_labels, test_losses, color=node_colors, alpha=0.7)
    ax3.set_ylabel('Test Loss')
    ax3.set_title('Test Loss Comparison')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot 2: Training Loss Convergence
    if num_nodes > 0:
        plot_width = 18 if num_nodes > 3 else 5 * num_nodes
        plt.figure(figsize=(plot_width, 5))
        for i, result in enumerate(node_results_list):
            plt.subplot(1, num_nodes, i + 1)
            training_losses = result.get('training_losses')
            if training_losses:
                iterations = range(1, len(training_losses) + 1)
                plt.plot(iterations, training_losses, 'o-', color=node_colors[i % len(node_colors)], linewidth=2, markersize=5)
                final_loss = training_losses[-1]
                plt.text(0.05, 0.95, f'Final Loss: {final_loss:.4f}\nIterations: {len(training_losses)}', 
                         transform=plt.gca().transAxes, verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor=node_colors[i % len(node_colors)], alpha=0.1))
            else:
                plt.text(0.5, 0.5, "No training loss data", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.xlabel('Iterations')
            plt.ylabel('Training Loss')
            plt.title(f'Node {result.get("node_idx", i) + 1} Training Convergence\n(Best λ = {result.get("best_lambda", float("nan")):.4f})')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Federated Learning Execution Functions
def run_federated_learning(data_files, test_file, local_iterations=5, 
                          fed_tolerance=1e-6, fed_max_rounds=50, 
                          model_tolerance=1e-6):
    """
    Run federated learning training, generating a report and visualization.
    
    Parameters:
    - data_files: Data file paths for each node
    - test_file: Test data file path
    - local_iterations: Local iterations per round
    - fed_tolerance: Global convergence tolerance
    - fed_max_rounds: Maximum federated learning rounds
    - model_tolerance: Model convergence tolerance (passed to LassoCoordinateDescent)
    
    Returns:
    - Simplified core federated learning results
    """
    print(f"Starting Federated Learning Training ({local_iterations} local iterations per round)")
    
    aggregator = TrustedAggregator(
        tolerance=fed_tolerance, 
        max_rounds=fed_max_rounds, 
        local_iterations=local_iterations
    )
    
    raw_results = aggregator.federated_training(data_files, test_file, 
                                                model_tolerance=model_tolerance)
    
    if not raw_results: 
        print("Federated learning training did not produce valid results or was aborted.")
        return {}

    aggregator.generate_report(raw_results) 
    
    aggregator.visualize_results(raw_results) 
    

    return {
        'scenario_name': f'{local_iterations}_iterations',
        'local_iterations': local_iterations,
        'n_rounds': raw_results.get('n_rounds', 0),
        'final_test_loss': raw_results.get('final_test_loss', float('nan')),
        'n_nonzero_beta': len(raw_results.get('nonzero_indices', [])),
        'best_lambdas_per_node': raw_results.get('best_lambdas', []),
        'global_vs_individual_metrics': raw_results.get('global_vs_individual_metrics', [])
    }



def main():
    result1 = analyze_single_node(node_idx=0, data_file="regression_data_node1.csv")
    result2 = analyze_single_node(node_idx=1, data_file="regression_data_node2.csv")
    result3 = analyze_single_node(node_idx=2, data_file="regression_data_node3.csv")

    results = [result1, result2, result3]

    create_summary_visualizations(results)

    run_federated_learning(data_files=["regression_data_node1.csv", "regression_data_node2.csv", "regression_data_node3.csv"], 
                           test_file="test_data.csv", local_iterations=5)
    run_federated_learning(data_files=["regression_data_node1.csv", "regression_data_node2.csv", "regression_data_node3.csv"], 
                           test_file="test_data.csv", local_iterations=10)


if __name__ == "__main__":
    main()
