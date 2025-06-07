import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sns
from federated_lasso import LassoCoordinateDescent, load_and_prepare_data, train_validation_split, tune_lambda
import warnings

# Configure Matplotlib to display minus sign correctly
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class TrustedAggregator:
    """
    Trusted Aggregator for Federated Lasso Regression
    
    This class implements the aggregator functionality in federated learning framework, 
    coordinating training process across multiple data nodes
    """
    
    def __init__(self, tolerance=1e-6, max_rounds=100, local_iterations=5):
        """
        Initialize trusted aggregator
        
        Parameters:
        - tolerance: Global convergence tolerance
        - max_rounds: Maximum federated learning rounds  
        - local_iterations: Local iterations per round
        """
        self.tolerance = tolerance
        self.max_rounds = max_rounds
        self.local_iterations = local_iterations
        self.global_beta = None
        self.global_intercept = None
        self.convergence_history = []
        self.round_number = 0
        
    def weighted_average(self, betas_list, intercepts_list, sample_sizes):
        """
        Calculate weighted average coefficients
        
        Weights are proportional to sample sizes of each node
        
        Parameters:
        - betas_list: List of coefficients from each node
        - sample_sizes: Sample sizes of each node
        
        Returns:
        - Weighted averaged global coefficients
        """
        total_samples = sum(sample_sizes)
        weights = [size / total_samples for size in sample_sizes]
        
        # Calculate weighted average
        global_beta = np.zeros_like(betas_list[0])
        global_intercept = 0
        for i, beta in enumerate(betas_list):
            global_beta += weights[i] * beta
            global_intercept += weights[i] * intercepts_list[i]
            
        return global_beta, global_intercept
    
    def check_convergence(self, new_beta):
        """
        Check if global model has converged
        
        Parameters:
        - new_beta: New global coefficients
        
        Returns:
        - Whether converged
        """
        if self.global_beta is None:
            return False
        
        change = np.linalg.norm(new_beta - self.global_beta)
        self.convergence_history.append(change)
        
        return change < self.tolerance
    
    def federated_training(self, data_files, test_file, model_tolerance=1e-6):
        """
        Execute federated learning training process
        
        Parameters:
        - data_files: Data file paths for each node
        - test_file: Test data file path
        - model_tolerance: Convergence tolerance for LassoCoordinateDescent internal model fitting.
        
        Returns:
        - Training results dictionary
        """
        # Load and preprocess data for each node
        nodes_data = []
        sample_sizes = []
        
        for i, data_file in enumerate(data_files):
            X, y = load_and_prepare_data(data_file)
            X_train, X_val, y_train, y_val = train_validation_split(X, y)
            
            nodes_data.append({
                'X_train': X_train,
                'X_val': X_val, 
                'y_train': y_train,
                'y_val': y_val,
                'n_samples': X_train.shape[0]
            })
            sample_sizes.append(X_train.shape[0])
        
        # Select best λ for each node
        best_lambdas = []
        
        for i, node_data in enumerate(nodes_data):
            current_best_lambda, _ = tune_lambda(
                node_data['X_train'], node_data['y_train'],
                node_data['X_val'], node_data['y_val'],
                verbose=False,
                model_tolerance=model_tolerance
            )
            best_lambdas.append(current_best_lambda)
        
        print("Best λ values chosen for each node:")
        for i, lam in enumerate(best_lambdas):
            print(f"  Node {i+1}: λ = {lam:.6f}")

        # Initialize models for each node
        node_models = []
        for i, selected_lambda in enumerate(best_lambdas):
            model = LassoCoordinateDescent(lambda_reg=selected_lambda, tol=model_tolerance)
            node_models.append(model)
        
        # Main federated learning loop
        round_losses = []
        X_test, y_test = load_and_prepare_data(test_file)
        
        converged_in_round = -1
        for round_num in range(self.max_rounds):
            local_betas = []
            local_intercepts = []
            
            for i, (model, node_data) in enumerate(zip(node_models, nodes_data)):
                local_beta, local_intercept = model.fit_partial(
                    node_data['X_train'], 
                    node_data['y_train'],
                    n_iterations=self.local_iterations,
                    initial_beta=self.global_beta
                )
                local_betas.append(local_beta)
                local_intercepts.append(local_intercept)
                
            new_global_beta, new_global_intercept = self.weighted_average(local_betas, local_intercepts, sample_sizes)
            
            converged = self.check_convergence(new_global_beta)
            
            self.global_beta = new_global_beta
            self.global_intercept = new_global_intercept
            self.round_number = round_num + 1
            
            temp_global_model = LassoCoordinateDescent(lambda_reg=0, tol=model_tolerance)
            temp_global_model.set_coefficients(self.global_beta, self.global_intercept)

            try:
                y_test_pred = temp_global_model.predict(X_test)
                global_test_loss = mean_squared_error(y_test, y_test_pred)
                round_losses.append(global_test_loss)
            except Exception:
                round_losses.append(np.nan)
            
            for model_node in node_models:
                model_node.set_coefficients(self.global_beta, self.global_intercept)
            
            if converged:
                converged_in_round = self.round_number
                break
        
        # Print summary of the final round after the loop
        last_round_ran = self.round_number
        last_loss_val = round_losses[-1] if round_losses and not np.isnan(round_losses[-1]) else 'N/A'
        last_change_val = self.convergence_history[-1] if self.convergence_history else 'N/A'
        
        print(f"\nFederated Learning Completed.")
        print(f"Final Round Status (Round {last_round_ran}):")
        
        if isinstance(last_loss_val, float):
            print(f"  Global test loss: {last_loss_val:.6f}")
        else:
            print(f"  Global test loss: {last_loss_val}")
        
        if isinstance(last_change_val, float):
            print(f"  Coefficient change: {last_change_val:.8f}")
        else:
            print(f"  Coefficient change: {last_change_val}")

        if converged_in_round != -1:
            print(f"Algorithm converged at round {converged_in_round}.")
        else:
            print(f"Reached maximum rounds ({self.max_rounds}), training ended.")

        # Final model creation and evaluation
        final_global_model = LassoCoordinateDescent(lambda_reg=0, tol=model_tolerance)
        final_global_model.set_coefficients(self.global_beta, self.global_intercept)
        y_test_pred_final = final_global_model.predict(X_test)
        final_test_loss = mean_squared_error(y_test, y_test_pred_final)
        nonzero_indices = final_global_model.get_nonzero_indices()
        
        individual_models = []
        individual_results = []
        for i, (node_data, current_best_lambda) in enumerate(zip(nodes_data, best_lambdas)):
            individual_model = LassoCoordinateDescent(lambda_reg=current_best_lambda, tol=model_tolerance)
            individual_model.fit(node_data['X_train'], node_data['y_train'])
            y_pred_individual = individual_model.predict(X_test)
            individual_test_loss = mean_squared_error(y_test, y_pred_individual)
            individual_models.append(individual_model)
            individual_results.append({
                'node': i+1,
                'best_lambda': current_best_lambda,
                'test_loss': individual_test_loss,
                'nonzero_indices': individual_model.get_nonzero_indices(),
                'n_nonzero_beta': len(individual_model.get_nonzero_indices())
            })
        
        # Calculate comparison metrics: Global vs. Individual Nodes
        global_vs_individual_metrics = []
        if final_global_model and individual_models:
            confusion_matrices = self.compute_confusion_matrices(
                final_global_model, 
                individual_models
            )
            for i, cm in enumerate(confusion_matrices):
                accuracy = 0
                precision = 0
                recall = 0
                if cm.sum() > 0:
                    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
                    # Precision: TP / (TP + FP)
                    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
                    # Recall: TP / (TP + FN)
                    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
                global_vs_individual_metrics.append({
                    'node': i + 1,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'confusion_matrix': cm.tolist()
                })

        results = {
            'best_lambdas': best_lambdas,
            'final_global_model': final_global_model,
            'final_test_loss': final_test_loss,
            'nonzero_indices': nonzero_indices,
            'n_rounds': self.round_number,
            'convergence_history': self.convergence_history,
            'round_losses': round_losses,
            'individual_models': individual_models,
            'individual_results': individual_results,
            'sample_sizes': sample_sizes,
            'aggregation_iteration': self.local_iterations,
            'global_vs_individual_metrics': global_vs_individual_metrics
        }
        return results
    
    def compute_confusion_matrices(self, global_model, individual_models, threshold=1e-8):
        """
        Compute confusion matrices between global and individual models
        
        Treat non-zero/zero status of regression coefficients as binary classification problem
        
        Parameters:
        - global_model: Global model
        - individual_models: List of individual models
        - threshold: Threshold for determining zero coefficients
        
        Returns:
        - List of confusion matrices
        """
        global_nonzero = np.abs(global_model.beta) > threshold
        confusion_matrices = []
        
        for i, individual_model in enumerate(individual_models):
            individual_nonzero = np.abs(individual_model.beta) > threshold
            
            # Compute confusion matrix
            cm = confusion_matrix(global_nonzero.astype(int), individual_nonzero.astype(int))
            confusion_matrices.append(cm)
            
        return confusion_matrices
    
    def generate_report(self, results):
        """
        Generate detailed federated learning results report
        
        Parameters:
        - results: Training results dictionary
        """
        print("\nFEDERATED LEARNING RESULTS REPORT")
        
        # 1. Report λ values for each node
        print("\n1. Regularization parameters λ selected by each data node:")
        for i, lambda_val in enumerate(results['best_lambdas']):
            print(f"   Node {i+1}: λ = {lambda_val:.6f}")
        
        # 2. Report non-zero coefficients of final aggregated model
        nonzero_indices = results['nonzero_indices']
        print(f"\n2. Non-zero regression coefficient indices in final aggregated model:")
        print(f"   Number of non-zero coefficients: {len(nonzero_indices)}")
        print(f"   Non-zero coefficient indices: {list(map(int, nonzero_indices))}")
        
        # 3. Report test losses
        print(f"\n3. Test loss comparison:")
        print(f"   Final aggregated model test loss: {results['final_test_loss']:.6f}")
        
        print(f"\n   Individual model test losses (trained independently):")
        for result in results['individual_results']:
            print(f"   Node {result['node']}: {result['test_loss']:.6f}")
        
        # 4. Report training statistics
        print(f"\n4. Training statistics:")
        print(f"   Federated learning rounds: {results['n_rounds']}")
        print(f"   Local iterations per round: {results['aggregation_iteration']}")
        
        if results.get('convergence_history') and results['convergence_history']:
            final_change = results['convergence_history'][-1]
            print(f"   Final convergence change: {final_change:.8f}")
    
    def visualize_results(self, results):
        """
        Visualize federated learning results (plots are shown, not saved to file).
        
        Parameters:
        - results: Training results dictionary
        """
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1 = axes1[0, 0]
        if results.get('convergence_history') and self.tolerance:
            ax1.semilogy(range(1, len(results['convergence_history']) + 1), 
                        results['convergence_history'], 'b-o', linewidth=2, markersize=6)
            ax1.axhline(y=self.tolerance, color='r', linestyle='--', label=f'Tolerance ({self.tolerance})')
            ax1.set_xlabel('Federated Learning Rounds')
            ax1.set_ylabel('Coefficient Change (log scale)')
            ax1.set_title('Convergence History')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No convergence history data", ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Convergence History')
        
        ax2 = axes1[0, 1]
        individual_results = results.get('individual_results', [])
        individual_nonzero_counts = [res.get('n_nonzero_beta', 0) for res in individual_results]
        global_nonzero_count = len(results.get('nonzero_indices', []))
        
        node_names = [f'Node {i+1}' for i in range(len(results.get('best_lambdas', [])))]
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow']
        
        all_counts = individual_nonzero_counts + [global_nonzero_count]
        all_names = node_names + ['Global']
        bar_colors = colors[:len(node_names)] + ['purple']
        
        ax2.bar(all_names, all_counts, color=bar_colors, alpha=0.7)
        ax2.set_ylabel('Number of Non-zero Coefficients')
        ax2.set_title('Non-zero Coefficients Count')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes1[1, 0]
        individual_test_losses = [res.get('test_loss', float('nan')) for res in individual_results]
        global_test_loss = results.get('final_test_loss', float('nan'))
        
        all_losses = individual_test_losses + [global_test_loss]
        ax3.bar(all_names, all_losses, color=bar_colors, alpha=0.7)
        ax3.set_ylabel('Test Loss')
        ax3.set_title('Test Loss Comparison')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes1[1, 1]
        round_losses = results.get('round_losses', [])
        if round_losses and not all(np.isnan(loss) for loss in round_losses if loss is not None):
            valid_losses = [loss for loss in round_losses if loss is not None and not np.isnan(loss)]
            valid_rounds = [i+1 for i, loss in enumerate(round_losses) if loss is not None and not np.isnan(loss)]
            if valid_rounds: 
                ax4.plot(valid_rounds, valid_losses, 'g-o', linewidth=2, markersize=6)
                ax4.set_xlabel('Federated Learning Rounds')
                ax4.set_ylabel('Global Test Loss')
                ax4.set_title('Global Test Loss over Rounds')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, "No valid global loss data", ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Global Test Loss over Rounds')
        else:
            ax4.text(0.5, 0.5, "No global loss data", ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Global Test Loss over Rounds')
            
        plt.tight_layout()
        plt.show()
        
        global_model_for_cm = results.get('final_global_model')
        individual_models_for_cm = results.get('individual_models', [])
        metrics_map = {m['node']: m for m in results.get('global_vs_individual_metrics', [])}
        num_nodes_for_cm = len(individual_models_for_cm)

        if global_model_for_cm and num_nodes_for_cm > 0:
            confusion_matrices = self.compute_confusion_matrices(
                global_model_for_cm, 
                individual_models_for_cm
            )
            
            plot_width_cm = 6 * num_nodes_for_cm
            if num_nodes_for_cm > 3: 
                plot_width_cm = 18 
            
            fig2, axes2 = plt.subplots(1, num_nodes_for_cm, figsize=(plot_width_cm, 5), squeeze=False)
            
            for i, (cm, ax) in enumerate(zip(confusion_matrices, axes2.flat)):
                node_num = i + 1
                metrics = metrics_map.get(node_num)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'shrink': 0.8})
                ax.set_title(f'Confusion Matrix: Node {node_num} vs Global')
                ax.set_xlabel('Individual Model')
                ax.set_ylabel('Global Model')
                
                if metrics:
                    metrics_text = (f"Accuracy: {metrics['accuracy']:.4f}\n"
                                    f"Precision: {metrics['precision']:.4f}\n"
                                    f"Recall: {metrics['recall']:.4f}")

                    ax.text(0.5, -0.4, metrics_text, 
                           transform=ax.transAxes, ha='center', fontsize=9, va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.show()
        else:
             print("\nCannot generate confusion matrix plot: Missing global model or individual model data.")

if __name__ == "__main__":
    pass