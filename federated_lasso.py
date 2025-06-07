import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class LassoCoordinateDescent:
    """
    Coordinate Descent Algorithm for Lasso Regression
    
    This class implements the coordinate descent algorithm for Lasso regression, 
    including soft thresholding and convergence checking.
    """
    
    def __init__(self, lambda_reg=1.0, max_iter=100, tol=1e-6):
        """
        Initialize Lasso regression model
        
        Parameters:
        - lambda_reg: Regularization parameter λ
        - max_iter: Maximum number of iterations  
        - tol: Convergence tolerance
        """
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None
        self.intercept = None
        self.training_losses = []
        
    def soft_threshold(self, z, gamma):
        """
        Soft thresholding function, core operation of coordinate descent for Lasso
        
        Formula: soft_threshold(z, γ) = sign(z) * max(|z| - γ, 0)
        
        Parameters:
        - z: Input value
        - gamma: Threshold parameter
        
        Returns:
        - Soft thresholded value
        """
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0.0
    
    
    def fit(self, X, y):
        """
        Train Lasso regression model using coordinate descent algorithm
        
        Algorithm steps:
        1. Standardize feature matrix X and response variable y
        2. Initialize coefficient vector β
        3. Iteratively update each coefficient until convergence
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - y: Response variable (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Data standardization
        # This ensures all features are on the same scale, preventing features with large ranges from dominating
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) 

        # Handle cases where standard deviation is zero
        zero_std_mask = (X_std == 0)
        X_std[zero_std_mask] = 1.0
        
        X_normalized = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_centered = y - y_mean
        
        # Initialize coefficients
        self.beta = np.zeros(n_features)
        
        # Reset training loss record
        self.training_losses = []
        
        # Main coordinate descent loop
        for iteration in range(self.max_iter):
            beta_old = self.beta.copy()
            
            # Initialize residual
            residual = y_centered - X_normalized @ self.beta

            # Coordinate descent update for each feature
            for j in range(n_features):
                # Calculate residual excluding j-th feature
                # r = y - X_{-j} * β_{-j}
                # Do not directly use self.beta[-j], because slicing creates a copy
                # this reduces the computation time
                residual += X_normalized[:, j] * self.beta[j]
                
                # Calculate optimal update for j-th feature
                # z_j = X_j^T * r / n   because here the data is already standardized, so no need to divide x^t*x.
                z_j = np.dot(X_normalized[:, j], residual) / n_samples
                
                # Apply soft thresholding to update coefficient
                # β_j = soft_threshold(z_j, λ)
                self.beta[j] = self.soft_threshold(z_j, self.lambda_reg / n_samples)

                # Update residual
                residual -= X_normalized[:, j] * self.beta[j]

            
            # Calculate and record training loss for current iteration
            temp_intercept = y_mean - np.dot(X_mean, self.beta / X_std)
            temp_beta = self.beta / X_std
            temp_loss = self._calculate_loss_with_params(X, y, temp_beta, temp_intercept)
            self.training_losses.append(temp_loss)
            
            # Check convergence
            if np.linalg.norm(self.beta - beta_old) < self.tol:
                break
            
        # Calculate intercept
        self.intercept = y_mean - np.dot(X_mean, self.beta / X_std)
        
        # Convert coefficients back to original scale
        self.beta = self.beta / X_std
        
        # Store standardization parameters for prediction
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        
    def _calculate_loss_with_params(self, X, y, beta, intercept):
        """
        Calculate loss with given coefficients and intercept
        """
        y_pred = X @ beta + intercept
        mse_loss = np.mean((y - y_pred) ** 2)
        l1_penalty = self.lambda_reg * np.sum(np.abs(beta))
        return mse_loss + l1_penalty
    
    def predict(self, X):
        """
        Make predictions using trained model
        
        Parameters:
        - X: Input feature matrix
        
        Returns:
        - Predicted values
        """
        return X @ self.beta + self.intercept
    
    def get_nonzero_indices(self, threshold=1e-8):
        """
        Get indices of non-zero coefficients
        
        Due to numerical precision, we use a small threshold to determine if coefficient is zero
        
        Parameters:
        - threshold: Threshold for determining zero coefficients
        
        Returns:
        - List of indices with non-zero coefficients
        """
        return np.where(np.abs(self.beta) > threshold)[0]

    # Federated learning related methods
    def fit_partial(self, X, y, n_iterations=5, initial_beta=None):
        """
        Perform specified number of coordinate descent iterations for federated learning
        
        Parameters:
        - X: Feature matrix
        - y: Target variable
        - n_iterations: Number of iterations
        - initial_beta: Initial coefficients (from aggregator)
        
        Returns:
        - Updated coefficients
        """
        n_samples, n_features = X.shape
        
        # Data standardization
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)

        zero_std_mask = (X_std == 0)
        X_std[zero_std_mask] = 1.0

        X_normalized = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_centered = y - y_mean
        
        # Initialize coefficients
        if initial_beta is not None:
            # Convert global coefficients to normalized space
            self.beta = initial_beta * X_std
        else:
            self.beta = np.zeros(n_features)
        
        # Store standardization parameters
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        
        # Perform specified coordinate descent iterations
        for iteration in range(n_iterations):
            # Initialize residual
            residual = y_centered - X_normalized @ self.beta

            # Coordinate descent update for each feature
            for j in range(n_features):
                # Calculate residual excluding j-th feature
                residual += X_normalized[:, j] * self.beta[j]
                
                # Calculate optimal update for j-th feature
                z_j = np.dot(X_normalized[:, j], residual) / n_samples
                
                # Apply soft thresholding to update coefficient
                self.beta[j] = self.soft_threshold(z_j, self.lambda_reg / n_samples)

                # Update residual
                residual -= X_normalized[:, j] * self.beta[j]
        
        # Calculate intercept
        self.intercept = y_mean - np.dot(X_mean, self.beta / X_std)
        
        # Convert coefficients back to original scale and return
        beta_original_scale = self.beta / X_std
        return beta_original_scale.copy(), self.intercept
    
    def set_coefficients(self, beta, intercept=None):
        """
        Set model coefficients, used for receiving global model from aggregator
        
        Parameters:
        - beta: Coefficient vector
        - intercept: Intercept term
        """
        self.beta = beta.copy()
        if intercept is not None:
            self.intercept = intercept

def load_and_prepare_data(file_path):
    """
    Load and preprocess dataset
    
    This function reads CSV file, separates features and target variable, and performs basic data validation
    
    Parameters:
    - file_path: Path to CSV file
    
    Returns:
    - X: Feature matrix
    - y: Target variable
    """
    data = pd.read_csv(file_path)
    
    # Separate features and target variable
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    
    return X, y

def train_validation_split(X, y):
    """
    Split data into training and validation sets: first 80% train, last 20% validation)
    
    Parameters:
    - X: Feature matrix
    - y: Target variable  
    
    Returns:
    - X_train, X_val, y_train, y_val: Split datasets
    """
    n_samples = X.shape[0]
    n_train = int(n_samples * 0.8)
    
    # Sequential split: first 80% train, last 20% validation
    X_train = X[:n_train]
    X_val = X[n_train:]
    y_train = y[:n_train]
    y_val = y[n_train:]
    
    return X_train, X_val, y_train, y_val

def tune_lambda(X_train, y_train, X_val, y_val, lambda_candidates=None, verbose=True, model_tolerance=1e-6):
    """
    Tune regularization parameter λ
    
    Use validation set to select the best λ value, the most important hyperparameter in Lasso regression
    
    Parameters:
    - X_train, y_train: Training set
    - X_val, y_val: Validation set
    - lambda_candidates: List of λ candidate values (default: np.logspace(-4, 4, 50))
    - verbose: Whether to print the detailed tuning table
    - model_tolerance: Model convergence tolerance for LassoCoordinateDescent
    
    Returns:
    - best_lambda: Best λ value
    - validation_losses: List of validation losses
    """
    if lambda_candidates is None:
        lambda_candidates = np.logspace(-4, 4, 50)
        
    validation_losses = []
    
    if verbose:
        print("Starting λ parameter tuning:")
    
    # Store tuning process data
    tuning_data = []
    
    for lambda_val in lambda_candidates:
        # Train model
        model = LassoCoordinateDescent(lambda_reg=lambda_val, tol=model_tolerance)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_loss = mean_squared_error(y_val, y_val_pred)
        validation_losses.append(val_loss)
        
        # Store data
        tuning_data.append((lambda_val, val_loss))
    
    # Select best λ
    best_idx = np.argmin(validation_losses)
    best_lambda = lambda_candidates[best_idx]
    best_loss = validation_losses[best_idx]
    
    if verbose:
        # Create complete tuning results table
        print("\n" + "=" * 80)
        print("Complete λ Parameter Tuning Results")
        print("=" * 80)
        
        # Use pandas to create beautiful table
        df_results = pd.DataFrame(tuning_data, columns=['λ Value', 'Validation Loss'])
        df_results['Rank'] = df_results['Validation Loss'].rank(ascending=True).astype(int)
        df_results['Best'] = df_results.index == best_idx
        df_results['Status'] = df_results['Best'].map({True: '★ BEST', False: ''})
        
        # Reorder columns
        df_results = df_results[['λ Value', 'Validation Loss', 'Rank', 'Status']]
        
        # Set pandas display options to show all rows
        with pd.option_context('display.max_rows', None, 
                              'display.max_columns', None,
                              'display.width', None,
                              'display.float_format', '{:.6f}'.format):
            print(df_results.to_string(index=True))
        
        print("\n" + "=" * 80)
        print(f"Tuning Summary:")
        print(f"• Total λ values tested: {len(lambda_candidates)}")
        print(f"• Best λ value: {best_lambda:.6f}")
        print(f"• Lowest validation loss: {best_loss:.6f}")
        print(f"• Best rank: 1")
        print("=" * 80)
    
    return best_lambda, validation_losses

if __name__ == "__main__":
    pass
