import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test



def plot_loss(result_dir,k = 5):
    """
    Plot the training and validation loss curves for all folds.
    
    Parameters:
    result_dir (str): Directory containing the results of all folds.
    k (int): Number of folds.
    """
    all_train_losses = []
    all_val_losses = []
    for i in range(k):
        i+=1
        log_dir = os.path.join(result_dir, f'fold_{i}', 'log.csv')
        log_data = pd.read_csv(log_dir)
        train_losses = log_data['loss'].values
        val_losses = log_data['val_loss'].values
        if i == 0:
            all_train_losses = [train_losses]
            all_val_losses = [val_losses]
        else:
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)

    
    plt.figure(figsize=(12, 8))
    
    # Define color maps for train and validation losses
    train_colors = plt.cm.Blues(np.linspace(0.4, 1, len(all_train_losses)))
    val_colors = plt.cm.Oranges(np.linspace(0.4, 1, len(all_val_losses)))
    
    for i, (train_losses, val_losses) in enumerate(zip(all_train_losses, all_val_losses)):
        plt.plot(train_losses, color=train_colors[i], label=f'Train Loss Fold {i+1}')
        plt.plot(val_losses, color=val_colors[i], linestyle='--', label=f'Val Loss Fold {i+1}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    


def plot_c_index(result_dir,k = 5):
    """
    Plot the C-index curves for all folds.
    
    Parameters:
    result_dir (str): Directory containing the results of all folds.
    k (int): Number of folds.
    """
    all_c_indices = []
    for i in range(k):
        i+=1
        log_dir = os.path.join(result_dir, f'fold_{i}', 'log.csv')
        log_data = pd.read_csv(log_dir)
        c_indices = log_data['c_index'].values
        if i == 0:
            all_c_indices = [c_indices]
        else:
            all_c_indices.append(c_indices)
            
    plt.figure(figsize=(12, 8))
    c_index_colors = plt.cm.Purples(np.linspace(0.4, 1, len(all_c_indices)))
    
    for i, c_indices in enumerate(all_c_indices):
        plt.plot(c_indices, color=c_index_colors[i], label=f'C-index Fold {i+1}')
    
    plt.xlabel('Epoch')
    plt.ylabel('C-index')
    plt.title('C-index Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_lr(result_dir,k = 5):
    """
    Plot the learning rate curves for all folds.
    
    Parameters:
    result_dir (str): Directory containing the results of all folds.
    k (int): Number of folds.
    """
    all_lrs = []
    for i in range(k):
        i+=1
        log_dir = os.path.join(result_dir, f'fold_{i}', 'log.csv')
        log_data = pd.read_csv(log_dir)
        lrs = log_data['lr'].values
        if i == 0:
            all_lrs = [lrs]
        else:
            all_lrs.append(lrs)
            
    plt.figure(figsize=(12, 8))
    lr_colors = plt.cm.Greens(np.linspace(0.4, 1, len(all_lrs)))
    
    for i, lrs in enumerate(all_lrs):
        plt.plot(lrs, color=lr_colors[i], label=f'Learning Rate Fold {i+1}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_km(result_dir,k = 5):
    for i in range(k):
        i+=1
        df = pd.read_csv(f'{result_dir}/fold_{i}/best_risk.csv')
        median_risk = df['risk_score'].median()

        df['risk_group'] = np.where(df['risk_score'] > median_risk, 'High Risk', 'Low Risk')

        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10, 6))
        kmf.fit(df[df['risk_group'] == 'High Risk']['time'], 
                df[df['risk_group'] == 'High Risk']['event'], 
                label="High Risk")
        kmf.plot()
        kmf.fit(df[df['risk_group'] == 'Low Risk']['time'], 
                df[df['risk_group'] == 'Low Risk']['event'], 
                label="Low Risk")
        kmf.plot()
        results = logrank_test(df[df['risk_group'] == 'High Risk']['time'], 
                            df[df['risk_group'] == 'Low Risk']['time'],
                            df[df['risk_group'] == 'High Risk']['event'],
                            df[df['risk_group'] == 'Low Risk']['event'])
        p_value = results.p_value
        p_value_str = f"p = {p_value:.4f}"
        plt.text(0.7, 0.1, p_value_str, transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.title("Kaplan-Meier Survival Curve by Risk Group")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.legend()
        plt.show()
        plt.close()
        
        
def mean_cindex(
    result_dir,
    k=5
):
    all_c_indices = list()
    for i in range(k):
        i+=1
        log_dir = os.path.join(result_dir, f'fold_{i}', 'log.csv')
        log_data = pd.read_csv(log_dir)
        c_indices = log_data['c_index'].values
        best_c_index = np.max(c_indices)

        all_c_indices.append(best_c_index)
            
    return np.mean(all_c_indices), all_c_indices