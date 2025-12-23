import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class LossPlotter:
    """
    Plots training and validation losses.
    """
    def __init__(self, save_path):
        self.save_path = save_path
        self.history = {}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def update(self, metrics: dict):
        """
        Update the history with new metrics.
        Args:
            metrics (dict): Dictionary of metric names and values.
        """
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def plot(self):
        """
        Generate plots based on the history.
        """
        # --- Training Losses Plot ---
        save_file_path_train = os.path.join(os.path.dirname(self.save_path), 'losses_train.png')
        plt.figure(figsize=(12, 6))
        
        # Standard keys to look for
        keys_to_plot = [
            ('total', 'Total Loss'),
            ('reconstruction_loss', 'Reconstruction Loss'),
            ('reconstruction', 'Reconstruction Loss'),
            ('vq_loss_top', 'VQ Loss Top'),
            ('vq_loss_bottom', 'VQ Loss Bottom'),
            ('vq', 'VQ Loss'),
            ('codebook', 'Codebook Loss'),
            ('commitment', 'Commitment Loss')
        ]

        for key, label in keys_to_plot:
            if key in self.history:
                plt.plot(self.history[key], label=label)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_file_path_train)
        plt.close()

        # --- Validation Losses Plot ---
        # Check if any validation keys exist
        val_keys = [k for k in self.history.keys() if k.startswith('val_')]
        
        if val_keys:
            save_file_path_val = os.path.join(os.path.dirname(self.save_path), 'losses_val.png')
            plt.figure(figsize=(12, 6))
            
            for key in val_keys:
                label = key.replace('val_', '').replace('_', ' ').title()
                plt.plot(self.history[key], label=label)
            
            # Find best validation epoch if val_total exists
            if 'val_total' in self.history:
                val_losses = self.history['val_total']
                best_val_idx = np.argmin(val_losses)
                best_val_loss = val_losses[best_val_idx]
                
                plt.axvline(x=best_val_idx, color='r', linestyle=':', alpha=0.7, label=f'Best Val (Epoch {best_val_idx+1})')
                plt.scatter(best_val_idx, best_val_loss, color='red', zorder=5)
                plt.annotate(f'Best: {best_val_loss:.4f}', 
                             xy=(best_val_idx, best_val_loss), 
                             xytext=(10, 10), textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Validation Losses')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_file_path_val)
            plt.close()
