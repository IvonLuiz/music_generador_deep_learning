import os
import torch

class ModelCheckpoint:
    """
    Saves the model and optimizer state.
    """
    def __init__(self, save_path, model, optimizer, mode="min"):
        """
        Args:
            save_path (str): Path to save the model.
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            mode (str): "min" or "max". If "min", saves best model when metric decreases.
        """
        self.save_path = save_path
        self.model = model
        self.optimizer = optimizer
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else float('-inf')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def step(self, epoch, current_loss, metric_value=None):
        """
        Args:
            epoch (int): Current epoch.
            current_loss (float): Loss to save with the checkpoint.
            metric_value (float): Metric to check for "best" model. If None, uses current_loss.
        """
        if metric_value is None:
            metric_value = current_loss

        # Save latest
        save_dict = {
            'model_state': self.model.state_dict(),
            'epoch': epoch,
            'optimizer_state': self.optimizer.state_dict(),
            'loss': current_loss
        }
        torch.save(save_dict, self.save_path)
        
        # Check if best
        if self.mode == "min":
            improved = metric_value < self.best_score
        else:
            improved = metric_value > self.best_score
            
        if improved:
            self.best_score = metric_value
            best_save_path = os.path.join(os.path.dirname(self.save_path), "best_model.pth")
            torch.save(save_dict, best_save_path)
            print(f"New best metric: {self.best_score:.4f}. Saved best model.")
