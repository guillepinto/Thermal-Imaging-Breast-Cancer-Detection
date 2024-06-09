# wandb essentials
import wandb

# Pytorch essentials
import torch

from utils import DEVICE

def validate(model, test_loader, loss_fn, accuracy_fn, recall_fn, epoch):
    """ 
    Evaluate the model on the test dataset and log the performance metrics.

    Parameters:
    model (torch.nn.Module): The model to be evaluated.
    test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    loss_fn (function): Loss function used to compute the loss.
    accuracy_fn (function): Function to compute accuracy.
    epoch (int): The current epoch number.

    Returns:
    val_accuracy (float): The average accuracy over the test dataset.
    """
    model.eval()

    # Run the model on some test examples
    num_batches = len(test_loader)
    val_loss, val_accuracy = 0, 0

    # Disable gradient calculation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_loss += loss_fn(outputs, labels.unsqueeze(1).float()).item()
            val_accuracy += accuracy_fn(outputs, labels.unsqueeze(1).float())
            val_recall += recall_fn(outputs, labels.unsqueeze(1).float())
 
        # Average the metrics over all batches
        val_loss /= num_batches
        val_accuracy /= num_batches
        val_recall /= num_batches

        # Log the evaluation metrics at the end of batches
        wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_accuracy": val_accuracy})
        print(f"val loss: {val_loss:.3f} accuracy: {val_accuracy:.3f} [after {num_batches} batches]")
    return val_loss