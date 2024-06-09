# Pytorch essentials
import torch

# wandb essentials
import wandb

# Utils
from utils import DEVICE
from validation import validate

def train_log(loss, accuracy, step, current):
    """ Log the metrics for the current batch into wandb

    Args:
        loss: the value of the loss at current batch
        accuracy: the value of the accuracy at current batch
        step: actual step
        current: actual batch
    """

    # Where the magic happens
    wandb.log({"step":step, "train_loss": loss, "train_accuracy": accuracy})
    print(f"train loss: {loss:.3f} accuracy: {accuracy:.3f} [after {current} batches]")

def train_batch(images, labels, model, optimizer, criterion, metrics_fn):
    """Train the model on a single bacth of the dataloader.

    Args:
      images (torch.Tensor): A batch of input images.
      labels (torch.Tensor): Corresponding labels for the input images.
      model (torch.nn.Module): The model to be trained.
      optimizer (torch.optim.Optimizer): The optimizer used for training.
      criterion (callable): The loss function.
      metrics_fn (callable): The function to calculate metrics (e.g., accuracy).

    Returns:
        loss: the value of the loss at current batch
        accuracy: the value of the accuracy at current batch
    """

    images, labels = images.to(DEVICE), labels.to(DEVICE)

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels.unsqueeze(1).float())
    accuracy = metrics_fn(outputs, labels.unsqueeze(1).float())

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss, accuracy

def train(model, train_loader, val_loader, criterion, optimizer, accuracy_fn, epochs):
    """
    Train the given model using the specified data loader, criterion, optimizer, and metric function.

    Parameters:
    model (torch.nn.Module): The neural network model to be trained.
    loader (torch.utils.data.DataLoader): The data loader providing training batches.
    criterion (torch.nn.Module): The loss function used to compute the loss.
    optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
    metric_fn (callable): The function used to compute the training metric (e.g., accuracy).
    
    Notes:
    - This function sets the model to training mode and iterates over the data loader.
    - The function computes the loss and accuracy for each batch, updates the model parameters,
      and logs metrics using `train_log` every `n_prints` batches.
    - The step count is incremented after each logging operation.
    """

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Initialize the step counter 
    step = 0
    best_val_loss = float('inf')
    patience = 5

    # 4 means that I am going to make 4 logs of the metrics when training
    n_prints = int(len(train_loader)/4)

    # Run training and track with wandb
    for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")

      train_loss, train_accuracy = 0, 0
      model.train()
      for batch, (images, labels) in enumerate(train_loader):

          loss, accuracy = train_batch(images, labels, model, optimizer, criterion, accuracy_fn)
          train_loss += loss.item()
          train_accuracy += accuracy

          # Report metrics every n_prints batch
          if batch % n_prints == n_prints-1:
              train_log(train_loss/(batch+1), train_accuracy/(batch+1), step, batch)
              # Increment the step after logging
              step += 1
            
      # and validate its performance per epoch
      val_loss = validate(model, val_loader, criterion, accuracy_fn, epoch=t)

      # Early stopping    
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 5  # Reset patience counter
      else:
        patience -= 1
        if patience == 0:
            break