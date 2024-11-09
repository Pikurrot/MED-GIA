import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from model import HelicobacterClassifier
from utils import HelicoDatasetClassification
from sklearn.model_selection import KFold


def train(model, loss_function, optimizer, train_loader, val_loader, device, num_epochs=10):
    """
    Train the model on the given dataset for the specified number of epochs.

    :param model: The model to train
    :param loss_function: The loss function to use
    :param optimizer: The optimizer to use
    :param train_loader: The training data loader
    :param val_loader: The validation data loader
    :param num_epochs: The number of epochs to train for
    """
    model = model.to(device) 
    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, data in enumerate(train_loader): # Data is a tuple ([B, C, H, W], [B])
            img, label = data
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

if __name__ == "__main__":
    # Initialize wandb
    wandb.login(key="07313fef21f9b32f2fb1fb00a2672258c8b5c3d4")
    wandb.init(project="MED-GIA")
    
    # Set hyperparameters
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 8,
        "batch_size": 256,
        "optimizer" : "adam",
        "k_folds": 5
    }

    print("num_epochs: ", wandb.config["epochs"])
    print("batch_size: ", wandb.config["batch_size"])
    print("learning_rate: ", wandb.config["learning_rate"])
    print("k_folds: ", wandb.config["k_folds"])
    
    # Load the dataset
    dataset = HelicoDatasetClassification()
    

    k_folds = wandb.config["k_folds"]
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------")
        
        # Sample elements randomly from a given list of indices, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        # Define data loaders for training and validation
        train_loader = DataLoader(dataset, batch_size=wandb.config["batch_size"], sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=wandb.config["batch_size"], sampler=val_subsampler)
        
        # Initialize the model
        model = HelicobacterClassifier()
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])
        
        # Train the model
        train(model, loss_function, optimizer, train_loader, val_loader, device, num_epochs=wandb.config["epochs"])
        
        # Save the model for each fold
        torch.save(model.state_dict(), f"HelicobacterClassifier_fold{fold}.pth")
        wandb.save(f"HelicobacterClassifier_fold{fold}.pth")
