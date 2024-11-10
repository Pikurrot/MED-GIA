import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from model import HelicobacterClassifier
from utils import HelicoDatasetClassification
from sklearn.model_selection import StratifiedKFold
from collections import Counter


def train(model, loss_function, optimizer, train_loader, val_loader, device, num_epochs=10, patience=5, min_delta=0.01, fold=None):
    """
    Train the model on the given dataset for the specified number of epochs.

    :param model: The model to train
    :param loss_function: The loss function to use
    :param optimizer: The optimizer to use
    :param train_loader: The training data loader
    :param val_loader: The validation data loader
    :param num_epochs: The number of epochs to train for
    :param patience: The number of epochs to wait for improvement before stopping
    :param min_delta: The minimum change in the monitored quantity to qualify as an improvement
    :param fold: The current fold number (for saving models)
    """
    model = model.to(device)
    print("Starting training")
    best_loss = float('inf')
    patience_counter = 0
    best_model_path = f"best_model_fold{fold}.pth" if fold is not None else "best_model.pth"
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, data in enumerate(train_loader):
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
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    img, label = data
                    img = img.to(device)
                    label = label.to(device)
                    output = model(img)
                    loss = loss_function(output, label)
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Accuracy: {accuracy}%")
            wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss, "accuracy": accuracy})
            
            # Early Stopping
            if avg_val_loss < best_loss - min_delta:
                print(f"Epoch {epoch + 1}: Validation loss improved from {best_loss:.4f} to {avg_val_loss:.4f}. Saving model.")
                best_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)  # Save the best model
            else:
                patience_counter += 1
                print(f"Epoch {epoch + 1}: Validation loss did not improve. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    # Load the best model before returning
    model.load_state_dict(torch.load(best_model_path))
    print(f"Best validation loss: {best_loss}")


if __name__ == "__main__":
    # Initialize wandb
    wandb.login(key="07313fef21f9b32f2fb1fb00a2672258c8b5c3d4")
    wandb.init(project="MED-GIA")
    
    # Set hyperparameters
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 40,
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
    
    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    k_folds = wandb.config["k_folds"]
     
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    train_labels = [label for _, label in train_dataset]
    
    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(train_dataset, train_labels)):
        print(f"FOLD {fold}")
        print("Train label distribution:", Counter([train_labels[i] for i in train_idx]))
        print("Validation label distribution:", Counter([train_labels[i] for i in val_idx]))
        print("--------------------------------")
        
        # Sample elements randomly from a given list of indices, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        # Define data loaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], sampler=train_subsampler)
        val_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], sampler=val_subsampler)
        
        # Initialize the model
        model = HelicobacterClassifier()
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])
        
        # Train the model
        train(model, loss_function, optimizer, train_loader, val_loader, device, num_epochs=wandb.config["epochs"])
        
        # Save the model for each fold
        model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
        torch.save(model.state_dict(), f"HelicobacterClassifier_fold{fold}.pth")
        wandb.save(f"HelicobacterClassifier_fold{fold}.pth")
    
    # Final training on the entire training dataset
    final_train_loader = DataLoader(train_dataset, batch_size=wandb.config["batch_size"], shuffle=True)
    final_model = HelicobacterClassifier().to(device)
    final_loss_function = nn.CrossEntropyLoss()
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=wandb.config["learning_rate"])
    
    train(final_model, final_loss_function, final_optimizer, final_train_loader, None, device, num_epochs=wandb.config["epochs"])
    
    # Save the final model
    torch.save(final_model.state_dict(), "HelicobacterClassifier_final.pth")
    wandb.save("HelicobacterClassifier_final.pth")