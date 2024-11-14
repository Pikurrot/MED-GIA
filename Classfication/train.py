import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from model import HelicobacterClassifier
from utils import HelicoDatasetClassification
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np


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
            img, label, _ = data
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
                    img, label,_ = data
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

            # Compute confusion matrix
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    img, label, _ = data
                    img = img.to(device)
                    label = label.to(device)
                    output = model(img)
                    _, predicted = torch.max(output.data, 1)
                    all_labels.extend(label.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

            cm = confusion_matrix(all_labels, all_predictions)
            print(f"Confusion Matrix:\n{cm}")
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                                                       y_true=np.array(all_labels),
                                                                       preds=np.array(all_predictions),
                                                                       class_names=[str(i) for i in range(len(cm))])})
            
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
    
    if val_loader is not None:
        torch.save(model.state_dict(), best_model_path)
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
    dataset = HelicoDatasetClassification(patient_id=True)
    
    k_folds = wandb.config["k_folds"]
     
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    train_labels = [label for _, label, _ in dataset]
    train_patient_ids = [patient_id for _, _, patient_id in dataset]
    torch.manual_seed(42)
    
    for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(dataset, train_labels, groups=train_patient_ids)):
        print(f"FOLD {fold}")
        print("Train label distribution:", Counter([train_labels[i] for i in train_idx]))
        print("Validation label distribution:", Counter([train_labels[i] for i in val_idx]))
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
        train(model, loss_function, optimizer, train_loader, val_loader, device, num_epochs=wandb.config["epochs"],fold=fold)
        
        # Save the model for each fold
        model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
        torch.save(model.state_dict(), f"HelicobacterClassifier_fold{fold}.pth")
        wandb.save(f"HelicobacterClassifier_fold{fold}.pth")
