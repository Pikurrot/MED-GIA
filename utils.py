import torch

def extract_images_autoencoder(path:str)-> torch.Tensor:
    """
    Function to extract images for the autoencoder
    Requirements: * Only images from healthy patients
                  * Using Spreadsh
    """
