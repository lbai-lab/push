import os
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets


if __name__ == "__main__":
    train_data = datasets.MNIST(
        root = os.getcwd(),
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = os.getcwd(), 
        train = False, 
        transform = ToTensor(),
    )
  
# Dataset

