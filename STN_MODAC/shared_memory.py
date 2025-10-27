import os
import torch
import time


def create_tensor_file(name):
    """
    Function to create a tensor file with an absolute path and default size (empty tensor).
    """
    # Get the absolute path based on the current working directory
    abs_path = os.path.join(os.getcwd(), str(name) + '.pt')

    if os.path.exists(abs_path):
        print(f"Tensor file '{abs_path}' already exists.")
        empty_tensor = torch.empty(0)  # Empty tensor (no elements)
        torch.save(empty_tensor, abs_path)  # Save tensor to file
        print(f"Created new empty tensor file.")
    else:
        # Create an empty tensor
        print(f"Tensor file '{abs_path}' not found. Creating new tensor file.")
        empty_tensor = torch.empty(0)  # Empty tensor (no elements)
        torch.save(empty_tensor, abs_path)  # Save tensor to file
        print(f"Created new empty tensor file.")

    return abs_path


def write_to_tensor_file(name, data):
    """
    Function to write tensor data to a file.
    """
    abs_path = os.path.join(os.getcwd(), name)
    torch.save(data, abs_path)
    print(f"Written data to tensor file:, {data.shape}")


def read_from_tensor_file(name):
    """
    Function to read tensor data from a file without requiring shape or dtype.
    """
    abs_path = os.path.join(os.getcwd(), name)
    data = torch.load(abs_path)
    print(f"Read data from tensor file:", data.shape)
    return data

def remove_tensor_file(name):
    """
    Function to remove the tensor file.
    """
    abs_path = os.path.join(os.getcwd(), name)

    if os.path.exists(abs_path):
        os.remove(abs_path)
        print(f"Tensor file '{abs_path}' has been removed.")
    else:
        print(f"Tensor file '{abs_path}' does not exist.")
