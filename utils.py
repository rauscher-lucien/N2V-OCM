import os
import warnings
import glob
import random
import torch
import numpy as np
import tifffile
import pickle
import matplotlib.pyplot as plt

def create_result_dir(project_dir, name='new_results'):

    os.makedirs(project_dir, exist_ok=True)
    results_dir = os.path.join(project_dir, name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    checkpoints_dir = os.path.join(project_dir, name, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    return results_dir, checkpoints_dir


def create_train_val_dir(results_dir):

    os.makedirs(results_dir, exist_ok=True)
    train_dir = os.path.join(results_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(results_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)

    return train_dir, val_dir


def normalize_dataset(dataset):
    all_means = []
    all_stds = []
    all_sizes = []

    # Compute mean, std, and size for each stack
    for stack in dataset:
        all_means.append(np.mean(stack))
        all_stds.append(np.std(stack))
        all_sizes.append(stack.size)

    # Convert lists to numpy arrays for easier computation
    array_means = np.array(all_means)
    array_stds = np.array(all_stds)
    array_sizes = np.array(all_sizes)

    # Compute weighted average of mean and std based on array sizes
    total_size = np.sum(array_sizes)
    weighted_mean = np.sum(array_means * array_sizes) / total_size
    weighted_std = np.sqrt(np.sum(array_stds**2 * array_sizes) / total_size)

    # Set global mean and std
    mean = weighted_mean
    std = weighted_std

    # Compute global minimum and maximum over the entire dataset
    global_min = np.min([np.min(stack) for stack in dataset])
    global_max = np.max([np.max(stack) for stack in dataset])

    # Apply global normalization to the entire dataset using the global min and max
    normalized_dataset = []
    for stack in dataset:
        # Normalize each slice in the stack using the global mean and std
        stack_normalized = (stack - mean) / std

        # Normalize each slice in the stack using the global min and max
        stack_normalized = (stack - global_min) / (global_max - global_min)

        # Clip and normalize to [0, 1] for each slice in the stack using the global min and max
        stack_normalized = np.clip(stack_normalized, 0, 1)

        normalized_dataset.append(stack_normalized.astype(np.float32))

    return normalized_dataset


def compute_and_save_global_stats(root_folder_path, save_path='normalization_params.pkl'):
    all_means = []
    all_stds = []
    for subdir, _, files in os.walk(root_folder_path):
        for filename in files:
            if filename.lower().endswith('.tiff'):
                file_path = os.path.join(subdir, filename)
                stack = tifffile.imread(file_path)
                all_means.append(np.mean(stack))
                all_stds.append(np.std(stack))
                
    mean = np.mean(all_means)
    std = np.mean(all_stds)
    
    # Save the computed mean and std to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    
    print(f"Normalization parameters saved to {save_path}")
    return mean, std


from concurrent.futures import ProcessPoolExecutor

def process_image(file_path):
    """
    Function to read an image file and compute its mean and standard deviation.
    """
    if file_path.lower().endswith('.tiff'):
        stack = tifffile.imread(file_path)
        return np.mean(stack), np.std(stack)
    return None

def compute_and_save_global_stats2(root_folder_path, save_path='normalization_params.pkl'):
    # Prepare a list of all TIFF files
    tiff_files = [os.path.join(subdir, f)
                  for subdir, _, files in os.walk(root_folder_path)
                  for f in files if f.lower().endswith('.tiff')]
    
    # Initialize lists to store results
    all_means = []
    all_stds = []
    
    # Use ProcessPoolExecutor to parallelize image processing
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_image, tiff_files))
    
    # Filter out None results and separate means and stds
    results = [r for r in results if r is not None]
    all_means, all_stds = zip(*results)
    
    # Compute global mean and std
    mean = np.mean(all_means)
    std = np.mean(all_stds)
    
    # Save the computed mean and std to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    
    print(f"Normalization parameters saved to {save_path}")
    return mean, std


def denormalize_image(normalized_img, mean, std):
    """
    Denormalizes an image back to its original range using the provided mean and standard deviation.

    Parameters:
    - normalized_img: The image to be denormalized.
    - mean: The mean used for the initial normalization.
    - std: The standard deviation used for the initial normalization.

    Returns:
    - The denormalized image.
    """
    original_img = (normalized_img * std) + mean
    return original_img.astype(np.float32)


def load_normalization_params(data_dir):
    """
    Loads the mean and standard deviation values from a pickle file located in the specified data directory.

    Parameters:
    - data_dir: Path to the directory containing the 'normalization_params.pkl' file.

    Returns:
    - A tuple containing the mean and standard deviation values.
    """
    # Construct the path to the pickle file
    load_path = os.path.join(data_dir, 'normalization_params.pkl')
    
    # Load the parameters from the pickle file
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    
    mean = params['mean']
    std = params['std']
    
    return mean, std



def load_min_max_params(data_dir):
    """
    Loads the global minimum and maximum values from a pickle file located in the specified data directory.

    Parameters:
    - data_dir: Path to the directory containing the 'min_max_params.pkl' file.

    Returns:
    - A tuple containing the global minimum and maximum values.
    """
    # Construct the path to the pickle file
    load_path = os.path.join(data_dir, 'min_max_params.pkl')
    
    # Load the parameters from the pickle file
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    
    global_min = params['global_min']
    global_max = params['global_max']
    
    return global_min, global_max




def plot_intensity_distribution(image_array, block_execution=True):
    """
    Plots the intensity distribution and controls execution flow based on 'block_execution'.
    """
    # Create a new figure for each plot to avoid conflicts
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(image_array.flatten(), bins=50, color='blue', alpha=0.7)
    ax.set_title('Intensity Distribution')
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if block_execution:
        plt.show()
    else:
        plt.draw()
        plt.pause(1)  # Allows GUI to update
        plt.close(fig)  # Close the new figure explicitly


def get_file_path(local_path, remote_path):

    path = ''

    # Detect the operating system
    if os.name == 'nt':  # Windows
        path = local_path
    else:  # Linux and others
        path = remote_path
    
    if not os.path.exists(path):
        warnings.warn(f"Project directory '{path}' not found. Please verify the path.")
        return
    print(f"Using file path: {path}")

    return path


def clip_extremes(data, lower_percentile=0, upper_percentile=100):
    """
    Clip pixel values to the specified lower and upper percentiles.
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)



def compute_global_min_max_and_save(dataset_path):
    """
    Computes and saves the global minimum and maximum values across all TIFF stacks
    in the given directory and its subdirectories, saving the results in the same directory.

    Parameters:
    - dataset_path: Path to the directory containing the TIFF files.
    """
    global_min = float('inf')
    global_max = float('-inf')
    for subdir, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                filepath = os.path.join(subdir, filename)
                stack = tifffile.imread(filepath)
                stack_min = np.min(stack)
                stack_max = np.max(stack)
                global_min = min(global_min, stack_min)
                global_max = max(global_max, stack_max)
    
    # Define the save_path in the same directory as the dataset
    save_path = os.path.join(dataset_path, 'min_max_params.pkl')

    # Save the computed global minimum and maximum to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'global_min': global_min, 'global_max': global_max}, f)
    
    print(f"Global min and max parameters saved to {save_path}")
    return global_min, global_max



def plot_intensity_line_distribution(image, title='1', bins=200):
    plt.figure(figsize=(10, 5))

    if isinstance(image, torch.Tensor):
        # Ensure it's on the CPU and convert to NumPy
        image = image.detach().numpy()

    # Use numpy.histogram to bin the pixel intensity data, using the global min and max
    intensity_values, bin_edges = np.histogram(image, bins=bins, range=(np.min(image), np.max(image)))
    # Calculate bin centers from edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.plot(bin_centers, intensity_values, label='Pixel Intensity Distribution')
    
    plt.title('Pixel Intensity Distribution ' + title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()



def compute_global_mean_and_std(dataset_path, checkpoints_path):
    """
    Computes and saves the global mean and standard deviation across all TIFF stacks
    in the given directory and its subdirectories, saving the results in the same directory.

    Parameters:
    - dataset_path: Path to the directory containing the TIFF files.
    """
    all_means = []
    all_stds = []
    for subdir, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                filepath = os.path.join(subdir, filename)
                stack = tifffile.imread(filepath)
                all_means.append(np.mean(stack))
                all_stds.append(np.std(stack))
                
    global_mean = np.mean(all_means)
    global_std = np.mean(all_stds)
    
    # Define the save_path in the same directory as the dataset
    save_path = os.path.join(checkpoints_path, 'normalization_params.pkl')

    # Save the computed global mean and standard deviation to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'mean': global_mean, 'std': global_std}, f)
    
    print(f"Global mean and std parameters saved to {checkpoints_path}")
    return global_mean, global_std


def crop_tiff_depth_to_divisible(path, divisor):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(root, file)
                with tifffile.TiffFile(file_path) as tif:
                    images = tif.asarray()
                    depth = images.shape[0]
                    
                    # Check if depth is divisible by divisor
                    if depth % divisor != 0:
                        # Calculate new depth that is divisible
                        new_depth = depth - (depth % divisor)
                        cropped_images = images[:new_depth]
                        
                        # Save the cropped TIFF stack
                        tifffile.imwrite(file_path, cropped_images, photometric='minisblack')
                        print(f'Cropped and saved: {file_path}')


def get_device():
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda:0")
    else:
        print("GPU is not available")
        device = torch.device("cpu")
    
    return device


def print_tiff_filenames(root_folder_path):
    """
    Prints the filenames of TIFF files in the specified folder and its subdirectories.
    
    Parameters:
    - root_folder_path: Path to the root folder containing TIFF stack files.
    """
    for subdir, _, files in os.walk(root_folder_path):
        sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
        for filename in sorted_files:
            print(filename)