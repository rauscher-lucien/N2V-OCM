import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch
import copy

from utils import *

class Normalize(object):
    """
    Normalize a single-channel image using mean and standard deviation.

    Args:
        mean (float): Mean for the channel.
        std (float): Standard deviation for the channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Normalize a single-channel image with dimensions (1, H, W).

        Args:
            img (numpy.ndarray): Image to be normalized, expected to be in the format (1, H, W).

        Returns:
            numpy.ndarray: Normalized image.
        """
        # Normalize the image by subtracting the mean and dividing by the standard deviation
        normalized_img = (img - self.mean) / self.std
        return normalized_img



class LogScale(object):
    """
    Apply logarithmic scaling to a single-channel image.

    Args:
        epsilon (float): A small value added to the input to avoid logarithm of zero.
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

    def __call__(self, img):
        """
        Apply logarithmic scaling to a single-channel image with dimensions (1, H, W).

        Args:
            img (numpy.ndarray): Image to be transformed, expected to be in the format (1, H, W).

        Returns:
            numpy.ndarray: Logarithmically scaled image.
        """
        # Apply logarithmic scaling with epsilon to avoid log(0)
        scaled_img = np.log(img + self.epsilon)
        return scaled_img



class LogScaleAndNormalize(object):
    """
    Apply logarithmic scaling followed by Z-score normalization to a single-channel image.

    Args:
        mean (float): Mean of the log-scaled data.
        std (float): Standard deviation of the log-scaled data.
        epsilon (float): A small value added to the input to avoid logarithm of zero.

    """

    def __init__(self, mean, std, epsilon=1e-10):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def __call__(self, img):
        """
        Apply logarithmic scaling followed by Z-score normalization to a single-channel image with dimensions (1, H, W).

        Args:
            img (numpy.ndarray): Image to be transformed, expected to be in the format (1, H, W).

        Returns:
            numpy.ndarray: Transformed image.
        """
        # Apply logarithmic scaling
        log_scaled_img = np.log(img + self.epsilon)
        log_scaled_mean = np.log(self.mean + self.epsilon)
        log_scaled_std = np.log(self.std + self.epsilon)

        # Normalize the log-scaled image
        normalized_img = (log_scaled_img - log_scaled_mean) / log_scaled_std
        return normalized_img



class RandomCrop(object):
    """
    Randomly crop a single-channel image to a specified size.
    
    Args:
        output_size (tuple): The target output size (height, width).
    """

    def __init__(self, output_size=(64, 64)):
        """
        Initializes the RandomCrop transformer with the desired output size.

        Parameters:
        - output_size (tuple): The target output size (height, width).
        """
        self.output_size = output_size

    def __call__(self, img):
        """
        Apply random cropping to a single-channel image with dimensions (1, H, W).

        Parameters:
        - img (numpy.ndarray): The image to be cropped, expected to be in the format (1, H, W).

        Returns:
        - numpy.ndarray: Randomly cropped image.
        """
        # Ensure that we're working with a single-channel image
        if img.ndim != 3 or img.shape[0] != 1:
            raise ValueError("Input image must have dimensions (1, H, W).")

        _, h, w = img.shape
        new_h, new_w = self.output_size

        if h > new_h and w > new_w:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            cropped_img = img[:, top:top+new_h, left:left+new_w]
        else:
            # If the image is smaller than the crop size, padding is required
            padding_top = (new_h - h) // 2 if new_h > h else 0
            padding_left = (new_w - w) // 2 if new_w > w else 0
            padding_bottom = new_h - h - padding_top if new_h > h else 0
            padding_right = new_w - w - padding_left if new_w > w else 0

            cropped_img = np.pad(img, ((0, 0), (padding_top, padding_bottom), (padding_left, padding_right)),
                                 mode='constant', constant_values=0)  # Can modify padding mode and value if needed

        return cropped_img




class RandomHorizontalFlip:
    """
    Apply random horizontal flipping to a single-channel image.
    
    Args:
        None needed for initialization.
    """

    def __call__(self, img):
        """
        Apply random horizontal flipping to a single-channel image with dimensions (1, H, W).
        
        Args:
            img (numpy.ndarray): The image to potentially flip, expected to be in the format (1, H, W).
        
        Returns:
            numpy.ndarray: Horizontally flipped image, if applied.
        """
        # Ensure that we're working with a single-channel image
        if img.ndim != 3 or img.shape[0] != 1:
            raise ValueError("Input image must have dimensions (1, H, W).")

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2)  # Flip along the width axis, which is axis 2 for (1, H, W)
        return img



class N2V_mask_generator:
    def __init__(self, perc_pixel=0.156, n2v_neighborhood_radius=5):
        """
        Initializes the N2V_mask_generator with:
        - perc_pixel: Percentage of pixels to be masked.
        - n2v_neighborhood_radius: Neighborhood radius to consider for mask generation.
        """
        self.perc_pixel = perc_pixel
        self.n2v_neighborhood_radius = n2v_neighborhood_radius

    def __call__(self, slice_data):
        """
        Apply a mask to a single-channel image by setting a specified percentage of pixels 
        to their neighborhood values, and return the original image as a label.

        Args:
            slice_data (numpy.ndarray): Image to be masked, expected to be in the format (1, H, W).

        Returns:
            dict: A dictionary containing the masked image, the original label, and the mask, all in the format (1, H, W).
        """
        if slice_data.ndim != 3 or slice_data.shape[0] != 1:
            raise ValueError("Input image must have dimensions (1, H, W).")

        h, w = slice_data.shape[1], slice_data.shape[2]
        num_pix = int(h * w * self.perc_pixel)
        assert num_pix >= 1, "Number of blind-spot pixels is below one."

        mask = np.ones(slice_data.shape, dtype=np.float32)
        masked_image = copy.deepcopy(slice_data)

        # Generate random coordinates for the mask
        coords = self.generate_random_coords(h, w, num_pix)
        radius = self.n2v_neighborhood_radius

        # Apply the neighborhood masking
        for y, x in coords:
            # Get a random neighbor within the neighborhood radius
            dy = np.random.randint(-radius // 2, radius // 2 + 1)
            dx = np.random.randint(-radius // 2, radius // 2 + 1)
            ny, nx = y + dy, x + dx

            # Handle boundary conditions
            ny = min(max(ny, 0), h - 1)
            nx = min(max(nx, 0), w - 1)

            # Set the masked image pixel to the value of the neighbor
            masked_image[0, y, x] = slice_data[0, ny, nx]
            mask[0, y, x] = 0  # Update mask to reflect masked locations

        return {'input': masked_image, 'label': slice_data, 'mask': mask}

    def generate_random_coords(self, h, w, num_pix):
        """
        Generates random coordinates for masking within a single-channel image.

        Parameters:
        - h (int): Height of the image.
        - w (int): Width of the image.
        - num_pix (int): Number of pixels to mask.

        Returns:
        - list of tuples: Random coordinates within the array.
        """
        ys = np.random.randint(0, h, num_pix)
        xs = np.random.randint(0, w, num_pix)
        return list(zip(ys, xs))



class ToTensor(object):
    """
    Convert dictionaries containing single-channel images to PyTorch tensors. This class is specifically
    designed to handle dictionaries where each value is a single-channel image formatted as (1, H, W).
    """

    def __call__(self, data):
        """
        Convert a dictionary of single-channel images to PyTorch tensors, maintaining the channel position.

        Args:
            data (dict): The input must be a dictionary where each value is a single-channel image
            in the format (1, H, W).

        Returns:
            dict: Each converted image as a PyTorch tensor in the format (1, H, W).
        """
        def convert_image(img):
            # Check image dimensions and convert to tensor
            if img.ndim != 3 or img.shape[0] != 1:
                raise ValueError("Unsupported image format: each image must be 2D with a single channel (1, H, W).")
            return torch.from_numpy(img.astype(np.float32))

        # Ensure data is a dictionary of images
        if isinstance(data, dict):
            return {key: convert_image(value) for key, value in data.items()}
        else:
            raise TypeError("Input must be a dictionary of single-channel images.")

        return converted_tensors



import numpy as np
import copy
import numpy.ma as ma

class N2V_mask_generator_median(object):

    def __init__(self, perc_pixel=0.198, n2v_neighborhood_radius=5):
        self.perc_pixel = perc_pixel
        self.local_sub_patch_radius = n2v_neighborhood_radius  # Radius for local neighborhood

    @staticmethod
    def __get_stratified_coords2D__(coord_gen, box_size, shape):
        box_count_y = int(np.ceil(shape[1] / box_size))
        box_count_x = int(np.ceil(shape[2] / box_size))
        x_coords = []
        y_coords = []
        for i in range(box_count_y):
            for j in range(box_count_x):
                y, x = next(coord_gen)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if y < shape[1] and x < shape[2]:
                    y_coords.append(y)
                    x_coords.append(x)
        return np.array(y_coords), np.array(x_coords)

    @staticmethod
    def __rand_float_coords2D__(boxsize):
        while True:
            yield np.random.rand() * boxsize, np.random.rand() * boxsize

    def __call__(self, data):
        shape = data.shape  # Determine the shape of the input data
        assert len(shape) == 3 and shape[0] == 1, "Input data must have shape (1, height, width)"

        self.dims = len(shape) - 1  # Number of spatial dimensions (excluding the channel dimension)

        num_pix = int(np.product(shape[1:]) / 100.0 * self.perc_pixel)
        assert num_pix >= 1, "Number of blind-spot pixels is below one. At least {}% of pixels should be replaced.".format(100.0 / np.product(shape[1:]))

        self.box_size = np.round(np.sqrt(100 / self.perc_pixel)).astype(int)
        self.get_stratified_coords = self.__get_stratified_coords2D__
        self.rand_float = self.__rand_float_coords2D__(self.box_size)

        label = data  # Input data as the label
        input_data = copy.deepcopy(label)
        mask = np.ones(label.shape, dtype=np.float32)  # Initialize mask

        coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size, shape=shape)
        indexing = (np.full(coords[0].shape, 0), coords[0], coords[1])
        indexing_mask = (np.full(coords[0].shape, 0), coords[0], coords[1])

        value_manipulation = self.pm_median()
        input_val = value_manipulation(input_data[0], coords, self.dims)

        input_data[indexing] = input_val
        mask[indexing_mask] = 0

        plot_first_depth_first_channel(mask)

        return {'input': input_data, 'label': label, 'mask': mask}

    def pm_median(self):
        def patch_median(patch, coords, dims):
            patch_wo_center = self.mask_center(ndims=dims)
            vals = []
            for coord in zip(*coords):
                sub_patch, crop_neg, crop_pos = self.get_subpatch(patch, coord, self.local_sub_patch_radius)
                slices = [slice(-n, s - p) for n, p, s in zip(crop_neg, crop_pos, patch_wo_center.shape)]
                sub_patch_mask = patch_wo_center[tuple(slices)]
                vals.append(np.median(sub_patch[sub_patch_mask]))
            return vals
        return patch_median

    def mask_center(self, ndims=2):
        size = self.local_sub_patch_radius * 2 + 1
        patch_wo_center = np.ones((size,) * ndims)
        patch_wo_center[self.local_sub_patch_radius, self.local_sub_patch_radius] = 0
        return ma.make_mask(patch_wo_center)

    def get_subpatch(self, patch, coord, local_sub_patch_radius, crop_patch=True):
        crop_neg, crop_pos = 0, 0
        if crop_patch:
            start = np.array(coord) - local_sub_patch_radius
            end = start + local_sub_patch_radius * 2 + 1
            crop_neg = np.minimum(start, 0)
            crop_pos = np.maximum(0, end - patch.shape)
            start -= crop_neg
            end -= crop_pos
        else:
            start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
            end = start + local_sub_patch_radius * 2 + 1
            shift = np.minimum(0, patch.shape - end)
            start += shift
            end += shift

        slices = [slice(s, e) for s, e in zip(start, end)]
        return patch[tuple(slices)], crop_neg, crop_pos

# Dummy function for plotting, replace with actual plotting function if needed
def plot_first_depth_first_channel(mask):
    pass




class CropToMultipleOf16Inference(object):
    """
    Crop an image to ensure its height and width are multiples of 16.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        _, h, w = img.shape  # Assuming img is a numpy array with shape (H, W, C) or (H, W)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        # Crop the image
        cropped_image = img[:, id_y, id_x]

        return cropped_image


class CropToMultipleOf16Video(object):
    """
    Crop a stack of images and a single image to ensure their height and width are multiples of 16.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            data (tuple): A tuple where the first element is an input stack with shape (4, H, W, 1)
                          and the second element is a target image with shape (H, W, 1).

        Returns:
            tuple: A tuple containing the cropped input stack and target image.
        """
        input_stack, target_img = data

        # Crop the input stack
        cropped_input_stack = [self.crop_image(frame) for frame in input_stack]

        # Crop the target image
        cropped_target_img = self.crop_image(target_img)

        return np.array(cropped_input_stack), cropped_target_img

    def crop_image(self, img):
        """
        Crop a single image to make its dimensions multiples of 16.

        Args:
            img (numpy.ndarray): Single image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        h, w = img.shape[:2]  # Assuming img is a numpy array with shape (H, W, C) or (H, W)

        # Compute new dimensions to be multiples of 16
        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Crop the image
        cropped_image = img[top:top+new_h, left:left+new_w]

        return cropped_image



class CropToMultipleOf32Inference(object):
    """
    Crop each slice in a stack of images to ensure their height and width are multiples of 32.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, stack):
        """
        Args:
            stack (numpy.ndarray): Stack of images to be cropped, with shape (Num_Slices, H, W).

        Returns:
            numpy.ndarray: Stack of cropped images.
        """

        num_slices, h, w = stack.shape

        # Compute new dimensions to be multiples of 32
        new_h = h - (h % 32)
        new_w = w - (w % 32)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h).astype(np.int32)
        id_x = np.arange(left, left + new_w).astype(np.int32)

        # Crop each slice in the stack
        cropped_stack = np.zeros((num_slices, new_h, new_w), dtype=stack.dtype)
        for i in range(num_slices):
            cropped_stack[i] = stack[i, id_y, :][:, id_x]

        return cropped_stack





class ToNumpy(object):

    def __call__(self, data):

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    




class BackTo01Range(object):
    """
    Normalize a tensor to the range [0, 1] based on its own min and max values.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor: A tensor with any range of values.
        
        Returns:
            A tensor normalized to the range [0, 1].
        """
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Avoid division by zero in case the tensor is constant
        if (max_val - min_val).item() > 0:
            # Normalize the tensor to [0, 1] based on its dynamic range
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else:
            # If the tensor is constant, set it to a default value, e.g., 0, or handle as needed
            normalized_tensor = tensor.clone().fill_(0)  # Here, setting all values to 0

        return normalized_tensor



class ToTensorInference(object):
    """
    Convert images or batches of images to PyTorch tensors, handling both single images
    and tuples of images (input_img, target_img). The input is expected to be in the format
    (b, h, w, c) for batches or (h, w, c) for single images, and it converts them to
    PyTorch's (b, c, h, w) format or (c, h, w) for single images.
    """

    def __call__(self, data):
        """
        Convert input images or a tuple of images to PyTorch tensors, adjusting the channel position.

        Args:
            data (numpy.ndarray or tuple of numpy.ndarray): The input can be a single image (h, w, c),
            a batch of images (b, h, w, c), or a tuple of (input_img, target_img) in similar formats.

        Returns:
            torch.Tensor or tuple of torch.Tensor: The converted image(s) as PyTorch tensor(s) in the
            format (c, h, w) for single images or (b, c, h, w) for batches. If input is a tuple, returns
            a tuple of tensors.
        """
        def convert_image(img):
            if img.ndim == 3:
                return torch.from_numpy(img.astype(np.float32))
            else:
                raise ValueError("Unsupported image format: must be (h, w, c) or (b, h, w, c).")

        # Check if the input is a tuple of images
        if isinstance(data, tuple):
            return tuple(convert_image(img) for img in data)
        else:
            return convert_image(data)
        

class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation, then convert it to 16-bit format.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        """
        Initialize with mean and standard deviation.
        
        Args:
            mean (float or tuple): Mean for each channel.
            std (float or tuple): Standard deviation for each channel.
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Denormalize the image and convert it to 16-bit format.
        
        Args:
            img (numpy array): Normalized image.
        
        Returns:
            numpy array: Denormalized 16-bit image.
        """
        # Denormalize the image by reversing the normalization process
        img_denormalized = (img * self.std) + self.mean

        # Scale the image to the range [0, 65535] and convert to 16-bit unsigned integer
        img_16bit = img_denormalized.astype(np.uint16)
        
        return img_16bit