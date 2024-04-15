import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch
import copy


import numpy as np

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
    def __init__(self, perc_pixel=0.198, n2v_neighborhood_radius=5):
        """
        Initializes the N2V_mask_generator with:
        - perc_pixel: Percentage of pixels to be masked.
        - n2v_neighborhood_radius: Neighborhood radius to consider for mask generation.
        """
        self.perc_pixel = perc_pixel
        self.n2v_neighborhood_radius = n2v_neighborhood_radius

    def __call__(self, slice_data):
        """
        Apply a mask to a single-channel image, setting a specified percentage of pixels to zero,
        and return the original image as a label.
        
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

        # Apply the mask to the image
        for y, x in coords:
            masked_image[0, y, x] = 0  # Mask pixels are set to zero
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




import torch
import numpy as np

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
        h, w = img.shape[:2]  # Assuming img is a numpy array with shape (H, W, C) or (H, W)

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
        cropped_image = img[id_y, id_x]

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



