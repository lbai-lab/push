import torch
import torch.nn as nn
import push.push as ppush
import torch.nn.functional as F
from torch.utils.data import Dataset
from push.bayes.ensemble import mk_optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os

# =============================================================================
# Dataset
# =============================================================================

class SineDataset(Dataset):
    def __init__(self, N, D, begin, end):
        self.xs = torch.linspace(begin, end, N).reshape(N, D)
        self.ys = torch.sin(self.xs[:, 0]).reshape(-1, 1) 

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class SineWithNoiseDataset(Dataset):
    def __init__(self, N, D, begin, end, noise_std=0.0005):
        self.xs = torch.linspace(begin, end, N).reshape(N, D)
        true_ys = torch.sin(self.xs[:, 0]).reshape(-1, 1)
        mean = torch.zeros(true_ys.size()[0])
        # Create the tensor with the specified entries
        std = torch.pow(torch.arange(0, N), 0.9) * noise_std
        noise = torch.normal(mean, std).view(-1,1)
        self.ys = true_ys + noise

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class CustomMNISTDataset(Dataset):
    def __init__(self, root, numbers=[0, 1], train=False, transform=None, limit=10):
        self.root = root
        self.train = train
        self.transform = transform
        self.numbers = numbers
        self.limit = limit

        # Download MNIST dataset
        self.mnist_dataset = datasets.MNIST(root=root, train=train, transform=transforms.ToTensor(), download=False)
        
        # Filter images based on selected numbers
        indices = np.isin(self.mnist_dataset.targets.numpy(), self.numbers)
        self.data = self.mnist_dataset.data[indices][:self.limit]
        self.targets = self.mnist_dataset.targets[indices][:self.limit]

    def __len__(self):
        return self.limit

    def __getitem__(self, idx):
        image, label = self.data[idx], int(self.targets[idx])
        
        # Convert to PIL Image
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label

class SelectMNISTDataset(Dataset):
    def __init__(self, root, numbers=[0, 1], num_entries_per_digit=None, train=False, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.numbers = numbers
        self.num_entries_per_digit = num_entries_per_digit

        # Download MNIST dataset
        self.mnist_dataset = datasets.MNIST(root=root, train=train, transform=transforms.ToTensor(), download=True)

        # Filter images based on selected numbers and number of entries per digit
        indices = []
        for digit in self.numbers:
            num_entries = self.num_entries_per_digit
            digit_indices = np.where(self.mnist_dataset.targets.numpy() == digit)[0][:num_entries]
            indices.extend(digit_indices)

        self.data = self.mnist_dataset.data[indices]
        self.targets = self.mnist_dataset.targets[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], int(self.targets[idx])

        # Convert to PIL Image
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class AddImpulseNoise(object):
    def __init__(self, prob=0.05):
        """
        Initializes the impulse noise addition function.
        
        Parameters:
        prob (float): The probability of an individual pixel being affected by impulse noise.
                      Should be between 0 and 1.
        """
        self.prob = prob
        
    def __call__(self, tensor):
        """
        Adds impulse noise to the given tensor.
        
        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which noise will be added.
        
        Returns:
        torch.Tensor: The tensor with impulse noise added.
        """
        if len(tensor.size()) == 3:  # Assuming CxHxW
            channels, height, width = tensor.size()
        else:
            raise ValueError("Tensor is not in CxHxW format")
        
        mask = torch.rand(channels, height, width) < self.prob
        pepper_mask = (torch.rand(channels, height, width) < 0.5) & mask
        salt_mask = mask ^ pepper_mask
        
        tensor = tensor.clone()  # Clone to avoid modifying the original tensor
        tensor[pepper_mask] = 0  # Setting the pepper pixels to black
        tensor[salt_mask] = 1  # Setting the salt pixels to white
        
        return tensor
    
    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'

class AddShotNoise(object):
    def __init__(self, scale=1.0):
        """
        Initializes the shot noise addition function.
        
        Parameters:
        scale (float): A scaling factor that adjusts the intensity of the noise.
        """
        self.scale = scale
        
    def __call__(self, tensor):
        """
        Adds shot noise to the given tensor.
        
        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which noise will be added.
        
        Returns:
        torch.Tensor: The tensor with shot noise added.
        """
        if tensor.min() < 0 or tensor.max() > 1:
            raise ValueError("Input tensor should have values between 0 and 1")
        
        # Shot noise follows a Poisson distribution. The lambda parameter (mean) of the distribution
        # is proportional to the pixel values of the image. We scale the noise to control intensity.
        noise = torch.poisson(tensor * self.scale) / self.scale
        
        # Ensure we don't go out of bounds
        return torch.clamp(noise, 0, 1)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale})'

class AddDefocusBlur(object):
    def __init__(self, radius=5):
        """
        Initializes the defocus blur addition function.
        
        Parameters:
        radius (int): The radius of the blur kernel. A larger radius will result in a stronger blur.
        """
        self.radius = radius
        self.kernel = self._create_circular_kernel(radius)
        
    def _create_circular_kernel(self, radius):
        """
        Creates a circular kernel to simulate defocus blur.
        
        Parameters:
        radius (int): The radius of the circular kernel.
        
        Returns:
        torch.Tensor: A circular kernel as a 2D tensor.
        """
        diameter = 2 * radius + 1
        L = np.arange(diameter) - radius
        X, Y = np.meshgrid(L, L)
        circular_kernel = np.sqrt(X**2 + Y**2) <= radius
        circular_kernel = torch.tensor(circular_kernel / circular_kernel.sum(), dtype=torch.float32)
        return circular_kernel
    
    def __call__(self, tensor):
        """
        Applies defocus blur to the given tensor.
        
        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which blur will be added.
        
        Returns:
        torch.Tensor: The tensor with defocus blur added.
        """
        C, H, W = tensor.shape
        if C not in [1, 3]:
            raise ValueError("Input tensor must have 1 or 3 channels.")
        
        kernel = self.kernel.to(tensor.device)
        kernel = kernel.expand(C, 1, *kernel.shape)  # Match the tensor's shape (C, H, W)
        padding = self.radius
        
        blurred_tensor = F.conv2d(tensor.unsqueeze(0), kernel, padding=padding, groups=C)
        return blurred_tensor.squeeze(0)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(radius={self.radius})'

class AddGlassBlur(object):
    def __init__(self, radius=1, prob=1.0):
        """
        Initializes the glass blur addition function.

        Parameters:
        radius (int): The radius of the neighborhood from which pixels are randomly sampled.
                      Determines the strength of the blur.
        prob (float): Probability of applying the blur to a pixel. Allows for partial application
                      of the effect for a more varied appearance.
        """
        self.radius = radius
        self.prob = prob

    def __call__(self, tensor):
        """
        Applies glass blur to the given tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which the blur will be added.

        Returns:
        torch.Tensor: The tensor with glass blur added.
        """
        if tensor.dim() != 3:
            raise ValueError("Input tensor must be 3D (C x H x W)")

        C, H, W = tensor.shape
        blurred_tensor = tensor.clone()

        for h in range(H):
            for w in range(W):
                if random.random() < self.prob:
                    # Randomly select a pixel within the neighborhood
                    dh = random.randint(-self.radius, self.radius)
                    dw = random.randint(-self.radius, self.radius)
                    h_new = min(max(h + dh, 0), H - 1)
                    w_new = min(max(w + dw, 0), W - 1)

                    # Apply the blur
                    blurred_tensor[:, h, w] = tensor[:, h_new, w_new]

        return blurred_tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(radius={self.radius}, prob={self.prob})'

class AddMotionBlur(object):
    def __init__(self, length=10, angle=45):
        """
        Initializes the motion blur addition function.

        Parameters:
        length (int): The length of the motion blur, representing the blur strength.
        angle (float): The angle of the motion blur in degrees, with 0 degrees starting
                       from the horizontal line to the right and increasing in the
                       counterclockwise direction.
        """
        self.length = length
        self.angle = angle
        self.kernel = self._create_motion_blur_kernel(length, angle)
        
    def _create_motion_blur_kernel(self, length, angle):
        """
        Creates a motion blur kernel.

        Parameters:
        length (int): The length of the blur.
        angle (float): The angle of the blur in degrees.

        Returns:
        torch.Tensor: A 2D tensor representing the motion blur kernel.
        """
        # Angle adjustment
        angle = -angle % 360
        rad = np.deg2rad(angle)

        # Kernel size
        half_length = length // 2
        kernel_size = length
        if kernel_size % 2 == 0:
            kernel_size += 1  # Make sure the kernel size is odd

        # Create a blank kernel
        kernel = torch.zeros((kernel_size, kernel_size))
        
        # Calculate the line equation parameters
        cos_rad = np.cos(rad)
        sin_rad = np.sin(rad)
        if cos_rad == 0:
            sin_rad = 1 if sin_rad > 0 else -1
        
        # Fill in the kernel
        for x in range(-half_length, half_length + 1):
            for y in range(-half_length, half_length + 1):
                if abs(sin_rad * x - cos_rad * y) < np.sqrt(2) / 2:
                    kernel[y + half_length, x + half_length] = 1
        
        # Normalize the kernel
        kernel /= kernel.sum()
        return kernel.view(1, 1, *kernel.size())

    def __call__(self, tensor):
        """
        Applies motion blur to the given tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which blur will be added.

        Returns:
        torch.Tensor: The tensor with motion blur added.
        """
        C, H, W = tensor.shape
        padded_tensor = F.pad(tensor, (self.length, self.length, self.length, self.length), mode='reflect')
        blurred_tensor = F.conv2d(padded_tensor.unsqueeze(0), self.kernel.repeat(C, 1, 1, 1),
                                  padding=0, groups=C)
        return blurred_tensor.squeeze(0)

    def __repr__(self):
        return f'{self.__class__.__name__}(length={self.length}, angle={self.angle})'

class AddZoomBlur(object):
    def __init__(self, intensity=5, steps=10):
        """
        Initializes the zoom blur addition function.

        Parameters:
        intensity (int): The intensity of the zoom effect. Higher values result in a more
                         pronounced blur. This controls how much the image is scaled.
        steps (int): The number of steps to use in the zoom. More steps will create a
                     smoother, more continuous blur effect.
        """
        self.intensity = intensity
        self.steps = steps

    def __call__(self, tensor):
        """
        Applies zoom blur to the given tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which the blur will be added.

        Returns:
        torch.Tensor: The tensor with zoom blur added.
        """
        if tensor.dim() != 3:
            raise ValueError("Input tensor must be 3D (C x H x W)")

        C, H, W = tensor.shape
        blurred_tensor = tensor.clone()
        center = (H / 2, W / 2)

        for step in range(1, self.steps + 1):
            # Scale factor decreases with each step to simulate zooming out
            scale_factor = 1 + self.intensity * step / self.steps

            # Calculate new size and ensure it's at least 1 pixel
            new_H, new_W = max(int(H / scale_factor), 1), max(int(W / scale_factor), 1)
            
            # Resize and then pad to maintain original size
            resized = F.interpolate(tensor.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False)
            padded = F.pad(resized, self._calculate_padding(H, W, new_H, new_W))

            # Accumulate the effect
            blurred_tensor += padded.squeeze(0)
        
        # Normalize to keep the tensor within valid range
        return blurred_tensor / (self.steps + 1)

    def _calculate_padding(self, original_H, original_W, new_H, new_W):
        """
        Calculates padding to center the scaled image within the original dimensions.

        Parameters:
        original_H (int), original_W (int): Dimensions of the original image.
        new_H (int), new_W (int): Dimensions of the scaled image.

        Returns:
        tuple: Padding values for the left, right, top, and bottom.
        """
        pad_vertical = (original_H - new_H) // 2
        pad_horizontal = (original_W - new_W) // 2
        return (pad_horizontal, pad_horizontal, pad_vertical, pad_vertical)

    def __repr__(self):
        return f'{self.__class__.__name__}(intensity={self.intensity}, steps={self.steps})'

class AddGaussianBlur(object):
    def __init__(self, kernel_size=5, sigma=1.0):
        """
        Initializes the Gaussian blur addition function.

        Parameters:
        kernel_size (int): The size of the kernel. It should be an odd number to ensure
                           that the kernel has a center pixel.
        sigma (float): The standard deviation of the Gaussian kernel. A larger sigma
                       will result in a blurrier image.
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """
        Creates a Gaussian kernel.

        Parameters:
        kernel_size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
        torch.Tensor: A 2D tensor representing the Gaussian kernel.
        """
        arange = torch.arange(kernel_size) - kernel_size // 2
        x, y = torch.meshgrid(arange, arange)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def __call__(self, tensor):
        """
        Applies Gaussian blur to the given tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which the blur will be added.

        Returns:
        torch.Tensor: The tensor with Gaussian blur added.
        """
        if tensor.dim() != 3:
            raise ValueError("Input tensor must be 3D (C x H x W)")

        C, _, _ = tensor.shape
        padded_tensor = F.pad(tensor, (self.kernel_size//2, self.kernel_size//2,
                                       self.kernel_size//2, self.kernel_size//2),
                              mode='reflect')
        blurred_tensor = F.conv2d(padded_tensor.unsqueeze(0), self.kernel.repeat(C, 1, 1, 1),
                                  padding=0, groups=C)
        return blurred_tensor.squeeze(0)

    def __repr__(self):
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})'

class AdjustContrast(object):
    def __init__(self, factor=1.0):
        """
        Initializes the contrast adjustment function.
        
        Parameters:
        factor (float): The factor by which the contrast will be adjusted. Values greater
                        than 1.0 increase contrast, values less than 1.0 decrease contrast,
                        and a value of 1.0 leaves the contrast unchanged.
        """
        self.factor = factor

    def __call__(self, tensor):
        """
        Adjusts the contrast of the given tensor.
        
        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which the contrast adjustment
                               will be applied.
                               
        Returns:
        torch.Tensor: The tensor with adjusted contrast.
        """
        if tensor.dim() != 3:
            raise ValueError("Input tensor must be 3D (C x H x W)")

        # Calculate the mean of the tensor
        mean = torch.mean(tensor, dim=(1, 2), keepdim=True)
        
        # Adjust the contrast
        tensor = (tensor - mean) * self.factor + mean
        
        # Clip the values to maintain valid pixel range
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        return tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'

class AdjustSaturation(object):
    def __init__(self, factor=1.0):
        """
        Initializes the saturation adjustment function.
        
        Parameters:
        factor (float): The factor by which the saturation will be adjusted. Values greater
                        than 1.0 increase saturation, values less than 1.0 decrease saturation,
                        and a value of 1.0 leaves the saturation unchanged.
        """
        self.factor = factor

    def __call__(self, tensor):
        """
        Adjusts the saturation of the given tensor.
        
        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which the saturation adjustment
                               will be applied. Expected to be in RGB format.
                               
        Returns:
        torch.Tensor: The tensor with adjusted saturation.
        """
        if tensor.dim() != 3 or tensor.size(0) != 3:
            raise ValueError("Input tensor must be 3D and have 3 channels (RGB)")

        # Convert image to grayscale
        grayscale = tensor.mean(dim=0, keepdim=True)
        
        # Blend the grayscale image back with the original image
        tensor = (1 - self.factor) * grayscale + self.factor * tensor
        
        # Clip the values to maintain valid pixel range
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        return tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor})'

class AddElasticTransform(object):
    def __init__(self, alpha=1, sigma=0.1, interpolation='bilinear'):
        """
        Initializes the elastic transformation function.

        Parameters:
        alpha (float): The scaling factor for the displacement vectors. Controls the intensity
                       of the deformation.
        sigma (float): The standard deviation of the Gaussian filter applied to the displacement
                       vectors. Determines the smoothness of the deformation.
        interpolation (str): The interpolation method used when resampling. Options are
                             'bilinear', 'nearest', etc.
        """
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation

    def __call__(self, tensor):
        """
        Applies elastic transformation to the given tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which the transformation
                               will be applied.

        Returns:
        torch.Tensor: The tensor with elastic transformation applied.
        """
        B, C, H, W = tensor.shape

        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (H, W)) * self.alpha
        dy = np.random.uniform(-1, 1, (H, W)) * self.alpha
        dz = np.zeros_like(dx)

        # Smooth the displacement fields
        dx = torch.tensor(gaussian_filter(dx, sigma=self.sigma, mode='constant'))
        dy = torch.tensor(gaussian_filter(dy, sigma=self.sigma, mode='constant'))
        dz = torch.tensor(dz)

        # Create flow field
        flow = np.stack([dx, dy, dz], axis=0)
        flow = torch.tensor(flow, dtype=tensor.dtype).permute(1, 2, 0)

        # Apply displacement field
        grid = torch.nn.functional.affine_grid(flow.unsqueeze(0), tensor.size(), align_corners=False)
        transformed_tensor = torch.nn.functional.grid_sample(tensor, grid, mode=self.interpolation, align_corners=False)

        return transformed_tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma})'

class Pixelate(object):
    def __init__(self, scale_factor=0.1):
        """
        Initializes the pixelation function.

        Parameters:
        scale_factor (float): Determines the degree of pixelation. It's a factor by which
                              the image's dimensions are reduced during the pixelation process.
                              A smaller scale_factor results in fewer, larger "pixels" in the
                              pixelated image. Should be between 0 and 1.
        """
        self.scale_factor = scale_factor

    def __call__(self, tensor):
        """
        Applies pixelation to the given tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which pixelation will be applied.
                               Expected to have a shape of (C, H, W), where C is the number of
                               channels, H is the height, and W is the width of the image.

        Returns:
        torch.Tensor: The tensor with pixelation applied.
        """
        if tensor.dim() != 3:
            raise ValueError("Input tensor must be 3D (C x H x W)")

        C, H, W = tensor.shape
        # Calculate the downsampled dimensions
        new_H, new_W = int(H * self.scale_factor), int(W * self.scale_factor)

        # Downscale and then upscale using nearest neighbor interpolation
        downsampled = F.interpolate(tensor.unsqueeze(0), size=(new_H, new_W), mode='nearest')
        upsampled = F.interpolate(downsampled, size=(H, W), mode='nearest')

        return upsampled.squeeze(0)

    def __repr__(self):
        return f'{self.__class__.__name__}(scale_factor={self.scale_factor})'

class AddSnowEffect(object):
    def __init__(self, snow_level=0.1, snow_amount=0.3, max_snow_diameter=3):
        """
        Initializes the snow effect function.

        Parameters:
        snow_level (float): Controls the intensity of the snow effect. Higher values result
                            in brighter snow. Value should be between 0 and 1.
        snow_amount (float): Controls the density of snowflakes. Higher values result in
                             more snow. Value should be between 0 and 1.
        max_snow_diameter (int): The maximum diameter of snowflakes in pixels.
        """
        self.snow_level = snow_level
        self.snow_amount = snow_amount
        self.max_snow_diameter = max_snow_diameter

    def __call__(self, tensor):
        """
        Applies the snow effect to the given tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which the snow effect
                               will be applied. Expected to be in C x H x W format.

        Returns:
        torch.Tensor: The tensor with snow effect applied.
        """
        if tensor.dim() != 3:
            raise ValueError("Input tensor must be 3D (C x H x W)")

        C, H, W = tensor.shape
        # Generate random snow
        snow = torch.rand((H, W)) < self.snow_amount

        # Make snowflakes larger
        if self.max_snow_diameter > 1:
            snow = snow.float().unsqueeze(0).unsqueeze(0)
            snow = F.max_pool2d(snow, kernel_size=self.max_snow_diameter, stride=1, padding=self.max_snow_diameter//2)
            snow = snow.squeeze(0).squeeze(0)

        # Apply snow effect
        snow_mask = snow * self.snow_level
        enhanced_tensor = tensor + snow_mask

        # Ensure tensor is still within valid range
        enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)

        return enhanced_tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(snow_level={self.snow_level}, snow_amount={self.snow_amount}, max_snow_diameter={self.max_snow_diameter})'

class AddFogEffect(object):
    def __init__(self, intensity=0.5, vertical_gradient=True):
        """
        Initializes the fog effect function.

        Parameters:
        intensity (float): Controls the overall intensity of the fog. Values closer to 0
                           result in lighter fog, while values closer to 1 produce denser fog.
        vertical_gradient (bool): If True, the fog will be denser at the top and lighter
                                  at the bottom. If False, the fog density will be uniform.
        """
        self.intensity = intensity
        self.vertical_gradient = vertical_gradient

    def __call__(self, tensor):
        """
        Applies the fog effect to the given tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which the fog effect
                               will be applied. Expected to be in C x H x W format.

        Returns:
        torch.Tensor: The tensor with fog effect applied.
        """
        if tensor.dim() != 3:
            raise ValueError("Input tensor must be 3D (C x H x W)")

        C, H, W = tensor.shape
        fog_tensor = torch.ones_like(tensor)  # White fog layer

        if self.vertical_gradient:
            # Create a vertical gradient
            gradient = torch.linspace(1, 0, H).unsqueeze(1).repeat(1, W)
            if C == 3:  # For RGB images, replicate the gradient for each channel
                gradient = gradient.unsqueeze(0).repeat(3, 1, 1)
            fog_intensity = self.intensity * gradient
        else:
            fog_intensity = self.intensity * torch.ones((C, H, W))

        # Blend the original image with the fog layer based on fog intensity
        enhanced_tensor = tensor + (fog_tensor - tensor) * fog_intensity

        # Ensure tensor is still within valid range
        enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)

        return enhanced_tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(intensity={self.intensity}, vertical_gradient={self.vertical_gradient})'
    
class AdjustBrightness(object):
    def __init__(self, brightness_factor=1.0):
        """
        Initializes the brightness adjustment function.
        
        Parameters:
        brightness_factor (float): The factor by which the brightness will be adjusted.
                                   Values greater than 1.0 make the image brighter,
                                   values less than 1.0 make the image darker,
                                   and a value of 1.0 leaves the brightness unchanged.
        """
        self.brightness_factor = brightness_factor

    def __call__(self, tensor):
        """
        Adjusts the brightness of the given tensor.
        
        Parameters:
        tensor (torch.Tensor): The input tensor (image) to which the brightness adjustment
                               will be applied.
                               
        Returns:
        torch.Tensor: The tensor with adjusted brightness.
        """
        if tensor.dim() != 3:
            raise ValueError("Input tensor must be 3D (C x H x W)")

        # Adjust the brightness
        adjusted_tensor = tensor * self.brightness_factor
        
        # Clip the values to maintain valid pixel range
        adjusted_tensor = torch.clamp(adjusted_tensor, 0.0, 1.0)
        
        return adjusted_tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(brightness_factor={self.brightness_factor})'


# =============================================================================
# Architecture
# =============================================================================

class RegNet(nn.Sequential):
    def __init__(self, dimensions, input_dim=1, output_dim=1, apply_var=True):
        super(RegNet, self).__init__()
        self.dimensions = [input_dim, dimensions, output_dim]        
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1]))
            if i < len(self.dimensions) - 2:
                self.add_module('relu%d' % i, torch.nn.ReLU())

        if output_dim == 2:
            self.add_module('var_split', SplitDim(correction=apply_var))

class BiggerNN(nn.Module):
    def __init__(self, n, input_dim, output_dim, hidden_dim):
        super(BiggerNN, self).__init__()
        self.minis = []
        self.n = n
       
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        for i in range(0, n):
            self.minis += [MiniNN(hidden_dim)]
            self.add_module("mini_layer"+str(i), self.minis[-1])
        self.fc = nn.Linear(hidden_dim, output_dim)
            
    def forward(self, x):
        x = self.input_layer(x)
        for i in range(0, self.n):
            x = self.minis[i](x)
        return self.fc(x)


class MiniNN(nn.Module):
    def __init__(self, D):
        super(MiniNN, self).__init__()
        self.fc1 = nn.Linear(D, D)
        self.fc2 = nn.Linear(D, D)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = self.fc2(x)
        return x

class TwoMoonsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input layer
        self.fc1 = nn.Linear(2, 64)
        self.dropout1 = nn.Dropout(p=0.2)

        # hidden layers
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.2)
        
        # output layer
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

class GenericNet(nn.Module):
    def __init__(self, input_dim):
        super(GenericNet, self).__init__()
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=1)
        )
        # Additional layers can be added here based on the architecture
        # Apply Xavier uniform initialization to the weights
        self.init_weights(self.net)

    def forward(self, x):
        return self.net(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class Model(nn.Module):
    def __init__(self, beta):
        super(Model, self).__init__()
        self.prior = GenericNet(input_dim=1)  # Specify the input dimension
        self.trainable = GenericNet(input_dim=1)  # Specify the input dimension
        self.beta = beta

    def forward(self, x):
        x1 = self.prior(x)
        x2 = self.trainable(x)
        return self.beta * x1 + x2



class PriorNet(nn.Module):
    def __init__(self, beta, model, *args):
        super(PriorNet, self).__init__()
        self.prior = model(*args)
        self.trainable = model(*args)
        self.beta = beta

    def forward(self, x):
        x1 = self.prior(x)
        x2 = self.trainable(x)
        return self.beta * x1 + x2
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


# =============================================================================
# Model
# =============================================================================

def push_dist_example(num_ensembles, mk_nn, *args):
    # Create a communicating ensemble of particles using mk_nn(*args) as a template.
    with ppush.PusH(mk_nn, *args) as push_dist:
        pids = []
        for i in range(num_ensembles):
            # 1. Each particle has a unique `pid`.
            # 2. `mk_optim` is a particle specific optimization method as in standard PyTorch.
            # 3. Create a particle on device 0.
            # 4. `receive` determines how each particle .
            # 5. `state` is the state associated with each particle.
            pids += [push_dist.p_create(mk_optim, device=0, receive={}, state={})]


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    num_ensembles = 4
    input_dim = 1
    output_dim = 1
    hidden_dim = 64
    n = 4
    push_dist_example(num_ensembles, BiggerNN, n, input_dim, output_dim, hidden_dim)
