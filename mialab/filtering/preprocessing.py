"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        # Z-score normalization
        mask = img_arr > 0 # consider only non-zero voxels (avoid background interference)
        if np.any(mask):
            mean_val = np.mean(img_arr[mask])
            std_val = np.std(img_arr[mask])
            if std_val > 0:
                img_arr[mask] = (img_arr[mask] - mean_val) / std_val
            else:
                img_arr[mask] = img_arr[mask] - mean_val # avoid division by zero

        img_out = sitk.GetImageFromArray(img_arr.astype(np.float32))
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)


class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image.
        """
        mask = params.img_mask  # the brain mask
        mask_uint8 = sitk.Cast(mask, sitk.sitkUInt8)

        # Apply the brain mask to remove skull
        # Multiply the image by the mask (where mask=1 is brain, mask=0 is skull/background)
        skull_stripped = sitk.Mask(image, mask_uint8, outsideValue=0, maskingValue=0)

        return skull_stripped

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """

        if params is None:
            raise ValueError("ImageRegistrationParameters are required")

        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different

        # Set up the resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(atlas)  # Use atlas as reference for output spacing, size, etc.
        resampler.SetTransform(transform)
        
        # For ground truth (label images), use nearest neighbor interpolation to preserve labels
        # For intensity images, use linear interpolation
        if is_ground_truth:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)  # Background label is 0
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0.0)

        registered_image = resampler.Execute(image)
        
        return registered_image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)


class BiasFieldCorrectionParameters(pymia_fltr.FilterParams):
    """Bias field correction parameters."""

    def __init__(self, convergence_threshold: float = 0.01,
                 max_iterations: int = 20,
                 number_fitting_levels: int = 3):
        """Initializes a new instance of the BiasFieldCorrectionParameters.

        Args:
            convergence_threshold (float): The convergence threshold for N4 correction. Default is 0.01 (more lenient for speed).
            max_iterations (int): The maximum number of iterations per fitting level. Default is 20 (reduced for speed).
            number_fitting_levels (int): The number of fitting levels. Default is 3 (reduced for speed).
        """
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.number_fitting_levels = number_fitting_levels


class BiasFieldCorrection(pymia_fltr.Filter):
    """Represents a bias field correction filter."""

    def __init__(self):
        """Initializes a new instance of the BiasFieldCorrection class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: BiasFieldCorrectionParameters = None) -> sitk.Image:
        """Executes bias field correction using N4ITK.

        Args:
            image (sitk.Image): The image.
            params (BiasFieldCorrectionParameters): The parameters for bias field correction.

        Returns:
            sitk.Image: The bias field corrected image.
        """
        if params is None:
            params = BiasFieldCorrectionParameters()

        # Cast to float
        image_float = sitk.Cast(image, sitk.sitkFloat32)

        # Create mask
        mask_filter = sitk.OtsuThresholdImageFilter()
        mask = mask_filter.Execute(image_float)

        # Apply N4 correction
        # Note: The number of fitting levels is determined by the length of the iterations list
        # N4 bias correction can be slow - using optimized parameters for reasonable speed
        print(f"  Applying N4 bias field correction (max_iter={params.max_iterations}, levels={params.number_fitting_levels})...")
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetConvergenceThreshold(params.convergence_threshold)
        # Set iterations for each fitting level - the list length determines number of levels
        corrector.SetMaximumNumberOfIterations([params.max_iterations] * params.number_fitting_levels)
        # Note: SetShrinkFactor is not available in all SimpleITK versions, using default shrink factor

        corrected = corrector.Execute(image_float, mask)
        print(f"  N4 bias field correction completed.")

        return corrected

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'BiasFieldCorrection:\n' \
            .format(self=self)
