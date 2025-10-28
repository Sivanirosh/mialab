"""The post-processing module contains classes for image filtering mostly applied after a classification.

Image post-processing aims to alter images such that they depict a desired representation.
"""
import warnings
from typing import Dict, Optional, Tuple

# import numpy as np
# import pydensecrf.densecrf as crf
# import pydensecrf.utils as crf_util
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk
import numpy as np


class PostProcessingParams(pymia_fltr.FilterParams):
    """Parameters for post-processing filter."""
    
    def __init__(self, min_component_size: int = 5,
                 per_tissue_sizes: Optional[Dict[str, int]] = None,
                 kernel_radius: Tuple[int, int, int] = (1, 1, 1),
                 retention_threshold: float = 0.5):
        """Initializes a new instance of the PostProcessingParams class.
        
        Args:
            min_component_size (int): Global minimum component size in voxels (default: 5).
            per_tissue_sizes : Dictionary mapping tissue names to minimum sizes. If provided, overrides global min_component_size for specific tissues.
            kernel_radius (Tuple[int, int, int]): Kernel radius for morphological operations (default: (1, 1, 1)).
            retention_threshold (float): Skip postprocessing if less than this fraction of voxels would be retained (default: 0.5).
        """
        self.min_component_size = min_component_size
        self.per_tissue_sizes = per_tissue_sizes or {}
        self.kernel_radius = kernel_radius
        self.retention_threshold = retention_threshold


class ImagePostProcessing(pymia_fltr.Filter):
    """Represents a post-processing filter."""

    # Map label indices to tissue names for per-tissue size configuration
    LABEL_TO_TISSUE = {
        1: 'WhiteMatter',
        2: 'GreyMatter',
        3: 'Hippocampus',
        4: 'Amygdala',
        5: 'Thalamus'
    }

    def __init__(self):
        """Initializes a new instance of the ImagePostProcessing class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: PostProcessingParams = None) -> sitk.Image:
        """Post-processes a segmentation image.

        Args:
            image (sitk.Image): The image.
            params (PostProcessingParams): The parameters.

        Returns:
            sitk.Image: The post-processed image.
        """
        # Use default parameters if none provided
        if params is None:
            params = PostProcessingParams()

        # Convert to numpy for processing
        img_array = sitk.GetArrayFromImage(image)
        processed_array = img_array.copy()
        
        # Get unique labels (excluding background)
        unique_labels = np.unique(img_array)
        unique_labels = unique_labels[unique_labels > 0]  # exclude background (0)
        
        # Apply connected component analysis for each label
        for label in unique_labels:
            # Determine minimum size for this specific label
            tissue_name = self.LABEL_TO_TISSUE.get(int(label), None)
            if tissue_name and tissue_name in params.per_tissue_sizes:
                min_size = params.per_tissue_sizes[tissue_name]
            else:
                min_size = params.min_component_size
            
            # Create binary mask for this label
            label_mask = (img_array == label).astype(np.uint8)
            initial_voxel_count = np.sum(label_mask)
            
            # Convert to SimpleITK image for connected component analysis
            label_img = sitk.GetImageFromArray(label_mask)
            label_img.CopyInformation(image)
            
            # Ensure the label image is uint8 type to avoid deprecation warnings
            label_img = sitk.Cast(label_img, sitk.sitkUInt8)
            
            # Apply connected component filter
            cc_filter = sitk.ConnectedComponentImageFilter()
            cc_img = cc_filter.Execute(label_img)
            
            # Get label statistics to find component sizes
            label_stats = sitk.LabelShapeStatisticsImageFilter()
            label_stats.Execute(cc_img)
            
            # Keep only components larger than minimum size
            cc_array = sitk.GetArrayFromImage(cc_img)
            filtered_mask = np.zeros_like(cc_array)
            
            for cc_label in label_stats.GetLabels():
                if label_stats.GetNumberOfPixels(cc_label) >= min_size:
                    filtered_mask[cc_array == cc_label] = 1
            
            # Check retention threshold
            retained_voxels = np.sum(filtered_mask)
            retention_ratio = retained_voxels / initial_voxel_count if initial_voxel_count > 0 else 0
            
            # Skip morphological operations if too many voxels were removed
            if retention_ratio < params.retention_threshold:
                warnings.warn(
                    f"Label {label} ({tissue_name or 'unknown'}): Retention ratio {retention_ratio:.2%} "
                    f"is below threshold {params.retention_threshold:.2%}. Skipping morphological closing."
                )
                # Keep the filtered result without morphological closing
                processed_array[filtered_mask == 1] = label
                processed_array[(img_array == label) & (filtered_mask == 0)] = 0
                continue
            
            # Apply morphological closing to fill small holes
            closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
            closing_filter.SetKernelRadius(params.kernel_radius)
            closing_filter.SetForegroundValue(1)
            
            filtered_img = sitk.GetImageFromArray(filtered_mask.astype(np.uint8))
            filtered_img.CopyInformation(image)
            closed_img = closing_filter.Execute(filtered_img)
            closed_array = sitk.GetArrayFromImage(closed_img)
            
            # Update the processed array
            processed_array[closed_array == 1] = label
            processed_array[(img_array == label) & (closed_array == 0)] = 0  # Set removed voxels to background
        
        # Convert back to SimpleITK image
        result_img = sitk.GetImageFromArray(processed_array.astype(np.uint8))
        result_img.CopyInformation(image)
        
        return result_img

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImagePostProcessing:\n' \
            .format(self=self)


# class DenseCRFParams(pymia_fltr.FilterParams):
#     """Dense CRF parameters."""
#     def __init__(self, img_t1: sitk.Image, img_t2: sitk.Image, img_proba: sitk.Image):
#         """Initializes a new instance of the DenseCRFParams
#
#         Args:
#             img_t1 (sitk.Image): The T1-weighted image.
#             img_t2 (sitk.Image): The T2-weighted image.
#             img_probability (sitk.Image): The posterior probability image.
#         """
#         self.img_t1 = img_t1
#         self.img_t2 = img_t2
#         self.img_probability = img_probability
#
#
# class DenseCRF(pymia_fltr.Filter):
#     """A dense conditional random field (dCRF).
#
#     Implements the work of Krähenbühl and Koltun, Efficient Inference in Fully Connected CRFs
#     with Gaussian Edge Potentials, 2012. The dCRF code is taken from https://github.com/lucasb-eyer/pydensecrf.
#     """
#
#     def __init__(self):
#         """Initializes a new instance of the DenseCRF class."""
#         super().__init__()
#
#     def execute(self, image: sitk.Image, params: DenseCRFParams = None) -> sitk.Image:
#         """Executes the dCRF regularization.
#
#         Args:
#             image (sitk.Image): The image (unused).
#             params (FilterParams): The parameters.
#
#         Returns:
#             sitk.Image: The filtered image.
#         """
#
#         if params is None:
#             raise ValueError('Parameters are required')
#
#         img_t2 = sitk.GetArrayFromImage(params.img_t1)
#         img_ir = sitk.GetArrayFromImage(params.img_t2)
#         img_probability = sitk.GetArrayFromImage(params.img_probability)
#
#         # some variables
#         x = img_probability.shape[2]
#         y = img_probability.shape[1]
#         z = img_probability.shape[0]
#         no_labels = img_probability.shape[3]
#
#         img_probability = np.rollaxis(img_probability, 3, 0)
#
#         d = crf.DenseCRF(x * y * z, no_labels)  # width, height, nlabels
#         U = crf_util.unary_from_softmax(img_probability)
#         d.setUnaryEnergy(U)
#
#         stack = np.stack([img_t2, img_ir], axis=3)
#
#         # Create the pairwise bilateral term from the above images.
#         # The two `s{dims,chan}` parameters are model hyper-parameters defining
#         # the strength of the location and image content bi-laterals, respectively.
#
#         # higher weight equals stronger
#         pairwise_energy = crf_util.create_pairwise_bilateral(sdims=(1, 1, 1), schan=(1, 1), img=stack, chdim=3)
#
#         # `compat` (Compatibility) is the "strength" of this potential.
#         compat = 10
#         # compat = np.array([1, 1], np.float32)
#         # weight --> lower equals stronger
#         # compat = np.array([[0, 10], [10, 1]], np.float32)
#
#         d.addPairwiseEnergy(pairwise_energy, compat=compat,
#                             kernel=crf.DIAG_KERNEL,
#                             normalization=crf.NORMALIZE_SYMMETRIC)
#
#         # add location only
#         # pairwise_gaussian = crf_util.create_pairwise_gaussian(sdims=(.5,.5,.5), shape=(x, y, z))
#         #
#         # d.addPairwiseEnergy(pairwise_gaussian, compat=.3,
#         #                     kernel=dcrf.DIAG_KERNEL,
#         #                     normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#         # compatibility, kernel and normalization
#         Q_unary = d.inference(10)
#         # Q_unary, tmp1, tmp2 = d.startInference()
#         #
#         # for _ in range(10):
#         #     d.stepInference(Q_unary, tmp1, tmp2)
#         #     print(d.klDivergence(Q_unary) / (z* y*x))
#         # kl2 = d.klDivergence(Q_unary) / (z* y*x)
#
#         # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
#         map_soln_unary = np.argmax(Q_unary, axis=0)
#         map_soln_unary = map_soln_unary.reshape((z, y, x))
#         map_soln_unary = map_soln_unary.astype(np.uint8)  # convert to uint8 from int64
#         # Saving int64 with SimpleITK corrupts the file for Windows, i.e. opening it raises an ITK error:
#         # Unknown component type error: 0
#
#         img_out = sitk.GetImageFromArray(map_soln_unary)
#         img_out.CopyInformation(params.img_t1)
#         return img_out