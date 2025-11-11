"""Tissue-aware postprocessing heuristics for predicted segmentations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


LabelId = int
Radius = Optional[Tuple[int, int, int]]


@dataclass(frozen=True)
class LabelPostprocessingRecipe:
    """Morphological and connected-component parameters for a given label."""

    opening_radius: Radius = (1, 1, 1)
    closing_radius: Radius = None
    keep_components: int = 2
    min_volume_voxels: int = 0
    allow_fewer_components: bool = True

    def apply(self, mask: sitk.Image) -> sitk.Image:
        """Run the configured clean-and-select pipeline on a binary mask."""

        processed = sitk.Cast(mask, sitk.sitkUInt8)

        if self.opening_radius:
            processed = sitk.BinaryMorphologicalOpening(processed, self.opening_radius)

        if self.closing_radius:
            processed = sitk.BinaryMorphologicalClosing(processed, self.closing_radius)

        connected = sitk.ConnectedComponent(processed)
        relabeled = self._relabel_components(connected, self.min_volume_voxels)

        # Fallback if everything was filtered out by the volume threshold
        if self.allow_fewer_components and self.min_volume_voxels > 0 and self._num_components(relabeled) == 0:
            relabeled = self._relabel_components(connected, minimum_size=0)

        num_components = self._num_components(relabeled)
        if num_components == 0:
            empty = sitk.Image(mask.GetSize(), mask.GetPixelIDValue())
            empty.CopyInformation(mask)
            return empty

        num_to_keep = min(self.keep_components, num_components) if self.keep_components > 0 else num_components
        selected = sitk.BinaryThreshold(relabeled, 1, num_to_keep, 1, 0)
        return sitk.Cast(selected, mask.GetPixelID())

    @staticmethod
    def _relabel_components(image: sitk.Image, minimum_size: int) -> sitk.Image:
        return sitk.RelabelComponent(image, sortByObjectSize=True, minimumObjectSize=minimum_size)

    @staticmethod
    def _num_components(image: sitk.Image) -> int:
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        return int(stats.GetMaximum())


class PostProcessingParams(pymia_fltr.FilterParams):
    """Filter parameters describing how to postprocess each label."""

    def __init__(
        self,
        recipes: Optional[Dict[LabelId, LabelPostprocessingRecipe]] = None,
        copy_unhandled_labels: bool = True,
    ):
        self.recipes = recipes if recipes is not None else default_recipes()
        self.copy_unhandled_labels = copy_unhandled_labels


class ImagePostProcessing(pymia_fltr.Filter):
    """Applies label-specific morphological cleanup to a segmentation."""

    def __init__(self, default_recipes: Optional[Dict[LabelId, LabelPostprocessingRecipe]] = None):
        super().__init__()
        self._default_params = PostProcessingParams(default_recipes)

    def execute(self, image: sitk.Image, params: Optional[PostProcessingParams] = None) -> sitk.Image:
        """Postprocess a multi-label segmentation."""

        parameter_set = params if isinstance(params, PostProcessingParams) else self._default_params
        recipes = parameter_set.recipes

        output = sitk.Image(image.GetSize(), image.GetPixelIDValue())
        output.CopyInformation(image)

        handled_labels = set()
        for label_id, recipe in recipes.items():
            label_mask = self._extract_label_mask(image, label_id)
            if self._mask_is_empty(label_mask):
                continue

            cleaned = recipe.apply(label_mask)
            if self._mask_is_empty(cleaned):
                continue

            output = sitk.Add(output, sitk.Multiply(cleaned, label_id))
            handled_labels.add(label_id)

        if parameter_set.copy_unhandled_labels:
            for label_id in self._discover_labels(image):
                if label_id == 0 or label_id in handled_labels:
                    continue
                passthrough_mask = self._extract_label_mask(image, label_id)
                if self._mask_is_empty(passthrough_mask):
                    continue
                output = sitk.Add(output, sitk.Multiply(passthrough_mask, label_id))

        return output

    @staticmethod
    def _extract_label_mask(image: sitk.Image, label_id: LabelId) -> sitk.Image:
        return sitk.BinaryThreshold(image, label_id, label_id, 1, 0)

    @staticmethod
    def _mask_is_empty(mask: sitk.Image) -> bool:
        stats = sitk.StatisticsImageFilter()
        stats.Execute(mask)
        return stats.GetSum() == 0

    @staticmethod
    def _discover_labels(image: sitk.Image) -> Iterable[LabelId]:
        array_view = sitk.GetArrayViewFromImage(image)
        return np.unique(array_view)

    def __str__(self) -> str:
        return "ImagePostProcessing"


def default_recipes() -> Dict[LabelId, LabelPostprocessingRecipe]:
    """Return default heuristics tuned for the five tissue classes."""

    return {
        1: LabelPostprocessingRecipe(
            opening_radius=(1, 1, 1),
            closing_radius=(1, 1, 1),
            keep_components=2,
            min_volume_voxels=5000,
        ),
        2: LabelPostprocessingRecipe(
            opening_radius=(1, 1, 1),
            closing_radius=(1, 1, 1),
            keep_components=2,
            min_volume_voxels=5000,
        ),
        3: LabelPostprocessingRecipe(
            opening_radius=(1, 1, 1),
            closing_radius=None,
            keep_components=2,
            min_volume_voxels=200,
        ),
        4: LabelPostprocessingRecipe(
            opening_radius=(1, 1, 1),
            closing_radius=None,
            keep_components=2,
            min_volume_voxels=150,
        ),
        5: LabelPostprocessingRecipe(
            opening_radius=(1, 1, 1),
            closing_radius=None,
            keep_components=2,
            min_volume_voxels=400,
        ),
    }


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