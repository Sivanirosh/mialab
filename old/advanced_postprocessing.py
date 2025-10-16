"""Advanced post-processing methods for testing spatial regularization hypotheses.

This module contains different post-processing approaches to test the hypothesis that
spatial regularization improves segmentation performance.
"""

import numpy as np
import SimpleITK as sitk
import pymia.filtering.filter as pymia_fltr
from scipy import ndimage
from sklearn.mixture import GaussianMixture


class ConnectedComponentPostProcessing(pymia_fltr.Filter):
    """Post-processing using connected component analysis with different strategies."""
    
    def __init__(self, min_component_size: int = 10, strategy: str = 'remove_small'):
        """Initialize the connected component post-processing.
        
        Args:
            min_component_size: Minimum size of components to keep
            strategy: 'remove_small', 'keep_largest', or 'size_threshold'
        """
        super().__init__()
        self.min_component_size = min_component_size
        self.strategy = strategy
    
    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Execute connected component post-processing."""
        img_array = sitk.GetArrayFromImage(image)
        processed_array = img_array.copy()
        
        unique_labels = np.unique(img_array)
        unique_labels = unique_labels[unique_labels > 0]  # exclude background
        
        for label in unique_labels:
            label_mask = (img_array == label).astype(np.uint8)
            label_img = sitk.GetImageFromArray(label_mask)
            label_img.CopyInformation(image)
            
            # Connected component analysis
            cc_filter = sitk.ConnectedComponentImageFilter()
            cc_img = cc_filter.Execute(label_img)
            
            label_stats = sitk.LabelShapeStatisticsImageFilter()
            label_stats.Execute(cc_img)
            
            cc_array = sitk.GetArrayFromImage(cc_img)
            filtered_mask = np.zeros_like(cc_array)
            
            if self.strategy == 'remove_small':
                # Remove components smaller than threshold
                for cc_label in label_stats.GetLabels():
                    if label_stats.GetNumberOfPixels(cc_label) >= self.min_component_size:
                        filtered_mask[cc_array == cc_label] = 1
                        
            elif self.strategy == 'keep_largest':
                # Keep only the largest component
                if label_stats.GetLabels():
                    largest_label = max(label_stats.GetLabels(), 
                                      key=lambda x: label_stats.GetNumberOfPixels(x))
                    filtered_mask[cc_array == largest_label] = 1
                    
            elif self.strategy == 'size_threshold':
                # Remove components smaller than percentage of largest
                if label_stats.GetLabels():
                    largest_size = max(label_stats.GetNumberOfPixels(cc_label) 
                                     for cc_label in label_stats.GetLabels())
                    threshold = largest_size * 0.1  # 10% of largest component
                    
                    for cc_label in label_stats.GetLabels():
                        if label_stats.GetNumberOfPixels(cc_label) >= threshold:
                            filtered_mask[cc_array == cc_label] = 1
            
            # Update processed array
            processed_array[img_array == label] = 0  # Clear original label
            processed_array[filtered_mask == 1] = label  # Set filtered regions
        
        result_img = sitk.GetImageFromArray(processed_array.astype(np.uint8))
        result_img.CopyInformation(image)
        return result_img


class MorphologicalPostProcessing(pymia_fltr.Filter):
    """Post-processing using morphological operations."""
    
    def __init__(self, operation: str = 'closing', kernel_size: int = 1):
        """Initialize morphological post-processing.
        
        Args:
            operation: 'closing', 'opening', 'both'
            kernel_size: Size of morphological kernel
        """
        super().__init__()
        self.operation = operation
        self.kernel_size = kernel_size
    
    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Execute morphological post-processing."""
        img_array = sitk.GetArrayFromImage(image)
        processed_array = img_array.copy()
        
        unique_labels = np.unique(img_array)
        unique_labels = unique_labels[unique_labels > 0]
        
        for label in unique_labels:
            label_mask = (img_array == label).astype(np.uint8)
            
            if self.operation in ['opening', 'both']:
                # Opening (erosion followed by dilation) - removes small noise
                label_mask = ndimage.binary_opening(
                    label_mask, 
                    structure=ndimage.generate_binary_structure(3, 1),
                    iterations=self.kernel_size
                ).astype(np.uint8)
            
            if self.operation in ['closing', 'both']:
                # Closing (dilation followed by erosion) - fills small holes
                label_mask = ndimage.binary_closing(
                    label_mask,
                    structure=ndimage.generate_binary_structure(3, 1),
                    iterations=self.kernel_size
                ).astype(np.uint8)
            
            # Update processed array
            processed_array[img_array == label] = 0
            processed_array[label_mask == 1] = label
        
        result_img = sitk.GetImageFromArray(processed_array.astype(np.uint8))
        result_img.CopyInformation(image)
        return result_img


class GaussianSmoothingPostProcessing(pymia_fltr.Filter):
    """Post-processing using Gaussian smoothing of probability maps."""
    
    def __init__(self, sigma: float = 1.0):
        """Initialize Gaussian smoothing post-processing.
        
        Args:
            sigma: Standard deviation for Gaussian kernel
        """
        super().__init__()
        self.sigma = sigma
    
    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Execute Gaussian smoothing post-processing.
        
        Note: This method assumes we have access to probability maps.
        For now, we'll apply smoothing to the label image directly.
        """
        # Convert to float for smoothing
        img_float = sitk.Cast(image, sitk.sitkFloat32)
        
        # Apply Gaussian smoothing
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(self.sigma)
        smoothed_img = gaussian_filter.Execute(img_float)
        
        # Convert back to labels (simple argmax-like operation)
        # This is a simplified approach - in practice, you'd want to use probability maps
        result_img = sitk.Cast(smoothed_img + 0.5, sitk.sitkUInt8)  # Round to nearest integer
        
        return result_img


class CombinedPostProcessing(pymia_fltr.Filter):
    """Combined post-processing using multiple techniques."""
    
    def __init__(self, use_cc: bool = True, use_morphology: bool = True, 
                 min_component_size: int = 10, morphology_kernel: int = 1):
        """Initialize combined post-processing.
        
        Args:
            use_cc: Whether to use connected component analysis
            use_morphology: Whether to use morphological operations
            min_component_size: Minimum component size for CC analysis
            morphology_kernel: Kernel size for morphological operations
        """
        super().__init__()
        self.use_cc = use_cc
        self.use_morphology = use_morphology
        self.min_component_size = min_component_size
        self.morphology_kernel = morphology_kernel
    
    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Execute combined post-processing."""
        result = image
        
        if self.use_morphology:
            morph_filter = MorphologicalPostProcessing('closing', self.morphology_kernel)
            result = morph_filter.execute(result)
        
        if self.use_cc:
            cc_filter = ConnectedComponentPostProcessing(self.min_component_size, 'remove_small')
            result = cc_filter.execute(result)
        
        return result


def analyze_connected_components(segmentation: sitk.Image) -> dict:
    """Analyze connected components in a segmentation.
    
    Args:
        segmentation: Binary or multi-label segmentation image
        
    Returns:
        Dictionary with component analysis results
    """
    img_array = sitk.GetArrayFromImage(segmentation)
    unique_labels = np.unique(img_array)
    unique_labels = unique_labels[unique_labels > 0]
    
    analysis_results = {}
    
    for label in unique_labels:
        label_mask = (img_array == label).astype(np.uint8)
        label_img = sitk.GetImageFromArray(label_mask)
        label_img.CopyInformation(segmentation)
        
        # Connected component analysis
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_img = cc_filter.Execute(label_img)
        
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(cc_img)
        
        component_sizes = []
        for cc_label in label_stats.GetLabels():
            component_sizes.append(label_stats.GetNumberOfPixels(cc_label))
        
        analysis_results[f'label_{label}'] = {
            'num_components': len(component_sizes),
            'component_sizes': component_sizes,
            'largest_component': max(component_sizes) if component_sizes else 0,
            'smallest_component': min(component_sizes) if component_sizes else 0,
            'mean_component_size': np.mean(component_sizes) if component_sizes else 0,
            'total_voxels': sum(component_sizes)
        }
    
    return analysis_results


def post_processing_comparison_study(segmentation: sitk.Image, ground_truth: sitk.Image = None):
    """Compare different post-processing approaches.
    
    Args:
        segmentation: Original segmentation
        ground_truth: Ground truth segmentation (optional)
        
    Returns:
        Dictionary with results from different post-processing methods
    """
    methods = {
        'original': segmentation,
        'cc_remove_small': ConnectedComponentPostProcessing(10, 'remove_small').execute(segmentation),
        'cc_keep_largest': ConnectedComponentPostProcessing(10, 'keep_largest').execute(segmentation),
        'morph_closing': MorphologicalPostProcessing('closing', 1).execute(segmentation),
        'morph_opening': MorphologicalPostProcessing('opening', 1).execute(segmentation),
        'morph_both': MorphologicalPostProcessing('both', 1).execute(segmentation),
        'combined': CombinedPostProcessing().execute(segmentation)
    }
    
    results = {}
    
    for method_name, processed_seg in methods.items():
        # Analyze connected components
        cc_analysis = analyze_connected_components(processed_seg)
        
        results[method_name] = {
            'segmentation': processed_seg,
            'connected_components': cc_analysis
        }
        
        # If ground truth is available, compute metrics
        if ground_truth is not None:
            try:
                from pymia.evaluation import metric
                dice_metric = metric.DiceCoefficient()
                hausdorff_metric = metric.HausdorffDistance(percentile=95)
                
                # Compute metrics for each label
                img_array = sitk.GetArrayFromImage(processed_seg)
                gt_array = sitk.GetArrayFromImage(ground_truth)
                unique_labels = np.unique(gt_array)
                unique_labels = unique_labels[unique_labels > 0]
                
                metrics = {}
                for label in unique_labels:
                    pred_mask = (img_array == label).astype(np.uint8)
                    gt_mask = (gt_array == label).astype(np.uint8)
                    
                    pred_img = sitk.GetImageFromArray(pred_mask)
                    pred_img.CopyInformation(ground_truth)
                    gt_img = sitk.GetImageFromArray(gt_mask)
                    gt_img.CopyInformation(ground_truth)
                    
                    dice_score = dice_metric.calculate(pred_img, gt_img)
                    hausdorff_dist = hausdorff_metric.calculate(pred_img, gt_img)
                    
                    metrics[f'label_{label}'] = {
                        'dice': dice_score,
                        'hausdorff': hausdorff_dist
                    }
                
                results[method_name]['metrics'] = metrics
                
            except ImportError:
                print("Metrics calculation requires pymia.evaluation module")
    
    return results


def create_post_processing_experiment_configs():
    """Create experiment configurations for testing different post-processing approaches."""
    from experiment_framework import ExperimentConfig
    
    configs = []
    
    # Baseline - no post-processing
    baseline = ExperimentConfig("baseline_no_postproc", 
                               "Baseline without any post-processing")
    baseline.postprocessing_params = {'simple_post': False}
    configs.append(baseline)
    
    # Simple post-processing (current implementation)
    simple = ExperimentConfig("simple_postproc", 
                             "Simple post-processing with CC analysis and morphological closing")
    simple.postprocessing_params = {'simple_post': True}
    configs.append(simple)
    
    # Connected components only
    cc_only = ExperimentConfig("cc_only_postproc", 
                              "Connected component analysis only")
    cc_only.postprocessing_params = {
        'simple_post': False,
        'cc_post': True,
        'cc_strategy': 'remove_small',
        'cc_min_size': 10
    }
    configs.append(cc_only)
    
    # Morphological operations only  
    morph_only = ExperimentConfig("morph_only_postproc", 
                                 "Morphological operations only")
    morph_only.postprocessing_params = {
        'simple_post': False,
        'morph_post': True,
        'morph_operation': 'closing',
        'morph_kernel': 1
    }
    configs.append(morph_only)
    
    # Aggressive connected components
    cc_aggressive = ExperimentConfig("cc_aggressive_postproc", 
                                   "Aggressive connected component analysis")
    cc_aggressive.postprocessing_params = {
        'simple_post': False,
        'cc_post': True,
        'cc_strategy': 'keep_largest',
        'cc_min_size': 50
    }
    configs.append(cc_aggressive)
    
    return configs