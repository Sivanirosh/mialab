"""3D Segmentation Visualizer for MIA Experiments.

This module provides visualization capabilities for viewing 3D brain tissue segmentations
from experiment results using VTK.
"""

import os
import argparse
import glob
from typing import Optional, List, Dict
import numpy as np
import vtk
import SimpleITK as sitk


# Brain tissue label mapping
LABEL_TO_TISSUE = {
    1: 'WhiteMatter',
    2: 'GreyMatter',
    3: 'Hippocampus',
    4: 'Amygdala',
    5: 'Thalamus'
}

# Color scheme for brain tissues (RGB values 0-255)
TISSUE_COLORS = {
    'WhiteMatter': [255, 255, 255],      # White
    'GreyMatter': [200, 200, 200],       # Light grey
    'Hippocampus': [0, 255, 0],          # Green
    'Amygdala': [255, 255, 0],           # Yellow
    'Thalamus': [255, 0, 0]              # Red
}


class SegmentationVisualizer:
    """3D visualization of brain tissue segmentations."""
    
    def __init__(self, spacing: tuple = (1.0, 1.0, 1.0)):
        """Initialize the visualizer.
        
        Args:
            spacing: Voxel spacing (x, y, z) in mm. Default is (1.0, 1.0, 1.0).
        """
        self.spacing = spacing
        self.colors = vtk.vtkNamedColors()
        self.renderer = vtk.vtkOpenGLRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_interactor = vtk.vtkRenderWindowInteractor()
        
    def load_segmentation(self, file_path: str) -> np.ndarray:
        """Load segmentation from file (.mha or .nii.gz format).
        
        Args:
            file_path: Path to the segmentation file (.mha or .nii.gz).
            
        Returns:
            numpy array of the segmentation.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Segmentation file not found: {file_path}")
        
        image = sitk.ReadImage(file_path)
        spacing = image.GetSpacing()
        self.spacing = (spacing[0], spacing[1], spacing[2])
        
        array = sitk.GetArrayFromImage(image)
        return array
    
    def extract_surface(self, data_array: np.ndarray, label_value: int, 
                       color_name: str, smooth_iterations: int = 15) -> Optional[vtk.vtkActor]:
        """Extract 3D surface mesh for a specific label.
        
        Args:
            data_array: 3D numpy array of segmentation.
            label_value: Label value to extract.
            color_name: Name of the color for this tissue.
            smooth_iterations: Number of smoothing iterations.
            
        Returns:
            VTK actor for the surface, or None if no surface found.
        """
        # Convert to uint8 for VTK
        label_mask = (data_array == label_value).astype(np.uint8)
        
        # Check if label exists
        if np.sum(label_mask) == 0:
            return None
        
        # Import data to VTK
        data_importer = vtk.vtkImageImport()
        data_string = label_mask.tobytes()
        data_importer.CopyImportVoidPointer(data_string, len(data_string))
        data_importer.SetDataScalarTypeToUnsignedChar()
        data_importer.SetNumberOfScalarComponents(1)
        
        [h, w, z] = label_mask.shape
        data_importer.SetDataExtent(0, z - 1, 0, w - 1, 0, h - 1)
        data_importer.SetWholeExtent(0, z - 1, 0, w - 1, 0, h - 1)
        data_importer.SetDataSpacing(self.spacing[0], self.spacing[1], self.spacing[2])
        
        # Extract surface using marching cubes
        surface_extractor = vtk.vtkDiscreteMarchingCubes()
        surface_extractor.SetInputConnection(data_importer.GetOutputPort())
        surface_extractor.SetValue(0, 1)  # Extract label value 1 (which is our binary mask)
        surface_extractor.Update()
        
        if surface_extractor.GetOutput().GetNumberOfPoints() == 0:
            return None
        
        # Smooth the surface
        smooth_filter = vtk.vtkSmoothPolyDataFilter()
        smooth_filter.SetInputConnection(surface_extractor.GetOutputPort())
        smooth_filter.SetNumberOfIterations(smooth_iterations)
        smooth_filter.SetRelaxationFactor(0.2)
        smooth_filter.FeatureEdgeSmoothingOff()
        smooth_filter.BoundarySmoothingOn()
        smooth_filter.Update()
        
        # Create triangle strips
        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(smooth_filter.GetOutputPort())
        
        # Create mapper
        mapper = vtk.vtkOpenGLPolyDataMapper()
        mapper.SetInputConnection(stripper.GetOutputPort())
        mapper.ScalarVisibilityOff()
        
        # Create actor
        actor = vtk.vtkOpenGLActor()
        actor.SetMapper(mapper)
        
        # Set color
        color = TISSUE_COLORS.get(color_name, [128, 128, 128])
        actor.GetProperty().SetColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        actor.GetProperty().SetDiffuseColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        actor.GetProperty().SetOpacity(0.8)
        
        return actor
    
    def visualize(self, segmentation_array: np.ndarray,
                  labels_to_show: Optional[List[int]] = None,
                  title: str = "Brain Tissue Segmentation",
                  save_screenshot: Optional[str] = None,
                  start_interactive: bool = True):
        """Visualize the segmentation.
        
        Args:
            segmentation_array: 3D numpy array of segmentation.
            labels_to_show: List of label values to show. If None, show all labels.
            title: Window title.
            save_screenshot: Optional path to save a rendered screenshot (PNG).
            start_interactive: Whether to start the interactive render loop.
        """
        # Clear previous actors
        self.renderer.RemoveAllViewProps()
        
        if labels_to_show is None:
            labels_to_show = list(LABEL_TO_TISSUE.keys())
        
        # Extract surfaces for each tissue type
        actors_added = 0
        for label_value in labels_to_show:
            tissue_name = LABEL_TO_TISSUE.get(label_value, f'Label{label_value}')
            actor = self.extract_surface(segmentation_array, label_value, tissue_name)
            if actor is not None:
                self.renderer.AddActor(actor)
                actors_added += 1
                print(f"  Added {tissue_name} (label {label_value})")
        
        if actors_added == 0:
            print("Warning: No surfaces found to display!")
            return
        
        # Set up renderer
        self.renderer.SetBackground(0.1, 0.1, 0.15)  # Dark blue-gray background
        
        # Set up render window
        if self.render_window.GetRenderers().GetNumberOfItems() == 0:
            self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)
        self.render_window.SetWindowName(title)
        
        # Set up interactor
        style = vtk.vtkInteractorStyleMultiTouchCamera()
        self.render_interactor.SetInteractorStyle(style)
        self.render_interactor.SetRenderWindow(self.render_window)
        
        if save_screenshot:
            self.render_window.SetOffScreenRendering(True)
        else:
            self.render_window.SetOffScreenRendering(False)

        # Reset camera to show all actors
        self.renderer.ResetCamera()
        
        # Start rendering
        print(f"\nDisplaying {actors_added} tissue surfaces")
        print("Controls:")
        print("  - Left click + drag: Rotate")
        print("  - Right click + drag: Zoom")
        print("  - Middle click + drag: Pan")
        print("  - Close window to exit\n")
        
        self.render_interactor.Initialize()
        self.render_window.Render()

        if save_screenshot:
            self._save_screenshot(self.render_window, save_screenshot)

        if start_interactive and not save_screenshot:
            self.render_interactor.Start()
    
    def visualize_comparison(
        self,
        prediction_array: np.ndarray,
        prediction_pp_array: Optional[np.ndarray] = None,
        ground_truth_array: Optional[np.ndarray] = None,
        labels_to_show: Optional[List[int]] = None,
        title: str = "Segmentation Comparison",
        save_screenshot: Optional[str] = None,
        start_interactive: bool = True,
    ):
        """Visualize segmentation arrays side by side (prediction, post-processed, ground truth)."""

        panels: List[tuple[str, np.ndarray]] = [("Prediction", prediction_array)]

        if prediction_pp_array is not None:
            panels.append(("Post-processed", prediction_pp_array))

        if ground_truth_array is not None:
            panels.append(("Ground Truth", ground_truth_array))

        if len(panels) < 2:
            # Nothing to compare, fall back to single visualization
            self.visualize(
                prediction_array,
                labels_to_show=labels_to_show,
                title=title,
                save_screenshot=save_screenshot,
                start_interactive=start_interactive,
            )
            return

        render_window = vtk.vtkRenderWindow()
        if save_screenshot:
            render_window.SetOffScreenRendering(True)

        num_panels = len(panels)
        renderers: List[vtk.vtkRenderer] = []
        shared_camera = vtk.vtkCamera()

        if labels_to_show is None:
            labels_to_show = list(LABEL_TO_TISSUE.keys())

        for idx, (panel_title, data_array) in enumerate(panels):
            renderer = vtk.vtkOpenGLRenderer()
            renderer.SetActiveCamera(shared_camera)
            renderer.SetBackground(0.1, 0.1, 0.15)

            actors_added = 0
            for label_value in labels_to_show:
                tissue_name = LABEL_TO_TISSUE.get(label_value, f"Label{label_value}")
                actor = self.extract_surface(data_array, label_value, tissue_name)
                if actor is not None:
                    actor.GetProperty().SetOpacity(0.8)
                    renderer.AddActor(actor)
                    actors_added += 1

            # Add text overlay
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(panel_title)
            text_actor.SetPosition(10, 10)
            text_actor.GetTextProperty().SetFontSize(20)
            text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
            renderer.AddActor2D(text_actor)

            viewport_min_x = idx / num_panels
            viewport_max_x = (idx + 1) / num_panels
            renderer.SetViewport(viewport_min_x, 0.0, viewport_max_x, 1.0)

            render_window.AddRenderer(renderer)
            renderers.append(renderer)

        render_window.SetSize(350 * num_panels, 600)
        render_window.SetWindowName(title)

        render_interactor = vtk.vtkRenderWindowInteractor()
        style = vtk.vtkInteractorStyleMultiTouchCamera()
        render_interactor.SetInteractorStyle(style)
        render_interactor.SetRenderWindow(render_window)

        for renderer in renderers:
            renderer.ResetCamera()

        print(f"\nDisplaying comparison view with {num_panels} panels.")
        print("Controls:")
        print("  - Left click + drag: Rotate")
        print("  - Right click + drag: Zoom")
        print("  - Middle click + drag: Pan")
        print("  - Close window to exit\n")

        render_interactor.Initialize()
        render_window.Render()

        if save_screenshot:
            self._save_screenshot(render_window, save_screenshot)

        if start_interactive and not save_screenshot:
            render_interactor.Start()

    @staticmethod
    def _save_screenshot(render_window: vtk.vtkRenderWindow, file_path: str) -> None:
        """Save the current render window contents to a PNG file."""

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(render_window)
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(file_path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        print(f"Screenshot saved to: {file_path}")


def find_segmentation_files(experiment_dir: str, subject_id: Optional[str] = None) -> Dict[str, str]:
    """Find segmentation files in an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory.
        subject_id: Optional subject ID to filter by.
        
    Returns:
        Dictionary mapping file types to file paths.
    """
    files = {}
    
    # Look for segmentation files
    if subject_id:
        pattern_seg = os.path.join(experiment_dir, f"{subject_id}_SEG.mha")
        pattern_seg_pp = os.path.join(experiment_dir, f"{subject_id}_SEG-PP.mha")
        if os.path.exists(pattern_seg):
            files['segmentation'] = pattern_seg
        if os.path.exists(pattern_seg_pp):
            files['segmentation_pp'] = pattern_seg_pp
    else:
        # Find all segmentation files
        seg_files = glob.glob(os.path.join(experiment_dir, "*_SEG.mha"))
        seg_pp_files = glob.glob(os.path.join(experiment_dir, "*_SEG-PP.mha"))
        if seg_files:
            files['segmentation'] = seg_files[0]  # Use first one
        if seg_pp_files:
            files['segmentation_pp'] = seg_pp_files[0]
    
    return files


def main():
    """Main entry point for the visualizer."""
    parser = argparse.ArgumentParser(
        description='3D Visualization of Brain Tissue Segmentations from MIA Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize segmentation from experiment directory
  python -m mia_experiments.core.segmentation_visualizer \\
    --experiment-dir ./ablation_experiments/exp_00_baseline_none
  
  # Visualize specific subject
  python -m mia_experiments.core.segmentation_visualizer \\
    --experiment-dir ./ablation_experiments/exp_00_baseline_none \\
    --subject 118528
  
  # Visualize with postprocessing
  python -m mia_experiments.core.segmentation_visualizer \\
    --experiment-dir ./ablation_experiments/exp_00_baseline_none \\
    --subject 118528 \\
    --postprocessed
  
  # Visualize specific file
  python -m mia_experiments.core.segmentation_visualizer \\
    --file ./ablation_experiments/exp_00_baseline_none/118528_SEG.mha
  
  # Compare prediction with ground truth (requires ground truth file)
  python -m mia_experiments.core.segmentation_visualizer \\
    --file ./ablation_experiments/exp_00_baseline_none/118528_SEG.mha \\
    --ground-truth ./data/test/118528/labels_native.nii.gz
        """
    )
    
    parser.add_argument(
        '--experiment-dir',
        type=str,
        help='Path to experiment directory containing segmentation files'
    )
    parser.add_argument(
        '--subject',
        type=str,
        help='Subject ID to visualize (optional)'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Direct path to segmentation .mha file'
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        help='Path to ground truth segmentation file for comparison'
    )
    parser.add_argument(
        '--postprocessed',
        action='store_true',
        help='Use postprocessed segmentation (SEG-PP.mha) instead of SEG.mha'
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        help='Labels to visualize (1=WhiteMatter, 2=GreyMatter, 3=Hippocampus, 4=Amygdala, 5=Thalamus)',
        choices=['1', '2', '3', '4', '5', 'all'],
        default=['all']
    )
    parser.add_argument(
        '--spacing',
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help='Voxel spacing in mm (default: 1.0 1.0 1.0)'
    )
    
    args = parser.parse_args()
    
    # Determine which file to load
    seg_file = None
    
    if args.file:
        seg_file = args.file
    elif args.experiment_dir:
        files = find_segmentation_files(args.experiment_dir, args.subject)
        if args.postprocessed and 'segmentation_pp' in files:
            seg_file = files['segmentation_pp']
        elif 'segmentation' in files:
            seg_file = files['segmentation']
        else:
            print(f"Error: No segmentation files found in {args.experiment_dir}")
            if args.subject:
                print(f"  Looking for subject: {args.subject}")
            return 1
    else:
        parser.print_help()
        return 1
    
    if not seg_file or not os.path.exists(seg_file):
        print(f"Error: Segmentation file not found: {seg_file}")
        return 1
    
    # Parse labels
    if 'all' in args.labels:
        labels_to_show = None
    else:
        labels_to_show = [int(l) for l in args.labels]
    
    # Create visualizer
    visualizer = SegmentationVisualizer(spacing=tuple(args.spacing))
    
    # Load segmentation
    print(f"Loading segmentation from: {seg_file}")
    seg_array = visualizer.load_segmentation(seg_file)
    print(f"Segmentation shape: {seg_array.shape}")
    print(f"Unique labels: {sorted(np.unique(seg_array))}")
    
    # Load ground truth if provided
    if args.ground_truth:
        if not os.path.exists(args.ground_truth):
            print(f"Warning: Ground truth file not found: {args.ground_truth}")
            print("Displaying prediction only...")
            visualizer.visualize(seg_array, labels_to_show, 
                               title=os.path.basename(seg_file))
        else:
            print(f"Loading ground truth from: {args.ground_truth}")
            gt_array = visualizer.load_segmentation(args.ground_truth)
            print(f"Ground truth shape: {gt_array.shape}")
            visualizer.visualize_comparison(seg_array, gt_array, labels_to_show,
                                          title=f"Comparison: {os.path.basename(seg_file)}")
    else:
        # Visualize single segmentation
        title = os.path.basename(seg_file)
        if args.subject:
            title = f"{args.subject} - {title}"
        visualizer.visualize(seg_array, labels_to_show, title=title)
    
    return 0


if __name__ == '__main__':
    exit(main())

