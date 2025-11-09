"""
3D Segmentation Visualizer - Updated for MIA Experiments.

This script provides a simple interface to visualize brain tissue segmentations.
It can load .mha files from experiment results or .npy files for backward compatibility.
"""

import argparse
import os
import sys
import numpy as np
import SimpleITK as sitk

# Try to import the new visualizer
try:
    from mia_experiments.core.segmentation_visualizer import SegmentationVisualizer
    USE_NEW_VISUALIZER = True
except ImportError:
    USE_NEW_VISUALIZER = False
    print("Warning: New visualizer not available, using legacy mode")
    import vtk


def display_surface_models_legacy(data_matrix, spacing=(1.0, 1.0, 1.0)):
    """Legacy display function for backward compatibility."""
    data_matrix = data_matrix.astype(np.uint8)

    dataImporter = vtk.vtkImageImport()
    data_string = data_matrix.tobytes()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    [h, w, z] = data_matrix.shape
    dataImporter.SetDataExtent(0, z - 1, 0, w - 1, 0, h - 1)
    dataImporter.SetWholeExtent(0, z - 1, 0, w - 1, 0, h - 1)
    dataImporter.SetDataSpacing(spacing[0], spacing[1], spacing[2])

    colors = vtk.vtkNamedColors()
    colors.SetColor("WhiteMatter", [255, 255, 255, 255])
    colors.SetColor("GreyMatter", [200, 200, 200, 255])
    colors.SetColor("Hippocampus", [0, 255, 0, 255])
    colors.SetColor("Amygdala", [255, 255, 0, 255])
    colors.SetColor("Thalamus", [255, 0, 0, 255])

    def extract(color, isovalue):
        skinExtractor = vtk.vtkDiscreteMarchingCubes()
        skinExtractor.SetInputConnection(dataImporter.GetOutputPort())
        skinExtractor.SetValue(0, isovalue)
        skinExtractor.Update()
        if skinExtractor.GetOutput().GetNumberOfPoints() == 0:
            return None

        smooth = vtk.vtkSmoothPolyDataFilter()
        smooth.SetInputConnection(skinExtractor.GetOutputPort())
        smooth.SetNumberOfIterations(15)
        smooth.SetRelaxationFactor(0.2)
        smooth.FeatureEdgeSmoothingOff()
        smooth.BoundarySmoothingOn()
        smooth.Update()

        skinStripper = vtk.vtkStripper()
        skinStripper.SetInputConnection(smooth.GetOutputPort())

        skinMapper = vtk.vtkOpenGLPolyDataMapper()
        skinMapper.SetInputConnection(skinStripper.GetOutputPort())
        skinMapper.ScalarVisibilityOff()

        skin = vtk.vtkOpenGLActor()
        skin.SetMapper(skinMapper)
        skin.GetProperty().SetDiffuseColor(colors.GetColor3d(color))
        skin.GetProperty().SetSpecular(.3)
        skin.GetProperty().SetSpecularPower(20)
        return skin

    renderer = vtk.vtkOpenGLRenderer()
    
    # Add brain tissue surfaces (labels: 1=WhiteMatter, 2=GreyMatter, 3=Hippocampus, 4=Amygdala, 5=Thalamus)
    actor = extract("WhiteMatter", 1)
    if actor:
        renderer.AddActor(actor)
    actor = extract("GreyMatter", 2)
    if actor:
        renderer.AddActor(actor)
    actor = extract("Hippocampus", 3)
    if actor:
        renderer.AddActor(actor)
    actor = extract("Amygdala", 4)
    if actor:
        renderer.AddActor(actor)
    actor = extract("Thalamus", 5)
    if actor:
        renderer.AddActor(actor)
    
    renderer.SetBackground(0.1, 0.1, 0.15)

    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleMultiTouchCamera()
    renderInteractor.SetInteractorStyle(style)
    renderInteractor.SetRenderWindow(renderWin)

    renderWin.SetSize(800, 600)
    renderWin.SetWindowName("Brain Tissue Segmentation")

    # Launching the renderer
    renderInteractor.Initialize()
    renderWin.Render()
    renderInteractor.Start()


def load_segmentation(file_path):
    """Load segmentation from file (supports .mha and .npy)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith('.mha') or file_path.endswith('.mhd'):
        # Load SimpleITK image
        image = sitk.ReadImage(file_path)
        array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        return array, spacing
    elif file_path.endswith('.npy'):
        # Load numpy array (legacy)
        array = np.load(file_path)
        return array, (1.0, 1.0, 1.0)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='3D Visualization of Brain Tissue Segmentations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize from .mha file (experiment result)
  python show3d.py --file ./ablation_experiments/exp_00_baseline_none/118528_SEG.mha
  
  # Visualize from .npy file (legacy)
  python show3d.py --file segmentation.npy
  
  # Visualize specific labels only
  python show3d.py --file segmentation.mha --labels 1 2 3
        """
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Path to segmentation file (.mha, .mhd, or .npy)'
    )
    parser.add_argument(
        '--labels',
        type=int,
        nargs='+',
        help='Labels to visualize (1=WhiteMatter, 2=GreyMatter, 3=Hippocampus, 4=Amygdala, 5=Thalamus)'
    )
    parser.add_argument(
        '--spacing',
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help='Voxel spacing in mm (default: 1.0 1.0 1.0, only used for legacy .npy files)'
    )
    
    args = parser.parse_args()
    
    # Determine file to load
    if args.file:
        file_path = args.file
    elif len(sys.argv) == 1:
        # No arguments, try default
        if os.path.exists('segmentation.npy'):
            file_path = 'segmentation.npy'
        else:
            parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1
    
    # Load segmentation
    print(f"Loading segmentation from: {file_path}")
    try:
        data_matrix, spacing = load_segmentation(file_path)
        print(f"Segmentation shape: {data_matrix.shape}")
        print(f"Spacing: {spacing}")
        print(f"Unique labels: {sorted(np.unique(data_matrix))}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1
    
    # Use new visualizer if available
    if USE_NEW_VISUALIZER:
        print("\nUsing enhanced visualizer...")
        visualizer = SegmentationVisualizer(spacing=spacing)
        visualizer.visualize(data_matrix, labels_to_show=args.labels, 
                           title=os.path.basename(file_path))
    else:
        print("\nUsing legacy visualizer...")
        display_surface_models_legacy(data_matrix, spacing=spacing)
    
    return 0


if __name__ == '__main__':
    exit(main())