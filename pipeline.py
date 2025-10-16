"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, 
         config_dict: dict = None):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # Default configuration if none provided
    if config_dict is None:
        config_dict = {
            'preprocessing': {
                'skullstrip_pre': True,
                'normalization_pre': True,
                'registration_pre': True,
                'coordinates_feature': True,
                'intensity_feature': True,
                'gradient_intensity_feature': True
            },
            'postprocessing': {
                'simple_post': True
            },
            'forest': {
                'n_estimators': 10,
                'max_depth': 10,
                'max_features': None
            }
        }

    # Extract configuration
    pre_process_params = config_dict.get('preprocessing', {})
    post_process_params = config_dict.get('postprocessing', {})
    forest_params = config_dict.get('forest', {})

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    # Setup random forest with configurable parameters
    n_features = images[0].feature_matrix[0].shape[1]
    max_features = forest_params.get('max_features', n_features)
    if max_features is None:
        max_features = n_features
    
    forest = sk_ensemble.RandomForestClassifier(
        max_features=max_features,
        n_estimators=forest_params.get('n_estimators', 10),
        max_depth=forest_params.get('max_depth', 10),
        random_state=42,  # For reproducibility
        n_jobs=-1  # Use all available cores
    )

    print(f'Training Random Forest with {forest_params.get("n_estimators", 10)} trees, '
          f'max_depth={forest_params.get("max_depth", 10)}, '
          f'max_features={max_features} (out of {n_features} total features)')

    start_time = timeit.default_timer()
    forest.fit(data_train, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    # Save configuration used for this experiment
    import json
    with open(os.path.join(result_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        predictions = forest.predict(img.feature_matrix[0])
        probabilities = forest.predict_proba(img.feature_matrix[0])
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=True)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')

        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    
    # Write results with custom CSV format that includes experiment metadata
    result_file = os.path.join(result_dir, 'results.csv')
    write_custom_results_csv(evaluator.results, result_file, config_dict)
    
    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()


def write_custom_results_csv(results, result_file: str, config_dict: dict):
    """Write results to CSV with consistent format and embedded metadata."""
    import pandas as pd
    from collections import defaultdict
    
    # Extract configuration for metadata
    preprocessing = config_dict.get('preprocessing', {})
    postprocessing = config_dict.get('postprocessing', {})
    
    # Group flat Results by (id_, label) to collect metrics per row
    grouped_results = defaultdict(lambda: defaultdict(dict))  # {id_: {label: {metric: value}}}
    
    for result in results:  # Iterate the list of Result objects
        subject_id = result.id_  # Correct attr: str, e.g., "118528" or "118528-PP"
        label_name = result.label  # str, e.g., "GREYMATTER"
        metric_name = result.metric  # str, e.g., "DICE" or "HDRFDST95"
        metric_value = result.value  # float, e.g., 0.85
        
        # Collect into dict (flexible for any metrics)
        grouped_results[subject_id][label_name][metric_name] = metric_value
    
    # Flatten to rows
    rows = []
    for subject_id, labels_dict in grouped_results.items():
        for label_name, metrics_dict in labels_dict.items():
            # Extract Dice and Hausdorff (handle naming variations like "DICE", "HD95")
            dice_value = metrics_dict.get('DICE', metrics_dict.get('Dice', None))
            hausdorff_value = None
            for key in metrics_dict:
                if 'Hausdorff' in key or 'HD' in key or 'HDRFDST' in key:
                    hausdorff_value = metrics_dict[key]
                    break
            
            row = {
                'SUBJECT': subject_id,
                'LABEL': label_name,
                'DICE': dice_value if dice_value is not None else '',
                'HDRFDST': hausdorff_value if hausdorff_value is not None else '',
                'normalization': preprocessing.get('normalization_pre', False),
                'skull_stripping': preprocessing.get('skullstrip_pre', False),
                'registration': preprocessing.get('registration_pre', False),
                'postprocessing': postprocessing.get('simple_post', False),
                'coordinates_feature': preprocessing.get('coordinates_feature', False),
                'intensity_feature': preprocessing.get('intensity_feature', False),
                'gradient_intensity_feature': preprocessing.get('gradient_intensity_feature', False)
            }
            rows.append(row)
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(rows)
    df.to_csv(result_file, index=False, sep=',')
    
    print(f'Results saved to {result_file} with {len(rows)} measurements')


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    
    # Load configuration if provided
    config_dict = None
    if hasattr(args, 'config_file') and args.config_file:
        import json
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)

    
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, config_dict)