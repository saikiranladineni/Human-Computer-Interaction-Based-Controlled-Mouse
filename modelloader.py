import os
import logging
from face_detection_model import FaceDetectionModel
from landmark_detection_model import LandmarkDetectionModel
from head_pose_estimation_model import HeadPoseEstimationModel
from gaze_estimation_model import GazeEstimationModel

def load_models(model_paths, device, threshold):
    logger = logging.getLogger("ModelLoader")

    models = {
        "face": FaceDetectionModel(model_paths['FaceDetectionModel'], device, threshold),
        "landmark": LandmarkDetectionModel(model_paths['LandmarkRegressionModel'], device, threshold),
        "head_pose": HeadPoseEstimationModel(model_paths['HeadPoseEstimationModel'], device, threshold),
        "gaze": GazeEstimationModel(model_paths['GazeEstimationModel'], device, threshold),
    }

    for model_name, model in models.items():
        if not os.path.isfile(model_paths[model_name]):
            logger.error(f"Model file not found: {model_paths[model_name]}")
            exit(1)
    
    return models

