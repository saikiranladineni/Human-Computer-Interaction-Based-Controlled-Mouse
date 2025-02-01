import argparse

def build_argparser():
    parser = argparse.ArgumentParser(description="Gaze-controlled Mouse Controller")
    
    parser.add_argument("-fd", "--faceDetectionModel", type=str, required=True,
                        help="Path to Face Detection model (XML)")
    parser.add_argument("-lr", "--landmarkRegressionModel", type=str, required=True,
                        help="Path to Landmark Regression model (XML)")
    parser.add_argument("-hp", "--headPoseEstimationModel", type=str, required=True,
                        help="Path to Head Pose Estimation model (XML)")
    parser.add_argument("-ge", "--gazeEstimationModel", type=str, required=True,
                        help="Path to Gaze Estimation model (XML)")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to input video file or 'cam' for webcam")
    parser.add_argument("-flags", "--previewFlags", nargs='+', default=[],
                        help="Specify preview flags: ff (Face), fl (Landmarks), fh (Pose), fg (Gaze)")
    parser.add_argument("-prob", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for face detection")
    parser.add_argument("-d", "--device", type=str, default='CPU',
                        help="Device to run inference on: CPU, GPU, FPGA, MYRIAD")
    parser.add_argument("-o", "--output_path", default='./results/', type=str,
                        help="Directory for output files")

    return parser
