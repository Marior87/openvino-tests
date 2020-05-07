"""
Demo application to run multiple OpenVINO detection models.

Same methodology apply for capturing from other video sources, code should be modified accordingly.
"""

import cv2
import detector
from argparse import ArgumentParser

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file. Use CAM to use webcam stream")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable."
                             "Sample will look for a suitable plugin for device specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering (0.5 by default)")
    return parser


def run_app(args):

    # Routes to actual models (modify properly). This is in order to be "ready-to-go" for the app:
    FACE_DETECTION = 'intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml'
    FACE_LM_2 = 'intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml'
    EMOTIONS = 'intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml'
    AGE_GENDER = 'intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml'

    # Get networks and input dimensions.
    face_net, h_face, w_face = detector.get_detector(FACE_DETECTION, device=args.device, cpu_extension=args.cpu_extension)
    lm2_net, h_lm2, w_lm2 = detector.get_detector(FACE_LM_2, device=args.device, cpu_extension=args.cpu_extension)
    emotions_net, h_emotions, w_emotions = detector.get_detector(EMOTIONS, device=args.device, cpu_extension=args.cpu_extension)
    age_gender_net, h_age_gender, w_age_gender = detector.get_detector(AGE_GENDER, device=args.device, cpu_extension=args.cpu_extension)

    if args.input != 'CAM':
        try:
            input_stream = cv2.VideoCapture(args.input)
            length = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))

            # Check if input is an image or video file:
            if length > 1:
                single_image_mode = False
            else:
                single_image_mode = True

        except:
            print('Not supported image or video file format. Please pass a supported one.')
            exit()

    else:
        input_stream = cv2.VideoCapture(0)
        single_image_mode = False

    while(input_stream.isOpened()):
        ret, frame = input_stream.read()
        
        # This line is because my video file (don't really know why) was saved horizontaly, leaving commented code in case you face the same problem.
        #frame = cv2.rotate(frame,rotateCode=cv2.ROTATE_90_CLOCKWISE)
        if ret == True:
            try:
                # Get face(s) bounding boxe(s):
                results_bb = detector.get_bounding_boxes(frame, face_net, h_face, w_face, confidence=args.prob_threshold)
                # Extract face(s) crop(s):
                crops = detector.get_crops(frame, results_bb)

                # Get absolute position of landmarks in the crop(s):
                pts_abs_lm = detector.get_landmarks(crops, lm2_net, h_lm2, w_lm2)

                # Get age and gender from face(s) crop(s):
                results_age_gender = detector.get_age_gender(crops, age_gender_net, h_age_gender, w_age_gender)
                
                # Get emotions from face(s) crop(s):
                emotions = detector.get_emotions(crops, emotions_net, h_emotions, w_emotions, language='en')
                
                # Create string(s) to show in the video with annotations:
                texts = []
                for resultado in zip(results_age_gender, emotions):
                    texts.append("age: "+str(resultado[0][0])+", "+str(resultado[0][1])+", "+str(resultado[1]))
                
                # Draw bounding boxe(s) and landmarks on frame:
                img = detector.draw_bb_points_from_crop_to_frame(frame, results_bb, pts_abs_lm)

                # Draw texts:
                img = detector.draw_texts(img, texts, results_bb)

                # Show image:
                cv2.imshow('frame',img)
                if cv2.waitKey(1) & 0xFF == ord('q') or single_image_mode:
                    break
            except:
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    input_stream.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Run the app
    run_app(args)


if __name__ == '__main__':
    main()