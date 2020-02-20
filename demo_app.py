"""
Demo application to run multiple OpenVINO detection models.

Same methodology apply for capturing from other video sources, code should be modified accordingly.
"""

import cv2
from detector import get_age_gender, draw_texts, get_emotions, get_detector, get_bounding_boxes, get_crops, get_landmarks, draw_bb_points_from_crop_to_frame, draw_bounding_box, draw_points

# Routes to actual models (modify properly)
FACE_DETECTION = 'intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml'
FACE_LM_2 = 'intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml'
EMOTIONS = 'intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml'
AGE_GENDER = 'intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml'

# Get networks and input dimensions.
face_net, h_face, w_face = get_detector(FACE_DETECTION)
lm2_net, h_lm2, w_lm2 = get_detector(FACE_LM_2)
emotions_net, h_emotions, w_emotions = get_detector(EMOTIONS)
age_gender_net, h_age_gender, w_age_gender = get_detector(AGE_GENDER)

# Put here a path to a video file (should be recognizable by OpenCV)
path_to_video = 'some/path/to/video/file'

cap = cv2.VideoCapture(path_to_video)

while(cap.isOpened()):
    ret, frame = cap.read()
    # This line is because my video file (don't really know why) was saved horizontaly, leaving commented code in case you face the same problem.
    #frame = cv2.rotate(frame,rotateCode=cv2.ROTATE_90_CLOCKWISE)
    if ret == True:
        try:
            # Get face(s) bounding boxe(s):
            results_bb = get_bounding_boxes(frame, face_net, h_face, w_face)

            # Extract face(s) crop(s):
            crops = get_crops(frame, results_bb)

            # Get absolute position of landmarks in the crop(s):
            pts_abs_lm = get_landmarks(crops, lm2_net, h_lm2, w_lm2)

            # Get age and gender from face(s) crop(s):
            results_age_gender = get_age_gender(crops, age_gender_net, h_age_gender, w_age_gender)
            
            # Get emotions from face(s) crop(s):
            emotions = get_emotions(crops, emotions_net, h_emotions, w_emotions, language='es')
            
            # Create string(s) to show in the video with annotations:
            textos = []
            for resultado in zip(results_age_gender, emotions):
                textos.append("edad: "+str(resultado[0][0])+", "+str(resultado[0][1])+", "+str(resultado[1]))
            
            # Draw bounding boxe(s) and landmarks on frame:
            img = draw_bb_points_from_crop_to_frame(frame, results_bb, pts_abs_lm)

            # Draw texts:
            img = draw_texts(img, textos, results_bb)

            # Show image:
            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

cap.release()
cv2.destroyAllWindows()