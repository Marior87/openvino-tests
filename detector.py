from handle_input import preprocessing
from inference import Network
import cv2
import numpy as np

# Path to CPU_EXTENSION, should be something like this:
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_detector(model_path, device='CPU', cpu_extension = CPU_EXTENSION):
    """
    Function to get a model loaded to a network and able to use some other useful functions.

    Args:
        model_path (string): Path to OpenVINO IR .xml file (assuming the .bin file is on the same location)
        device (string): One between 'CPU' and 'GPU', currently only supporting 'CPU'
        cpu_extension (string): Path to cpu extension file (device dependant).

    Returns:
        model_net = A Network instance ready to be used for inference.
        h_model = Expected input height for the model.
        w_model = Expected input width for the model.
    """
    model_net = Network()
    model_net.load_model(model_path,device,cpu_extension)
    model_shape = model_net.get_input_shape()
    h_model = model_shape[2]
    w_model = model_shape[3]
    return model_net, h_model, w_model

def get_bounding_boxes(frame, model_net, model_h, model_w, confidence=0.5):
    """
    Function to obtain bounding boxes when a frame is passed to a Network instance.

    Args:
        frame: Frame (or image) to process.
        model_net: A Network instance with model loaded.
        model_h: Input height expected for the model.
        model_w: Output width expected for the model.
        confidence: Confidence to consider successful a detection.

    Returns:
        results_bb: A list with len = number of bounding boxes, and each element contains
                    the four boundig box coordinates (xmin, ymin, xmax, ymax).
    """
    pp_frame = preprocessing(frame, model_h, model_w)
    model_net.async_inference(pp_frame)
    results_bb = []
    if model_net.wait() == 0:
        prev_results = model_net.extract_output()
        for p_r in prev_results[0,0]:
            if p_r[2] > confidence:
                results_bb.append(p_r[3:])
    return results_bb

def get_landmarks(crops, model_net, model_h, model_w):
    """
    Function to obtain landmarks from crop(s) (or image) when this is passed to a Network instance.
    Note:   Here we denote 'crop' because most of this kind of models require to process a small
            image, usually where a face was previously detected.

    Args:
        crops: Crops of a frame (or image) to process.
        model_net: A Network instance with model loaded.
        model_h: Input height expected for the model.
        model_w: Output width expected for the model.

    Returns:
        pts_abs = A list with the absolute position (ie regarding input shape) of each landmark.
    """
    pts_abs = []
    for crop in crops:
        h_crop = crop.shape[0]
        w_crop = crop.shape[1]
        pp_frame = preprocessing(crop, model_h, model_w)
        model_net.async_inference(pp_frame)
        p_abs = []
        if model_net.wait() == 0:
            results = model_net.extract_output()
            results = np.squeeze(results)
            i = 0
            while i < len(results):
                pt_abs = (results[i]*w_crop, results[i+1]*h_crop)
                p_abs.append(pt_abs)
                i = i + 2
        pts_abs.append(p_abs)
    return pts_abs

def draw_bounding_box(frame, result_bb, color =(0,255,0)):
    """
    Function to draw a bounding box over a frame on each detection in result_bb.

    Args:
        frame: A frame (or image) where to draw the bounding boxes.
        result_bb: A list of bounding box coordinates.
        color: Bounding box color (in BGR).

    Returns:
        img: Image with all the bounding boxes drawed.
    """

    height = frame.shape[0]
    width = frame.shape[1]
    for detection in result_bb:  
        xmin = int(detection[0]*width) # Top left
        ymin = int(detection[1]*height)
        xmax = int(detection[2]*width) # Bottom Right
        ymax = int(detection[3]*height)

        img = cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),color=color,thickness=1)
    return img

def draw_points(frame, result_lm, color=(0,0,255), thickness=-1):
    """
    Function to draw a points over a frame on each detection in result_lm.

    Args:
        frame: A frame (or image) where to draw the bounding boxes.
        result_lm: A list of points (or landmarks) coordinates.
        color: Bounding box color (in BGR).

    Returns:
        img: Image with all the points drawed.
    """

    height = frame.shape[0]
    width = frame.shape[1]
    for pt in result_lm:
        img = cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, color, thickness=thickness)
    return img

def get_crops(frame, result_bb):
    """
    Function to obtain crops from a frame given a list of bounding boxes.

    Args:
        frame: Frame (or image) where to obtain the crops.
        result_bb: A list of bounding box coordinates.

    Results:
        crops: A list with crops from the frame.
    """
    height = frame.shape[0]
    width = frame.shape[1]
    crops = []

    for result in result_bb: 
        xmin = int(result[0]*width) # Top left
        ymin = int(result[1]*height)
        xmax = int(result[2]*width) # Bottom Right
        ymax = int(result[3]*height)
        crops.append(frame[ymin:ymax, xmin:xmax])
    return crops

def draw_bb_points_from_crop_to_frame(frame, results_bb, pts_abs_lm):
    """
    Function to draw bounding boxes and face landmarks in a frame.

    Args:
        frame: Frame (or image) where to draw the bounding boxes.
        results_bb: List of bounding box coordinates.
        pts_abs_lm: List of absolute points of landmarks coordinates in the crop.

    Returns:
        img: Image with bounding boxes and face landmarks drawed.
    """

    assert len(results_bb) == len(pts_abs_lm), "Error en detecciones"

    img = frame.copy()
    h = frame.shape[0]
    w = frame.shape[1]

    for i in range(len(results_bb)):
        xmin = results_bb[i][0]*w
        ymin = results_bb[i][1]*h
        for point in pts_abs_lm[i]:
            x = int(point[0] + xmin)
            y = int(point[1] + ymin)
            img = draw_points(img, [(x,y)])

    img = draw_bounding_box(img, results_bb)

    return img

def get_emotions(crops, model_net, model_h, model_w, language='en'):
    """
    Function to get emotions from faces crops.

    Args:
        crop: A list of crops from a frame (or image).
        model_net: A Network instance of a suitable model.
        model_h: Input height expected for the model.
        model_w: Output width expected for the model.
        language (string): One of 'en' for english (default) or 'es' for spanish.

    Returns:
        emotions: A list of emotions strings for analyzed crops.
    """

    if language == 'en':
        emotions_tuple = ('neutral', 'happy', 'sad', 'surprise', 'anger')
    elif language == 'es':
        emotions_tuple = ('neutral', 'feliz', 'triste', 'sorprendido', 'furioso')
    else:
        print('Error, language should be one of "en" or "es"')
        exit()

    emotions = []
    for crop in crops:
        h_crop = crop.shape[0]
        w_crop = crop.shape[1]
        pp_frame = preprocessing(crop, model_h, model_w)
        model_net.async_inference(pp_frame)

        if model_net.wait() == 0:
            results = model_net.extract_output()
            results = np.squeeze(results)
            idx_max = np.argmax(results)
            emotion = emotions_tuple[idx_max]
        emotions.append(emotion)
    return emotions

def draw_texts(frame, 
                texts, 
                results_bb, 
                font = cv2.FONT_HERSHEY_SIMPLEX, 
                font_size = 0.5, 
                font_color = (255,255,255), 
                font_thickness = 2,
                offset_x = -50,
                offset_y = -10):
    """
    Function to draw text over a bounding box.

    Args:
        frame: Frame (or image) where to draw text.
        texts: A list of text, should be the same lenght as the quantity of bounding boxes.
        results_bb: A list of bounding box coordinates, which each one is in the form (xmin, ymin, xmax, ymax).
        font: OpenCV font style
        font_size: OpenCV font size.
        font_color: BGR font color tuple.
        fonr_thickness: Font line thickness.
        offset_x: Offset in x position of text regarding the upper left corner
        offset_y: Offset in y position of text regarding the upper left corner

    Returns:
        img: Image with required text drawn.
    """


    assert len(results_bb) == len(texts), "Number of bounding boxes ({}) not the same as number of texts ({})".format([len(results_bb), len(texts)])
    
    h = frame.shape[0]
    w = frame.shape[1]

    img = frame.copy()

    for i in range(len(results_bb)):
        xmin = int(results_bb[i][0]*w)
        ymin = int(results_bb[i][1]*h)
        xmax = int(results_bb[i][2]*w)
        ymax = int(results_bb[i][3]*h)

        img = cv2.putText(img, texts[i],(xmin+offset_x,ymin+offset_y), font, font_size, font_color, font_thickness)

    return img

def get_age_gender(crops, model_net, model_h, model_w, language='es'):
    """
    Function to get emotions from faces crops.

    Args:
        crop: A list of crops from a frame (or image).
        model_net: A Network instance of a suitable model.
        model_h: Input height expected for the model.
        model_w: Output width expected for the model.
        language (string): One of 'en' for english (default) or 'es' for spanish.

    Returns:
        age_gender: A list of tuples with identifyed (age, gender) for each crop.
    """
    if language == 'en':
        gender_list = ('female', 'male')
    elif language == 'es':
        gender_list = ('mujer', 'hombre')
    else:
        print('Error, language should be one of "en" or "es"')
        exit()

    age_gender =[]
    for crop in crops:
        h_crop = crop.shape[0]
        w_crop = crop.shape[1]
        pp_frame = preprocessing(crop, model_h, model_w)
        model_net.async_inference(pp_frame)
        if model_net.wait() == 0:
            outputs = model_net.exec_network.requests[0].outputs
            age = np.round(np.squeeze(outputs['age_conv3'])*100,2)
            idx_gender = np.argmax(np.squeeze(outputs['prob']))
            gender = gender_list[idx_gender]
            age_gender.append((age, gender))
            
    return age_gender
