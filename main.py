import cv2
import torch
import torch.nn as nn
import numpy as np
import supervision as sv

from attibute_analysis_detector import VehicleAttribute, ColorDetector
from yolo_detector import YoloDetector

# Device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "")


def calculate_average_pooling(img, size_of_pool=3):
    height, width, channel = img.shape
    if height < size_of_pool or width < size_of_pool:
        return

    img = torch.from_numpy(img).float().to(device)
    img = torch.permute(img, (2, 0, 1))         # Change the input size to (channel, height, width)

    # Average Pooling
    pool = nn.AvgPool2d(kernel_size=size_of_pool, stride=size_of_pool).to(device)
    output = pool(img)
    output = torch.permute(output, (1, 2, 0)).detach().cpu().numpy()  # Back to size of (height, width, channel)
    return output


if __name__ == '__main__':
    # Define all required objects
    yolo = YoloDetector(yolo_model="asset/model/yolo/yolov8l.pt")
    vehicle_attribute = VehicleAttribute()
    color_detector = ColorDetector()
    bounding_box_annotator = sv.BoundingBoxAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"])
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#FFFFFF"]),
        text_color=sv.Color.BLACK,
        text_scale=0.35,
        text_padding=2
    )

    cap = cv2.VideoCapture("asset/video/sample#2.mp4")
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        h, w = frame.shape[:2]
        return_value = []       # Will be filled with dictionary of color & type
        return_color = []       # Will be filled with only color
        return_type = []        # Will be filled with only type

        result_car_detection = yolo.track(frame, classes=[1, 2, 3, 5, 7], conf=0.65)
        detections = sv.Detections.from_ultralytics(result_car_detection)
        if detections.tracker_id is None:
            continue

        # For loop of vehicle detections
        for x1, y1, x2, y2 in detections.xyxy:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mid_x, mid_y = int(x1 + ((x2 - x1) / 2)), int(y1 + ((y2 - y1) / 2))

            car = frame[y1:y2, x1:x2]
            car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)

            # Attribute Analysis
            result = next(vehicle_attribute.get_attribute(car))[0]  # use next() because it generator
            attributes = result['attributes']
            """
                variable 'attributes' will output like this,
                
                OUTPUT CAN BE VARY : 
                    - "Color: (golden, prob: 0.8583461046218872), Type: (sedan)"
                    - "Color: (golden, prob: 0.8583461046218872), Type: None"
                    - "Color: None, Type: (sedan)"
                    - "Color: None, Type: None"
                TYPE    : String
                
            """

            ###########################
            # Below is the code for separate the prediction output (String) to dictionary of value
            ###########################

            color_value = None
            color_conf = None
            type_value = None
            type_conf = None

            if len(attributes.split(",")) <= 2:       # Filter if both none
                color_value = "unknown"
                color_conf = -1

                type_value = "unknown"
                type_conf = -1
            else:
                index_of_word_type = attributes.index("Type")

                color = attributes[:index_of_word_type]             # Filter 'Color' section
                if color.find("(") != -1:
                    _color_value = color[color.find("(") + 1:color.find(",")]
                    _color_conf = color[color.find(":", 10) + 2:color.find(")", 10)]

                    color_value = _color_value
                    color_conf = float(_color_conf)
                else:
                    color_value = "unknown"
                    color_conf = -1

                type = attributes[index_of_word_type:]              # Filter 'Type' section
                if type.find("(") != -1:
                    _type_value = type[type.find("(") + 1:type.find(",")]
                    _type_conf = type[type.find(":", 10) + 2:type.find(")", 10)]

                    type_value = _type_value
                    type_conf = float(_type_conf)

                else:
                    type_value = "unknown"
                    type_conf = -1

            # Color Detector
            """
                Even though the vehicle attributes paddle library has color detection
                the result is not satisfactory. We'll use third party way of predicting
                color of the car.
                    
                PAPER   : Robust color object detection using spatial-color joint probability functions
                URL     : https://ieeexplore.ieee.org/abstract/document/1315057
            """
            h_car, w_car = car.shape[:2]
            # Crop to middle of cropped image of the car
            crop_percentage = 0.2
            x1_crop = 0 + h_car * crop_percentage
            y1_crop = 0 + w_car * crop_percentage
            x2_crop = h_car - h_car * crop_percentage
            y2_crop = w_car - w_car * crop_percentage
            x1_crop, y1_crop, x2_crop, y2_crop = int(x1_crop), int(y1_crop), int(x2_crop), int(y2_crop)

            cropped_area = car[y1_crop:y2_crop, x1_crop:x2_crop]       # Cropped region of interest (one car only)

            # Average pooling
            average_pooling = calculate_average_pooling(cropped_area)
            if average_pooling is None:                             # If average pooling error skip this car image
                continue
            median = np.median(average_pooling, axis=(0, 1))        # Get the median value of average pooling result

            p_col, p_cat = color_detector.predict(median)           # Predict color (accepted input: RGB array)
            color_value = p_col

            return_value.append({
                "color": {
                    "value": color_value,
                    "conf": color_conf
                },
                "type": {
                    "value": type_value,
                    "conf": type_conf
                }
            })

            return_color.append(color_value)
            return_type.append(type_value)

        # Supervision annotation
        labels = [
            f"{class_name} | {type} {color}"
            for tracker_id, class_name, confidence, type, color
            in zip(detections.tracker_id, detections['class_name'], detections.confidence, return_type, return_color)
        ]
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        cv2.imshow("webcam", annotated_frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
