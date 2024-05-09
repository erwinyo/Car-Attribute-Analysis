import cv2
import numpy as np

from yolo_detector import YoloDetector
from nafnet import NAFNet
from attibute_analysis_detector import VehicleAttribute, ColorDetector

if __name__ == '__main__':
    vehicle_detector = YoloDetector(yolo_model="asset/model/yolo/yolov8l.pt")
    # license_plate_detector = YoloDetector(yolo_model="asset/model/license_plate/lpr_fast.pt")
    # deblur = NAFNet()
    vehicle_attribute = VehicleAttribute()
    color_detector = ColorDetector()

    # Set up the NAFNet
    # opt_path = 'nafnet/options/test/REDS/NAFNet-width64.yml'
    # opt = deblur.parse(opt_path)
    # opt['dist'] = False
    # model = deblur.create_model(opt)
    # deblur.set_model(model=model)

    cap = cv2.VideoCapture("asset/video/sample#1.mp4")
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break
        h, w = frame.shape[:2]
        return_value = []

        padding = 0
        result_car_detection = vehicle_detector.track(frame, classes=[1, 2, 3, 5, 7], conf=0.8)
        boxes = result_car_detection.boxes.xyxy.clone().tolist()
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1, x2, y2 = x1-padding, y1-padding, x2+padding, y2+padding
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            car = frame[y1:y2, x1:x2]
            car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)

            # Attribute Analysis
            result = next(vehicle_attribute.get_attribute(car))[0]
            attributes = result['attributes']

            color_value = None
            color_conf = None
            type_value = None
            type_conf = None

            if len(attributes.split(",")) <= 2:
                color_value = "unknown"
                color_conf = -1

                type_value = "unknown"
                type_conf = -1
            else:
                index_of_word_type = attributes.index("Type")

                color = attributes[:index_of_word_type]
                if color.find("(") != -1:
                    _color_value = color[color.find("(") + 1:color.find(",")]
                    _color_conf = color[color.find(":", 10) + 2:color.find(")", 10)]

                    color_value = _color_value
                    color_conf = float(_color_conf)
                else:
                    color_value = "unknown"
                    color_conf = -1

                type = attributes[index_of_word_type:]
                if type.find("(") != -1:
                    _type_value = type[type.find("(") + 1:type.find(",")]
                    _type_conf = type[type.find(":", 10) + 2:type.find(")", 10)]

                    type_value = _type_value
                    type_conf = float(_type_conf)

                else:
                    type_value = "unknown"
                    type_conf = -1

            # Color Detector
            h_car, w_car = car.shape[:2]
            perc = 0.2
            # Calculate average color
            small_area = car[(int(0+h_car*perc)): (int(h_car-h_car*perc)), (int(0+w_car*perc)): (int(w_car-w_car*perc))]
            cv2.imshow("we", small_area)
            rgb = np.mean(small_area, axis=(0, 1))  # Average across rows and columns
            p_col, p_cat = color_detector.predict(rgb)
            color_value = p_cat

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

            # Annotate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255))
            # Put text on the image
            cv2.putText(frame, f"Color: {color_value}", (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Type: {type_value}", (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("webcam", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
