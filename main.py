import cv2

from yolo_detector import YoloDetector
from nafnet import NAFNet

if __name__ == '__main__':
    vehicle_detector = YoloDetector(yolo_model="asset/model/yolo/yolov8l.pt")
    license_plate_detector = YoloDetector(yolo_model="asset/model/license_plate/lpr_fast.pt")
    deblur = NAFNet()

    # Set up the NAFNet
    opt_path = 'nafnet/options/test/REDS/NAFNet-width64.yml'
    opt = deblur.parse(opt_path)
    opt['dist'] = False
    model = deblur.create_model(opt)
    deblur.set_model(model=model)

    cap = cv2.VideoCapture("asset/video/sample#4.mp4")
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        result_car_detection = vehicle_detector.track(frame, classes=[1, 2, 3, 5, 7], conf=0.8)
        boxes = result_car_detection.boxes.xyxy.clone().tolist()
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # car = frame[y1:y2, x1:x2]
            # frame[y1:y2, x1:x2] = deblur.run(car)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255))

        result_license_plate_detection = license_plate_detector.detect(frame, conf=0.4)
        boxes = result_license_plate_detection.boxes.xyxy.clone().tolist()
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            license_plate = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = deblur.run(license_plate)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255))

        cv2.imshow("webcam", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
