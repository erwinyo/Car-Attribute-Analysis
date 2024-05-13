import cv2
import supervision as sv

from yolo_detector import YoloDetector


if __name__ == '__main__':
    yolo = YoloDetector(yolo_model="../asset/model/yolo/yolov8l.pt")
    bounding_box_annotator = sv.BoundingBoxAnnotator(
        color=sv.ColorPalette.DEFAULT
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.ROBOFLOW
    )

    cap = cv2.VideoCapture("../asset/video/sample#2.mp4")
    while True:
        has_frame, frame = cap.read()
        if not has_frame:
            break

        result = yolo.track(frame, classes=[1, 2, 3, 5, 7], conf=0.65)
        detections = sv.Detections.from_ultralytics(result)
        # Supervision annotation
        labels = [
            f"{tracker_id} {class_name} {confidence:.2f}"
            for tracker_id, class_name, confidence
            in zip(detections.tracker_id, detections['class_name'], detections.confidence)
        ]

        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        for x1, y1, x2, y2 in detections.xyxy:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mid_x, mid_y = int(x1+((x2-x1) / 2)), int(y1+((y2-y1) / 2))
            anchor = sv.Point(x=mid_x, y=y2-10)
            annotated_frame = sv.draw_text(
                scene=annotated_frame,
                text="Hello, world!",
                text_anchor=anchor,
                text_color=sv.Color.WHITE
            )

        cv2.imshow("webcam", annotated_frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()
