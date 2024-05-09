import paddleclas


class VehicleAttribute:
    def __init__(self):
        self.model = paddleclas.PaddleClas(model_name="vehicle_attribute")

    def get_attribute(self, frame):
        result = self.model.predict(frame, predict_type="cls")
        return result