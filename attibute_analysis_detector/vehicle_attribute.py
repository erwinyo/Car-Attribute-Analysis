import paddleclas


class VehicleAttribute:
    def __init__(self):
        self.model = paddleclas.PaddleClas(model_name="vehicle_attribute")

    def get_attribute(self, frame):
        """
        Get the vehicle attributes

        Args:
            frame (np.ndarray): numpy array of frame (RGB format)

        Returns:
            generator: generator with each generate is dictionary of vehicle attributes

        """
        result = self.model.predict(frame, predict_type="cls")
        return result
