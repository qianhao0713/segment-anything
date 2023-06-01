class ModelData(object):
    # Name of input node
    INPUT_NAME = "Input"
    # CHW format of model input
    INPUT_SHAPE = (1, 3, 1088, 1920)
    # Name of output node
    OUTPUT_NAME = "NMS"

    @staticmethod
    def get_input_channels():
        return ModelData.INPUT_SHAPE[1]

    @staticmethod
    def get_input_height():
        return ModelData.INPUT_SHAPE[2]

    @staticmethod
    def get_input_width():
        return ModelData.INPUT_SHAPE[3]