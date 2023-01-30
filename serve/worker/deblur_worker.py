import cv2
from pydantic import BaseModel

from models.deblur_gan.interface import load_deblur_model
from models.deblur_gan.model.config import DeBlurConfig
from serve.worker.utils import read_image_from_binary, write_image_to_binary

class DeBlurWorkerConfig(BaseModel):
    device: str = 'cpu'

class DeBlurWorker:
    def __init__(self, worker_config: DeBlurWorkerConfig = DeBlurWorkerConfig(),
                 model_config: DeBlurConfig = DeBlurConfig()):
        self.worker_config = worker_config
        self.model_config = model_config

        self.model = load_deblur_model(self.worker_config.device, self.model_config)

    def process(self, input_image):
        image = read_image_from_binary(input_image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.model(image)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return write_image_to_binary(result)
