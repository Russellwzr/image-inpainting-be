import cv2
from pydantic import BaseModel

from models.big_lama.interface import load_lama_model
from models.big_lama.model.config import LaMaConfig
from serve.worker.utils import read_image_from_binary, write_image_to_binary


class LaMaWorkerConfig(BaseModel):
    device: str = 'cpu'

class LaMaWorker:
    def __init__(self, worker_config: LaMaWorkerConfig = LaMaWorkerConfig(), model_config: LaMaConfig = LaMaConfig()):
        self.worker_config: LaMaWorkerConfig = worker_config
        self.model_config = model_config

        self.model = load_lama_model(self.worker_config.device, self.model_config)

    def input_check(self, images, masks) -> bool:
        flag = True
        for i, tmp in enumerate(zip(images, masks)):
            img, msk = tmp
            if img.shape[:2] != msk.shape:
                flag = False
                masks[i] = cv2.resize(msk, (img.shape[1], img.shape[0]))
        return flag

    def process(self, input_image, input_mask):
        input_image = read_image_from_binary(input_image, cv2.IMREAD_COLOR)
        images = [cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)]
        masks = [read_image_from_binary(input_mask, cv2.IMREAD_GRAYSCALE)]
        results = self.model(images, masks)
        return [write_image_to_binary(result) for result in results]
