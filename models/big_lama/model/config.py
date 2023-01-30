from pydantic import BaseModel


class LaMaConfig(BaseModel):
    model_path = r'models/big_lama/weight/big-lama.pt'
    pad_mod: int = 8
    pad_to_square: bool = False
    resize_limit: int = 512
    pad_min_size: int = 128
