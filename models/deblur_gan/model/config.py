from pydantic import BaseModel


class DeBlurConfig(BaseModel):
    norm_layer:str = 'instance'
    weight_path = r'models\deblur_gan\weight\fpn_inception.h5'
