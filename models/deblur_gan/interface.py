from models.deblur_gan.model.config import DeBlurConfig
from models.deblur_gan.model.predict import Predictor


# def download_weight(path='models/deblur_gan/weight/big-lama.pt')
# model pth url
# https://drive.google.com/uc?export=view&id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR

def load_deblur_model(device='cpu', config=DeBlurConfig()):
    model = Predictor(device, config)
    return model
