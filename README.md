# Image Inpainting Back-End

This repository is modified from [Xyfuture/hpc_backbone](https://github.com/Xyfuture/hpc_backbone) for easier and faster deployment and test:

* Change the backend framework to Flask
* Delete Redis Message Queue

Note: "Using Flask in this way is by far the easiest way to start serving your PyTorch models, but it will not work for a use case with high performance requirements." [DEPLOYING PYTORCH IN PYTHON VIA A REST API WITH FLASK](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)

**Related Deep Learning Models：**

* LaMa Image Inpainting, Resolution-robust Large Mask Inpainting with Fourier Convolutions, WACV 2022
  * Code for this paper: https://github.com/saic-mdal/lama

- DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better, ICCV 2019
  - Code for this paper: https://github.com/VITA-Group/DeblurGANv2


## How to run this project

### Step 1

Download model weight to  `models/big_lama/weight/` and  `models/deblur_gan/weight/`

- [Lama](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt)
- [DebulrGANv2](https://drive.google.com/uc?export=view&id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR)

### Step 2

Install dependencies：

```
pip install -r requirements.txt
```

### Step 3

Start server

```
python app.py
```

