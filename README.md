A project on detecting bounding box over the chessboard present in the picture.

The project has three distinctive parts:
1) generation of synthetic chessboards in PyBullet, using publicly available images
2) data augmentation, i.e. rotating, adding shadows etc
3) training convolution neural networks (mobile-net and resnet) to output a bounding box

Generation of synthetic chessboards is done in `chessboard.py`, but it uses data that is not available in this repository, except for some samples, which are stored in `public_data/samples`.
It uses PyBullet, which is a lightweight framework for 3D. (I could've probably used Blender)

Training is done in `training.py`, which uses PyTorch tools to load dataset and prepare it for training. Some features of training include:

- wing loss instead of standard MSE loss to prevent vanishing gradient
- additional component of loss function which penalises non-square angles and differences in rotation angles with the target, apart from raw L^2
- combining many distinct sources of data into one Dataset
- versioning models and picking up training

Whole pipeline can be tested by running `test.sh`, which serves as a kind of CI/CD rapid-(ish) script.

At the moment (29/10) detection does not work spectacularly well, but it outputs a broad shape of chessboard which could be presumably later refined with opencv.

____

Installation:

Due to requirement on numpy < 1.26 imposed by pybullet, it is recommended to create a separate `uv` venv.

```shell
uv venv
uv pip install -r requirements.txt
```

REMEMBER to add +cpu to `torch` and `torchvision` if you do not have GPU on your machine.

_____

Results:

one can inspect some results in `checkpoints`, where I left some pictures of how network is performing. 
Generated data can be checked in `public_data/samples`.