# FUCOS-tensorflow
A simple implementation of fucos on tensorflow

## Requirements
[Tensorflow](https://www.tensorflow.org/install/install_windows)

[tqdm](https://pypi.python.org/pypi/tqdm)

[VGG16 weights](https://www.cs.toronto.edu/~frossard/post/vgg16/)

The datasets for training, validation and testing can be found [here](https://onedrive.live.com/?id=9DDAAD6A86CCD831%218240&cid=9DDAAD6A86CCD831). 

## Usages
To train our network from scratch
```
python fucos.py train path/to/save/folder
```
To test an existing model
```
python fucos.py test path/to/checkpoint path/to/testing/image/folder
```
## Acknowlegements 
This implementation doesn't follow exactly what's on the paper. For more information please check out the [original paper](https://www.google.co.kr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiSkbmwnYvUAhVMHJQKHcYtBNkQFggmMAA&url=http%3A%2F%2Fwww.cv-foundation.org%2Fopenaccess%2Fcontent_cvpr_2016%2Fpapers%2FBruce_A_Deeper_Look_CVPR_2016_paper.pdf&usg=AFQjCNEe8O6bslD8hTTiGPfedAl0MmsoFA&sig2=mCwVrLi_c6dxRiBqAPOydA).
