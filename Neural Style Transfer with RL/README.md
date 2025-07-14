<br>
<img src="https://github.com/nazianafis/Resources/blob/main/NST/NST-gif.gif" alt="header" align="right" width="270"/>

# Neural-Style-Transfer (NST)

Neural Style Transfer is the ability to create a new image (known as a pastiche) based on two input images: one representing the content and the other representing the artistic style.

This repository contains a lightweight PyTorch implementation of art style transfer discussed in the seminal paper by [Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) To make the model faster and more accurate, a pre-trained VGG19 model is used.

## Overview <a name="overview"></a>

Neural style transfer is a technique that is used to take two images—a content image and a style reference image—and blend them together so that output image looks like the content image, but “painted” in the style of the style reference image.

## How does it work?<a name="working"></a>

1. We take content and style images as input and pre-process them.
2. Next, we load VGG19 which is a pre-trained CNN (convolutional neural network).
    1. Starting from the network's input layer, the first few layer activations represent low-level features like colors, and textures. As we step through the network, the final few layers represent higher-level features—like eyes.
    2. In this case, we use `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` for style representation, and `conv4_2` for content representation.    
![](https://github.com/nazianafis/Resources/blob/main/NST/NST-architecture.png)

3. We begin by cloning the content image and then iteratively changing its style. Then, we set our task as an optimization problem where we try to minimize:
    1. **content loss**, which is the L2 distance between the content and the generated image,
    2. **style loss**, which is the sum of L2 distances between the Gram matrices of the representations of the content image and the style image, extracted from different layers of VGG19.
    3. **total variation loss**, which is used for spatial continuity between the pixels of the generated image, thereby denoising it and giving it visual coherence.
4. Finally, we set our gradients and optimize using the L-BFGS algorithm to get the desired output.
