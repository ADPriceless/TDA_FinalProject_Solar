# Models 
---

## U-Net
### Architecture
- Made of an encoder (downsampler) and an encoder (upsampler) \[[1](#sources)\]
- Best to use a pretrained model as the encoder \[[1](#sources)\]
- Deeper layers of the encoder capture increasingly abstract representations \[[5](#sources)\]
  - Deeper into the encoder, the spacial resolution (width and height of the layer's input) decreases but  the depth (i.e. feature space) increases
  - This allows the model to learn low-level and high-level features \[[6](#sources)\]
- Skip connections between encoder and decoder... \[[5](#sources)\]
- The output is a convolution with *n*-channels representing *n*-classes \[[5](#sources)\] (in this case, one channel is required)

### Implementation Notes
#### Available Pretrained Models
- Using PyTorch package (because it was easier to create custom data and label loading)
  - Pre-trained models may be loaded using TorchVision and PyTorch `torch.hub` packages \[[2](#sources)\] e.g.
    - ResNet-50 which is a popular CNN architecture \[[3](#sources)\]
    - MobileNet
  - Or they may be loaded via Hugging Face's `timm` module \[[4](#sources)\]

#### Data Preparation
- The custom dataset `__getitem__` method should return \[[7](#sources)\]:
>- image: torchvision.tv_tensors.Image of shape [3, H, W], a pure tensor, or a PIL Image of size [H, W]
>- target: a dict containing the following fields
>  - boxes, torchvision.tv_tensors.BoundingBoxes of shape [N, 4]: the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
>  - labels, integer torch.Tensor of shape [N]: the label for each bounding box. 0 represents always the background class.
>  - image_id, int: an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
>  - area, float torch.Tensor of shape [N]: the area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
>  - iscrowd, uint8 torch.Tensor of shape [N]: instances with iscrowd=True will be ignored during evaluation.
>  - (optionally) masks, torchvision.tv_tensors.Mask of shape [N, H, W]: the segmentation masks for each one of the objects

- Currently, the custom dataset returns a pure tensor, so is compliant
- However, the current custom dataset returns a pure tensor. This may need to be changed to a dict of:
  - `label`: integer torch.Tensor of shape [2]: background and solar panel (background is always label 0 \[[7](#sources)\])
  - `masks`: torchvision.tv_tensors.Mask of shape [2, H, W]

#### How to Fine-Tune Pretrained Models
- Use the whole model and fine tune
  - Load the data in a format that the model can use
  - Load the pre-trained weights
  - Freeze the model's weights (i.e. no training)
  - Use a custom head on the output to train
- Use the frozen encoder and create a custom decoder
  - Similar to the above, but with a custom trainable decoder which must have the cross-connections to the encoder blocks
  - This sounds more awkward to implement, and would require more training
- OR train from scratch?
  - May take too long and produce worse results

### Optimiser
- Use Adam by default, since (from experience) it often leads to better results than SGD or batch gradient descent

### Loss Function
- Crossentropy \[[4](#sources)\]
  - Could binary crossentropy be used since there are only two classes in this case: solar panel and no solar panel?

### Metrics
- Mean IoU \[[2](#sources)\]
- Pixelwise accuracy \[[2](#sources)\]
---

## Sources
1. “Image segmentation | TensorFlow Core,” TensorFlow. https://www.tensorflow.org/tutorials/images/segmentation
2. “Models and pre-trained weights — Torchvision main documentation,” pytorch.org. https://pytorch.org/vision/master/models.html
3. S. Mukherjee, “The Annotated ResNet-50,” Medium, Aug. 18, 2022. https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758
4. Ruman, “Part 1: Ultimate Guide to Fine-Tuning in PyTorch : Pre-trained Model and Its Configuration,” Medium, Nov. 07, 2023. https://rumn.medium.com/part-1-ultimate-guide-to-fine-tuning-in-pytorch-pre-trained-model-and-its-configuration-8990194b71e (accessed Mar. 18, 2024).
5. “U-Net Architecture Explained,” GeeksforGeeks, Jun. 08, 2023. https://www.geeksforgeeks.org/u-net-architecture-explained/
6. D. N. R, “U-Net Demystified: A Gentle Introduction,” Medium, Oct. 09, 2023. https://medium.com/@deepaknr015/u-net-demystified-a-gentle-introduction-2c96778126d2.
7. “TorchVision Object Detection Finetuning Tutorial — PyTorch Tutorials 1.5.0 documentation,” pytorch.org. https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
8. G. Kasmi et al., “A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays and installation metadata,” Scientific Data, vol. 10, no. 1, Jan. 2023, doi: https://doi.org/10.1038/s41597-023-01951-4
9. M. Kleebauer, C. Marz, C. Reudenbach, and M. Braun, “Multi-Resolution Segmentation of Solar Photovoltaic Systems Using Deep Learning,” Remote Sensing, vol. 15, no. 24, p. 5687, Jan. 2023, doi: https://doi.org/10.3390/rs15245687
10. H. Jiang et al., “Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery,” Earth System Science Data, vol. 13, no. 11, pp. 5389–5401, Nov. 2021, doi: https://doi.org/10.5194/essd-13-5389-2021