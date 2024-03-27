"""Segmentation app:
Allows user to upload an image and a mask, then makes a prediction,
compares it to the mask and calculates the IoU"""

from typing import Optional, Any

import numpy as np
from PIL import Image
import streamlit as st
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image, resize
from torchvision.utils import draw_segmentation_masks

from models.fcn_factory import FcnFactory


def main():
    """Displays the solar panel segmentation app"""
    st.title("Solar Panel Segmentation")
    # Load model
    model_path = 'saved_models/model_20240324_152229_4'
    model, input_transforms = load_model_and_transforms(model_path)
    prediction = None
    mask = None
    # Upload input image and mask as two columns
    col1, col2 = st.columns(2)
    with col1:
        image = upload_photo('Choose an input to for the model...')
        if image is not None:
            if image.mode == 'RGB':
                st.header('Input Image')
                display_photo(image)
                st.header('Prediction Overlay')
                prediction = predict(image, input_transforms, model)
                display_segmentation_mask(image, prediction)
            else:
                wrong_mode_notification('RGB')
    with col2:
        mask = upload_photo('Choose a mask to compare the model output to')
        if mask is not None:
            if mask.mode == 'L':
                st.header('Target Mask')
                display_photo(mask)
                st.header('Target Overlay')
                display_segmentation_mask(image, mask)
            else:
                wrong_mode_notification('L')
    if prediction is not None and mask is not None:
        st.header(f'IoU Score: {calc_iou(prediction, mask):.2f}')


def load_model_and_transforms(model_path) -> tuple[torch.nn.Module, Any]:
    """Load the model and its input transforms"""
    factory = FcnFactory(n_classes=2)
    loaded_model = factory.make_fcn('resnet50')
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    return loaded_model, factory.input_transforms


def upload_photo(msg: str) -> Optional[Image.Image]:
    """Upload a photo and return it as a PIL Image"""
    uploaded_file = st.file_uploader(
        msg, type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        image = None
    return image


def display_photo(img: Image) -> None:
    """Display a photo"""
    st.image(img, caption='Uploaded Image.', use_column_width=True)


def wrong_mode_notification(mode: str) -> None:
    """Display a notification when the wrong Image mode is selected"""
    mode_str = 'an RGB' if mode == 'RGB' else 'a Greyscale'
    st.write(f'Please upload {mode_str} image.')


def predict(image: Image.Image, transforms: Any, model: torch.nn.Module) -> torch.Tensor:
    """Predict the segmentation mask of an image"""
    processed_img = transforms(to_tensor(image))
    # Expand to shape [batch_size, n_classes, h, w] required for model
    processed_img = processed_img.expand([1, *processed_img.shape])
    # The output contains two classes: background (class 0) and solar panels
    # (class 1). Extract solar panels the class.
    output = model(processed_img)['out']
    return output > 0.5


def display_segmentation_mask(image: Image.Image, mask: Image.Image | torch.Tensor) -> None:
    """Display the segmentation mask (predicted or true) of an image"""
    # Rescale output back to original size
    if isinstance(mask, Image.Image):
        mask = to_tensor(mask) > 0.5
    elif isinstance(mask, torch.Tensor):
        mask = format_prediction(mask, image.size)
    else:
        raise TypeError(
            f'Expected mask to be of type Image.Image or torch.Tensor, got {type(mask)}'
        )
    # Overlay mask onto image
    processed_img = (to_tensor(image) * 255).type(torch.uint8)
    overlaid = to_pil_image(draw_segmentation_masks(processed_img, mask))
    st.image(overlaid, caption='Segmented Image.', use_column_width=True)


def format_prediction(pred: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Format the prediction to be of shape [h, w] and chosen size"""
    return resize(pred, size)[0, 1, :, :]


def calc_iou(pred: torch.BoolTensor, target: Image.Image) -> float:
    """Calculate the IoU for a single sample"""
    pred = format_prediction(pred, target.size).numpy()
    target = to_tensor(target).numpy()
    intersection = np.sum(np.logical_and(pred, target))
    union = np.sum(np.logical_or(pred, target))
    # Check if union is zeros.
    if union == 0.:
        # This means that the prediction and output
        # are both all zeros.
        return 1.
    return intersection / union


if __name__ == "__main__":
    main()
