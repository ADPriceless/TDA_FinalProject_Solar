"""Segmentation app:
Displays a single image from a test dataset with options to:
- retrieve new images;
- overlay the mask, predicion and IoU onto it;
- and change the threshold for the prediction.
"""


import logging
from pathlib import Path
import random
import sys
from typing import Union

from PIL import Image
import streamlit as st
import torch
from torch import Tensor
from torchvision.transforms.functional import to_tensor, to_pil_image, resize
from torchvision.utils import draw_segmentation_masks

import app_state


def load_sample_image() -> Image.Image:
    """Open the chosen image and apply whichever overlay is selected."""
    _load_img_mask_paths()
    input_img = _open_image(astype='tensor')
    opt = st.session_state.overlay
    if opt == 'True Mask':
        return _overlay_mask_on_img(
            input_img, _open_mask(), overlay_colours=['yellow']
        )
    if opt == 'Prediction':
        prediction = _predict(input_img)
        return _overlay_mask_on_img(
            input_img, prediction, overlay_colours=['yellow']
        )
    if opt == 'IoU':
        intersection, union = _calc_intersection_and_union(input_img)
        _iou_score(intersection, union)
        iou_mask = _iou_mask(intersection, union)
        return _overlay_mask_on_img(
            input_img, iou_mask, overlay_colours=['yellow', 'purple']
        )
    return _open_image(astype='image')


def _load_img_mask_paths():
    files = tuple(st.session_state.test_ds_path.glob('*.png'))
    file = str(files[st.session_state.sample_idx])
    if file.endswith('_label.png'):
        st.session_state.img_path = file[:-len('_label.png')] + '.png'
        st.session_state.mask_path = file
    else:
        st.session_state.img_path = file
        st.session_state.mask_path = file[:-4] + '_label.png'


def _open_image(astype: str = 'image') -> Union[Tensor, Image.Image]:
    img = Image.open(st.session_state.img_path)
    if astype == 'image':
        return img
    if astype == 'tensor':
        # `to_tensor` seems to convert to float, so convert back to uint8
        return (to_tensor(img) * 255).to(torch.uint8)
    raise ValueError(f'Unknown value for `astype`: {astype}')


def _overlay_mask_on_img(
    img: Tensor, mask: Tensor, overlay_colours: list[str]
) -> Image.Image:
    overlaid = draw_segmentation_masks(
        img, mask, alpha=st.session_state.overlay_alpha, colors=overlay_colours
    )
    return to_pil_image(overlaid)


def _open_mask() -> Tensor:
    return to_tensor(Image.open(st.session_state.mask_path)) > 0.


def _predict(input_img: Tensor) -> Tensor:
    if not st.session_state.prediction_valid:
        # Prevent the app from recalculating the prediction over and
        # again (Streamlit reruns entire script for any UI change).
        # `prediction_valid` is reset whenever a new image is grabbed.
        st.session_state.output = _clf_output(input_img)
        st.session_state.prediction_valid = True
    pred = (torch.softmax(st.session_state.output, dim=1) * 100.) > st.session_state.clf_threshold
    return resize(pred, input_img.shape[-2:])[0, 1] # return [h x w] of prediction


def _clf_output(input_img: Tensor) -> Tensor:
    processed_img = st.session_state.in_transforms(input_img)
    # Expand to shape [batch_size, n_classes, h, w] required for model
    processed_img = processed_img.expand([1, *processed_img.shape])
    # The output contains two classes: background (class 0) and solar panels
    # (class 1). Extract solar panels the class.
    return st.session_state.classifier(processed_img)['out']


def _calc_intersection_and_union(input_img: Tensor) -> tuple[Tensor, Tensor]:
    pred = _predict(input_img) # make a prediction if haven't already
    true = _open_mask()
    intersection = true.logical_and(pred)
    union = true.logical_or(pred)
    return intersection, union


def _iou_mask(intersection: Tensor, union: Tensor) -> Tensor:
    union = union.logical_xor(intersection) # Remove union mask where intersection is
    return torch.concat((intersection, union), dim=0)


def _iou_score(intersection: Tensor, union: Tensor) -> None:
    union_sum = union.sum().item()
    # If union is zero, neither pred or mask have solar panels. So,
    # model has made perfect prediction. Also prevent zero-division.
    if union_sum == 0.:
        st.session_state.iou = 1.
    st.session_state.iou = intersection.sum().item() / union_sum


def choose_random_img_mask_pair():
    """Select a random sample from the dataset."""
    st.session_state.sample_idx = random.randint(0, (2 * st.session_state.num_samples) -1)
    st.session_state.prediction_valid = False # reset for every new image


def description() -> str:
    """Return the description text associated with the current overlay."""
    opt = st.session_state.overlay
    if opt == 'IoU':
        prefix = f'#### IoU score: {st.session_state.iou:.2f}'
    else:
        prefix = ''
    return prefix + st.session_state.descriptions[opt]


def main():
    """Defines the UI and connects it to the app logic."""
    app_state.initialise_state()
    # Title
    st.markdown(
        f"<h1 style='text-align: center'>{st.session_state.title}</h1>",
        unsafe_allow_html=True
    )
    sample_img = load_sample_image()
    st.image(sample_img, st.session_state.img_caption, use_column_width=True)
    with st.container(border=True):
        left, middle, right = st.columns(3)
        left.button(
            'Show another image!', on_click=choose_random_img_mask_pair
        )
        middle.selectbox(
            'Overlay', st.session_state.overlay_opts, key='overlay'
        )
        right.slider(
            'Confidence Threshold (%)',
            min_value=10, max_value=90, step=10,
            value=st.session_state.default_clf_threshold,
            key='clf_threshold',
            help="""The higher the confidence,
                the more sure the model is there's a solar panel.
                \n\nGreater than 50% means it thinks the highlighted 
                area is more likely to be a solar panel than not."""
        )
    with st.container(border=True):
        st.write(description())


def log_resources():
    """Log files in app_resources dir, but count the images"""
    image_count = 0
    logging.debug('Directories:')
    for file in Path('app_resources').glob('**/*'):
        if file.suffix == '.png':
            image_count += 1
        else:
            logging.debug(file)
    logging.debug('Image count: %d', image_count)


if __name__ == "__main__":
    # check secrets.toml works
    logging.basicConfig(level=logging.DEBUG)
    logging.info('Branch streamlit_server_debugging')
    if 'verify_this_exists' not in st.secrets:
        logging.warning('secrets.toml not updated!')
    # Using these logs to probe project directory and
    # config before running the app itself.
    logging.debug('Python version: %s', sys.version)
    log_resources()
    main()
