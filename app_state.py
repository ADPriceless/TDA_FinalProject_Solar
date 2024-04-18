"""Functions to initialise the state of the app."""

from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms.v2 as tv_transforms

from models.fcn_factory import FcnFactory
from preprocess.transforms import rgba_to_rgb


def initialise_state():
    """Initialises the state for the app. Edit values to alter app properties."""
    initial_state = {
        'title': 'Solar Panel Segmentation',
        'test_ds_path': Path('app_resources\\test_ds'),
        'num_samples': 465,
        'sample_idx': 0,
        'img_path': None,
        'mask_path': None,
        'img_caption': '',
        'overlay_alpha': 0.3,
        'overlay_opts': ['None', 'True Mask', 'Prediction', 'IoU'],
        'overlay': 'None',
        'clf_path': Path('app_resources\\pretrained_models\\pretrained_solar_panel_clf.pt'),
        'default_clf_threshold': 50,
        'prediction_valid': False,
        'iou': 0.0,
    }
    for k, v in initial_state.items():
        if k not in st.session_state:
            setattr(st.session_state, k, v)
    # 'descriptions' setup after because it refers to 'iou'
    if 'descriptions' not in st.session_state:
        st.session_state.descriptions = {
            'None':
                """The images are taken from a collection of images the
                model is not trained on.
                \n\nThe dataset which the model was trained on is:
                \n\n*Jiang Hou, Yao Lingand Liu Yujun, ‘Multi-resolution
                dataset for photovoltaic panel segmentation from satellite
                and aerial imagery’. Zenodo, Aug. 09, 2021. doi:
                10.5281/zenodo.5171712.*""",
            'True Mask':
                """The true mask is the area in which the solar panels
                actually are.
                \n\nHowever, you may notice some errors in these - any
                errors like this in the training data will affect the
                quality of the model.""",
            'Prediction':
                """This is the prediction made by the model.""",
            'IoU':
                """\n\n The *Intersection Over Union* (IoU) score is the
                metric used to evaluate each prediction. It is simply the
                value of the intersection divided by the union, where:
                \n\n- The *intersection* is the area where the prediction
                and the true mask both agree that there is a solar panel
                (yellow).
                \n\n- The *union* is the total area covered by either the
                prediction or the mask (purple).
                \n\nThe more purple there is, the more mistakes the model
                has made.""",
        }
    factory = FcnFactory(n_classes=2)
    if 'classifier' not in st.session_state:
        st.session_state.classifier = load_clf_from(factory)
    if 'in_transforms' not in st.session_state:
        st.session_state.in_transforms = load_input_transforms_from(factory)


def load_clf_from(factory: FcnFactory) -> torch.nn.Module:
    """Load pretrained classifier from factory."""
    loaded_clf = factory.make_fcn('resnet50')
    loaded_clf.load_state_dict(torch.load(
        st.session_state.clf_path,
        map_location=torch.device('cpu')
    ))
    loaded_clf.eval()
    return loaded_clf


def load_input_transforms_from(factory: FcnFactory) -> tv_transforms.Transform:
    """Load the input transforms for the pretrained classifier
    from the factory."""
    input_transforms = [tv_transforms.Lambda(rgba_to_rgb)]
    input_transforms.append(factory.input_transforms)
    return tv_transforms.Compose(input_transforms)
