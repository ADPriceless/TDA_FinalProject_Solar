"""Segmentation app:
Allows user to upload an image and a mask, then makes a prediction,
compares it to the mask and calculates the IoU"""

from dataclasses import dataclass
from typing import Optional

from PIL import Image
import streamlit as st
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image, resize
import torchvision.transforms.v2 as tv_transforms
from torchvision.utils import draw_segmentation_masks

from models.fcn_factory import FcnFactory
from preprocess.transforms import rgba_to_rgb


@dataclass
class ImageModel:
    """Base class to store data for an image."""
    data: Optional[Image.Image]
    caption: str
    use_col_width: bool

    def __init__(
        self, caption: str, data: Image.Image = None, use_col_width=True
    ) -> None:
        self.caption = caption
        self.data = data
        self.use_col_width = use_col_width

    @property
    def mode(self):
        """Return the mode of the image e.g. RGB, L, etc."""
        if self.data:
            return self.data.mode
        return None


@dataclass
class UploadImageModel(ImageModel):
    """A class to store data for the images that the user uploads."""
    upload_msg: str
    supported_modes: list[str]
    supported_types: Optional[list[str]]

    def __init__(
        self, caption: str, upload_msg: str, supported_modes: list[str],
        supported_types: tuple[str] = None, data: Image.Image = None, use_col_width=True
    ) -> None:
        super().__init__(caption, data, use_col_width)
        self.upload_msg = upload_msg
        self.supported_modes = supported_modes
        if supported_types is None:
            self.supported_types = ('png', 'jpg', 'jpeg')
        else:
            self.supported_types = supported_types

    @property
    def mode_supported(self) -> bool:
        """Returns true if the """
        if self.data is None:
            return False
        return self.data.mode in self.supported_modes

    def convert_to(self, mode: str):
        """Convert the image to specified mode."""
        if self.data is not None:
            self.data = self.data.convert(mode)


@dataclass
class OverlayImageModel(ImageModel):
    """A class to store data for images where a mask is overlaid."""
    alpha: float
    colours: list[str]

    def __init__(
        self, caption: str, data: Image.Image = None, use_col_width=True,
        alpha=0.5, colours='yellow'
    ) -> None:
        super().__init__(caption, data, use_col_width)
        self.alpha = alpha
        self.colours = colours


@dataclass(init=True)
class AppModel:
    """Stores the data for the application."""
    # General
    title: str
    # Image
    upload_header: str
    input_image: UploadImageModel
    mask_image: UploadImageModel
    # Classifier
    solar_clf_path: str
    input_trans: list
    # Overlaid Images
    overlay_header: str
    pred_overlay: OverlayImageModel
    mask_overlay: OverlayImageModel
    # Scoring
    scoring_header: str
    iou_overlay: OverlayImageModel


def make_default_app_model() -> AppModel:
    """Initialises the AppModel. Edit values to alter app properties."""
    return AppModel(
        # General
        title='Solar Panel Segmentation',
        # Image
        upload_header='Upload an Image and/or Mask',
        input_image = UploadImageModel(
            caption='Uploaded image.',
            upload_msg='Upload an image to input to the classifier.',
            supported_modes=['RGB', 'RGBA', 'P']
        ),
        mask_image=UploadImageModel(
            caption='Uploaded mask.',
            upload_msg='Upload a mask.',
            supported_modes=['L']
        ),
        # Classifier
        solar_clf_path='saved_models/model_20240328_074804_4',
        input_trans=[tv_transforms.Lambda(rgba_to_rgb)],
        # Overlaid Images
        overlay_header='Prediction vs. True Mask',
        pred_overlay=OverlayImageModel('Prediction'),
        mask_overlay=OverlayImageModel('True Mask'),
        # Scoring
        scoring_header='IoU Score: ',
        iou_overlay=OverlayImageModel(
            caption='IoU: Yelow = Intersection, Purple = Union',
            colours=['yellow', 'purple']
        ),
    )


class View:
    """A class to handle displaying items on the page."""
    def __init__(self) -> None:
        # __init__ defines page layout
        pass

    def show_title(self, title: str) -> None:
        st.title(title)

    def setup_columns(self):
        return st.columns(2)

    def show_header(self, header: str) -> None:
        st.header(header)

    def get_input_image(self, message: str, supported_types: list[str]):
        return st.file_uploader(message, type=supported_types)

    def show_image(self, img: ImageModel):
        st.image(
            img.data,
            caption=img.caption,
            use_column_width=img.use_col_width
        )

    def show_mode_not_supported_warning(self, mode: str):
        st.warning(f'Image mode "{mode}" not supported as model input')

    def warning(self, msg):
        st.warning(msg)


class UploadController:
    """Handles logic related to uploading and displaying images."""
    def __init__(self, view: View, model: AppModel) -> None:
        self.model = model
        self.view = view

    def upload_show_input_img(self):
        """Upload and show the input image."""
        self._upload_photo_to(self.model.input_image)
        if not self.model.input_image.mode_supported:
            self.view.show_mode_not_supported_warning(self.model.input_image.mode)
            return
        # Always convert to RGB since colour mode 'P' is
        # ok for displaying but will not be accepted by
        # the classifier
        self.model.input_image.convert_to('RGB')
        self.view.show_image(self.model.input_image)

    def _upload_photo_to(self, input_img: UploadImageModel) -> None:
        uploaded_file = self.view.get_input_image(input_img.upload_msg, input_img.supported_types)
        if uploaded_file is not None:
            input_img.data = Image.open(uploaded_file)
        else:
            input_img.data = None

    def upload_show_mask_img(self):
        """Upload and show the mask."""
        self._upload_photo_to(self.model.mask_image)
        if not self.model.mask_image.mode_supported:
            self.view.show_mode_not_supported_warning(self.model.mask_image.mode)
            return
        self.view.show_image(self.model.mask_image)


class ClfController:
    """A class to coordinate interactions between the classifier and 
    view."""
    def __init__(self, view: View, model: AppModel) -> None:
        self.view = view
        self.model = model
        self.factory = None
        self.classifier = None
        self.prediction = None
        self._load_solar_clf()
        self._load_input_transforms()

    def _load_solar_clf(self) -> None:
        self.factory = FcnFactory(n_classes=2)
        loaded_clf = self.factory.make_fcn('resnet50')
        loaded_clf.load_state_dict(torch.load(
            self.model.solar_clf_path, map_location=torch.device('cpu')
        ))
        loaded_clf.eval()
        self.classifier = loaded_clf

    def _load_input_transforms(self) -> None:
        self.model.input_trans.append(self.factory.input_transforms)
        self.model.input_trans = tv_transforms.Compose(self.model.input_trans)

    def show_prediction(self):
        """Displays the classifier's prediction overlaid onto the input image."""
        self._predict()
        self._make_segmentation(self.model.pred_overlay, self._format_prediction())
        self.view.show_image(self.model.pred_overlay)

    def _predict(self) -> torch.Tensor:
        """Predict the segmentation mask of an image."""
        processed_img = self.model.input_trans(to_tensor(self.model.input_image.data))
        # Expand to shape [batch_size, n_classes, h, w] required for model
        processed_img = processed_img.expand([1, *processed_img.shape])
        # The output contains two classes: background (class 0) and solar panels
        # (class 1). Extract solar panels the class.
        output = self.classifier(processed_img)['out']
        self.prediction = torch.softmax(output, dim=1) > 0.5

    def _make_segmentation(self, img: OverlayImageModel, mask: torch.Tensor) -> None:
        """Display the segmentation mask (predicted or true) of an image"""
        # Overlay mask onto image
        processed_img = (to_tensor(self.model.input_image.data) * 255).type(torch.uint8)
        segmented = draw_segmentation_masks(
            processed_img, mask, alpha=img.alpha, colors=img.colours
        )
        img.data = to_pil_image(segmented)

    def _format_prediction(self) -> None:
        """Format the prediction to be of shape [h, w] and chosen size"""
        # PIL is WxH, where as torch is HxW
        to_size = self.model.input_image.data.size[1], self.model.input_image.data.size[0]
        return resize(self.prediction, to_size)[0, 1]

    def show_mask(self):
        """Show the mask overlaid on the input image."""
        self._make_segmentation(self.model.mask_overlay, self._format_true_mask())
        self.view.show_image(self.model.mask_overlay)

    def _format_true_mask(self) -> torch.Tensor:
        return to_tensor(self.model.mask_image.data) > 0.

    def calc_iou(self) -> float:
        """Calculate the IoU for a single sample"""
        pred_mask = self._format_prediction()
        true_mask = self._format_true_mask()
        intersection = true_mask.logical_and(pred_mask).sum()
        union = true_mask.logical_or(pred_mask).sum()
        # Check if union is zeros.
        if union == 0.:
            # This means that the prediction and output
            # are both all zeros.
            return 1.
        return intersection / union

    def show_iou(self):
        """Show the IoU overlaid on the input image."""
        pred_mask  = self._format_prediction()
        true_mask = self._format_true_mask()
        intersection = true_mask.logical_and(pred_mask)
        union = true_mask.logical_or(pred_mask)
        # Remove union mask where intersection is
        union = union.logical_xor(intersection)
        masks = torch.concat((intersection, union), dim=0)
        self._make_segmentation(self.model.iou_overlay, masks)
        self.view.show_image(self.model.iou_overlay)


class MainController:
    """Class to coordinate interactions between the other controllers,
    the model and the view."""
    def __init__(self, view: View, model: AppModel) -> None:
        self.view = view
        self.model = model
        self.upload_controller = UploadController(view, model)
        self.clf_controller = ClfController(view, model)

    def run(self):
        """The overarching logic to run the application."""
        self.view.show_title(self.model.title)
        self.view.show_header(self.model.upload_header)
        left, right = self.view.setup_columns()
        with left:
            self.upload_controller.upload_show_input_img()
        with right:
            self.upload_controller.upload_show_mask_img()
        self.view.show_header(self.model.overlay_header)
        left, right = self.view.setup_columns()
        if not self._img_valid():
            return
        with left:
            self.clf_controller.show_prediction()
        if not (self._img_mask_valid() and self._img_mask_same_size()):
            return
        with right:
            self.clf_controller.show_mask()
        iou = self.clf_controller.calc_iou()
        self.view.show_header(self.model.scoring_header + f'{iou:.2f}')
        self.clf_controller.show_iou()

    def _img_valid(self) -> bool:
        return self.model.input_image.mode_supported

    def _img_mask_valid(self) -> bool:
        if self._img_valid() and self._mask_valid():
            return True
        self.view.warning('Image or mask not valid')
        return False

    def _mask_valid(self) -> bool:
        return self.model.mask_image.mode_supported

    def _img_mask_same_size(self) -> bool:
        if self.model.input_image.data.size == self.model.mask_image.data.size:
            return True
        self.view.warning('Image and mask must have same width and height')
        return False


def main():
    """Displays the solar panel segmentation app"""
    app_model = make_default_app_model()
    app_view = View()
    controller = MainController(app_view, app_model)
    controller.run()


if __name__ == "__main__":
    main()
