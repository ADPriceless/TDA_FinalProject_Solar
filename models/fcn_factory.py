"""Custom head on ResNet-50 for 2 class (`__background__` and `solar_panel`)
pixel-wise classification."""


from torch import nn
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import fcn_resnet50, FCN_ResNet50_Weights


class FcnFactory:
    """Factory class to create Fully Convolutional Network (FCN) models.
    Call `make_fcn` to create a model."""
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.weights = None
        self.backbone = None
        self.classifier = None

    def make_fcn(self, name: str) -> _SimpleSegmentationModel:
        """Creates an FCN model specified by `name`, where `name` can be:
        `resnet50`."""
        if name.lower() == 'resnet50':
            self.weights = FCN_ResNet50_Weights
            self.backbone = fcn_resnet50(weights=self.weights.DEFAULT).backbone
            self.classifier = self._resnet50_classifier()
        else:
            raise NotImplementedError(f'FCN model {name} not implemented')
        self._freeze_backbone()
        return _SimpleSegmentationModel(self.backbone, self.classifier)

    def _resnet50_classifier(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), # uses defaults
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, self.n_classes, kernel_size=3, stride=1),
        )

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_input_transforms(self):
        """Returns the input transforms for the model that was 
        last created."""
        return self.weights.transforms
