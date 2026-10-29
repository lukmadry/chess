from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small, resnet50, ResNet50_Weights


def create_input_transform(input_size):
	_input_transform = transforms.Compose(
					[transforms.ToTensor(),
					 transforms.Resize((input_size, input_size), antialias=True),
					 transforms.Normalize(
						mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
					 ])
	return _input_transform


class PretrainedSmallChessBoardRegressor(nn.Module):
	def __init__(self, output_dim=8, train_backbone=False):
		super().__init__()
		weights = MobileNet_V3_Small_Weights.DEFAULT
		self.backbone = mobilenet_v3_small(weights=weights).features
		self.train_backbone = train_backbone
		if not train_backbone:
			for param in self.backbone.parameters():
				param.requires_grad = False

		in_features = self.backbone[-1].out_channels
		self.pool = nn.AdaptiveAvgPool2d((1,1))

		self.head = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features, 256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(256, output_dim)
			)

	def train(self, mode=True):
		super().train(mode)
		if not self.train_backbone:
			self.backbone.eval()

	def forward(self, x, with_preprocessing=False):
		if with_preprocessing:
			raise NotImplementedError

		features = self.backbone(x)
		features = self.pool(features)
		return self.head(features)


class PretrainedChessboardRegressor(nn.Module):
	def __init__(self, output_dim=8, train_backbone=False):
		super().__init__()
		self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
		self.train_backbone = train_backbone
		if not train_backbone:
			for param in self.backbone.parameters():
				param.requires_grad = False
		num_features = self.backbone.fc.in_features

		self.backbone.fc = nn.Identity() # what is this ?

		self.head = nn.Sequential(
			nn.Linear(num_features, 256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(256, output_dim)
			)

	def forward(self, x):
		features = self.backbone(x)
		return self.head(features)

	def train(self, mode=True):
		super().train(mode)
		if not self.train_backbone:
			self.backbone.eval()
