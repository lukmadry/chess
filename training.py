import time
import numpy as np
import os
import tempfile
import argparse
import glob
import json
import torch
import argparse
import datetime
from itertools import chain
from random import choices, sample
from functools import partial
from copy import deepcopy

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision.models import resnet34, ResNet34_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms

from helpers import display_grid_with_labels, convert_boxes_to_cas, cas_to_boxes, model_path_from_date, canonical_corner_order
from augmentation import final_image_shifts_dict

# TODO move some stuff to a separate file, loss functions to losses.py, training and testing loops to loops.py or sth
# TODO you unflatten and then flatten ground truths, this is very redundant, fix it
# TODO the same way you diagnosis error in pixels in test loop, add diagnostics of angles

def create_input_transform(input_size):
	_input_transform = transforms.Compose(
					[transforms.ToTensor(),
					 transforms.Resize((input_size, input_size), antialias=True),
					 transforms.Normalize(
						mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
					 ])
	return _input_transform


class AnisothropicWingLoss(nn.Module):
	def __init__(self, w_pixels=25, epsilon_pixels=2, orig_size=(640, 480)):
		super(AnisothropicWingLoss, self).__init__()
		self.w_pixels = w_pixels
		self.epsilon_pixels = epsilon_pixels

		width, height = orig_size
		self.register_buffer('scale', torch.tensor([width, height] * 4))
		self.c = w_pixels - w_pixels * torch.log(torch.tensor(1.0 + w_pixels / epsilon_pixels))

	def forward(self, pred, target):
		delta_pixels = (pred - target).abs() * self.scale
		
		small_mask = delta_pixels < self.w_pixels
		
		loss_small = self.w_pixels * torch.log(1.0 + delta_pixels[small_mask] / self.epsilon_pixels)
		loss_large = delta_pixels[~small_mask] - self.c
		
		loss = torch.empty_like(delta_pixels)
		loss[small_mask] = loss_small
		loss[~small_mask] = loss_large
		return loss.mean()

# create two classes that will have different behaviour of __getitem__ ?
class ChessDataset(Dataset):

	def __init__(self, images_dir, labels_dir=None, size=None, with_bright=False, filter_labels_params=False, 
		allowed_dist=None, convert_to_cas=False, normalised_labels=False, input_size=224, adaptive_size=False,
		height=480., width=640., max_angle=20., verify_angle=False, keep_orig_bb=False, return_normalised_image=True):
		"""
		if adaptive_size = True and normalised_labels = True, then we will load unnormalised labels in the initialisation phase and normalise them later

		"""
		self.images_dir = images_dir
		if labels_dir is None:
			labels_dir = os.path.join(images_dir, "ground_truths")
		self.labels_dir = labels_dir
		self.convert_to_cas = convert_to_cas
		self.return_normalised_image = return_normalised_image
		max_angle_rad = np.deg2rad(max_angle)

		if not isinstance(self.images_dir, str):
			raise ValueError("images_dir has to be a string. If you want to pass a list to create Dataset from different directories, use Subset from Torch")

		if adaptive_size and convert_to_cas:
			raise NotImplementedError("adaptive_size and convert_to_cas both set to True are not possible yet")

		if with_bright:
			self.images_list = glob.glob(os.path.join(self.images_dir, "*.jpg"))
		else:
			self.images_list = glob.glob(os.path.join(self.images_dir, "*.jpg"))
			self.images_list = [x for x in self.images_list if "bright" not in x]

		self.images_list = [x for x in self.images_list if os.path.isfile(x)] # excluding directories
		self.images_list = self._filter_images_with_no_labels()
		print(f"After filtering we have {len(self.images_list)} images")
		self.labels = []
		self.height = height
		self.width = width
		self.adaptive_size = adaptive_size # in this case we will always normalise to [0,1], but not here, only later in __getitem__
		self.normalised_labels = normalised_labels

		if adaptive_size:
			normalised_labels = False
			self.normalised_labels = True # weird and counterintuitive, TODO to be cleaned up

		if not convert_to_cas and normalised_labels:
			for l in self.images_list:
				local_lab = []
				labels_dict = self._load_labels_dict(l)

				for (x, y) in labels_dict["inner_board"]:
					local_lab.append([float(x)/height, float(y)/width])#  ([[float(x)/480., float(y)/640.] for (x,y) in labels_dict["inner_board"]])
				self.labels.append(local_lab)
		elif not convert_to_cas and not normalised_labels: 
			for l in self.images_list:
				local_lab = []
				labels_dict = self._load_labels_dict(l)

				for (x,y) in labels_dict["inner_board"]:
					local_lab.append([float(x), float(y)])
				self.labels.append(local_lab)
		elif convert_to_cas and not keep_orig_bb: # so converting to cas, TODO modify behaviour to account for flag keep_orig_bb
			for l in self.images_list:
				local_lab = []
				labels_dict = self._load_labels_dict(l)
				self.labels.append(
					convert_boxes_to_cas(
						labels_dict["inner_board"], 
						normalised_labels=normalised_labels, 
						height=height, 
						width=width,
						max_angle_rad=max_angle_rad))
		elif convert_to_cas and keep_orig_bb:
			self._orig_bb = []
			for l in self.images_list:
				local_lab = []
				labels_dict = self._load_labels_dict(l)
				# print(labels_dict)
				self.labels.append(
					convert_boxes_to_cas(
						labels_dict["inner_board"], 
						normalised_labels=normalised_labels, 
						height=height, 
						width=width,
						max_angle_rad=max_angle_rad))
				self._orig_bb.append(labels_dict['inner_board'])
		else:
			raise NotImplementedError

		if convert_to_cas and verify_angle and normalised_labels: # to make sure that we do not have angles outside of the prescribed range
			for l in self.labels:
				if abs(l[-1]) > 1 + 1e-4:
					raise ValueError("angle outside of proposed range") 
		elif convert_to_cas and verify_angle:
			for l in self.labels:
				if abs(l[-1]) > max_angle_rad + 1e-1: # a bit of leniency is accepted
					raise ValueError("angle outside of proposed range")  

		self._input_transform = create_input_transform(input_size) 

		if not len(self.labels) == len(self.images_list):
			raise ValueError(f"Size of image list and labels set are not the same, we have respectively {len(self.images_list)} and {len(self.labels)}")
		print("Succesfully loaded image list and labels")
		print(f"Loaded {len(self.images_list)} datapoints")

		if size is not None:
			self.images_list = self.images_list[:size]
			self.labels = self.labels[:size]

		if not convert_to_cas:
			self._unify_orientation_of_labels()

	def __len__(self):
		return len(self.images_list)

	def __getitem__(self, idx):
		# rename to synth get item ? either case you'll need to branch it
		im = Image.open(self.images_list[idx])
		_lab = self.labels[idx]
		if self.adaptive_size:
			width, height = im.size
			lab = []
			for (x, y) in _lab:
				lab.append([float(x)/height, float(y)/width])
		else:
			lab = _lab
		lab = np.array(lab).flatten()
		# lab = canonical_corner_order(lab)
		lab = torch.tensor(lab, dtype=torch.float32)

		if self.return_normalised_image:
			im = self._input_transform(im)

		# print(f"[ChessDataset {id(self)} | dir={self.images_dir}] returning im type: {type(im)}")
		# print("type of image returned by getitem")
		# print(type(im), type(lab))
		return im, lab

	def _filter_images_with_no_labels(self):
		new_images = []
		for f in self.images_list:
			filename = f.split("/")[-1].replace("_bright", "") # ?the last one not necessary?
			filename = os.path.join(self.labels_dir, filename.replace("jpg", "json"))
			if os.path.exists(filename):
				new_images.append(f)
		return new_images

	def _unify_orientation_of_labels(self):

		if self.convert_to_cas:
			raise ValueError("unifying orientation not applicable if targets are in CAS format")

		new_labels = []
		for lab in self.labels:
			_lab = canonical_corner_order(lab)
			new_labels.append(_lab)
		self.labels = new_labels

	def _load_labels_dict(self, image_fname):
		filename = image_fname.split("/")[-1].replace("_bright", "") # ?the last one not necessary?
		filename = os.path.join(self.labels_dir, filename.replace("jpg", "json"))
		with open(filename, "r") as f:
			labels_dict = json.load(f)
		return labels_dict

	def sample_raw_images(self, k, indices=None, normalized_targets=True, sample_orig_bb=True, impose_size=None):
		if indices is None:
			imgs_idx = choices(np.arange(len(self)), k=k)
		else:
			imgs_idx = choices(indices, k=k)
		images = []
		labels = []

		for idx in imgs_idx:
			im = Image.open(self.images_list[idx])
			old_h, old_w = im.size[:2]
			if impose_size is not None:
				height, width = impose_size
				im = im.resize((height, width))
			# im = np.array(im, dtype=np.uint8)
			if (self.convert_to_cas and not sample_orig_bb) or (not self.convert_to_cas):
				lab = self.labels[idx]
				if not normalized_targets and not self.adaptive_size: # will have to be adapted to adaptive size
					lab = [[int(self.height*x), int(self.width*y)] for x, y in lab]
				elif not normalized_targets and self.adaptive_size:
					if impose_size is not None:
						# height, width = impose_size
						lab = [[int(height*(x/float(old_h))), int(width*(y/float(old_w)))] for x, y in lab]
					else:
						lab = [[int(x), int(y)] for x, y in lab]
					# else: # we do not need to rescale anything because adaptive size already gives us unscaled 
						# height, width = im.size
						# lab = [[int(height*x), int(width*y)] for x, y in lab] # this is for drawing labels
				# lab = np.array(lab).flatten()
			elif (self.convert_to_cas and sample_orig_bb):
				lab = self._orig_bb[idx] # note: in this scenario they are not normalised
			
			images.append(im)
			labels.append(lab)

		# images = np.array(images)
		return images, labels

class CleanChessDatasetWrapper(Dataset):
	def __init__(self, subset, input_size=224, normalise=True, swap_labels=True):
		self.subset = subset
		self._swap = swap_labels
		self._input_transform = create_input_transform(input_size) if normalise else None

	def __len__(self):
		return len(self.subset)

	@property
	def indices(self):
		return self.subset.indices

	def __getitem__(self, idx):
		img, label = self.subset[idx]
		if self._input_transform is not None:
			img = self._input_transform(img)

		return img, label

class AugmentationChessDatasetWrapper(Dataset):
	def __init__(self, subset, transformation_list=None, 
		augmentation_factor=5, subsample_transforms=True, transformation_params=None, input_size=224):
		self.subset = subset # maybe make a distinction between Subset and Dataset ?

		if transformation_params is None:
			self.transformation_params = {k: {} for k in final_image_shifts_dict.keys()}
		else:
			self.transformation_params = transformation_params

		if transformation_list is None:
			transformation_list = list(final_image_shifts_dict.keys())

		self.tf_funcs = [final_image_shifts_dict[tr] for tr in transformation_list]
		self.tf_kwargs = [self.transformation_params[tr] for tr in transformation_list]
		self.num_tfs = len(transformation_list)

		self.augmentation_factor = augmentation_factor
		self.subsample_transforms = subsample_transforms

		self._input_transform = create_input_transform(input_size)

	def __len__(self):
		return len(self.subset) * self.augmentation_factor

	def __getitem__(self, idx):
		orig_idx = idx % len(self.subset)
		img, label = self.subset[orig_idx] # actually no

		if self.subsample_transforms and self.num_tfs > 0:
			# Pick k random transformations for this batch
			k = np.random.randint(0, self.num_tfs + 1)
			# Use zip to pair selected functions with their kwargs
			# (Note: use 'sample' if you want unique transforms, 'choices' if duplicates are allowed)
			active_tfs = sample(list(zip(self.tf_funcs, self.tf_kwargs)), k)
		else:
			active_tfs = zip(self.tf_funcs, self.tf_kwargs)

		if isinstance(img, Image.Image):
			img_np = np.array(img)
		else:
			img_np = img

		for func, kwargs in active_tfs:
			# print(f"applying function: {func.__name__}")
			# print(type(img))
			img_np = func(img_np, **kwargs)

		if isinstance(img_np, np.ndarray):
			img = Image.fromarray(img_np)
		else:
			img = img_np

		img = self._input_transform(img)
		return img, label


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
	def __init__(self, output_dim=8):
		super().__init__()
		self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
		num_features = self.backbone.fc.in_features

		self.backbone.fc = nn.Identity() # what is this ?

		self.head = nn.Sequential(
			nn.Linear(num_features, 256),
			nn.ReLu(),
			nn.Dropout(0.1),
			nn.Linear(256, output_dim)
			)

		def forward(self, x):
			features = self.backbone(x)
			return self.head(features)

def angle_penalty_loss(pred, gt, eps=1e-5):
	"""
	pred - torch tensor of shape (B, 8)
	gt - torch tensor of shape (B, 4, 2)

	todo : can be made more concise
	"""

	# square angles
	pred = pred.reshape(-1, 4, 2)
	gt = gt.reshape(-1, 4, 2)
	edges = torch.roll(pred, shifts=-1, dims=1) - pred
	norm_edges = edges / (torch.norm(edges, dim=2, keepdim=True) + eps)
	square_angles = torch.square((norm_edges * torch.roll(norm_edges, shifts=1, dims=1)).sum(dim=-1)).mean()

	# angle alignement
	gt_edges = torch.roll(gt, shifts=-1, dims=1) - gt
	gt_edges = gt_edges / (torch.norm(gt_edges, dim=2, keepdim=True) + eps)
	cos_sim = (1- (norm_edges * gt_edges).sum(dim=-1)).mean()

	return square_angles, cos_sim

def mse_loss_with_penalty(pred, gt, penalty_weight=0.5):
	square_penalty, cos_penalty = angle_penalty_loss(pred, gt)
	return nn.MSELoss(pred, gt) + penalty_weight*square_penalty + penalty_weight*cos_penalty


def _update_stats_of_gradients(model, grad_dict):
	for name, param in model.named_parameters():
		if param.requires_grad and param.grad is not None:
			grad_mean = param.grad.abs().mean().item()
			grad_dict[name]['mean'].append(grad_mean)

			grad_std = param.grad.abs().std().item()
			grad_dict[name]['std'].append(grad_std)
			
			grad_max = param.grad.abs().max().item()
			grad_dict[name]['max'].append(grad_max)
			
			grad_min = param.grad.abs().min().item()
			grad_dict[name]['min'].append(grad_min)

	return grad_dict

def _print_stats_of_gradient(grad_dict):
	for name in list(grad_dict.keys()):
		mean_mean_grad = np.mean(grad_dict[name]['mean'])
		max_mean_grad = np.max(grad_dict[name]['mean'])
		max_max_grad = np.max(grad_dict[name]['max'])

		print(f"Layer: {name}\nMean grad: {mean_mean_grad:.5f}\nMax value: {max_max_grad:.5f}")

# copied from PyTorch website, for my purposes probably sufficient
# these two can be probably merged somehow but I think it would decrease readability if done hastily
def train_loop(dataloader, model, base_loss_fn, optimizer, angle_penalty_weight=0.5, batch_size=32):
	size = len(dataloader.dataset)
	model.train()

	grad_dict = {}
	for name, param in model.named_parameters():
		if param.requires_grad:
			grad_dict[name] = {}
			for stat in ['mean', 'std', 'max', 'min']:
				grad_dict[name][stat] = []

	for batch, (X, y) in enumerate(dataloader):
		# Compute prediction and loss
		pred = model(X)
		mse_loss = base_loss_fn(pred, y)
		square_angles, cos_sim = angle_penalty_loss(pred, y)
		loss = (mse_loss + angle_penalty_weight * (square_angles + cos_sim))
		# Backpropagation
		loss.backward()
		_update_stats_of_gradients(model, grad_dict)
		optimizer.step()
		optimizer.zero_grad() # resetting gradients

		if batch % 5 == 0:
			loss, current = loss.item(), batch * batch_size + len(X)
			print(f"MSE loss: {mse_loss.item():>7f}  [{current:>5d}/{size:>5d}]")
			print(f"Square angle loss: {square_angles.item():>7f}\nAngle similarity loss: {cos_sim.item():>7f}")

	_print_stats_of_gradient(grad_dict)
	return grad_dict

def train_loop_cas(dataloader, model, base_loss_fn, optimizer, batch_size=32, angle_penalty_weight=1.):
	size = len(dataloader.dataset)
	model.train()

	grad_dict = {}
	for name, param in model.named_parameters():
		if param.requires_grad:
			grad_dict[name] = {}
			for stat in ['mean', 'std', 'max', 'min']:
				grad_dict[name][stat] = []

	for batch, (X, y) in enumerate(dataloader):
		pred = model(X)
		mse_loss_center = base_loss_fn(pred[:, :2], y[:, :2])
		mse_loss_size = base_loss_fn(pred[:, 2], y[:, 2])
		mse_loss_angle = base_loss_fn(pred[:, 3], y[:, 3])
		loss = mse_loss_center + mse_loss_size + angle_penalty_weight * mse_loss_angle
		loss.backward()
		_update_stats_of_gradients(model, grad_dict)
		optimizer.step()
		optimizer.zero_grad()

		if batch % 5 == 0:
			loss, current = loss.item(), batch * batch_size + len(X)
			print(f"Total MSE loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")
			print(f"Center loss: {mse_loss_center.item():>7f}\nSize loss: {mse_loss_size.item():>7f}\nAngle loss: {mse_loss_angle.item():>7f}")

	_print_stats_of_gradient(grad_dict)
	return grad_dict

# to be rewritten, or at least API of sample_raw_images will have to be
def _image_dumper(model, dataloader, grid, model_for_cas, fig_dir, **kwargs):
	val_indices = dataloader.dataset.indices
	if hasattr(dataloader.dataset, 'dataset'):
		orig_dataset = dataloader.dataset.dataset
	else:
		orig_dataset = dataloader.dataset.subset.dataset

	display_images_grid, target_grid = orig_dataset.sample_raw_images(int(grid**2), 
		indices=val_indices, 
		normalized_targets=False, 
		sample_orig_bb=True, 
		impose_size=None)
	image_for_model = [orig_dataset._input_transform(x) for x in display_images_grid]
	image_for_model = torch.stack(image_for_model)
	display_grid_with_labels(
			model=model, 
			display_images=display_images_grid, 
			image_array=image_for_model, 
			gt_list=target_grid, 
			image_grid_size=grid, 
			show=False, 
			savefig=True, 
			fig_dir=fig_dir,
			model_for_cas=model_for_cas)

# same caveat about merging here
# change mentions of mse in case of wing
def test_loop(dataloader, model, base_loss_fn, angle_penalty_weight=0.5, dump_image=True, grid=3, dump_image_path=None):
	model.eval()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss = 0
	test_mse_loss = 0
	test_square_angle_loss = 0
	test_cos_sim_loss = 0

	pixel_error_axis_one = 0
	pixel_error_axis_two = 0

	# Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
	# also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
	min_max_pred = [np.inf, 0]
	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			nr_it = pred.shape[0]
			mse_loss = base_loss_fn(pred, y)

			square_angles, cos_sim = angle_penalty_loss(pred, y)
			loss = mse_loss + angle_penalty_weight * (square_angles + cos_sim)

			test_loss += loss.item()
			test_mse_loss += mse_loss.item()
			test_square_angle_loss += square_angles.item()
			test_cos_sim_loss += cos_sim.item()
			axis_one_abs_err = torch.abs(pred[:, [0,2,4,6]] * 480 - y[:, [0,2,4,6]] * 480).sum().item()
			axis_two_abs_err = torch.abs(pred[:, [1,3,5,7]] * 640 - y[:, [1,3,5,7]] * 640).sum().item()
			pixel_error_axis_one += axis_one_abs_err
			pixel_error_axis_two += axis_two_abs_err

			if pred.max() > min_max_pred[1]:
				min_max_pred[1] = pred.max()

			if pred.min() < min_max_pred[0]:
				min_max_pred[0] = pred.min()

	test_loss /= num_batches
	test_mse_loss /= num_batches
	test_square_angle_loss /= num_batches
	test_cos_sim_loss /= num_batches
	pixel_error_axis_one /= (4 * size)
	pixel_error_axis_two /= (4 * size)
	print(f"Test Error: \nAvg pixel error height: {(pixel_error_axis_one):>0.1f}\nAvg pixel error width {(pixel_error_axis_two):>0.1f}")
	print(f"Average loss: {test_loss:>7f}\nAverage MSE loss: {test_mse_loss:>7f}")
	print(f"\nAverage square angle loss: {test_square_angle_loss:>7f}\nAverage cos sim loss: {test_cos_sim_loss:>7f}")
	print(f"Predictions are in the range of {min_max_pred}")

	if dump_image:
		_image_dumper(
			dataloader=dataloader, 
			grid=grid, 
			model_for_cas=False,
			model=model, 
			image_grid_size=grid, 
			show=False, 
			savefig=True, 
			fig_dir=dump_image_path)

	return {'mse_loss': test_mse_loss, 'square_angles_loss': test_square_angle_loss, 'cos_sim_loss': test_cos_sim_loss}, pixel_error_axis_one, pixel_error_axis_two

def test_loop_cas(dataloader, model, base_loss_fn, angle_penalty_weight=0.5, dump_image=True, grid=3, dump_image_path=None):
	model.eval()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)

	test_loss = 0
	test_center_loss = 0
	test_size_loss = 0
	test_angle_loss = 0

	pixel_error_axis_one = 0
	pixel_error_axis_two = 0

	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			nr_it = pred.shape[0]

			mse_loss_center = base_loss_fn(pred[:, :2], y[:, :2])
			mse_loss_size = base_loss_fn(pred[:, 2], y[:, 2])
			mse_loss_angle = base_loss_fn(pred[:, 3], y[:, 3])
			test_center_loss += mse_loss_center.item()
			test_size_loss += mse_loss_size.item()
			test_angle_loss += mse_loss_angle.item()

			pred_bb = torch.tensor(cas_to_boxes(pred))
			gt_bb = torch.tensor(cas_to_boxes(y))

			axis_one_abs_err = torch.abs(pred_bb[:, :, 0] - gt_bb[:, :, 0]).sum().item()
			axis_two_abs_err = torch.abs(pred_bb[:, :, 1] - gt_bb[:, :, 1]).sum().item()

			pixel_error_axis_one += axis_one_abs_err
			pixel_error_axis_two += axis_two_abs_err

	test_loss /= num_batches
	test_center_loss /= num_batches
	test_size_loss /= num_batches
	test_angle_loss /= num_batches
	pixel_error_axis_one /= (4 * size)
	pixel_error_axis_two /= (4 * size)

	print(f"Test loss: {test_loss:>7f}\nCenter loss: {test_center_loss:>7f}\nSize loss: {test_size_loss:>7f}\nAngle loss: {test_angle_loss:>7f}")
	print(f"Pixel error height: {pixel_error_axis_one:>7f}\nPixel error width: {pixel_error_axis_two:>7f}")

	if dump_image:
		_image_dumper(
			dataloader=dataloader, 
			grid=grid, 
			model_for_cas=True,
			model=model, 
			image_grid_size=grid, 
			show=False, 
			savefig=True, 
			fig_dir=dump_image_path)

	return {'center_loss': test_center_loss, 'size_loss': test_size_loss, 'angle_loss': test_angle_loss}, pixel_error_axis_one, pixel_error_axis_two

if __name__ == "__main__":
	# TODO split arguments into categories to make this wall of text more readable and easier to navigate
	parser = argparse.ArgumentParser()
	parser.add_argument("--synth_input_dir", type=str, nargs='+') # this will not be augmented as it is already augmented during image generation
	parser.add_argument("--auth_input_dir", type=str, nargs='+') # this will be augmented additionally. 
	parser.add_argument("--synth_labels_dir", type=str, nargs='+')
	parser.add_argument("--auth_labels_dir", type=str, nargs='+')

	parser.add_argument("--with_bright", action=argparse.BooleanOptionalAction)
	parser.add_argument("--train_size", type=float, default=0.75)
	parser.add_argument("--val_on_synth", action=argparse.BooleanOptionalAction)

	parser.add_argument("--model", type=str, default="mobile")
	parser.add_argument("--learning_rate", type=float, default=1e-3)
	parser.add_argument("--epochs", type=int, default=25)
	parser.add_argument("--train_backbone", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--size", type=int, help="mostly for testing")
	parser.add_argument("--angle_penalty", type=float, default=5)
	parser.add_argument("--input_size", default=224, type=int)
	parser.add_argument("--loss", type=str, default='mse')
	parser.add_argument("--cas", action=argparse.BooleanOptionalAction)

	parser.add_argument("--name", type=str)
	parser.add_argument("--load_from_checkpoint", action=argparse.BooleanOptionalAction)
	parser.add_argument("--last_model_date", type=str)
	parser.add_argument("--no_model", type=int)
	parser.add_argument("--epoch_to_restart", type=int, default=-1)

	parser.add_argument("--augmentation_factor", default=1, type=int)

	parser.add_argument("--test_run", action=argparse.BooleanOptionalAction)
	parser.add_argument("--test_run_marker", type=str)
	args = parser.parse_args()

	# TODO ADD A ROUTER FOR TRAINING
	if args.synth_input_dir is None and args.auth_input_dir is None:
		raise ValueError("you need to pass at least one of synth_input_dir or auth_input_dir")

	assert args.model in ["mobile", "resnet", "from_scratch"]
	if args.test_run and args.test_run_marker is None:
		raise ValueError("To do a test run you need to give test_run_marker, otherwise the script will not be able to check if it passed correctly.")

	if args.loss == 'wing' and args.cas:
		raise NotImplementedError("WingLoss and center-angle-size together not supported yet.")

	if args.cas:
		output_dim = 4
		train_loop_fn = train_loop_cas
		test_loop_fn = test_loop_cas
		print("Will train model with output structured as center, angle, size")
	else:
		output_dim = 8
		train_loop_fn = train_loop
		test_loop_fn = test_loop
		print("will train model with output structured as coordinates of a bounding box")

	if args.model == "mobile":
		regressor = PretrainedSmallChessBoardRegressor(output_dim=output_dim, train_backbone=args.train_backbone)
	elif args.model == "resnet":
		regressor = PretrainedChessboardRegressor(output_dim=output_dim, train_backbone=args.train_backbone)
	else:
		raise NotImplementedError

	synth_chess_dataset_kwargs = dict(
			with_bright=args.with_bright, 
			size=args.size,
			convert_to_cas=args.cas,
			normalised_labels=True,
			input_size=args.input_size,
			keep_orig_bb=True)

	auth_chess_dataset_kwargs = deepcopy(synth_chess_dataset_kwargs)
	auth_chess_dataset_kwargs['adaptive_size'] = True
	auth_chess_dataset_kwargs['return_normalised_image'] = False

	val_loaders = []

	synth_train_datasets = []
	synth_val_datasets = []
	generator = torch.Generator().manual_seed(421312)
	if args.synth_input_dir is not None:

		if args.synth_labels_dir is None:
			synth_ld = [None] * len(args.synth_input_dir)
		else:
			synth_ld = args.synth_labels_dir

		for input_dir, lab_dir in zip(args.synth_input_dir, synth_ld):
			synth_chess_dataset = ChessDataset(images_dir=input_dir, labels_dir=lab_dir, **synth_chess_dataset_kwargs)
			synth_loc_size = len(synth_chess_dataset)
			synth_loc_train_size = int(args.train_size * synth_loc_size)
			synth_loc_val_size = synth_loc_size - synth_loc_train_size
			synth_chess_train, synth_chess_val = random_split(synth_chess_dataset, [synth_loc_train_size, synth_loc_val_size], generator=generator)
			synth_train_datasets.append(synth_chess_train)
			synth_val_datasets.append(synth_chess_val)

		if len(synth_val_datasets) > 1:
			synth_val_ds = ConcatDataset(synth_val_datasets)
		else:
			synth_val_ds = synth_val_datasets[0]

		synth_val_loader = DataLoader(
			synth_val_ds, batch_size=32, shuffle=False, num_workers=1, persistent_workers=True)
		val_loaders.append(('synth', synth_val_loader))

	auth_train_datasets = []
	auth_val_datasets = []

	if args.auth_input_dir is not None:

		if args.auth_labels_dir is None:
			auth_ld = [None] * len(args.auth_input_dir)
		else:
			auth_ld = args.auth_labels_dir

		for input_dir, lab_dir in zip(args.auth_input_dir, auth_ld):
			auth_chess_dataset_pure = ChessDataset(images_dir=input_dir, labels_dir=lab_dir, **auth_chess_dataset_kwargs)
			auth_loc_size = len(auth_chess_dataset_pure)
			auth_loc_train_size = int(args.train_size * auth_loc_size)
			auth_loc_val_size = auth_loc_size - auth_loc_train_size
			auth_chess_train_split, auth_chess_val_split = random_split(auth_chess_dataset_pure, [auth_loc_train_size, auth_loc_val_size], generator=generator)
			auth_chess_dataset_pure_train = CleanChessDatasetWrapper(auth_chess_train_split, input_size=args.input_size)
			auth_chess_dataset_pure_val = CleanChessDatasetWrapper(auth_chess_val_split, input_size=args.input_size)

			auth_chess_dataset_aug_train = AugmentationChessDatasetWrapper(auth_chess_train_split, 
				augmentation_factor=args.augmentation_factor, 
				input_size=args.input_size)
			auth_train_datasets.extend([auth_chess_dataset_pure_train, auth_chess_dataset_aug_train])

			auth_val_datasets.append(auth_chess_dataset_pure_val)

		if len(auth_val_datasets) > 1:
			auth_val_ds = ConcatDataset(auth_val_datasets)
		else:
			auth_val_ds = auth_val_datasets[0]

		auth_val_loader = DataLoader(
			auth_val_ds, batch_size=32, shuffle=False, num_workers=1, persistent_workers=True)
		val_loaders.append(('real', auth_val_loader))
		
	train_dataset = ConcatDataset(synth_train_datasets + auth_train_datasets)
	train_loader = DataLoader(
		train_dataset, batch_size=32, shuffle=True, num_workers=1, persistent_workers=True)

	# ideally this part won't be touched at all after successful split
	# at worst you can do a separate test loop on a synth dataset and authentic dataset
	if args.loss == 'mse':
		loss_fn = nn.MSELoss()
	elif args.loss == 'wing':
		loss_fn = AnisothropicWingLoss()
	else:
		raise ValueError("Loss has to be MSE or WingLoss, i.e. mse or wing")
	trainable_params = [p for p in regressor.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

	# os.makedirs("checkpoints", exist_ok=True)
	# TODO clean up the ifs below, it's a mess

	if not args.load_from_checkpoint:
		chckpt_path = os.path.join("checkpoints", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
		os.makedirs(chckpt_path, exist_ok=True)
	else:
		model_chckpt_path = model_path_from_date(args.last_model_date, args.epoch_to_restart, args.no_model)
		chckpt_path = os.path.dirname(model_chckpt_path)
		print("Will load the following model: ", chckpt_path)
		checkpoint = torch.load(model_chckpt_path, weights_only=True) # this to be modified !
		regressor.load_state_dict(checkpoint["model_state_dict"])

		# if "optimizer_state_dict" in checkpoint:
		# 	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

		print("Succesfully restored checkpoint: {}".format(chckpt_path))
		start_epoch = checkpoint["epoch"]
	# os.makedirs(chckpt_path, exist_ok=True)
	start_epoch = args.epoch_to_restart + 1
	# base_chckpt_path = os.path.dirname(chckpt_path)

	for t in range(start_epoch, start_epoch + args.epochs):
		print(f"Epoch: {t+1}\n----------------")
		train_loop_fn(train_loader, regressor, base_loss_fn=loss_fn, optimizer=optimizer, angle_penalty_weight=args.angle_penalty)
		
		for title, val_loader in val_loaders:
			print(f"Currently evaluating val: {title}")
			path_image = os.path.join(chckpt_path, f"grid_epoch_{title}_{t+1}.png")
			val_loss, val_pixel_error_axis_one, val_pixel_error_axis_two = test_loop_fn(val_loader, regressor, 
				base_loss_fn=loss_fn, angle_penalty_weight=args.angle_penalty, dump_image=True, dump_image_path=path_image)
		checkpoint = {
			"epoch": t,
			"model_state_dict": regressor.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			# "pixel_error_axis_one": val_pixel_error_axis_one,
			# "pixel_error_axis_two": val_pixel_error_axis_two,
			"model_name": args.model,
			# 'loss': val_loss,
		}
		torch.save(checkpoint, os.path.join(chckpt_path, f"epoch_{t}.pth"))
		scheduler.step(val_pixel_error_axis_one + val_pixel_error_axis_two)
		print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

	if args.test_run:
		with open(args.test_run_marker, "w") as f:
			f.write("SUCCESS")