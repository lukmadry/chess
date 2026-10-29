import glob
import json
import os
from numbers import Number
from random import choices, sample

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from augmentation import final_image_shifts_dict
from helpers import convert_boxes_to_cas, canonical_corner_order
from training import create_input_transform


class ChessDataset(Dataset):

	def __init__(self, images_dir, labels_dir=None, size=None, with_bright=False, filter_labels_params=False,
		allowed_dist=None, convert_to_cas=False, normalised_labels=False, input_size=224, adaptive_size=False,
		height=480., width=640., max_angle=20., verify_angle=False, keep_orig_bb=False, return_normalised_image=True,
		print_labels_for_test=False):
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
		self.adaptive_size = adaptive_size # in this case we will always normalise to [0,1]
		self.normalised_labels = normalised_labels

		if adaptive_size:
			# normalised_labels = False
			self.normalised_labels = True # weird and counterintuitive, TODO to be cleaned up
			self.image_sizes = []
		else:
			self.height = height
			self.width = width

		if not convert_to_cas and self.normalised_labels:
			for l in self.images_list:
				local_lab = []
				with Image.open(l) as img:
					width, height = img.size
				labels_dict = self._load_labels_dict(l)

				for (x, y) in labels_dict["inner_board"]:
					local_lab.append([float(x)/height, float(y)/width])#  ([[float(x)/480., float(y)/640.] for (x,y) in labels_dict["inner_board"]])
				self.labels.append(local_lab)
		elif not convert_to_cas and not self.normalised_labels:
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
					raise ValueError("labels did not convert to the normalised format")
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
			print("Will unify orientation of labels")
			self._unify_orientation_of_labels()

		if print_labels_for_test:
			print("Printing 5 ground truths for debugging")
			for k in range(5):
				print(self.labels[k])

	def __len__(self):
		return len(self.images_list)

	def __getitem__(self, idx):
		# rename to synth get item ? either case you'll need to branch it
		im = Image.open(self.images_list[idx])
		_lab = self.labels[idx]
		# if self.adaptive_size:
		# 	width, height = im.size
		# 	lab = []
		# 	for (x, y) in _lab:
		# 		lab.append([float(x)/height, float(y)/width])
		# else:
		# 	lab = _lab
		lab = np.array(_lab).flatten()
		# lab = canonical_corner_order(lab)
		lab = torch.tensor(lab, dtype=torch.float32)

		if self.return_normalised_image:
			im = self._input_transform(im)

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
		dtype = np.float32 if (self.normalised_labels or self.adaptive_size) else np.int32
		for lab in self.labels:
			_lab = canonical_corner_order(lab, dtype)
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

	@property
	def height(self):
		if self.adaptive_size:
			raise ValueError("adaptive_size is set to True, therefore images may have different sizes. Attribute .height not available")
		return self._height

	@height.setter
	def height(self, val):
		if not isinstance(val, Number) or val < 0:
			raise ValueError
		self._height = val

	@property
	def width(self):
		if self.adaptive_size:
			raise ValueError("adaptive_size is set to True, therefore images may have different sizes. Attribute .width not available")
		return self._width

	@width.setter
	def width(self, val):
		if not isinstance(val, Number) or val < 0:
			raise ValueError
		self._width = val


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
