import os
import json
import torch
import argparse
import datetime
import pickle

from copy import deepcopy

from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset

from src.helpers.tools import model_path_from_date
from src.dataset import ChessDataset, CleanChessDatasetWrapper, AugmentationChessDatasetWrapper
from src.loops import train_loop, train_loop_cas, test_loop, test_loop_cas
from src.losses import AnisothropicWingLoss
from src.models import PretrainedSmallChessBoardRegressor, PretrainedChessboardRegressor


# TODO move some stuff to a separate file, loss functions to losses.py, training and testing loops to loops.py or sth
# TODO you unflatten and then flatten ground truths, this is very redundant, fix it
# TODO the same way you diagnose error in pixels in test loop, add diagnostics of angles


# create two classes that will have different behaviour of __getitem__ ?


# copied from PyTorch website, for my purposes probably sufficient
# these two can be probably merged somehow but I think it would decrease readability if done hastily

# same caveat about merging here
# change mentions of mse in case of wing

if __name__ == "__main__":
	# TODO split arguments into categories to make this wall of text more readable and easier to navigate
	parser = argparse.ArgumentParser()
	parser.add_argument("--synth_input_dir", type=str, nargs='+') # this will not be augmented as it is already augmented during image generation
	parser.add_argument("--auth_input_dir", type=str, nargs='+') # this will be augmented additionally. 
	parser.add_argument("--synth_labels_dir", type=str, nargs='+')
	parser.add_argument("--auth_labels_dir", type=str, nargs='+')
	parser.add_argument("--synth_adaptive_size", action=argparse.BooleanOptionalAction, default=False)

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
	parser.add_argument("--cas", action=argparse.BooleanOptionalAction, default=False)

	parser.add_argument("--name", type=str)
	parser.add_argument("--load_from_checkpoint", action=argparse.BooleanOptionalAction)
	parser.add_argument("--last_model_date", type=str)
	parser.add_argument("--no_model", type=int)
	parser.add_argument("--epoch_to_restart", type=int, default=-1)

	parser.add_argument("--augmentation_factor", default=1, type=int)

	# some admin args
	parser.add_argument("--test_run", action=argparse.BooleanOptionalAction)
	parser.add_argument("--test_run_marker", type=str)
	parser.add_argument("--debug_mode", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--commit_hash", type=str)
	parser.add_argument("--dump_images", action=argparse.BooleanOptionalAction, default=True)
	args = parser.parse_args()

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

	print(f"Will train on the network {args.model}")
	synth_chess_dataset_kwargs = dict(
			with_bright=args.with_bright, 
			size=args.size,
			convert_to_cas=args.cas,
			normalised_labels=True,
			input_size=args.input_size,
			keep_orig_bb=True,
			print_labels_for_test=args.debug_mode)

	if args.synth_adaptive_size:
		print("will read synthetic images with adaptive size")
		synth_chess_dataset_kwargs['adaptive_size'] = True

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

	with open(os.path.join(chckpt_path, "kwargs"), "w") as f:
		json.dump(vars(args), f, default=str)
	for t in range(start_epoch, start_epoch + args.epochs):
		print(f"Epoch: {t+1}\n----------------")
		_, train_loss = train_loop_fn(train_loader, regressor, base_loss_fn=loss_fn, optimizer=optimizer, angle_penalty_weight=args.angle_penalty)
		epoch_error = {}
		epoch_error['train'] = train_loss
		for title, val_loader in val_loaders:
			print(f"Currently evaluating val: {title}")
			path_image = os.path.join(chckpt_path, f"grid_epoch_{title}_{t+1}.png")
			val_loss, val_pixel_error_axis_one, val_pixel_error_axis_two = test_loop_fn(val_loader, regressor, 
				base_loss_fn=loss_fn, angle_penalty_weight=args.angle_penalty, 
				dump_image=args.dump_images, dump_image_path=path_image)
			epoch_error[title] = {
			'val_loss': val_loss, 
			'val_pixel_error_axis_one': val_pixel_error_axis_one, 
			'val_pixel_error_axis_two': val_pixel_error_axis_two}

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
		with open(os.path.join(chckpt_path, f"errors_epoch_{t}.pth"), "wb") as f:
			pickle.dump(epoch_error, f)
		scheduler.step(val_pixel_error_axis_one + val_pixel_error_axis_two)
		print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

	if args.test_run:
		with open(args.test_run_marker, "w") as f:
			f.write("SUCCESS")