import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

from src.helpers.tools import cas_to_boxes, _update_stats_of_gradients, _print_stats_of_gradient
from src.helpers.vis import display_grid_with_labels
from src.losses import angle_penalty_loss
# from training import _image_dumper


# to be rewritten, or at least API of sample_raw_images will have to be
def _image_dumper(model, dataloader, grid, model_for_cas, fig_dir,  **kwargs):

	# dataset = dataloader.dataset
	# val_idx = choices(range(len(dataloader)), k=grid**2)
	# # we will force normalisation of labels and turn it off later
	# raw_images, target_grid = dataset.sample_raw_images(grid**2)
	# images_for_model = [dataset._input_transform(im) for im in raw_images]
	# for targets in target_grid:
	# 	if max(targets) > 1.001: # leaving a bit of a margin
	# 		raise NotImplementedError("A label that is not normalised found during the image dumping procedure.")

	val_indices = dataloader.dataset.indices
	if hasattr(dataloader.dataset, 'dataset'):
		orig_dataset = dataloader.dataset.dataset
	else:
		orig_dataset = dataloader.dataset.subset.dataset

	print("normalised labels: ", orig_dataset.normalised_labels)
	display_images_grid, target_grid = orig_dataset.sample_raw_images(int(grid**2),
		indices=val_indices,
		normalized_targets=True,
		sample_orig_bb=True,
		impose_size=None)
	image_for_model = [orig_dataset._input_transform(x) for x in display_images_grid]
	image_for_model = torch.stack(image_for_model)
	# print(target_grid)

	if not model_for_cas:
		display_grid_with_labels(
				model=model,
				image_array=image_for_model,
				raw_images=display_images_grid,
				gt_list=target_grid,
				image_grid_size=grid,
				show=False,
				savefig=True,
				fig_dir=fig_dir,
				model_for_cas=model_for_cas)


def train_loop(dataloader, model, base_loss_fn, optimizer, angle_penalty_weight=0.5, batch_size=32):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.train()

	grad_dict = {}
	for name, param in model.named_parameters():
		if param.requires_grad:
			grad_dict[name] = {}
			for stat in ['mean', 'std', 'max', 'min']:
				grad_dict[name][stat] = []

	total_loss = 0
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
			current =batch * batch_size + len(X)
			print(f"Main loss loss: {mse_loss.item():>7f}  [{current:>5d}/{size:>5d}]")
			print(f"Square angle loss: {square_angles.item():>7f}\nAngle similarity loss: {cos_sim.item():>7f}")

		loss_item = loss.item()
		total_loss += loss_item

	_print_stats_of_gradient(grad_dict)
	total_loss /= num_batches
	return grad_dict, total_loss


def train_loop_cas(dataloader, model, base_loss_fn, optimizer, batch_size=32, angle_penalty_weight=1.):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.train()

	grad_dict = {}
	total_loss = 0
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

		loss_item = loss.item()
		if batch % 5 == 0:
			current = batch * batch_size + len(X)
			print(f"Total MSE loss: {loss_item:>7f}   [{current:>5d}/{size:>5d}]")
			print(f"Center loss: {mse_loss_center.item():>7f}\nSize loss: {mse_loss_size.item():>7f}\nAngle loss: {mse_loss_angle.item():>7f}")


		total_loss += loss_item

	_print_stats_of_gradient(grad_dict)
	total_loss /= num_batches
	return grad_dict, total_loss


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
	print(f"Average loss: {test_loss:>7f}\nAverage main loss: {test_mse_loss:>7f}")
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


def test_loop_cas(dataloader, model, base_loss_fn, angle_penalty_weight=0.5, dump_image=False, grid=3, dump_image_path=None):
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
			fig_dir=dump_image_path,
			cas=True)

	return {'center_loss': test_center_loss, 'size_loss': test_size_loss, 'angle_loss': test_angle_loss}, pixel_error_axis_one, pixel_error_axis_two
