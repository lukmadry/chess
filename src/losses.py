import torch
from torch import nn


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
