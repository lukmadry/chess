import numpy as np
import os
import argparse
import glob
import json
import torch
import argparse
import datetime
import cv2
from random import choices
from PIL import Image
# from torch.utils.data import DataLoader, Dataset

from training import PretrainedSmallChessBoardRegressor, canonical_corner_order
from helpers import sample_imgs_and_display, model_path_from_date

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_dir")
	parser.add_argument("--image_grid_size", type=int, default=3)
	parser.add_argument("--epoch", type=int)
	parser.add_argument("--savefig", action=argparse.BooleanOptionalAction)
	parser.add_argument("--last_model_date", type=str)
	parser.add_argument("--no_model", type=int)
	parser.add_argument("--epoch_to_load", type=int, default=-1)
	args = parser.parse_args()

	
	chckpt_path = model_path_from_date(args.last_model_date, args.epoch_to_load, args.no_model)
	base_chckpt_path = os.path.dirname(chckpt_path)
	model = PretrainedSmallChessBoardRegressor()
	checkpoint = torch.load(chckpt_path, weights_only=True)
	model.load_state_dict(checkpoint["model_state_dict"])

	visualisation_path = os.path.join(base_chckpt_path, "vis")
	os.makedirs(visualisation_path, exist_ok=True)
	fig_path = os.path.join(visualisation_path, f"vis_{args.epoch}_{args.image_grid_size}.png")

	sample_imgs_and_display(
		model=model, 
		show=True, 
		fig_dir=fig_path, 
		savefig=args.savefig, 
		image_grid_size=args.image_grid_size, 
		image_dir=args.image_dir)