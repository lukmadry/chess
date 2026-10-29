import sys
import numpy as np

from src.dataset import ChessDataset
from src.helpers.tools import canonical_corner_order

# test_args = {'dates': } # needed to pass date argument

# TODO
# correct loading of ChessDataset and labels
def _get_min_max_of_label(lb):
	mmin, mmax = np.inf, -np.inf
	for x, y in lb:
		if mmin < min(x, y):
			mmin = min(x, y)
		if mmax > max(x, y):
			mmax = max(x, y)
	return mmin, mmax

def test_variants_of_dataset_load(image_dir, adaptive_dir):

	cd_normalised = ChessDataset(image_dir, convert_to_cas=False, normalised_labels=True, size=100)

	for lb in cd_normalised.labels:
		mmin, mmax = _get_min_max_of_label(lb)
		assert(mmin > 0 and mmax < 1.001)

	for im in cd_normalised.images_list:
		assert(isinstance(im, str)) # not normalised yet, raw image

	cd_adaptive_norm = ChessDataset(adaptive_dir, convert_to_cas=False, normalised_labels=True, size=100, adaptive_size=True)

	for lb in cd_adaptive_norm.labels:
		mmin, mmax = _get_min_max_of_label(lb)
		assert( mmax > 1)

	for i in range(100):
		im, lab = cd_adaptive_norm[i]
		mmin, mmax = _get_min_max_of_label(lab)
		assert(mmin >= 0 and mmax < 1.001)

	cd_adaptive_not_norm = ChessDataset(adaptive_dir, convert_to_cas=False, normalised_labels=False, size=100, adaptive_size=True)

	for lb in cd_adaptive_not_norm.labels:
		mmin, mmax = _get_min_max_of_label(lb)
		assert( mmax > 1)

	for i in range(100):
		im, lab = cd_adaptive_not_norm[i]
		mmin, mmax = _get_min_max_of_label(lab)
		assert( mmax > 1)

test_variants_of_dataset_load("chessboard_augment_very_low_yaw_with_pieces", "images_to_annotate")

# test if loading a model from checkpoint works correctly
# test.sh should be run after test passages so you should provide the date to test



# test of canonical order
def test_canonical_order_of_labels(unchanged, rev_lab, permuted, permut_desired_reordering):

	for lab in unchanged:
		lab = np.array(lab)
		assert(np.all(lab == canonical_corner_order(lab)))

	for lab in rev_lab:
		lab = np.array(lab)
		assert(np.all(lab[::-1] == canonical_corner_order(lab)))

	for lab, reord in zip(permuted, permut_desired_reordering):
		lab = np.array([lab[i] for i in reord])
		assert(np.all(lab) == canonical_corner_order(lab))


unchanged_labels = [
	[[-1, 1], [-1, 3], [2, 3], [2, 1]],
	[[-1, 1], [-1.2, 4], [2.3, 3.5], [2.3, 0.9]]
]

reversed_labels = [
	[[-1, 1], [-1, 3], [2, 3], [2, 1]][::-1]
]

permuted_labels = [
	[unchanged_labels[0][i] for i in [2, 3, 1, 0]]
]

reorder = [
	[3, 2, 0, 1]
]

test_canonical_order_of_labels(unchanged_labels, reversed_labels, permuted_labels, reorder)
# test if bb -> cas -> bb works correctly

