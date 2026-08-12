python chessboard.py --nr_chessboards 2 --nr_permutations 5 --output_dir chessboard_augment_small_test --yaw_range -10 10 
python training.py --input_dir chessboard_augment_low_yaw --labels_dir chessboard_augment_low_yaw/ground_truths --model mobile --cas --input_size 224 --epochs 10 --size 100
python training.py --load_from_checkpoint 
python training.py --input_dir chessboard_augment_low_yaw --labels_dir chessboard_augment_low_yaw/ground_truths --model mobile --no-cas --input_size 224 --epochs 10 --size 100 --no-cas
python training.py --load_from_checkpoint 
python visualise_model.py --model_dir checkpoints/Thursday,\ 30.\ July\ 2026\ 04:37PM --epoch 24 --image_dir chessboard_augment_low_yaw --image_grid_size 3