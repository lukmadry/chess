DATE_STR = $(date +%d/%m/%Y)
python chessboard.py --nr_chessboards 2 --nr_permutations 5 --output_dir chessboard_augment_small_test --yaw_range -10 10 
python chessboard.py --nr_chessboards 2 --half-synth --output_dir chessboard_halfsynth_test
python training.py --input_dir chessboard_augment_low_yaw --labels_dir chessboard_augment_low_yaw/ground_truths --model mobile --cas --input_size 224 --epochs 2 --size 100
python training.py --load_from_checkpoint --last_model_date $DATE_STR --input_dir chessboard_augment_low_yaw --model mobile --cas -input_size 224 --epoch_to_restart 2 --size 100 --epochs 2
python training.py --input_dir chessboard_augment_low_yaw --labels_dir chessboard_augment_low_yaw/ground_truths --model mobile --no-cas --input_size 224 --epochs 2 --size 100 --no-cas
python training.py --load_from_checkpoint --last_model_date $DATE_STR --input_dir chessboard_augment_low_yaw --model mobile --no-cas --input_size 224 --epochs 2 --size 100
python tests.py