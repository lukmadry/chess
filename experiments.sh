python training.py --synth_input_dir chessboard_augment_low_yaw --model mobile --no-cas --input_size 384 --epochs 50 --dump_images --loss wing
python training.py --synth_input_dir chessboard_augment_very_low_yaw_with_pieces --model mobile --no-cas --input_size 384 --epochs 50 --dump_images --loss wing
python training.py --synth_input_dir chessboard_augment_low_yaw --model resnet --no-cas --input_size 384 --epochs 50 --dump_images --loss wing
python training.py --synth_input_dir chessboard_augment_very_low_yaw_with_pieces --model resnet --no-cas --input_size 384 --epochs 50 --dump_images --loss wing

