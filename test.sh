DATE_STR=$(date +%d-%m-%Y)

mkdir -p "test_status_files"
FS_CHESSBOARD_STATUS_FILE="test_status_files/chess_from_scratch_${DATE_STR}.txt"
HS_CHESSBOARD_STATUS_FILE="test_status_files/half_synth_${DATE_STR}.txt"
TRAININGCAS_STATUS_FILE="test_status_files/training_cas_${DATE_STR}.txt"
TRAININGCAS_LOAD_STATUS_FILE="test_status_files/training_cas_load_${DATE_STR}.txt"
TRAININGBB_STATUS_FILE="test_status_files/training_bb_${DATE_STR}.txt"
TRAININGBB_LOAD_STATUS_FILE="test_status_files/training_bb_load_${DATE_STR}.txt"

TRAINING_AUTH_STATUS_FILE="test_status_files/training_bb_auth_${DATE_STR}.txt"
TRAINING_SYNT_AND_AUTH_STATUS_FILE="test_status_files/training_bb_synth_auth_${DATE_STR}.txt"

SUCCESS_COUNT=0
TEST_COUNT=0
LOG_FILE="output_$(date +%Y%m%d_%H%M%S).log"
exec 3>&1 > "${LOG_FILE}" 2>&1

echo "testing generating chessboard from scratch" >&3
((TEST_COUNT++))
python chessboard.py --nr_chessboards 2 --nr_permutations 5 --output_dir test_run_cbs --yaw_range -10 10 --test_run --test_run_marker $FS_CHESSBOARD_STATUS_FILE
if [ -f "${FS_CHESSBOARD_STATUS_FILE}" ] && [ "$(cat "${FS_CHESSBOARD_STATUS_FILE}")" = "SUCCESS" ]; then
	echo "SUCCESS: finished generating chessboard from scratch" >&3
	((SUCCESS_COUNT++))
else
	echo "generating chessboard from scratch failed" >&3
fi

echo "testing half synthetic chessboards" >&3
((TEST_COUNT++))
python chessboard.py --nr_chessboards 2 --half_synth --output_dir test_run_cbs_halfsynth --yaw_range -10 10 --test_run --test_run_marker $HS_CHESSBOARD_STATUS_FILE
if [ -f "${HS_CHESSBOARD_STATUS_FILE}" ] && [ "$(cat "${HS_CHESSBOARD_STATUS_FILE}")" = "SUCCESS" ]; then
	echo "SUCCESS: finished generating half synth chessboards" >&3
	((SUCCESS_COUNT++))
else
	echo "generating half synth chessboards failed" >&3
fi

echo "testing training on synthetic data only" >&3
((TEST_COUNT++))
python training.py --test_run --test_run_marker $TRAININGCAS_STATUS_FILE --synth_input_dir chessboard_augment_low_yaw --synth_labels_dir chessboard_augment_low_yaw/ground_truths --model mobile --cas --input_size 224 --epochs 2 --size 100
if [ -f "${TRAININGCAS_STATUS_FILE}" ] && [ "$(cat "${TRAININGCAS_STATUS_FILE}")" = "SUCCESS" ]; then
	echo "SUCCESS: center-angle-size training worked" >&3
	((SUCCESS_COUNT++))
else
	echo "FAILURE: center-angle-size training" >&3
fi

((TEST_COUNT++))
python training.py --test_run --test_run_marker $TRAININGCAS_LOAD_STATUS_FILE --load_from_checkpoint --last_model_date $DATE_STR --synth_input_dir chessboard_augment_low_yaw --model mobile --cas --input_size 224 --epoch_to_restart 2 --size 100 --epochs 2
if [ -f "${TRAININGCAS_LOAD_STATUS_FILE}" ] && [ "$(cat "${TRAININGCAS_LOAD_STATUS_FILE}")" = "SUCCESS" ]; then
	echo "SUCCESS: center-angle-size training with loading from checkpoint worked" >&3
	((SUCCESS_COUNT++))
else
	echo "FAILURE: center-angle-size training with loading from checkpoint" >&3
fi

((TEST_COUNT++))
python training.py --test_run --test_run_marker $TRAININGBB_STATUS_FILE --synth_input_dir chessboard_augment_low_yaw --synth_labels_dir chessboard_augment_low_yaw/ground_truths --model mobile --no-cas --input_size 224 --epochs 2 --size 100 --no-cas
if [ -f "${TRAININGBB_STATUS_FILE}" ] && [ "$(cat "${TRAININGBB_STATUS_FILE}")" = "SUCCESS" ]; then
	echo "SUCCESS: bounding box training worked" >&3
	((SUCCESS_COUNT++))
else
	echo "FAILURE: bounding box training" >&3
fi

((TEST_COUNT++))
python training.py --test_run --test_run_marker $TRAININGBB_LOAD_STATUS_FILE --load_from_checkpoint --last_model_date $DATE_STR --synth_input_dir chessboard_augment_low_yaw --model mobile --no-cas --input_size 224 --epochs 2 --size 100
if [ -f "${TRAININGBB_LOAD_STATUS_FILE}" ] && [ "$(cat "${TRAININGBB_LOAD_STATUS_FILE}")" = "SUCCESS" ]; then
	echo "SUCCESS: bounding box training with checkpoint loading worked" >&3
	((SUCCESS_COUNT++))
else
	echo "FAILURE: bounding box training with checkpoint loading" >&3
fi

echo "testing training on authentic data only" >&3
((TEST_COUNT++))
python training.py --test_run --test_run_marker $TRAINING_AUTH_STATUS_FILE --auth_input_dir images_to_annotate --model mobile --no-cas --input_size 224 --epochs 2 --size 100
if [ -f "${TRAINING_AUTH_STATUS_FILE}" ] && [ "$(cat "${TRAINING_AUTH_STATUS_FILE}")" = "SUCCESS" ]; then
	echo "SUCCESS: center-angle-size training on authentic data worked worked" >&3
	((SUCCESS_COUNT++))
else
	echo "FAILURE: center-angle-size training" >&3
fi

echo "testing training on synth + training"
((TEST_COUNT++))
python training.py --test_run --test_run_marker $TRAINING_SYNT_AND_AUTH_STATUS_FILE --synth_input_dir chessboard_augment_low_yaw --auth_input_dir images_to_annotate --model mobile --no-cas --input_size 224 --epochs 2 --size 100
if [ -f "${TRAINING_SYNT_AND_AUTH_STATUS_FILE}" ] && [ "$(cat "${TRAINING_SYNT_AND_AUTH_STATUS_FILE}")" = "SUCCESS" ]; then
	echo "SUCCESS: center-angle-size training on synth+authentic data worked worked" >&3
	((SUCCESS_COUNT++))
else
	echo "FAILURE: center-angle-size training" >&3
fi

rm -R test_run_cbs
rm -R test_run_cbs_halfsynth
rm $FS_CHESSBOARD_STATUS_FILE
rm $HS_CHESSBOARD_STATUS_FILE
rm $TRAININGCAS_STATUS_FILE
rm $TRAININGCAS_LOAD_STATUS_FILE
rm $TRAININGBB_STATUS_FILE
rm $TRAININGBB_LOAD_STATUS_FILE
rm $TRAINING_AUTH_STATUS_FILE
rm $TRAINING_SYNT_AND_AUTH_STATUS_FILE

echo "Passed tests: ${SUCCESS_COUNT}/${TEST_COUNT}"