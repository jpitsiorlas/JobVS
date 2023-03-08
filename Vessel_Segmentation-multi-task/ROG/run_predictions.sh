TASK_ID=0
GPU_IDs=0
BATCH_SIZE=1
FOLD=1
PORT=8
MODEL=ROG
WEIGHTS='/home/pitsiorl/Shiny_Icarus/Vessel_Segmentation-multi-task/ROG/ROG/final_jobvs/fold1/best_dice.pth.tar'


#############################################################################

EXPERIMENT='final_jobvs'
OUTPUT_DIR='/home/pitsiorl/Shiny_Icarus/Vessel_Segmentation-multi-task/ROG/'$MODEL'/'$EXPERIMENT
#############################################################################

#Test
CUDA_VISIBLE_DEVICES=$GPU_IDs python main.py \
    --port $PORT --model $MODEL \
    --fold $FOLD --task $TASK_ID \
    --gpu $GPU_IDs --batch $BATCH_SIZE \
    --name $OUTPUT_DIR \
    --test 

