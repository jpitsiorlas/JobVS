TASK_ID=0
GPU_IDs=0
BATCH_SIZE=1
FOLD=2
PORT=8
MODEL=ROG
WEIGHTS='/home/pitsiorl/Shiny_Icarus/Vessel_Segmentation-multi-task/ROG/ROG/fold1/best_dice.pth.tar'
# WEIGHTS='/media/SSD0/nfvalderrama/Vessel_Segmentation/data/model_weights/fold1_f48_ep300_4gpu_dice0_9059/model.pt'
# WEIGHTS='/media/SSD0/nfvalderrama/Vessel_Segmentation/data/model_weights/UNETR_model_best_acc.pth'

#############################################################################

EXPERIMENT='Multitask_Brain_Vessel'
OUTPUT_DIR='/home/pitsiorl/Shiny_Icarus/Vessel_Segmentation-multi-task/ROG/'$MODEL'/'$EXPERIMENT
OUTPUT_DIR_FREE_AT=$OUTPUT_DIR'/AT'
#############################################################################

#For the standard training
# CUDA_VISIBLE_DEVICES=$GPU_IDs python main.py \
#     --port $PORT --model $MODEL \
#     --fold $FOLD --task $TASK_ID \
#     --gpu $GPU_IDs --batch $BATCH_SIZE \
#     --name $OUTPUT_DIR \
#     --load_weights $WEIGHTS \
#     #--resume

# TASK_ID=0
# GPU_IDs=3
# BATCH_SIZE=1
# FOLD=2
# PORT=8
# MODEL=ROG
# WEIGHTS='/media/SSD0/nfvalderrama/Vessel_Segmentation/data/model_weights/ROG_brats_best_dice.pth.tar'
# # WEIGHTS='/media/SSD0/nfvalderrama/Vessel_Segmentation/data/model_weights/fold1_f48_ep300_4gpu_dice0_9059/model.pt'
# # WEIGHTS='/media/SSD0/nfvalderrama/Vessel_Segmentation/data/model_weights/UNETR_model_best_acc.pth'

# #############################################################################

# EXPERIMENT='Multitask_Brain_Vessel'
# OUTPUT_DIR='/media/SSD0/nfvalderrama/Vessel_Segmentation/exps/'$MODEL'/'$EXPERIMENT
# OUTPUT_DIR_FREE_AT=$OUTPUT_DIR'/AT'
# #############################################################################

# # For the standard training
# CUDA_VISIBLE_DEVICES=$GPU_IDs python main.py \
#     --port $PORT --model $MODEL \
#     --fold $FOLD --task $TASK_ID \
#     --gpu $GPU_IDs --batch $BATCH_SIZE \
#     --name $OUTPUT_DIR \
#     --load_weights $WEIGHTS \

#Test
CUDA_VISIBLE_DEVICES=$GPU_IDs python main.py \
    --port $PORT --model $MODEL \
    --fold $FOLD --task $TASK_ID \
    --gpu $GPU_IDs --batch $BATCH_SIZE \
    --name $OUTPUT_DIR \
    --test 

# Test AutoAttack
# python main.py \
#     --fold $FOLD --task $TASK_ID \
#     --gpu $GPU_IDs --batch $BATCH_SIZE \
#     --name $OUTPUT_DIR \
    # --test --adv

# # For the Free AT fine tuning
# python main.py \
#     --fold $FOLD --task $TASK_ID \
#     --gpu $GPU_IDs --batch $BATCH_SIZE \
#     --name $OUTPUT_DIR_FREE_AT \
#     --pretrained $OUTPUT_DIR \
#     --AT --ft

# # Test AT
# python main.py \
#     --fold $FOLD --task $TASK_ID \
#     --gpu $GPU_IDs --batch $BATCH_SIZE \
#     --name $OUTPUT_DIR_FREE_AT \
#     --test 

# Test AutoAttack AT
# python main.py \
#     --fold $FOLD --task $TASK_ID \
#     --gpu $GPU_IDs --batch $BATCH_SIZE \
#     --name $OUTPUT_DIR_FREE_AT \
#     --test --adv


