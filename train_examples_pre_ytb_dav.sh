################### step 1:  OneVOS pre training with Static datasets ####################
CUDA_VISIBLE_DEVICES=4,5,6,7 python "tools/train_accum.py"  \
--amp \
--exp_name exp1 \
--stage pre \
--model onevos_convmae \
--gpu_num 4 \
--batch_size 4


# ################### step 2: OneVOS main training with YTB and DAVIS datasets ####################
# CUDA_VISIBLE_DEVICES=4,5,6,7 python "tools/train_accum.py"  \
# --amp \
# --exp_name exp1 \
# --stage pre_ytb_dav \
# --model onevos_convmae \
# --gpu_num 4 \
# --batch_size 4


