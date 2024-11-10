# 测试davis16 val
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python /home/liwy/code/OneVOS-开源/Final_Update_Code/OneVOS/tools/eval.py \
--amp \
--exp_name "exp1" \
--stage pre_ytb_dav \
--model onevos_convmae \
--ckpt_path "./checkpoints_saves/exp1_onevos_convmae/PRE_YTB_DAV/ema_ckpt/save_step_196000.pth" \
--output_path "./output_test/" \
--dataset davis2016 \
--split val \
--gpu_num 4 \

# 测试davis17 val
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python /home/liwy/code/OneVOS-开源/Final_Update_Code/OneVOS/tools/eval.py \
# --amp \
# --exp_name "exp1" \
# --stage pre_ytb_dav \
# --model onevos_convmae \
# --ckpt_path  "./checkpoints_saves/exp1_onevos_convmae/PRE_YTB_DAV/ema_ckpt/save_step_196000.pth" \
# --output_path "./output_test/" \
# --dataset davis2017 \
# --split val \
# --gpu_num 4 \


# 测试davis17 test
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python /home/liwy/code/OneVOS-开源/Final_Update_Code/OneVOS/tools/eval.py \
# --amp \
# --exp_name "exp1" \
# --stage pre_ytb_dav \
# --model onevos_convmae \
# --ckpt_path  "./checkpoints_saves/exp1_onevos_convmae/PRE_YTB_DAV/ema_ckpt/save_step_200000.pth" \
# --output_path "./output_test/" \
# --dataset davis2017_test \
# --split test \
# --mem_cap 4 \
# --gpu_num 4 \


# 测试youtube vos 19
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python /home/liwy/code/OneVOS-开源/Final_Update_Code/OneVOS/tools/eval.py \
# --amp \
# --exp_name "exp1" \
# --stage pre_ytb_dav \
# --model onevos_convmae \
# --ckpt_path  "./checkpoints_saves/exp1_onevos_convmae/PRE_YTB_DAV/ema_ckpt/save_step_196000.pth" \
# --output_path "./output_test/" \
# --dataset youtubevos2019 \
# --split val \
# --mem_every 11 \
# --mem_cap 3 \
# --gpu_num 4 \


# 测试mose
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python /home/liwy/code/OneVOS-开源/Final_Update_Code/OneVOS/tools/eval.py \
# --amp \
# --exp_name "exp1" \
# --stage pre_ytb_dav \
# --model onevos_convmae \
# --ckpt_path  "./checkpoints_saves/exp1_onevos_convmae/PRE_YTB_DAV/ema_ckpt/save_step_196000.pth" \
# --output_path "./output_test/" \
# --dataset mose \
# --split val \
# --mem_every 5 \
# --gpu_num 4 \


# 测试 lvos val 
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python /home/liwy/code/OneVOS-开源/Final_Update_Code/OneVOS/tools/eval.py \
# --amp \
# --exp_name "exp1" \
# --stage pre_ytb_dav \
# --model onevos_convmae \
# --ckpt_path  "./checkpoints_saves/exp1_onevos_convmae/PRE_YTB_DAV/ema_ckpt/save_step_196000.pth" \
# --output_path "./output_test/" \
# --dataset mose \
# --split val \
# --gpu_num 4 \