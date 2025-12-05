export CUDA_VISIBLE_DEVICES=6
python main.py --model TIGER --dataset Baby --quant_method mm_rqvae --embedding_modality lfused

export CUDA_VISIBLE_DEVICES=7
python main.py --model RPG --dataset Beauty --quant_method opq