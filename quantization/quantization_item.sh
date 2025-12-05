python main.py \
  --model_name rqvae \
  --dataset_name Baby \
  --embedding_modality fused \
  --embedding_model text-embedding-3-large-clip-vit-base-patch32

# 多模态
python main.py \
  --model_name mm_rqvae \
  --dataset_name Baby \
  --text_embedding_model text-embedding-3-large \
  --image_embedding_model clip-vit-base-patch32 