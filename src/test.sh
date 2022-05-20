python test.py \
  --model_type concat \
  --backbone electra \
  --embedding_method cls_with_pooler \
  --model_path ../model/XQBert-0509-230316.pth \
  --test_data_path ../data/test.tsv \
  --max_length 512 \
  --pad_to_max_length \
  --batch_size 32 \
  --output_dir ../