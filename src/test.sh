python test.py \
  --model_type concat \
  --backbone roberta \
  --embedding_method cls_with_pooler \
  --model_path ../model/XQBert-0508-234021.pth \
  --test_data_path ../data/test.tsv \
  --max_length 512 \
  --pad_to_max_length \
  --batch_size 32 \
  --output_dir ../