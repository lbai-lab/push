python train.py -wb -g size6 --model transformer2 -t mswag_push -n 2 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 64 --mlp_dim 3072 --hidden_dim 768 || true   # 1
python train.py -wb -g size6 --model transformer2 -t mswag_push -n 4 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 32 --mlp_dim 3072 --hidden_dim 768 || true   # 1
python train.py -wb -g size6 --model transformer2 -t mswag_push -n 8 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 16 --mlp_dim 3072 --hidden_dim 768 || true   # 1
python train.py -wb -g size6 --model transformer2 -t mswag_push -n 16 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 8 --mlp_dim 3072 --hidden_dim 768 || true    # 2
python train.py -wb -g size6 --model transformer2 -t mswag_push -n 32 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 4 --mlp_dim 3072 --hidden_dim 768 || true   # 4
python train.py -wb -g size6 --model transformer2 -t mswag_push -n 64 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 2 --mlp_dim 3072 --hidden_dim 768 || true   # 8
python train.py -wb -g size6 --model transformer2 -t mswag_push -n 128 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 1 --mlp_dim 3072 --hidden_dim 768 || true   # 16

python train.py -wb -g size7 --model transformer2 -t mswag_push -n 16 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 8 --mlp_dim 3072 --hidden_dim 768 || true    # 2
python train.py -wb -g size7 --model transformer2 -t mswag_push -n 32 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 8 --mlp_dim 1536 --hidden_dim 768 || true   # 4
python train.py -wb -g size7 --model transformer2 -t mswag_push -n 64 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 8 --mlp_dim 1536 --hidden_dim 384 || true   # 8
python train.py -wb -g size7 --model transformer2 -t mswag_push -n 128 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 4 --mlp_dim 1536 --hidden_dim 384 || true   # 16
python train.py -wb -g size7 --model transformer2 -t mswag_push -n 256 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 4 --mlp_dim 768 --hidden_dim 384 || true   # 16
python train.py -wb -g size7 --model transformer2 -t mswag_push -n 512 -d 2 --pretrain_epochs 0 --swag_epochs 5 -dl 5120  --num_heads 12 --num_layers 4 --mlp_dim 768 --hidden_dim 192 || true   # 16
