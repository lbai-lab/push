#! /bin/bash

GROUP=${1:-"DEFAULT"}

python train.py -wb -g $GROUP --model transformer -t ensemble -n 1 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t ensemble -n 2 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t ensemble -n 4 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t ensemble -n 8 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 1 -d 1 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 2 -d 1 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 4 -d 1 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 8 -d 1 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t svgd -n 1 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t svgd -n 2 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t svgd -n 4 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model transformer -t svgd -n 8 -d 1 -e 10 -dl 5120  || true

python train.py -wb -g $GROUP --model cgcnn -t ensemble -n 1 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t ensemble -n 2 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t ensemble -n 4 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t ensemble -n 8 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 1 -d 1 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 2 -d 1 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 4 -d 1 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 8 -d 1 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t svgd -n 1 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t svgd -n 2 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t svgd -n 4 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model cgcnn -t svgd -n 8 -d 1 -e 10 -dl 800  || true

python train.py -wb -g $GROUP --model unet -t ensemble -n 1 -d 1 -e 10 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t ensemble -n 2 -d 1 -e 10 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t ensemble -n 4 -d 1 -e 10 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t ensemble -n 8 -d 1 -e 10 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 1 -d 1 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 2 -d 1 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 4 -d 1 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 8 -d 1 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t svgd -n 1 -d 1 -e 10 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t svgd -n 2 -d 1 -e 10 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t svgd -n 4 -d 1 -e 10 -dl 2000  || true
python train.py -wb -g $GROUP --model unet -t svgd -n 8 -d 1 -e 10 -dl 2000  || true

python train.py -wb -g $GROUP --model schnet -t ensemble -n 1 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t ensemble -n 2 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t ensemble -n 4 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t ensemble -n 8 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 1 -d 1 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 2 -d 1 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 4 -d 1 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 8 -d 1 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t svgd -n 1 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t svgd -n 2 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t svgd -n 4 -d 1 -e 10 -dl 800  || true
python train.py -wb -g $GROUP --model schnet -t svgd -n 8 -d 1 -e 10 -dl 800  || true

python train.py -wb -g $GROUP --model resnet -t ensemble -n 1 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t ensemble -n 2 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t ensemble -n 4 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t ensemble -n 8 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 1 -d 1 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 2 -d 1 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 4 -d 1 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t mswag --pretrain_epochs 3 --swag_epochs 10 -n 8 -d 1 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t svgd -n 1 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t svgd -n 2 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t svgd -n 4 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g $GROUP --model resnet -t svgd -n 8 -d 1 -e 10 -dl 5120  || true
