#! /bin/bash

GROUP=${1:-"DEFAULT"}

python train.py -wb -g $GROUP --model unet -t ensemble -n 2 -d 1
python train.py -wb -g $GROUP --model unet -t mswag --pretrain_epochs 3 --swag_epochs 3 -n 2 -d 1
python train.py -wb -g $GROUP --model unet -t svgd -n 2 -d 1

python train.py -wb -g $GROUP --model schnet -t ensemble -n 2 -d 1
python train.py -wb -g $GROUP --model schnet -t mswag --pretrain_epochs 3 --swag_epochs 3 -n 2 -d 1
python train.py -wb -g $GROUP --model schnet -t svgd -n 2 -d 1

python train.py -wb -g $GROUP --model cgcnn -t ensemble -n 2 -d 1
python train.py -wb -g $GROUP --model cgcnn -t mswag --pretrain_epochs 3 --swag_epochs 3 -n 2 -d 1
python train.py -wb -g $GROUP --model cgcnn -t svgd -n 2 -d 1

python train.py -wb -g $GROUP --model cnn -t ensemble -n 2 -d 1
python train.py -wb -g $GROUP --model cnn -t mswag --pretrain_epochs 3 --swag_epochs 3 -n 2 -d 1
python train.py -wb -g $GROUP --model cnn -t svgd -n 2 -d 1

python train.py -wb -g $GROUP --model resnet -t ensemble -n 2 -d 1
python train.py -wb -g $GROUP --model resnet -t mswag --pretrain_epochs 3 --swag_epochs 3 -n 2 -d 1
python train.py -wb -g $GROUP --model resnet -t svgd -n 2 -d 1

python train.py -wb -g $GROUP --model transformer -t ensemble -n 2 -d 1
python train.py -wb -g $GROUP --model transformer -t mswag --pretrain_epochs 3 --swag_epochs 3 -n 2 -d 1
python train.py -wb -g $GROUP --model transformer -t svgd -n 2 -d 1