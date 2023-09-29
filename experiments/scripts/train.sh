#! /bin/bash

GROUP=${1:-"DEFAULT"}
EPOCHS=5


# Vision
python train.py -wb -g $GROUP --model resnet -t ensemble_push -n 2 -d 1 -e $EPOCHS -dl 5120 || true # 852 MiB
python train.py -wb -g $GROUP --model resnet -t ensemble_push -n 2 -d 2 -e $EPOCHS  -dl 5120 || true  # 660 MiB x2 = 1320 MiB
python train.py -wb -g $GROUP --model resnet -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 1  -dl 5120  || true # 1018 MiB
python train.py -wb -g $GROUP --model resnet -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 2 -dl 5120  || true # 850 MiB x2 = 1700 MiB
python train.py -wb -g $GROUP --model resnet -t svgd_push -n 2 -d 1 -e $EPOCHS -mef -dl 5120  || true # 1694 MiB
python train.py -wb -g $GROUP --model resnet -t svgd_push -n 2 -d 2 -e $EPOCHS -mef-dl 5120  || true # 1604 + 556 = 2160 MiB

python train.py -wb -g $GROUP --model transformer -t ensemble_push -n 2 -d 1 -e $EPOCHS -dl 5120  || true # 420 MiB
python train.py -wb -g $GROUP --model transformer -t ensemble_push -n 2 -d 2 -e $EPOCHS -dl 5120  || true # 406 MiB x2 = 812 MiB
python train.py -wb -g $GROUP --model transformer -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 1 -dl 5120  || true # 430 Mib
python train.py -wb -g $GROUP --model transformer -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 2 -dl 5120  || true # 418 MiB x2 = 836 MiB
python train.py -wb -g $GROUP --model transformer -t svgd_push -n 2 -d 1 -e $EPOCHS -mef -dl 5120  || true # 482 MiB
python train.py -wb -g $GROUP --model transformer -t svgd_push -n 2 -d 2 -e $EPOCHS -mef -dl 5120  || true # 478 + 400 = 878 MiB

# Qchem

python train.py -wb -g $GROUP --model schnet -t ensemble_push -n 2 -d 1 -e $EPOCHS -dl 800  || true # 1320 MiB
python train.py -wb -g $GROUP --model schnet -t ensemble_push -n 2 -d 2 -e $EPOCHS -dl 800  || true # 1126 x2 = 2252 MiB
python train.py -wb -g $GROUP --model schnet -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 1 -dl 800  || true # 1350 MiB
python train.py -wb -g $GROUP --model schnet -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 2 -dl 800  || true # 1238 MiB
python train.py -wb -g $GROUP --model schnet -t svgd_push -n 2 -d 1 -e $EPOCHS -mef -dl 800  || true # 1960 MiB
python train.py -wb -g $GROUP --model schnet -t svgd_push -n 2 -d 2 -e $EPOCHS -mef -dl 800  || true # 1874 + 1086 = 2960 MiB

python train.py -wb -g $GROUP --model cgcnn -t ensemble_push -n 2 -d 1 -e $EPOCHS -dl 800  || true #  1562 MiB
python train.py -wb -g $GROUP --model cgcnn -t ensemble_push -n 2 -d 2 -e $EPOCHS -dl 800  || true # 1480 + 1482  = 2962 MiB
python train.py -wb -g $GROUP --model cgcnn -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 1 -dl 800  || true # 1542 MiB
python train.py -wb -g $GROUP --model cgcnn -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 2 -dl 800  || true # 1504 x2 = 3008 MiB
python train.py -wb -g $GROUP --model cgcnn -t svgd_push -n 2 -d 1 -e $EPOCHS -mef -dl 800  || true #  1684 MiB
python train.py -wb -g $GROUP --model cgcnn -t svgd_push -n 2 -d 2 -e $EPOCHS -mef -dl 800  || true # 1666 + 1422  = 3088 MiB

# SciMl
python train.py -wb -g $GROUP --model unet -t ensemble_push -n 2 -d 1 -e $EPOCHS -dl 2000  || true #  523 MiB
python train.py -wb -g $GROUP --model unet -t ensemble_push -n 2 -d 2 -e $EPOCHS -dl 2000  || true # 483 + 479  = 962 MiB
python train.py -wb -g $GROUP --model unet -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 1  -dl 2000  || true #  541 MiB
python train.py -wb -g $GROUP --model unet -t mswag_push --pretrain_epochs 3 --swag_epochs $EPOCHS -n 2 -d 2 -dl 2000  || true #  513 + 521 = 1034 MiB
python train.py -wb -g $GROUP --model unet -t svgd_push -n 2 -d 1 -e $EPOCHS -mef -dl 2000  || true # 771 MiB
python train.py -wb -g $GROUP --model unet -t svgd_push -n 2 -d 2 -e $EPOCHS -mef -dl 2000  || true # 703 + 449 = 1152 MiB