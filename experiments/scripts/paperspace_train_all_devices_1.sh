python train.py -wb -g a5000 --model schnet -t ensemble_push -n 1 -d 1 -e 10 -dl 800  || true
python train.py -wb -g a5000 --model schnet -t ensemble_push -n 2 -d 1 -e 10 -dl 800  || true
python train.py -wb -g a5000 --model schnet -t ensemble_push -n 4 -d 1 -e 10 -dl 800  || true
python train.py -wb -g a5000 --model schnet -t ensemble_push -n 8 -d 1 -e 10 -dl 800  || true
python train.py -wb -g a5000 --model schnet -t mswag_push -n 1 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 800  || true
python train.py -wb -g a5000 --model schnet -t mswag_push -n 2 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 800  || true
python train.py -wb -g a5000 --model schnet -t mswag_push -n 4 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 800  || true
python train.py -wb -g a5000 --model schnet -t mswag_push -n 8 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 800  || true
python train.py -wb -g a5000 --model schnet -t svgd_push -n 1 -d 1 -e 10 -dl 800  -mef || true
python train.py -wb -g a5000 --model schnet -t svgd_push -n 2 -d 1 -e 10 -dl 800  -mef || true
python train.py -wb -g a5000 --model schnet -t svgd_push -n 4 -d 1 -e 10 -dl 800  -mef || true
python train.py -wb -g a5000 --model schnet -t svgd_push -n 8 -d 1 -e 10 -dl 800  -mef || true
python train.py -wb -g a5000 --model transformer -t ensemble_push -n 1 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g a5000 --model transformer -t ensemble_push -n 2 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g a5000 --model transformer -t ensemble_push -n 4 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g a5000 --model transformer -t ensemble_push -n 8 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g a5000 --model transformer -t mswag_push -n 1 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 5120  || true
python train.py -wb -g a5000 --model transformer -t mswag_push -n 2 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 5120  || true
python train.py -wb -g a5000 --model transformer -t mswag_push -n 4 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 5120  || true
python train.py -wb -g a5000 --model transformer -t mswag_push -n 8 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 5120  || true
python train.py -wb -g a5000 --model transformer -t svgd_push -n 1 -d 1 -e 10 -dl 5120  -mef || true
python train.py -wb -g a5000 --model transformer -t svgd_push -n 2 -d 1 -e 10 -dl 5120  -mef || true
python train.py -wb -g a5000 --model transformer -t svgd_push -n 4 -d 1 -e 10 -dl 5120  -mef || true
python train.py -wb -g a5000 --model transformer -t svgd_push -n 8 -d 1 -e 10 -dl 5120  -mef || true
python train.py -wb -g a5000 --model cgcnn -t ensemble_push -n 1 -d 1 -e 10 -dl 800  || true
python train.py -wb -g a5000 --model cgcnn -t ensemble_push -n 2 -d 1 -e 10 -dl 800  || true
python train.py -wb -g a5000 --model cgcnn -t ensemble_push -n 4 -d 1 -e 10 -dl 800  || true
python train.py -wb -g a5000 --model cgcnn -t ensemble_push -n 8 -d 1 -e 10 -dl 800  || true
python train.py -wb -g a5000 --model cgcnn -t mswag_push -n 1 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 800  || true
python train.py -wb -g a5000 --model cgcnn -t mswag_push -n 2 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 800  || true
python train.py -wb -g a5000 --model cgcnn -t mswag_push -n 4 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 800  || true
python train.py -wb -g a5000 --model cgcnn -t mswag_push -n 8 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 800  || true
python train.py -wb -g a5000 --model cgcnn -t svgd_push -n 1 -d 1 -e 10 -dl 800  -mef || true
python train.py -wb -g a5000 --model cgcnn -t svgd_push -n 2 -d 1 -e 10 -dl 800  -mef || true
python train.py -wb -g a5000 --model cgcnn -t svgd_push -n 4 -d 1 -e 10 -dl 800  -mef || true
python train.py -wb -g a5000 --model cgcnn -t svgd_push -n 8 -d 1 -e 10 -dl 800  -mef || true
python train.py -wb -g a5000 --model resnet -t ensemble_push -n 1 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g a5000 --model resnet -t ensemble_push -n 2 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g a5000 --model resnet -t ensemble_push -n 4 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g a5000 --model resnet -t ensemble_push -n 8 -d 1 -e 10 -dl 5120  || true
python train.py -wb -g a5000 --model resnet -t mswag_push -n 1 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 5120  || true
python train.py -wb -g a5000 --model resnet -t mswag_push -n 2 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 5120  || true
python train.py -wb -g a5000 --model resnet -t mswag_push -n 4 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 5120  || true
python train.py -wb -g a5000 --model resnet -t mswag_push -n 8 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 5120  || true
python train.py -wb -g a5000 --model resnet -t svgd_push -n 1 -d 1 -e 10 -dl 5120  -mef || true
python train.py -wb -g a5000 --model resnet -t svgd_push -n 2 -d 1 -e 10 -dl 5120  -mef || true
python train.py -wb -g a5000 --model resnet -t svgd_push -n 4 -d 1 -e 10 -dl 5120  -mef || true
python train.py -wb -g a5000 --model resnet -t svgd_push -n 8 -d 1 -e 10 -dl 5120  -mef || true
python train.py -wb -g a5000 --model unet -t ensemble_push -n 1 -d 1 -e 10 -dl 2000 -cp || true
python train.py -wb -g a5000 --model unet -t ensemble_push -n 2 -d 1 -e 10 -dl 2000 -cp || true
python train.py -wb -g a5000 --model unet -t ensemble_push -n 4 -d 1 -e 10 -dl 2000 -cp || true
python train.py -wb -g a5000 --model unet -t ensemble_push -n 8 -d 1 -e 10 -dl 2000 -cp || true
python train.py -wb -g a5000 --model unet -t mswag_push -n 1 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 2000 -cp || true
python train.py -wb -g a5000 --model unet -t mswag_push -n 2 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 2000 -cp || true
python train.py -wb -g a5000 --model unet -t mswag_push -n 4 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 2000 -cp || true
python train.py -wb -g a5000 --model unet -t mswag_push -n 8 -d 1 --pretrain_epochs 3 --swag_epochs 10 -dl 2000 -cp || true
python train.py -wb -g a5000 --model unet -t svgd_push -n 1 -d 1 -e 10 -dl 2000 -cp -mef || true
python train.py -wb -g a5000 --model unet -t svgd_push -n 2 -d 1 -e 10 -dl 2000 -cp -mef || true
python train.py -wb -g a5000 --model unet -t svgd_push -n 4 -d 1 -e 10 -dl 2000 -cp -mef || true
python train.py -wb -g a5000 --model unet -t svgd_push -n 8 -d 1 -e 10 -dl 2000 -cp -mef || true
