import argparse

"""[Note]

python gen_train_script.py --model all -d 4 -cp
"""


def generate_one(args, model, f, devices):
    methods = ["ensemble_push", "mswag_push", "svgd_push"]
    if devices == 4:
        particles = [4, 8, 16, 32]
    elif devices == 2:
        particles = [2, 4, 8, 16]
    elif devices == 1:
        particles = [1, 2, 4, 8]

    cp = ""
    if model == "schnet" or model == "cgcnn":
        dl = 800
    elif model == "resnet" or model == "transformer":
        dl = 5120
    elif model == "unet" or model == "fno":
        dl = 2000
        if args.cloud_path:
            cp = "-cp"
        
    for method in methods:
        for p in particles:
            if method == "ensemble_push":
                f.write(f"python train.py -wb -g {args.group} --model {model} -t {method} -n {p} -d {devices} -e {args.epochs} -dl {dl} {cp} || true\n")
            elif method == "mswag_push":
                f.write(f"python train.py -wb -g {args.group} --model {model} -t {method} -n {p} -d {devices} --pretrain_epochs 3 --swag_epochs {args.epochs} -dl {dl} {cp} || true\n")
            else:
                f.write(f"python train.py -wb -g {args.group} --model {model} -t {method} -n {p} -d {devices} -e {args.epochs} -dl {dl} {cp} -mef || true\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", type=str, default="vit",
                        choices=[
                            "all",
                            "schnet",
                            "cgcnn",
                            "unet",
                            "fno",
                            "resnet",
                            "cnn",
                            "transformer"
                        ])
    parser.add_argument("-g", "--group", type=str, default='default')
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-d", "--devices", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("-cp", "--cloud_path", action='store_true')
    args = parser.parse_args()

    with open(f"paperspace_train_{args.model}_devices_{args.devices}.sh", "w") as f:
        if args.model == "all":
            models = ["schnet", "transformer", "cgcnn", "resnet", "unet", "fno"]
            for model in models:
                generate_one(args, model, f, args.devices)
        else:
            generate_one(args, args.model, f, args.devices)
