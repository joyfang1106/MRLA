import torch
from thop import profile
import torchvision
import models
import argparse
import os


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Compute model parameters and FLOPs')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50_mrlal',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50_mrlal)')
parser.add_argument('--input-size', '-i', default=224, type=int,
                    metavar='N', help='input image size (default: 224)')
parser.add_argument('--work-dir', default='work_dirs', type=str, 
                    help='the dir to save logs and models')
parser.add_argument('--log-dir', '-ld', default='None', type=str, 
                    help='the sub dir to save logs and models of a model architecture')



def main():
    global args
    args = parser.parse_args()
    model = models.__dict__[args.arch]() 
    print(model)
    input = torch.randn(1, 3, args.input_size, args.input_size)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input = input.to(device)
    # model.eval()
    flops, params = profile(model, inputs=(input, ))
    print("flops = ", flops)
    print("params = ", params)
    if args.log_dir is not None:
        save_dir = os.path.join(args.work_dir, args.log_dir)
        with open(os.path.join(save_dir, 'params.txt'), "w") as f:
            f.write("total params: {:d},  {:.2f}M \n".format(int(params), params/1000**2))
            f.write("flops: {:d},  {:.2f}B".format(int(flops), flops/1000**3))
            f.close()
    flops, params = clever_format([flops, params], "%.3f")
    print("flops = ", flops)
    print("params = ", params)

def clever_format(nums, format="%.2f"):
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1024 ** 4) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1024 ** 3) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1024 ** 2) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1024) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums


if __name__ == '__main__':
    main()