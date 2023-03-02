import torch
from thop import profile


'''
compute parameters and flops
'''

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

def compute_params(model_func):
    model = model_func
    # print(model)
    input = torch.randn(1, 3, 224, 224)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input = input.to(device)
    # model.eval()
    flops, params = profile(model, inputs=(input, ))
    print("flops = ", flops)
    print("params = ", params)
    flops, params = clever_format([flops, params], "%.3f")
    print("flops = ", flops)
    print("params = ", params)