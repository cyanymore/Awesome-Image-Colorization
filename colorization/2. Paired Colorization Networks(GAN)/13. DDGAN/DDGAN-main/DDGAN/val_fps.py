import torch
import time
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from module import Generator
from torchsummary import summary
from thop import profile, clever_format

# 查看模型结构
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Generator(3, 3, 32)
net = net.to(device).cuda()
summary(net, (3, 256, 256))

# 查看模型FLOPs和参数量
_input = torch.randn(1, 3, 256, 256).to(device)
flops, params = profile(net, inputs=(_input,))  # 自定义模块需要：custom_ops={'BasicBlock':model}
flops, params = clever_format([flops, params], '%.6f')
print('FLOPs:', flops, '\tparams:', params)


def compute_speed(model, input_size, device, iteration=100):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(50):
        model(input)

    print('=========Eval Forward Time=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--size", type=str, default="256,256", help="input size of model")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--model', type=str, default='Net')
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    args = parser.parse_args()

    h, w = map(int, args.size.split(','))
    compute_speed(net, (args.batch_size, args.num_channels, h, w), int(args.gpus), iteration=args.iter)
