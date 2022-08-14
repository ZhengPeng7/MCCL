import argparse
import torch
from time import time

from models.GCoNet import GCoNet


def main(args):
    # Init model
    device = torch.device("cuda")
    model = GCoNet()
    model.to(device)
    model.eval()

    N = 2
    with torch.no_grad():
        time_avg = 0.
        buf_iter = 100
        for i in range(1, 10000+1+buf_iter):
            inputs = torch.randn(N, 3, args.size, args.size).float().cuda()
            time_st = time()
            _ = model(inputs)
            if i > buf_iter:
                time_latest = time() - time_st
                time_avg += time_latest
                print(i, time_avg / (i - buf_iter) / N)
                print(i, time_latest / N)


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='GCoNet',
                        type=str,
                        help="Options: '', ''")
    parser.add_argument('--testsets',
                       default='CoCA+CoSOD3k+CoSal2015',
                       type=str,
                       help="Options: 'CoCA','CoSal2015','CoSOD3k','iCoseg','MSRC'")
    parser.add_argument('--size',
                        default=256,
                        type=int,
                        help='input size')
    parser.add_argument('--ckpt', default='./ckpt/gconet/final.pth', type=str, help='model folder')
    parser.add_argument('--pred_dir', default='/root/datasets/sod/preds/GCoNet_ext', type=str, help='Output folder')

    args = parser.parse_args()

    main(args)


