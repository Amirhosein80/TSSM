import time
import torch
from typing import List

def fps_calculator(net: torch.nn.Module, input_shape: List[int], ) -> float:
    """
    Calculate model FPS
    :param net: model
    :param input_shape: input feature map shape (C, H, W)
    :return: FPS
    """
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    net.eval()
    net.to(device)
    iterations = None

    inputs = torch.randn(1, *input_shape).cuda()
    with torch.no_grad():
        for _ in range(10):
            net(inputs)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    net(inputs)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            net(inputs)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    return FPS
