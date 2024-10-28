import math

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    base_log_dir = 'logs'
    experiment_name = 'experiment-3'
    sw = SummaryWriter(log_dir=f'{base_log_dir}/{experiment_name}')
    for iteration in range(0, 1000):
        sw.add_scalar('metric_1', iteration * 0.05, iteration)
        sw.add_scalar('metric_2', math.log(10 + iteration), iteration)
    sw.close()