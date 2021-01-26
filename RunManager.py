"""
created by PyCharm
date: 2021/1/14
time: 22:53
user: wkc
"""
import time
import datetime
import json
import os
import pandas as pd
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision


class RunManager:
    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_correct_num = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

        self.result_dir = './result'

    def begin_run(self, run, network, loader):
        """
        分配参数后开始运行 初始化参数 在tensorboard中显示图片数据
        :param run: 参数实例
        :param network: 神经网络实例
        :param loader: 数据加载器实例
        """
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images.to(getattr(run, 'device', 'run')))  # run中的device属性存在则使用,不存在则使用默认属性cpu

    def end_run(self):
        """
        每次指定参数运行结束 关闭tensorboard 参数初始化
        """
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        """
        epoch开始运行 初始化参数
        """
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_correct_num = 0

    def end_epoch(self):
        """
        epoch结束运行 计算loss和accuracy 向tensorboard中添加运行参数 统计运行结果
        :return: DataFrame: self.run_data
        """
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)  # len(self.loader.dataset) = 60000
        accuracy = self.epoch_correct_num / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        run_data_df = pd.DataFrame(self.run_data)

        return run_data_df

    def track_loss(self, loss):
        """
        累加损失
        :param loss: 每轮epoch的loss
        """
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_correct_num(self, preds, labels):
        """
        累加准确结果
        :param preds: 预测值
        :param labels: 真实值
        """
        self.epoch_correct_num += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        """
        得到预测准确的值
        :param preds: 预测值
        :param labels: 真实值
        :return: 预测值中预测准确的值
        """
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, file_name):
        """
        运行结果保存 默认路径 ./result
        :param file_name: 文件名
        """
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        result_path = os.path.join(self.result_dir, file_name)
        time_index = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        pd.DataFrame(self.run_data).to_csv(f'{result_path}_{time_index}.csv', index=False)
        with open(f'{result_path}_{time_index}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
