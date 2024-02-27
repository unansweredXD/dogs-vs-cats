import torch
import wandb
from torch import nn
from tqdm import tqdm

from src.metrics.metrics_monitor import MetricMonitor


class ModelTrainer(nn.Module):
    def __init__(self, device, train_loader, val_loader, model, criterion, optimizer, epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch

    def train_model(self, epoch):
        metric_monitor = MetricMonitor()
        self.model.train()
        stream = tqdm(self.train_loader)

        for i, (images, target) in enumerate(stream, start=1):
            metrics = self.get_metrics(images, target, metric_monitor)

            self.optimizer.zero_grad()
            metrics['loss'].backward()
            self.optimizer.step()
            stream.set_description(f'Epoch: {epoch}.\tTrain.\t {metric_monitor}')

        wandb.log(
            {
                'train_accuracy': metric_monitor.metrics['Accuracy']['avg'],
                'train_loss': metric_monitor.metrics['Loss']['avg'],
                'train_F1-score': metric_monitor.metrics['F1-score']['avg']
            }
        )

        self.model.eval()
        stream = tqdm(self.val_loader)

        with torch.no_grad():
            for i, (images, target) in enumerate(stream, start=1):
                metrics = self.get_metrics(images, target, metric_monitor)

                stream.set_description(f'Epoch: {epoch}.\tValidation.\t {metric_monitor}')

        wandb.log(
            {
                'validation_accuracy': metric_monitor.metrics['Accuracy']['avg'],
                'validation_loss': metric_monitor.metrics['Loss']['avg'],
                'validation_F1-score': metric_monitor.metrics['F1-score']['avg']
            }
        )

    def get_metrics(self, images, target, monitor):
        images = images.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        output = self.model(images)
        accuracy = MetricMonitor.calculate_accuracy(output, target)
        loss = self.criterion(output, target)
        f1 = MetricMonitor.model_f1_score(output, target)

        monitor.update('Accuracy', accuracy)
        monitor.update('Loss', loss.item())
        monitor.update('F1-score', f1)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'f1': f1
        }

    def start_training(self):
        for epoch in range(self.epoch):
            self.train_model(epoch + 1)
            torch.save(self.model.state_dict(), 'model.pt')
