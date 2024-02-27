from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.metrics = None
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'val': 0, 'count': 0, 'avg': 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric['val'] += val
        metric['count'] += 1
        metric['avg'] = metric['val'] / metric['count']

    def __str__(self):
        return ' | '.join(
            [
                '{metric_name}: {avg:.{float_precision}f}'.format(
                    metric_name=metric_name, avg=metric['avg'], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

    @staticmethod
    def calculate_accuracy(output, target):
        acc = ((output.argmax(dim=1) == target).float().mean())
        return acc

    @staticmethod
    def model_f1_score(y, y_pred):
        return f1_score(y.cpu().data.max(1)[1], y_pred.cpu())

    @staticmethod
    def model_matrix(y, y_pred):
        return confusion_matrix(y.data.max(1)[1], y_pred.cpu())
