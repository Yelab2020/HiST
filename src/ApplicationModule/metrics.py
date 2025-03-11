from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        

def calculate_metrics(preds, targets):
    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='weighted',zero_division=1)
    recall = recall_score(targets, preds, average='weighted',zero_division=1)
    f1 = f1_score(targets, preds, average='weighted',zero_division=1)
    return acc, precision, recall, f1