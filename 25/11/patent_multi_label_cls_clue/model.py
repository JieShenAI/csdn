import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from settings import PATENT_CLS_NAMES


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # 将 logits 转换为概率并得到预测结果
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    # 计算整体准确率
    accuracy = accuracy_score(labels, preds)

    # 计算每个类别的 Precision, Recall, F1
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    # 计算宏观平均指标
    # precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    # recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)

    # 构建结果字典
    metrics = {
        'accuracy': accuracy,
        # 'precision_macro': precision_macro,
        # 'recall_macro': recall_macro,
        'f1': f1_macro
    }

    # 添加每个类别的指标
    for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        metrics[f'f1_class_{i}'] = f

    return metrics


class PatentClassifier(nn.Module):
    def __init__(self, model_name):
        super(PatentClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(PATENT_CLS_NAMES))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def compute_loss(self, predictions, targets):
        # criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(predictions, targets)
        return loss

    # def forward(self, text_tokens, labels=None):
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            return {
                "loss": loss,
                "logits": logits,
            }
        return {"logits": logits}

