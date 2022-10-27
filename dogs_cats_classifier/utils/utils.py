import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, \
    auc

from tqdm import tqdm
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


def evaluate_model(model: LightningModule, dataloader: DataLoader, title: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)

    predictions = []
    ground_truth = []

    with tqdm(total=len(dataloader)) as pbar:
        for x, y in dataloader:
            x = x.to(device)

            y_pred = model(x)

            predictions.extend(y_pred.detach().to('cpu').numpy().tolist())
            ground_truth.extend(y.detach().to('cpu').numpy().tolist())

            pbar.update()

    predictions = np.array(predictions).flatten()
    class_predictions = predictions > 0.5
    ground_truth = np.array(ground_truth).flatten()

    print('Title:', title)
    accuracy = accuracy_score(ground_truth, class_predictions)
    print('Accuracy:', accuracy)
    print('Precision:', precision_score(ground_truth, class_predictions))
    print('Recall:', recall_score(ground_truth, class_predictions))
    print('AUC:', roc_auc_score(ground_truth, predictions))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # plot ROC-AUC
    fpr, tpr, _ = roc_curve(ground_truth, predictions)
    roc_auc = auc(fpr, tpr)

    lw = 2
    axs[0].plot(fpr, tpr, lw=lw, label=f'ROC curve (area = {roc_auc:0.3f})')
    axs[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle="--")
    axs[0].set_xlim(0.0, 1.0)
    axs[0].set_ylim(0.0, 1.05)
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].legend(loc='lower right')

    # plot confusion matrix
    cf_matrix = confusion_matrix(ground_truth, class_predictions, labels=[0, 1])
    sns.heatmap(cf_matrix, annot=True, fmt='d', xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'], ax=axs[1])
    axs[1].set_xlabel('Ground Truth')
    axs[1].set_ylabel('Model')
    axs[1].set_title(f'Acc:{accuracy:0.3f}')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'./reports/figures/{title}.png', dpi=300)
    plt.show()
