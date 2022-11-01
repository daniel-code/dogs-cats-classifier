import os

import numpy as np
import seaborn as sns
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, auc, accuracy_score, precision_score, recall_score, \
    roc_auc_score
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, output_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model
        self.model.eval()
        self.model.to(self.device)

        self.output_path = output_path

    def inference_dataloader(self, dataloader):
        predictions = []
        ground_truth = []

        with tqdm(total=len(dataloader)) as pbar:
            with torch.no_grad():
                for x, y in dataloader:
                    x = x.to(self.device)

                    y_pred = self.model(x)

                    predictions.extend(y_pred.detach().to('cpu').numpy().tolist())
                    ground_truth.extend(y.detach().to('cpu').numpy().tolist())

                    pbar.update()

        ground_truth = np.array(ground_truth).flatten()
        predictions = np.array(predictions).flatten()
        class_predictions = (predictions > 0.5).astype(int)

        return ground_truth, predictions, class_predictions

    def evaluate(self, dataloader, title, verbose=False):
        ground_truth, predictions, class_predictions = self.inference_dataloader(dataloader=dataloader)
        print('Title:', title)
        print('Accuracy:', accuracy_score(ground_truth, class_predictions))
        print('Precision:', precision_score(ground_truth, class_predictions))
        print('Recall:', recall_score(ground_truth, class_predictions))
        print('AUC:', roc_auc_score(ground_truth, predictions))

        self.plot_auc_confusion_matrix(ground_truth,
                                       predictions,
                                       title=title,
                                       output_path=self.output_path,
                                       verbose=verbose)

        return ground_truth, predictions, class_predictions

    def plot_auc_confusion_matrix(self, y_ture: np.ndarray, y_pred: np.ndarray, title: str, output_path, verbose=False):
        y_class = (y_pred > 0.5).astype(int)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # plot ROC-AUC
        fpr, tpr, _ = roc_curve(y_ture, y_pred)
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
        cf_matrix = confusion_matrix(y_ture, y_class, labels=[0, 1])
        sns.heatmap(cf_matrix, annot=True, fmt='d', xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'], ax=axs[1])
        axs[1].set_xlabel('Ground Truth')
        axs[1].set_ylabel('Model')
        axs[1].set_title(f'Acc:{accuracy_score(y_ture, y_class):0.3f}')

        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{title}.png'), dpi=300)
        if verbose:
            plt.show()

    def plot_images(self, df, title, output_path, n_cols=10, verbose=False):
        nrows = len(df.index) // n_cols + 1
        fig, axs = plt.subplots(nrows=nrows, ncols=n_cols, figsize=(n_cols, int(nrows * 2.5)))

        for i, (idx, data) in enumerate(df.iterrows()):
            image = Image.open(data['filename'])
            if nrows > 1:
                ax = axs[i // n_cols][i % n_cols]
            else:
                ax = axs[i % n_cols]

            ax.imshow(image)
            ax.set_title('cat' if data['ground_truth'] == 0 else 'dogs')

        for i in range(nrows * n_cols):
            if nrows > 1:
                ax = axs[i // n_cols][i % n_cols]
            else:
                ax = axs[i % n_cols]

            ax.axis('off')
        plt.tight_layout()

        plt.savefig(os.path.join(output_path, f'{title}_images.png'), dpi=300)
        if verbose:
            plt.show()
