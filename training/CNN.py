from pathlib import Path
import torch
import torch.nn.functional as f
import lightning as pl
from sklearn.metrics import precision_score, f1_score, confusion_matrix
import torch.nn as nn
import numpy as np

from utils.utility import cm_analysis


class CNN(pl.LightningModule):

    def __init__(self, input_dim, fold, classes_names, output_dim=3, learning_rate=1e-3):
        super(CNN, self).__init__()
        self.name = "CNN"
        self.classes_labels = classes_names
        self.fold = fold
        self.classes = output_dim

        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

        self.learning_rate = learning_rate
        self.loss_function = nn.CrossEntropyLoss()

        self.input_dim = 6
        self.dim = 64
        self.filter_size = 3
        self.window_size = input_dim

        # Convolution Branch
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.window_size, self.dim, self.filter_size),
            nn.BatchNorm1d(self.dim),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.1)
        )  # (_, 64, 2)

        self.fc1 = nn.Sequential(
            nn.Linear(self.dim * 2, 128),
            nn.PReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def id(self):
        return f"{self.name}_{self.window_size}"

    def forward(self, x):
        # input: (batch size, d_model, length)
        # x = x.view(-1, self.input_dim, self.window_size)
        x = self.conv1(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.compute_loss(y_hat, y)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Collect predictions and targets
        self.val_predictions.append(np.argmax(y_hat.cpu().numpy(), 1))
        self.val_targets.append(np.argmax(y.cpu().numpy(), 1))

        return val_loss

    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets from this epoch
        val_predictions = np.concatenate(self.val_predictions)
        val_targets = np.concatenate(self.val_targets)

        # Log or print confusion matrix and classification report
        precision = precision_score(val_targets, val_predictions, average='macro', zero_division=0)
        f1 = f1_score(val_targets, val_predictions, average='macro', zero_division=0)
        self.log("prec_macro", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("f1_score", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Clear stored values for the next epoch
        self.val_predictions.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.compute_loss(y_hat, y)
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Collect predictions and targets
        self.test_predictions.append(np.argmax(y_hat.cpu().numpy(), 1))
        self.test_targets.append(np.argmax(y.cpu().numpy(), 1))

        return test_loss

    def on_test_end(self):
        output_path = f"output/{self.id()}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        test_predictions = np.concatenate(self.test_predictions)
        test_target = np.concatenate(self.test_targets)
        cm_analysis(
            test_target,
            test_predictions,
            f"{output_path}/confusion_matrix_segments_fold_{self.fold}",
            range(self.classes),
            self.classes_labels,
            specific_title=f"Segments: {self.id()} fold {self.fold}"
        )
        self.fold += 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        return optimizer

    def compute_loss(self, y_hat, y):
        return self.loss_function.forward(y_hat, y)
