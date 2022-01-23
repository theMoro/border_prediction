"""
Author: Tobias Morocutti
Matr.Nr.: K12008172
Exercise 5
"""

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter
import os
import glob

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils

LEARNING_RATE = 0.0005
EPOCHS = 25
BATCH_SIZE = 32

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide1 = nn.Conv2d(2, 64, kernel_size=11, padding=5)
        self.wide3 = nn.Conv2d(64, 32, kernel_size=7, padding=3)

        self.conv1 = nn.Conv2d(2, 64, kernel_size=11, padding=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=9, padding=4)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)

    def forward(self, x):
        wide1 = self.wide1(x)
        wide1 = self.bn64(wide1)
        wide1 = F.selu(wide1)

        x = self.conv1(x)
        x = self.bn64(x)
        x = F.selu(x)

        x = self.conv2(x)
        x = self.bn64(x)
        x = F.selu(x)

        wide2 = self.wide3(x)
        wide2 = self.bn32(wide2)
        wide2 = F.selu(wide2)

        x = F.alpha_dropout(x, p=0.2)

        x = self.conv3(x + wide1)
        x = self.bn64(x)
        x = F.selu(x)

        x = self.conv4(x)
        x = self.bn32(x)
        x = F.selu(x)

        x = F.alpha_dropout(x, p=0.2)

        x = self.conv5(x + wide2)
        x = self.bn32(x)
        x = F.selu(x)

        return self.conv6(x)


def evaluate_model(model: nn.Module, data_loader: DataLoader, loss_function, phase):
    loss_result = 0
    all_output_predictions = []

    writer = SummaryWriter(log_dir=os.path.join("results", "experiment_00"))

    model.eval()

    for idx, (inputs, targets, masks, means, stds, sample_ids) in enumerate(tqdm(data_loader)):
        actual_batch_size = inputs.shape[0]
        masks = masks.to(device).view(actual_batch_size, -1)
        inputs = inputs.to(device).view(actual_batch_size, 2, 90, 90)

        if phase != 'test':
            targets = targets.to(device).view(actual_batch_size, 1, 90, 90)

        with torch.no_grad():
            preds = model(inputs)
            preds = preds.view(actual_batch_size, 1, 90, 90)

            denormalized_preds = denormalize(preds, means, stds) * 255
            denormalized_preds = torch.clip(denormalized_preds, 0, 255)

            if phase != 'test':
                targets = torch.clip(targets.float(), 0, 255)
                loss_result += loss_function(denormalized_preds, targets)

            denormalized_preds = denormalized_preds.view(-1, 8100)

            output_preds = []
            for i, pred in enumerate(denormalized_preds):
                output_pred = torch.masked_select(pred, masks[i] == 0)
                output_pred = torch.clip(output_pred, 0, 255)
                output_preds.append(output_pred)

            _ = [all_output_predictions.append(np.array(np.round(x.cpu().numpy(), decimals=0), dtype=np.uint8))
                 for x in output_preds]

            if idx == 0 and phase != 'test':
                tqdm.write(f'now showing images of set {phase}')
                input_images = (denormalize(inputs[:4, 0], means[:4], stds[:4])) * 255
                img_grid = torchvision.utils.make_grid(input_images)
                writer.add_image('Input images', img_grid)

                mask_images = masks[:4].view(-1, 90, 90) * 255
                img_grid = torchvision.utils.make_grid(mask_images)
                writer.add_image('Mask images', img_grid)

                target_images = targets[:4]
                img_grid = torchvision.utils.make_grid(target_images)
                writer.add_image('Target images', img_grid)

                pred_images = denormalized_preds[:4].view(-1, 90, 90)
                img_grid = torchvision.utils.make_grid(pred_images)
                writer.add_image('My Prediction images', img_grid)

                writer.close()

                # to visualize the first 4 images
                for i in range(4):
                    input_img = Image.fromarray(input_images[i].cpu().numpy())
                    known_img = Image.fromarray(mask_images[i].cpu().numpy())
                    target_img = Image.fromarray((target_images[i].cpu().numpy()[0]))
                    pred_img = Image.fromarray(pred_images[i].cpu().numpy().astype(np.uint8))
                    utils.visualize_images(input_img, known_img, target_img, pred_img)

    return loss_result / len(data_loader), all_output_predictions


def denormalize(data, means, stds):
    for i in range(data.shape[0]):
        if len(data.shape) == 4:
            data[i] = ((data[i].flatten() * stds[i]) + means[i]).view(data.shape[1], data.shape[2], data.shape[3])
        elif len(data.shape) == 3:
            data[i] = ((data[i].flatten() * stds[i]) + means[i]).view(data.shape[1], data.shape[2])

    return data


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, loss_function):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    lambda_lr = lambda epoch: 0.85 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    model.train()
    errors = []

    pbar = tqdm(total=len(train_loader))

    writer = SummaryWriter(log_dir=os.path.join("results", "experiment_00"))
    update = 0

    best_val_loss = -1
    patience_limit = 2
    patience = 0

    for epoch in range(EPOCHS):

        pbar.set_description('Epoch %d/%d' % (epoch, EPOCHS))
        pbar.reset()
        errors.append(0)

        for inputs, targets, masks, means, stds, sample_ids in train_loader:

            actual_batch_size = inputs.shape[0]

            inputs = inputs.to(device).view(actual_batch_size, 2, 90, 90)
            targets = targets.to(device).view(actual_batch_size, 1, 90, 90)

            preds = model(inputs)
            preds = preds.view(actual_batch_size, 1, 90, 90)

            denormalized_preds = denormalize(preds, means, stds) * 255

            denormalized_preds = torch.clip(denormalized_preds, 0, 255)
            targets = torch.clip(targets.float(), 0, 255)
            error = loss_function(denormalized_preds, targets)

            # Add l2 regularization
            l2_term = torch.mean(torch.stack([(param ** 2).mean()
                                              for param in model.parameters()]))
            # Compute final loss
            loss = error + l2_term * 1e-2

            errors[-1] += error.item()
            loss.backward()

            optimizer.step()

            # region TensorBoard
            if update % 50 == 0:
                writer.add_scalar(tag="training/main_loss",
                                  scalar_value=error.cpu(),
                                  global_step=update)
                writer.add_scalar(tag="training/l2_term",
                                  scalar_value=l2_term.cpu(),
                                  global_step=update)
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(),
                                  global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(model.parameters()):
                    writer.add_histogram(tag=f'training/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(model.parameters()):
                    writer.add_histogram(tag=f'training/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)

            update += 1
            # endregion

            optimizer.zero_grad()

            pbar.update()

        val_loss, _ = evaluate_model(model, val_loader, loss_function, phase='val_during_train')

        errors[-1] /= len(train_loader)
        tqdm.write('Epoch %d finished training loss: %g, validation loss: %g' % (epoch + 1, errors[-1], val_loss))
        tqdm.write('Updates: %d, Learning rate: %g' % (update, scheduler.get_last_lr()[0]))

        scheduler.step()

        # early stopping
        if val_loss > best_val_loss != -1:
            patience += 1

            if patience >= patience_limit and val_loss <= 200:
                tqdm.write("Training got stopped early!")
                torch.save(model, os.path.join("results/models", f"trained_model_stopped_early.pt"))
                break
        else:
            best_val_loss = val_loss
            patience = 0

    writer.close()
    pbar.close()

    model_nr = len(glob.glob(os.path.join("results/models", "**", "*.pt"), recursive=False))
    torch.save(model, os.path.join("results/models", f"trained_model_{model_nr}.pt"))
