import argparse
import json
import os
import shutil
import sys
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import configs.config_bert as CONFIG
from utils_data_bert import build_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='data', type=str)
parser.add_argument('--lr', type=str)
args = parser.parse_args()
EXPT_DIR = os.path.join(args.data_root, 'experiments', "ChartFC")
CONFIG.root = args.data_root
LEARNING_RATE = float(args.lr)


def make_experiment_directory(CONFIG):
    if not os.path.exists(EXPT_DIR):
        os.makedirs(EXPT_DIR)


def inline_print(text):
    """
    A simple helper to print text inline. Helpful for displaying training progress among other things.
    Args:
        text: Text to print inline
    """
    sys.stdout.write('\r' + text)
    sys.stdout.flush()


def train_epoch(model, train_loader, criterion, optimizer, epoch, config, val_loaders, test_loaders):
    model.train()

    correct = 0
    total = 0
    total_loss = 0
    accumulation_steps = 2

    for batch_idx, (txt, label, img, qid, ql, ocr, ocrl) in enumerate(train_loader):
        i = img.to("cuda")
        a = label.to("cuda") # answer as string
        p = model(i, txt, ql, ocr, ocrl)

        loss = criterion(p, a)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        p_scale = torch.sigmoid(p)
        pred_class = p_scale >= 0.5
        c = float(torch.sum(pred_class.float() == a))
        # print(f"pred_class.float(): {str(pred_class.float())}, a: {a}")

        correct += c
        a_numpy = a.cpu().detach().numpy()
        pred_class_numpy = pred_class.float().cpu().detach().numpy()
        f1_result_macro = f1_score(a_numpy, pred_class_numpy, average="macro")
        total += len(ql)
        total_loss += loss * len(ql)
        inline_print(
            f'Running {train_loader.dataset.split}, Processed {total} of {len(train_loader) * train_loader.batch_size} '
            f', Accuracy: {round(correct / total, 3)}, '
            f'F1 macro: {round(f1_result_macro, 3)}'
            f', Loss: {total_loss / total}, Learning Rate: {[param_group["lr"] for param_group in optimizer.param_groups]}'
        )

        # validate after steps = 250
        if batch_idx % 400 == 0 and batch_idx != 0:
            print(f'\nTrain Accuracy for Steps {batch_idx}: {correct / total}')
            print(f'{train_loader.dataset.split} F1 macro for Steps {batch_idx}: {f1_result_macro}\n')

            predict(model, val_loaders, epoch, steps=batch_idx)
            predict(model, test_loaders, epoch, steps=batch_idx)
            model.train()

    print(f'\nTrain Accuracy for Epoch {epoch + 1}: {correct / total}')
    print(f'{train_loader.dataset.split} F1 macro for Epoch {epoch + 1}: {f1_result_macro}\n')

    return correct/total


def predict(model, dataloaders, epoch, steps = "total"):
    model.eval()
    for data in dataloaders:
        correct = 0
        total = 0
        results = dict()
        with torch.no_grad():
            for q, a, i, qid, ql, ocr, ocrl in data:
                i = i.to("cuda")  # image
                a = a.to("cuda")  # answer as string
                p = model(i, q, ql, ocr, ocrl)
                _, idx = p.max(dim=1)

                p_scale = torch.sigmoid(p)
                pred_class = p_scale >= 0.5
                c = float(torch.sum(pred_class.float() == a))
                for qqid, curr_pred_class in zip(qid, pred_class):
                    qqid = int(qqid.item())
                    if qqid not in results:
                        results[qqid] = int(curr_pred_class)

                correct += c
                total += len(ql)

                a_numpy = a.cpu().detach().numpy()
                pred_class_numpy = idx.float().cpu().detach().numpy()
                f1_result_macro = f1_score(a_numpy, pred_class_numpy, average="macro")
                inline_print(
                    f'Running {data.dataset.split}, Processed {total} of {len(data) * data.batch_size} '
                    f', Accuracy: {round(correct / total, 3)}, '
                    f'F1 macro: {round(f1_result_macro, 3)}'
                )

        result_file = os.path.join(EXPT_DIR, f'results_{data.dataset.split}_{epoch + 1}.json')
        json.dump(results, open(result_file, 'w'))
        print(f"Saved {result_file}")

        print(f'\n{data.dataset.split} Accuracy for Epoch {epoch + 1}, Steps {steps}: {correct / total}')
        print(f'{data.dataset.split} F1 macro for Epoch {epoch + 1}, Steps {steps}: {f1_result_macro}\n')


def update_learning_rate(epoch, optimizer, config):
    # if epoch < len(config.lr_warmup_steps):
    #     optimizer.param_groups[0]['lr'] = config.lr_warmup_steps[epoch]*optimizer.param_groups[0]['lr']
    if epoch in config.lr_decay_epochs:
        optimizer.param_groups[0]['lr'] *= config.lr_decay_rate


def train(config, model, train_loader, val_loaders, test_loaders, optimizer, criterion, start_epoch):
    for epoch in range(start_epoch, config.max_epochs):
        update_learning_rate(epoch, optimizer, config)
        accuracy = train_epoch(model, train_loader, criterion, optimizer, epoch, config, val_loaders, test_loaders)
        curr_epoch_path = os.path.join(EXPT_DIR, str(epoch + 1) + '.pth')
        latest_path = os.path.join(EXPT_DIR, 'latest.pth')
        data = {'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr']}
        # torch.save(data, curr_epoch_path)
        torch.save(data, latest_path)

        if epoch % config.test_interval == 0 or epoch >= config.test_every_epoch_after:
            predict(model, val_loaders, epoch)
            predict(model, test_loaders, epoch)


def main():
    make_experiment_directory(CONFIG)

    # create model and data loaders
    train_data, val_data, test_data, n1, n2 = build_dataloaders(CONFIG)
    lut_dict = {'ans2idx': train_data.dataset.label2idx,
                'ques2idx': train_data.dataset.txt2idx,
                'maxlen': train_data.dataset.maxlen}
    json.dump(lut_dict, open(os.path.join(EXPT_DIR, 'LUT.json'), 'w'))
    shutil.copy(f'/scratch/users/k20116188/prefil/configs/config_uniter.py',
                os.path.join(EXPT_DIR, 'config_uniter.py'))

    model = CONFIG.model(n1, 1, CONFIG)
    print("Model Overview: ")
    print(model)
    model.to("cuda")

    # set optimizer, criterion
    optimizer = CONFIG.optimizer(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # start training
    train(CONFIG, model, train_data, val_data, test_data, optimizer, criterion, 0)


if __name__ == '__main__':
    main()
