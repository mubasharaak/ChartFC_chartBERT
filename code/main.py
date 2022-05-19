import argparse
import json
import os
import shutil
import sys
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, average_precision_score, recall_score
import configs.config_bert as CONFIG
from utils_data_bert import build_dataloaders
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--data_root', default='data', type=str)
parser.add_argument('--lr', type=str)
args = parser.parse_args()
EXPT_DIR = os.path.join(args.data_root, 'experiments', "bert_densenet_multiplication_baseline")
CONFIG.root = args.data_root
LEARNING_RATE = float(args.lr)
CONFIG.lr = LEARNING_RATE


def make_experiment_directory(CONFIG):
    if not os.path.exists(EXPT_DIR) and not args.evaluate and not args.resume:
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
    y_true = []
    y_pred = []

    for batch_idx, (txt, label, img, img_path, qid, ql, ocr, ocrl) in enumerate(train_loader):
        i = img.to("cuda")
        a = label.to("cuda") # answer as string
        p = model(i, txt, None) # @todo replace 'ocr' by None to ignore OCR extracted text of charts

        loss = criterion(p, a)
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        p_scale = torch.sigmoid(p)
        pred_class = p_scale >= 0.5
        c = float(torch.sum(pred_class.float() == a))

        # for i, entry in enumerate(pred_class):
        #     print(f"PREDICTION: {a[0] == entry.float()}")
        #     print(f"Text is : {txt[i]}")
        #     print(f"Img path is : {img_path[i]}")
        #     print(f"Label is : {a[i]}")
        #     print(f"Predicted is : {entry.float()}")
        #     print("______________________________________________________________________")

        correct += c
        a_numpy = a.cpu().detach().numpy()
        pred_class_numpy = pred_class.float().cpu().detach().numpy()
        # for calculating metrices later
        y_true.extend(a_numpy)
        y_pred.extend(pred_class_numpy)

        f1_result_macro = f1_score(a_numpy, pred_class_numpy, average="macro")
        total += len(ql)
        total_loss += loss * len(ql)
        inline_print(
            f'Running {train_loader.dataset.split}, Processed {total} of {len(train_loader) * train_loader.batch_size} '
            f', Accuracy: {round(correct / total, 3)}, '
            f'F1 macro: {round(f1_result_macro, 3)}'
            f', Loss: {total_loss / total}, Learning Rate: {[param_group["lr"] for param_group in optimizer.param_groups]}'
        )

        save_steps = 356
        if batch_idx % save_steps == 0 and batch_idx != 0:
            print(f'\nTrain Accuracy for Steps {batch_idx}: {correct / total}')
            print(f'{train_loader.dataset.split} F1 macro for Steps {batch_idx}: {f1_result_macro}\n')

            val_acc, val_f1_macro, val_f1_micro, val_prec_macro, val_prec_micro, val_recall_macro, val_recall_micro = predict(model, val_loaders, epoch, steps=batch_idx)
            test_acc, test_f1_macro, test_f1_micro, test_prec_macro, test_prec_micro, test_recall_macro, test_recall_micro = predict(model, test_loaders, epoch, steps=batch_idx)
            model.train()

            data = {'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr']}

            results_file = pd.read_csv(os.path.join(EXPT_DIR, "results.csv"))
            if all(val_acc > entry for entry in results_file["accuracy"]):
                best_path = os.path.join(EXPT_DIR, 'best_model.pth')
                torch.save(data, best_path)

            results_file = results_file.append({'epoch': str(epoch + 1) + '_' + str(save_steps),
                                                'dataset': "validation",
                                                'accuracy': val_acc, 'f1_macro': val_f1_macro, 'f1_micro': val_f1_micro,
                                                'recall_macro': val_recall_macro, 'recall_micro': val_recall_micro,
                                                'precision_macro': val_prec_macro, 'precision_micro': val_prec_micro,
                                                'learning_rate': CONFIG.lr, 'lr_decay_rate': CONFIG.lr_decay_rate,
                                                'lr_decay_step': CONFIG.lr_decay_step,
                                                'batch_size': CONFIG.batch_size},
                                               ignore_index=True)

            results_file = results_file.append({'epoch': str(epoch + 1) + '_' + str(save_steps),
                                                'dataset': "test",
                                                'accuracy': test_acc, 'f1_macro': test_f1_macro, 'f1_micro': test_f1_micro,
                                                'recall_macro': test_recall_macro, 'recall_micro': test_recall_micro,
                                                'precision_macro': test_prec_macro, 'precision_micro': test_prec_micro,
                                                'learning_rate': CONFIG.lr, 'lr_decay_rate': CONFIG.lr_decay_rate,
                                                'lr_decay_step': CONFIG.lr_decay_step,
                                                'batch_size': CONFIG.batch_size},
                                               ignore_index=True)
            results_file.to_csv(os.path.join(EXPT_DIR, "results.csv"), index=False)

    f1_result_macro = f1_score(y_true, y_pred, average="macro")
    print(f'\nTrain Accuracy for Epoch {epoch + 1}: {correct / total}')
    print(f'{train_loader.dataset.split} F1 macro for Epoch {epoch + 1}: {f1_result_macro}\n')

    return correct/total


def predict(model, dataloaders, epoch, steps = "total"):
    model.eval()
    for data in dataloaders:
        correct = 0
        total = 0
        results = dict()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for txt, a, i, img_path, qid, ql, ocr, ocrl in data:
                i = i.to("cuda")  # image
                a = a.to("cuda")  # answer as string
                p = model(i, txt, None)
                _, idx = p.max(dim=1)

                p_scale = torch.sigmoid(p)
                pred_class = p_scale >= 0.5
                c = float(torch.sum(pred_class.float() == a))

                # for i, entry in enumerate(pred_class):
                #     print(f"PREDICTION: {a[i] == entry.float()}")
                #     print(f"Text is : {txt[i]}")
                #     print(f"Img path is : {img_path[i]}")
                #     print(f"Label is : {a[i]}")
                #     print(f"Predicted is : {entry.float()}")
                #     print("______________________________________________________________________")

                for qqid, curr_pred_class in zip(qid, pred_class):
                    qqid = int(qqid.item())
                    if qqid not in results:
                        results[qqid] = int(curr_pred_class)

                correct += c
                total += len(ql)
                # for calculating metrices later
                a_numpy = a.cpu().detach().numpy()
                pred_class_numpy = pred_class.float().cpu().detach().numpy()
                y_true.extend(a_numpy)
                y_pred.extend(pred_class_numpy)

                inline_print(
                    f'Running {data.dataset.split}, Processed {total} of {len(data) * data.batch_size} '
                    f', Accuracy: {round(correct / total, 3)}, '
                )

        # result_file = os.path.join(EXPT_DIR, f'results_{data.dataset.split}_{epoch + 1}_{steps}.json')
        # json.dump(results, open(result_file, 'w'))
        # print(f"Saved {result_file}")

        f1_result_macro = f1_score(y_true, y_pred, average="macro")
        f1_result_micro = f1_score(y_true, y_pred, average="micro")
        precision_macro = average_precision_score(y_true, y_pred, average="macro")
        precision_micro = average_precision_score(y_true, y_pred, average="micro")
        recall_macro = recall_score(y_true, y_pred, average="macro")
        recall_micro = recall_score(y_true, y_pred, average="micro")

        print(f'\n{data.dataset.split} Accuracy for Epoch {epoch + 1}, Steps {steps}: {correct / total}')
        print(f'{data.dataset.split} F1 macro for Epoch {epoch + 1}, Steps {steps}: {f1_result_macro}\n')
        print(f'{data.dataset.split} F1 micro for Epoch {epoch + 1}, Steps {steps}: {f1_result_micro}\n')
        print(f'{data.dataset.split} Precision macro for Epoch {epoch + 1}, Steps {steps}: {precision_macro}\n')
        print(f'{data.dataset.split} Precision micro for Epoch {epoch + 1}, Steps {steps}: {precision_micro}\n')
        print(f'{data.dataset.split} Recall macro for Epoch {epoch + 1}, Steps {steps}: {recall_macro}\n')
        print(f'{data.dataset.split} Recall micro for Epoch {epoch + 1}, Steps {steps}: {recall_micro}\n')

        return correct / total, f1_result_macro, f1_result_micro, precision_macro, precision_micro, recall_macro, recall_micro


def update_learning_rate(epoch, optimizer, config):
    # if epoch < len(config.lr_warmup_steps):
    #     optimizer.param_groups[0]['lr'] = config.lr_warmup_steps[epoch]*optimizer.param_groups[0]['lr']
    if epoch in config.lr_decay_epochs:
        optimizer.param_groups[0]['lr'] *= config.lr_decay_rate


def train(config, model, train_loader, val_loaders, test_loaders, optimizer, criterion, start_epoch):
    for epoch in range(start_epoch, config.max_epochs):
        update_learning_rate(epoch, optimizer, config)
        train_epoch(model, train_loader, criterion, optimizer, epoch, config, val_loaders, test_loaders)
        data = {'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr']}
        # curr_epoch_path = os.path.join(EXPT_DIR, str(epoch + 1) + '.pth')
        # latest_path = os.path.join(EXPT_DIR, 'latest.pth')
        # torch.save(data, curr_epoch_path)
        # torch.save(data, latest_path)

        if epoch % config.test_interval == 0 or epoch >= config.test_every_epoch_after:
            val_acc, val_f1_macro, val_f1_micro, val_prec_macro, val_prec_micro, val_recall_macro, val_recall_micro = predict(model, val_loaders, epoch)
            test_acc, test_f1_macro, test_f1_micro, test_prec_macro, test_prec_micro, test_recall_macro, test_recall_micro = predict(model, test_loaders, epoch)

            results_file = pd.read_csv(os.path.join(EXPT_DIR, "results.csv"))
            if all(val_acc > entry for entry in results_file["accuracy"]):
                best_path = os.path.join(EXPT_DIR, 'best_model.pth')
                torch.save(data, best_path)

            results_file = results_file.append({'epoch': str(epoch + 1),
                                                'dataset': "validation",
                                                'accuracy': val_acc, 'f1_macro': val_f1_macro, 'f1_micro': val_f1_micro,
                                                'recall_macro': val_recall_macro, 'recall_micro': val_recall_micro,
                                                'precision_macro': val_prec_macro, 'precision_micro': val_prec_micro,
                                                'learning_rate': CONFIG.lr, 'lr_decay_rate': CONFIG.lr_decay_rate,
                                                'lr_decay_step': CONFIG.lr_decay_step,
                                                'batch_size': CONFIG.batch_size},
                                               ignore_index=True)

            results_file = results_file.append({'epoch': str(epoch + 1),
                                                'dataset': "test",
                                                'accuracy': test_acc, 'f1_macro': test_f1_macro, 'f1_micro': test_f1_micro,
                                                'recall_macro': test_recall_macro, 'recall_micro': test_recall_micro,
                                                'precision_macro': test_prec_macro, 'precision_micro': test_prec_micro,
                                                'learning_rate': CONFIG.lr, 'lr_decay_rate': CONFIG.lr_decay_rate,
                                                'lr_decay_step': CONFIG.lr_decay_step,
                                                'batch_size': CONFIG.batch_size},
                                               ignore_index=True)
            results_file.to_csv(os.path.join(EXPT_DIR, "results.csv"), index=False)


def main():
    make_experiment_directory(CONFIG)
    if args.evaluate or args.resume:
        CONFIG.lut_location = os.path.join(EXPT_DIR, 'LUT.json')
        train_data, val_data, test_data, n1, n2 = build_dataloaders(CONFIG)
    else:
        # create model and data loaders
        train_data, val_data, test_data, n1, n2 = build_dataloaders(CONFIG)
        lut_dict = {'ans2idx': train_data.dataset.label2idx,
                    'ques2idx': train_data.dataset.txt2idx,
                    'maxlen': train_data.dataset.maxlen}
        json.dump(lut_dict, open(os.path.join(EXPT_DIR, 'LUT.json'), 'w'))

        # first call
        # copy config
        if not os.path.exists(os.path.join(EXPT_DIR, 'config_bert.py')):
            shutil.copy(f'/scratch/users/k20116188/prefil/configs/config_bert.py',
                        os.path.join(EXPT_DIR, 'config_bert.py'))

    model = CONFIG.model(n1, 1, CONFIG)
    print("Model Overview: ")
    print(model)
    model.to("cuda")

    # save model summary
    if not os.path.exists(os.path.join(EXPT_DIR, 'model_summary.txt')):
        with open(os.path.join(EXPT_DIR, 'model_summary.txt'), "w") as file:
            print(model, file=file)
            file.close()

    # create and save empty results.csv file if not existing
    if not os.path.exists(os.path.join(EXPT_DIR, 'results.csv')):
        results = pd.DataFrame(columns=['epoch', 'dataset',
                                        'accuracy', 'f1_macro', 'f1_micro',
                                        'recall_macro', 'recall_micro',
                                        'precision_macro', 'precision_micro',
                                        'learning_rate', 'lr_decay_rate', 'lr_decay_step',
                                        'batch_size', 'weight_decay'])
        # create empty file
        results.to_csv(os.path.join(EXPT_DIR, 'results.csv'), index=False)
    else:
        # load empty file
        results = pd.read_csv(os.path.join(EXPT_DIR, 'results.csv'))

    # set optimizer, criterion
    optimizer = CONFIG.optimizer(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    start_epoch = 0

    if args.resume:
        resumed_data = torch.load(os.path.join(EXPT_DIR, 'latest.pth'))
        print(f"Resuming from epoch {resumed_data['epoch'] + 1}")
        model.load_state_dict(resumed_data['model_state_dict'])
        optimizer = CONFIG.optimizer(model.parameters(), lr=resumed_data['lr'])
        optimizer.load_state_dict(resumed_data['optim_state_dict'])
        start_epoch = resumed_data['epoch']

    # start training
    train(CONFIG, model, train_data, val_data, test_data, optimizer, criterion, start_epoch)


if __name__ == '__main__':
    main()
