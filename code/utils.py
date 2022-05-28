import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader


def encode_txt(txt, txt2idx, maxlen):
    """
    Encodes text input 'txt' into word vectors

    :param txt:
    :param txt2idx:
    :param maxlen:
    :return:
    """
    txt_vec = torch.zeros(maxlen).long()
    txt_words = word_tokenize(txt.lower())
    txt_len = len(txt_words)
    for i, word in enumerate(txt_words):
        txt_vec[i] = txt2idx.get(word, len(txt2idx))

    return txt_vec, txt_len


def encode_label(label):
    """
    Encodes labels into vector
    :param label:
    :return:
    """
    label_vec = torch.zeros((1,))
    if label == '0' or label == 0:
        label_vec[0] = 0.0
    elif label == '1' or label == 1:
        label_vec[0] = 1.0
    else:
        raise ValueError(f"Label {label} not in ['0', '1', 0, 1].")
    return label_vec


class ChartFCDataset(Dataset):
    def __init__(self, dataset, label2idx, txt2idx, maxlen, split, config):
        self.dataset = dataset
        self.txt2idx = txt2idx
        self.label2idx = label2idx
        self.maxlen = maxlen
        self.split = split
        self.config = config

        # transform image as specified in config.json file
        if self.split == 'train':
            self.prep = config.train_transform
        else:
            self.prep = config.test_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        txt = self.dataset[index]['question']
        txt_len = len(txt)
        txt_encode, txt_encode_len = encode_txt(txt, self.txt2idx, self.maxlen)
        label = encode_label(self.dataset[index]['answer'])

        # extract id for dataset entry
        idx = "".join(self.dataset[index]['image_filename'].split("_")[:2])
        idx = idx.replace("-", "")
        idx = idx.replace(".png", "")
        idx = idx.replace(".html.csv", "")
        idx = int(idx)

        img_path = os.path.join(self.config.root, self.config.dataset, 'images',
                                self.dataset[index]['image_filename'])

        # Load image and create image tensor
        img = Image.open(img_path).convert('RGB')  # set 'RGB' to 'L' if black-white image
        img_tensor = self.prep(img)

        if self.config.use_ocr:
            # Extract and prepare OCR text
            ocr_df = pd.DataFrame(self.dataset[index]['img_text'], columns=['x', 'y', 'w', 'h', 'x_label', 'y_label', 'text'])
            ocr_df = ocr_df.sort_values(by=['y', 'x']) # order by (1) x axis (=row) and by (2) y axis
            ocr_text = list(ocr_df["text"])
            ocr_text = " ".join([str(entry) for entry in ocr_text])
            ocr_text_len = len(ocr_text)
            return txt, txt_encode, label, img_tensor, img_path, idx, txt_len, ocr_text, ocr_text_len
        else:
            return txt, txt_encode, label, img_tensor, img_path, idx, txt_len, "", 0  # no ocr text returned


def tokenize(entry):
    text = entry['question'].lower()
    ocr_text = [str(e[-1]).lower() for e in entry['img_text']]
    ocr_text = " ".join(ocr_text)
    text = text + " " + ocr_text # using both claim and ocr text to create LUT for LSTM encoder
    return word_tokenize(text)


def build_lut(dataset):
    print("Building lookup table for claim and image text tokens")
    pool = ProcessPoolExecutor(max_workers=8)
    text = list(pool.map(tokenize, dataset, chunksize=1000))
    pool.shutdown()

    maxlen = max([len(t) for t in text])
    unique_tokens = set([tok for t in text for tok in t])
    ques2idx = {word: idx + 1 for idx, word in enumerate(unique_tokens)}  # save 0 for padding
    ans2idx = {'supports': 1, 'refutes': 0, '1': 1, '0': 0, 1: 1, 0: 0}

    print(f"lookup table for answers: {ans2idx}.")
    return ans2idx, ques2idx, maxlen


def collate_batch(data_batch):
    return torch.utils.data.dataloader.default_collate(data_batch)
    # return data_batch.sort(key=lambda x: x[-1], reverse=True)


def build_dataloaders(config):
    train_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.train_filename)))
    all_data = train_data.copy()

    for file in config.test_filenames.values():
        all_data.extend(json.load(open(os.path.join(config.root, config.dataset, 'qa', file))))
    for file in config.val_filenames.values():
        all_data.extend(json.load(open(os.path.join(config.root, config.dataset, 'qa', file))))

    if config.lut_location == '':
        ans2idx, ques2idx, maxlen = build_lut(all_data)
    else:
        lut = json.load(open(config.lut_location, 'r'))
        ans2idx = lut['ans2idx']
        ques2idx = lut['ques2idx']
        maxlen = lut['maxlen']

    n = int(config.data_subset * len(train_data))
    np.random.seed(config.data_subset)
    np.random.shuffle(train_data)
    train_data = train_data[:n]
    print(f"Training with {len(train_data)} samples in total.")

    # balanced sampling during training
    train_targets = [entry['answer'] for entry in train_data]
    supports_count = sum(train_targets)
    refutes_count = len(train_data) - supports_count
    class_sample_count = [refutes_count, supports_count]
    weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    samples_weights = weights[train_targets]
    # total data sampled = number of refutes samples*2
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, refutes_count * 2, replacement=False)

    train_dataset = ChartFCDataset(train_data, ans2idx, ques2idx, maxlen, 'train', config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_batch,
                                  num_workers=8, sampler=sampler)

    val_datasets = []
    for split in config.val_filenames:
        cqa_val_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.val_filenames[split])))
        val_datasets.append(ChartFCDataset(cqa_val_data, ans2idx, ques2idx, maxlen, split, config))

    val_dataloaders = []
    for vds in val_datasets:
        val_dataloaders.append(DataLoader(vds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch,
                                          num_workers=8))

    test_datasets = []
    for split in config.test_filenames:
        cqa_test_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.test_filenames[split])))
        test_datasets.append(ChartFCDataset(cqa_test_data, ans2idx, ques2idx, maxlen, split, config))

    test_dataloaders = []
    for tds in test_datasets:
        test_dataloaders.append(DataLoader(tds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch,
                                           num_workers=8))

    return train_dataloader, val_dataloaders, test_dataloaders, len(ques2idx) + 1, len(ans2idx) + 1
