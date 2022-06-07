import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader


num_dict = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    21: "twenty-one",
    22: "twenty-two",
    23: "twenty-three",
    24: "twenty-four",
    25: "twenty-five",
    26: "twenty-six",
    27: "twenty-seven",
    28: "twenty-eight",
    29: "twenty-nine",
    30: "thirty",
    31: "thirty-one",
    32: "thirty-two",
    33: "thirty-three",
    34: "thirty-four",
    35: "thirty-five",
    36: "thirty-six",
    37: "thirty-seven",
    38: "thirty-eight",
    39: "thirty-nine",
    40: "forty",
    41: "forty-one",
    42: "forty-two",
    43: "forty-three",
    44: "forty-four",
    45: "forty-five",
    46: "forty-six",
    47: "forty-seven",
    48: "forty-eight",
    49: "forty-nine",
    50: "fifty",
}

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

            if self.config.ocr_type == "concat":
                ocr_df = ocr_df.sort_values(by=['y', 'x']) # order by (1) x axis (=row) and by (2) y axis
                ocr_text = list(ocr_df["text"])
                ocr_text = " ".join([str(entry) for entry in ocr_text])
            elif self.config.ocr_type == "template_sentence":
                ocr_text = extract_ocr(ocr_df, self.config.ocr_type)
            else:
                return txt, txt_encode, label, img_tensor, img_path, idx, txt_len, None, 0  # no ocr text returned

            ocr_text_len = len(ocr_text)
            return txt, txt_encode, label, img_tensor, img_path, idx, txt_len, ocr_text, ocr_text_len
        else:
            return txt, txt_encode, label, img_tensor, img_path, idx, txt_len, None, 0  # no ocr text returned


def tokenize(entry):
    text = entry['question'].lower()
    ocr_text = [str(e[-1]).lower() for e in entry['img_text']]
    ocr_text = " ".join(ocr_text)
    text = text + " " + ocr_text # using both claim and ocr text to create LUT for LSTM encoder
    return word_tokenize(text)


def extract_ocr(ocr_df: pd.DataFrame, extraction_type: str):
    sample_df = ocr_df

    sample_df["x_mid"] = sample_df["x"] + (sample_df["w"] / 2)
    sample_df["y_mid"] = sample_df["y"] + (sample_df["h"] / 2)
    a = list(sample_df["x_mid"])
    sample_df["x_norm"] = [round(x / 5, 0) * 5 for x in a]
    a = list(sample_df["y_mid"])
    sample_df["y_norm"] = [round(x / 5, 0) * 5 for x in a]

    # filter based on column "x_label"/"y_label" if entry is a label
    x_label = sample_df[(sample_df["x_label"] == 1)]
    x_label = " ".join([str(x) for x in list(x_label["text"])])
    y_label = sample_df[sample_df["y_label"] == 1]
    y_label = " ".join([str(x) for x in list(y_label["text"])])
    column_names = [x_label, y_label]

    sample_df = sample_df.sort_values(by=["y_norm", "x_norm"])
    row_num = 0

    if extraction_type == "concatenation":
        img_text = sample_df["text"]  # is already sorted according to y and x coordinates
        evidence_content = " ; ".join(list([str(x) for x in img_text]))
    elif extraction_type == "template_sentence":
        sample_df_no_labels = sample_df[(sample_df["x_label"] == 0) & (sample_df["y_label"] == 0)]  # take out x and y labels
        sentences_list = []
        y_temp = 0
        for index, row in sample_df_no_labels.iterrows():
            if y_temp == 0:
                # save the first y_mid coordinate
                row_num += 1
                y_temp = row["y_mid"]
                row_temp = [row.to_dict()]
            else:
                # save all entries with -/+ 5 pixels for y_mid
                if (y_temp - 5) < row["y_mid"] < (y_temp + 5):
                    row_temp.append(row.to_dict())
                    y_temp = row["y_mid"]
                else:
                    # transfer previous entries to sentences_list
                    row_temp = pd.DataFrame(row_temp)
                    row_temp = row_temp.sort_values(by=["x_mid"])  # sort previous entries by their x_mid
                    row_temp = [str(x) for x in list(row_temp["text"])]
                    if len(row_temp) > 1:
                        # create template based sentence
                        sent_temp = "entry {} is: {} is {}; {} is {}.".format(num_dict[row_num], y_label,
                                                                                 " ".join(row_temp[:-1]), x_label,
                                                                                 row_temp[-1])
                        row_num += 1
                        sentences_list.append(sent_temp)
                    # new row
                    y_temp = row["y_mid"]
                    row_temp = [row.to_dict()]
        # save final entry
        row_temp = pd.DataFrame(row_temp)
        row_temp = row_temp.sort_values(by=["x_mid"])  # sort previous entries by their x_mid
        row_temp = [str(x) for x in list(row_temp["text"])]
        if len(row_temp) > 1:
            sent_temp = "entry {} is : {} is {} ; {} is {} .".format(num_dict[row_num], y_label,
                                                                     " ".join(row_temp[:-1]), x_label,
                                                                     row_temp[-1])

            sentences_list.append(sent_temp)

        evidence_content = " ".join(sentences_list)
    else:  # capture coordinates
        entries = []
        for j, row in sample_df.iterrows():
            row_entry = str(int(row["x_norm"])) + " " + str(int(row["y_norm"])) + " " + str(row[4]) + " " + \
                        str(row[5]) + " " + str(row[6])
            entries.append(row_entry)

        evidence_content = "###".join(entries)

    return evidence_content


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
