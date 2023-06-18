import io
import json
import multiprocessing
import os
import re
import time

import pandas as pd
import requests

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


# Loading TabFact TABLES
def load_tabfact_tables(directory: str):
    """
    Loads tabfact tables

    Parameters
    ----------
    directory: string
        Path to directory containing tables as .csv files, seperator of csv files is '#'

    Returns
    -------
    dict
        a dictionary where the table filename is the key and the value is the table saved as pd.DataFrame

    Raises
    ------
    AssertionError
        If number of retrieved tables is not correct.

    """
    assert type(directory) == str, "Parameter path must be a string"

    table_dict = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        table = pd.read_csv(f, sep="#")
        table_dict[filename] = table

    return table_dict


# Loading TabFact CLAIMS

def load_tabfact_samples(list_paths: list):
    """
    Loads all TabFact claims.

    Parameters
    ----------
    list_paths: list
        List of strings, each containing a path to a .json file with tabfact examples

    Returns
    -------
    list
        a list of all samples concatenated

    Raises
    ------
    AssertionError
        If number of retrieved claims is not correct.
    """
    assert type(list_paths[0]) == type(list_paths[1]) == type(
        list_paths[2]) == str, "Entries of parameter 'list_paths' must be strings."
    claims = {}

    for path in list_paths:
        with open(path, 'r') as file:
            data = json.load(file)
            claims = {**claims, **data}

    tabfact_claims_list = [claim for entry in list(claims.values()) for claim in entry[0]]
    assert len(tabfact_claims_list) == 117854, "Number of retrieved list must be 117854"

    return tabfact_claims_list


# Function to generate final TabFact claim dataset with (=> tabfact_joined):
### Lemmatized claims with column links
### Claims without lemmatization
### caption
### table ID
### ...

def create_tabfact_dataset_linked_initial(save_path=""):
    """
    Creates a tabfact dataset with
        i. claims after preprocessing (lemmatized and column entity linking applied)
        ii. labels
        iii. POS tagging
        iv. table caption
        v. initial claims (as they were collected from MTurk)
    If save_path given, dataframe is saved.

    Parameters
    -------
    save_path: string, optional
        Path where to save newly created sample set

    Returns
    -------
    dict
        a dictionary with the tabfact table ID as key and values containing all five information mentioned above

    Raises
    ------
    AssertionError
        If not error in loading data from tabfact repo
    """
    dataset_size = 16573
    if save_path:
        assert type(save_path) == str, "Parameter save_path has to be of type string"

    # Load preprocessed data (linked and lemmatized)
    with open(r'../data/preprocessed_data/preprocessed_data_sample.json', 'r', encoding='utf-8') as file:
        tabfact_linked = json.load(file)
    assert len(tabfact_linked) == dataset_size, "Error in loading tabfact linked claim list!"

    # load initial data r1
    with open(r'../data/tabfact_collected_data/r1_training_all.json', 'r', encoding='utf-8') as file:
        tabfact_init_1 = json.load(file)

    # load initial data r2
    with open(r'../data/tabfact_collected_data/r2_training_all.json', 'r', encoding='utf-8') as file:
        tabfact_init_2 = json.load(file)

    tabfact_init = {**tabfact_init_1, **tabfact_init_2}
    assert len(tabfact_init) == 16573, "Error in loading tabfact inital claim list!"

    # create new list of tabfact claims containing content from both lists
    tabfact_joined = {}
    for key, value in tabfact_linked.items():
        value_list = value.copy()
        claims_init = tabfact_init[key][0]  # list of claims at the first position in list of values
        value_list.append(claims_init)

        tabfact_joined[key] = value_list

    assert len(tabfact_joined) == dataset_size, "Error in joining tabfact lists!"

    # Save tabfact_joined dataset
    if save_path:
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(tabfact_joined, file, indent=4)

    return tabfact_joined


# Function to create subtables
def generate_subtable_tabfact_claim(claim, table) -> pd.DataFrame:
    """
    Uses entities linked in 'claim' to generate a subtable out of 'table'

    Parameters
    ----------
    claim: string
        Claim with linked entities by TabFact authors (see appendix in paper about entity linking)
    table: pd.DataFrame
        Initial table in TabFact dataset

    Returns
    -------
    pd.Dataframe
        Sub-table after claim matching

    Raises
    ------
    AssertionError
        (1) If any input is empty or None. (2) etc.
    """
    assert claim and len(claim) > 0, "Input 'claim' must be a non-empty string."
    assert len(table) > 0, "Input 'table' must be a non-empty dataframe."

    # Extract matched column indices out of claim and create table subset
    column_indices = []
    for linked_entity in re.findall('#(.+?);(.+?),(.+?)#', claim):
        column_index = int(linked_entity[2])
        if column_index != -1:
            column_indices.append(column_index)

    # filter and return subtable
    return table.iloc[:, list(set(column_indices))]


def create_save_subtable(input_zipped):
    """
    Function to create subtables for a given Tabfact table and corresponding claims.
    Each subtable is saved with the following filename: ''{tabfact_table_key}_{claim_index}.html.csv'

    claim_index is the index of the subtable claim form list of claims belonging to the tabfact table.
    """
    table_key, value = input_zipped
    total_count = 0
    empty_count = 0
    max_col_count = 0
    max_row_count = 0
    dtype_count = 0
    remaining_count = 0

    path_subtable = "../data/subtables/{}_{}.html.csv"
    max_col_len = 4
    max_row_len = 20

    # path = "../data/tabfact_collected_data/all_csv/{}".format(table_key)
    path = "C:/Users/k20116188/PycharmProjects/Table-Fact-Checking/data/all_csv/{}".format(table_key)
    table = pd.read_csv(path, sep="#")

    for i, claim in enumerate(value[0]):
        subtable = generate_subtable_tabfact_claim(claim, table)
        total_count += 1

        # only save tables which fulfill certain conditions
        if subtable.empty or len(subtable.columns) == 0:
            empty_count += 1
            continue
        elif len(subtable.columns) > max_col_len:
            max_col_count += 1
            continue
        elif len(subtable) > max_row_len:
            max_row_count += 1
            continue
        elif len(subtable.columns) > 1 and list(subtable.select_dtypes(exclude='object')) == []:
            dtype_count += 1
            continue
        elif len(subtable.columns) == 1 and list(subtable.select_dtypes(exclude='object')):
            dtype_count += 1
            continue
        else:
            remaining_count += 1
            # path = path_subtable.format(table_key.split(".html")[0], i)
            # subtable.to_csv(path, sep="#", index=False)

    return (total_count, empty_count, max_col_count, max_row_count, dtype_count, remaining_count)


def multiprocess_create_subtable(tabfact_dict: dict):
    """
    Uses multiprocessing on all available cpu cores to create subtables for tabfact.

    Parameters
    ----------
    tabfact_dict: dictionary
        Dictionary with tabfact table ids as keys and coresponding value[0] contains a list of linked claims belonging to table.
    """
    use_multiprocessing = False

    if use_multiprocessing:
        cores = multiprocessing.cpu_count()
        print(f"Number of cores is: {cores}.")

        start_time = time.time()
        pool = multiprocessing.Pool(cores)
        pool.map(create_save_subtable, zip(list(tabfact_dict.keys()), list(tabfact_dict.values())))
        pool.close()
        pool.join()

        end_time = time.time()
        print(end_time - start_time)
    else:
        total_count = 0
        empty_count = 0
        max_col_count = 0
        max_row_count = 0
        dtype_count = 0
        remaining_count = 0

        for key, val in tabfact_dict.items():
            x = create_save_subtable((key, val))
            # print(x)
            total_count += x[0]
            empty_count += x[1]
            max_col_count += x[2]
            max_row_count += x[3]
            dtype_count += x[4]
            remaining_count += x[5]

        print(total_count)
        print(empty_count)
        print(max_col_count)
        print(max_row_count)
        print(dtype_count)
        print(remaining_count)


def create_tablebert_files_subtables(path_to_claims, path_to_subtables, outputfile = "test.tsv"):

    return_df = pd.DataFrame(columns=["filename", "column_count", "column_names", "table_content", "claim", "label"])

    with open(path_to_claims, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    for index, entry in enumerate(dataset):
        claim = entry["question"]
        table_filename = entry["image_filename"].split(".png")[0]+".html.csv"
        label = entry["answer"]

        table = pd.read_csv(os.path.join(path_to_subtables, table_filename), encoding="utf-8", sep="#")
        column_names = table.columns
        column_count = len(column_names)

        sentence_list = []
        for index, row in table.iterrows():
            sentence = f"row {num_dict[index + 1]} is : "
            for j, row_entry in enumerate(row):
                sentence += f"{column_names[j]} is {row_entry} ; "
            sentence += ". "
            sentence = sentence.replace("; .", ".")
            sentence_list.append(sentence)

        append_row = {"filename": table_filename,
                     "column_count": len(column_names),
                     "column_names": " ".join(column_names),
                     "table_content": " ".join(sentence_list),
                     "claim": claim,
                     "label": label}

        return_df = return_df.append(append_row, ignore_index=True)

    return_df.to_csv(outputfile, sep='\t', header=False, index=False)


def create_tablebert_files_charttext(path_to_claims, path_to_subtables, outputfile = "test.tsv", type = "sentences"):

    return_df = pd.DataFrame(columns=["filename", "column_count", "column_names", "table_content", "claim", "label"])

    with open(path_to_claims, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    for index, entry in enumerate(dataset):
        claim = entry["question"]
        label = entry["answer"]
        table_filename = entry["image_filename"].split(".png")[0]+".html.csv"

        sample = entry["img_text"]
        sample_df = pd.DataFrame(sample)

        sample_df["x_mid"] = sample_df[0] + (sample_df[2] / 2)
        sample_df["y_mid"] = sample_df[1] + (sample_df[3] / 2)
        a = list(sample_df["x_mid"])
        sample_df["x_norm"] = [round(x / 5, 0) * 5 for x in a]
        a = list(sample_df["y_mid"])
        sample_df["y_norm"] = [round(x / 5, 0) * 5 for x in a]

        # filter based on column "x_label"/"y_label" if entry is a label
        x_label = sample_df[(sample_df[4] == 1)]
        x_label = " ".join([str(x) for x in list(x_label[6])])
        y_label = sample_df[sample_df[5] == 1]
        y_label = " ".join([str(x) for x in list(y_label[6])])
        column_names = [x_label, y_label]

        sample_df = sample_df.sort_values(by=["y_norm", "x_norm"])
        row_num = 0

        if type == "concatenation":
            img_text = sample_df[6] # is already sorted according to y and x coordinates
            evidence_content = " ; ".join(list([str(x) for x in img_text]))
        elif type == "sentences":
            sample_df_no_labels = sample_df[(sample_df[4] == 0) & (sample_df[5] == 0)]  # take out x and y labels
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
                    if row["y_mid"] > y_temp - 5 and row["y_mid"] < y_temp + 5:
                        row_temp.append(row.to_dict())
                        y_temp = row["y_mid"]
                    else:
                        # transfer previous entries to sentences_list
                        row_temp = pd.DataFrame(row_temp)
                        row_temp = row_temp.sort_values(by=["x_mid"])  # sort previous entries by their x_mid
                        row_temp = [str(x) for x in list(row_temp[6])]
                        if len(row_temp) > 1:
                            # create template based sentence
                            # sent_temp_0 = str(row_num) + " " + str(0) + " " + str(row[4]) + " " + str(row[5]) + " " + " ".join(row_temp[:-1])
                            # sent_temp_1 = str(row_num) + " " + str(1) + " " + str(row[4]) + " " + str(row[5]) + " " + str(row_temp[-1])
                            sent_temp = "entry {} is : {} is {} ; {} is {} .".format(num_dict[row_num], y_label,
                                                                                     " ".join(row_temp[:-1]), x_label,
                                                                                     row_temp[-1])
                            # sent_temp = "{} is {} when {} is {}.".format(y_label, " ".join(row_temp[:-1]), x_label,
                            #                                              row_temp[-1])
                            row_num += 1
                            sentences_list.append(sent_temp)
                            # sentences_list.append(sent_temp_0)
                            # sentences_list.append(sent_temp_1)
                        # new row
                        y_temp = row["y_mid"]
                        row_temp = [row.to_dict()]
            # save final entry
            row_temp = pd.DataFrame(row_temp)
            row_temp = row_temp.sort_values(by=["x_mid"])  # sort previous entries by their x_mid
            row_temp = [str(x) for x in list(row_temp[6])]
            if len(row_temp) > 1:
                sent_temp = "entry {} is : {} is {} ; {} is {} .".format(num_dict[row_num], y_label,
                                                                         " ".join(row_temp[:-1]), x_label,
                                                                         row_temp[-1])
                # sent_temp = "{} is {} when {} is {}.".format(y_label, " ".join(row_temp[:-1]), x_label, row_temp[-1])
                # sent_temp_0 = str(row_num) + " " + str(0) + " " + str(row[4]) + " " + str(row[5]) + " " + " ".join(
                #     row_temp[:-1])
                # sent_temp_1 = str(row_num) + " " + str(1) + " " + str(row[4]) + " " + str(row[5]) + " " + str(
                #     row_temp[-1])

                sentences_list.append(sent_temp)
                # sentences_list.append(sent_temp_0)
                # sentences_list.append(sent_temp_1)

            evidence_content = " ".join(sentences_list)
            # evidence_content = "###".join(sentences_list)

        else: # capture coordinates
            entries = []
            for j, row in sample_df.iterrows():
                row_entry = str(int(row["x_norm"])) + " " + str(int(row["y_norm"])) + " " + str(row[4]) + " " + \
                            str(row[5]) + " " + str(row[6])
                entries.append(row_entry)

            evidence_content = "###".join(entries)

        append_row = {"filename": table_filename,
                     "column_count": len(column_names),
                     "column_names": "; ".join(column_names),
                     "table_content": evidence_content,
                     "claim": claim,
                     "label": label}

        return_df = return_df.append(append_row, ignore_index=True)

    return_df.to_csv(outputfile, sep='\t', header=False, index=False)


if __name__ == '__main__':
    # data_joined = create_tabfact_dataset_linked_initial("../data/preprocessed_data/preprocessed_data_joined.json")
    with open(r"../data/preprocessed_data/preprocessed_data_joined.json", "r", encoding="utf-8") as file:
        data_joined = json.load(file)

    print(f"length of data_joined: {len(data_joined)}")
    multiprocess_create_subtable(data_joined)

    # create_tablebert_files_subtables("../data/qa/valid_barplot_seaborn.json", "../data/subtables", "../data/qa/dev.tsv")

    # create_tablebert_files_charttext("../data/qa/train_barplot_seaborn_imgtext_tesseract.json", "../data/subtables",
    #                                  "../data/qa/train_charttext_sentences_v5.tsv", type="sentences")