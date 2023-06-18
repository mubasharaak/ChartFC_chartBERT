import json
import multiprocessing
import os
import time

import pandas as pd


def extract_img_text(entry: dict):
    subtable_file = "_".join(entry["question_index"].split("_")[:2])+".html.csv"

    # open subtable
    subtable_path = f"../data/subtables/"
    subtable = pd.read_csv(os.path.join(subtable_path, subtable_file), sep="#", encoding="utf-8")

    # extract text
    result = [list(map(str, r)) for r in subtable.values.tolist()] + [subtable.columns.tolist()]

    return result


def main():
    with open(f"..\data\preprocessed_data\dataset.json", "r", encoding="utf-8") as file:
        dataset = json.load(file)

    cores = multiprocessing.cpu_count()
    print(f"{cores} cores used for multiprocessing.")

    start_time = time.time()
    pool = multiprocessing.Pool(cores)
    result = pool.map(extract_img_text, dataset)

    pool.close()
    pool.join()
    end_time = time.time()
    print(end_time - start_time)

    dataset_img_text = []
    for i, entry in enumerate(dataset):
        entry["img_text"] = result[i]
        dataset_img_text.append(entry)

    # save new dataset
    with open(f"..\data\preprocessed_data\dataset_img_text.json", "w", encoding="utf-8") as file:
        json.dump(dataset_img_text, file, indent=4)


if __name__ == '__main__':
    main()
