import copy
import random

import matplotlib.pyplot as plt
import json
import multiprocessing
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
from datawrapper import Datawrapper
from dateutil import parser
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("error")

# Variables for chart image creation
# "Principles of Effective Data Visualization":
# https://reader.elsevier.com/reader/sd/pii/S2666389920301896?token=971C6D0D384BB2233A457900D969028C3553D296086DAF13DA41191485ED6B98E7540965D085138459D6A3F46A4D54B6&originRegion=eu-west-1&originCreation=20221002130930
CHART_VIS_VARIABLES = {}

# (1) color of bars: blue, green, red
#   Recommendation: one color per barchart as this is recommended if no difference in categorical data
#   Recommendation: consider colors which are challenges for color blind people: green and red
CHART_VIS_VARIABLES["color"] = ["skyblue", "lightgreen", "pink"]

# (2) orientation of charts: horizontal, vertical
CHART_VIS_VARIABLES["orientation"] = ["h", "v"]

# (3) axis label location: below/top for x-axis, left/right for y-axis
CHART_VIS_VARIABLES["grid_style"] = ["white", "darkgrid", "whitegrid"]

# (4) font size: recommended 8-12 pt
# https://tex.stackexchange.com/questions/14988/which-font-should-be-used-for-diagrams-graphs-flow-charts
CHART_VIS_VARIABLES["font_scale"] = [1.0, 1.2]

# (6) bar width: thin, medium, thick #todo is already given by default


def rand_chart_variable_selection():
    result_dict = {"color": random.choice(CHART_VIS_VARIABLES["color"]),
                   "orientation": random.choice(CHART_VIS_VARIABLES["orientation"]),
                   "grid_style": random.choice(CHART_VIS_VARIABLES["grid_style"]),
                   "font_scale": random.choice(CHART_VIS_VARIABLES["font_scale"]),
                   }
    return result_dict


def load_properties() -> dict:
    """
    Load properties for datawrapper chart from file properties.json
    :return:
    """
    with open(r"../data/properties.json", "r", encoding="utf-8") as file:
        properties = json.load(file)
    return properties


def set_properties(num_rows: int, prop_dict: dict) -> dict:
    """
    Sets "chart-height" and "embed-height" parameters of dictionary prop_dict
    :param num_rows:
    :param prop_dict:
    :type prop_dict: dict
    """
    if num_rows < 11:
        height = 600
    elif num_rows > 15:
        height = 1000
    else:
        height = 800

    prop_dict["publish"]["chart-height"] = height
    prop_dict["publish"]["embed-height"] = height

    return prop_dict


def create_save_chart(filename: str):
    """
    Function which creates a chart using DataWrapper for a given filename containing a TabFact subtable
    :param filename: subtable filename used to create chart
    """
    assert filename, "A non-empty string must be given as filename."
    DW_ACCESS_TOKEN = "CwzLxkZqMDbzLNMYXa3Je7e5iLpAHe4cxM49ltlBMwApWvdLpmVTonrg3FhGFStf"
    chart_dir = r'..\data\charts'
    directory = r'..\data\subtables'

    if any([chartfile for chartfile in os.listdir(chart_dir) if filename.split(".html")[0] in chartfile]):
        print(f"Chart for {filename} already exists.")
        return None

    date_col = False
    dw = Datawrapper(access_token=DW_ACCESS_TOKEN)
    path = os.path.join(directory, filename)
    try:
        # Load subtable into a pd.DataFrame
        df = pd.read_csv(path, sep="#")
    except Exception as e:
        print(f"The following error occurred while creating chart for {filename}: {e}.")
        return None

    # deal with empty cells
    df = df.replace("-", np.nan)
    df = df.replace("n / a", np.nan)
    df = df.dropna()

    regex_1 = r'^(?:(?:31(\/|-|\.| \/ | - | \. | )(?:0?[13578]|1[02]|(?:jan|january|mar|march|may|jul|july|aug|august|oct|october|dec|december)))\1|(?:(?:29|30)(\/|-|\.| \/ | - | \. | )(?:0?[1,3-9]|1[0-2]|(?:jan|january|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.| \/ | - | \. | )(?:0?2|(?:feb|february))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.| \/ | - | \. | )(?:(?:0?[1-9]|(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september))|(?:1[0-2]|(?:oct|october|nov|november|dec|december)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$'
    regex_2 = r'/^(?:(?:1[6-9]|[2-9]\d)?\d{2})(?:(?:(\/|-|\.| \/ | - | \. | )(?:0?[13578]|1[02])\1(?:31))|(?:(\/|-|\.| \/ | - | \. | )(?:0?[13-9]|1[0-2])\2(?:29|30)))|(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00)))(\/|-|\.| \/ | - | \. | )0?2\3(?:29)|(?:(?:1[6-9]|[2-9]\d)?\d{2})(\/|-|\.| \/ | - | \. | )(?:(?:0?[1-9])|(?:1[0-2]))\4(?:0?[1-9]|1\d|2[0-8])$'
    regexes = [regex_1, regex_2]

    # preprocess dataframe
    for col_name, col_type in dict(df.dtypes).items():
        if col_type == object:
            # parse datetimes
            try:
                for regex in regexes:
                    r = re.compile(regex)
                    if not date_col and len(df[col_name])>0 and r.match(list(df[col_name])[0]) and list(df.select_dtypes(exclude=[np.number])) == [
                        col_name]:
                            df[col_name] = df[col_name].apply(lambda x: parser.parse(x).strftime("%y-%m-%d"))
                            df = df.sort_values(by=col_name)
                            date_col = col_name
            except Exception:
                continue

    # prepare chart creation
    cols = list(df.columns)
    if len(cols) == 1 and len(list(df.select_dtypes(exclude=[np.number]))) == 1:
        # if subtable only has one categorical column
        d = Counter(df.iloc[:, 0])
        if all(entry == 1 for entry in d.values()):  # don't create chart if all entries in list are equal one
            return None
        df = pd.DataFrame.from_dict(d, orient='index').reset_index()
        df.columns = cols + ["total"]

    elif len(list(df.select_dtypes(include='object'))) == 2 and not date_col:  # combine two categorical columns to one
        cat_cols = list(df.select_dtypes(include='object'))
        df["{} & {}".format(cat_cols[0], cat_cols[1])] = df[cat_cols[0]] + " & " + df[
            cat_cols[1]]  # @todo ask sv for better way for this
        df = df.drop(columns=cat_cols)

    elif len(
            list(df.select_dtypes(include='object'))) == 3 and not date_col:  # combine three categorical columns to one
        cat_cols = list(df.select_dtypes(include='object'))
        df["{} & {} & {}".format(cat_cols[0], cat_cols[1], cat_cols[2])] = df[cat_cols[0]] + " & " + df[
            cat_cols[1]] + " & " + df[cat_cols[2]]
        df = df.drop(columns=cat_cols)

    cols = list(df.columns)

    # decide chart type
    chart_type = "d3-bars"
    chart_type_path = "barplot-simple"
    count_num_columns = len(list(df.select_dtypes(include=[np.number])))

    if date_col and (count_num_columns == 1 or count_num_columns == 2):
        chart_type = "d3-lines"
        chart_type_path = "linechart"
    elif len(cols) > 2:
        if count_num_columns == 3 and len(list(df.select_dtypes(exclude=[np.number]))) == 0:
            chart_type = "d3-scatter-plot"
            chart_type_path = "scatterplot"
        else:
            chart_type = "d3-bars-split"
            chart_type_path = "barplot-complex"
    elif len(cols) == 2:
        if count_num_columns == 2:
            chart_type = "d3-scatter-plot"
            chart_type_path = "scatterplot"

    # create chart
    caption = ""
    for col in df.columns:
        caption += " vs '{}'".format(col) if caption else "'{}'".format(col)

    # set properties according to number of rows
    properties = set_properties(len(df), load_properties())
    chart_path = os.path.join(chart_dir, "{}_{}.png".format(filename.split(".html")[0], chart_type_path))

    # create chart
    if chart_type_path == "scatterplot" and len(df.columns) == 3:
        # own scatterplot if all three columns numerical
        df_scatter = df.copy()
        barplot = sns.scatterplot(x="0", y="1", hue="2", data=df_scatter.rename(
            columns={x: str(y) for x, y in zip(df_scatter.columns, range(0, len(df_scatter.columns)))}))
        barplot.set(xlabel=df.columns[0], ylabel=df.columns[1])
        barplot.set_title(caption)
        barplot.get_legend().set_title(df.columns[2])
        barplot.get_figure().savefig(chart_path, bbox_inches="tight")
        barplot.get_figure().clf()

    elif chart_type_path == "linechart" and len(df.columns) == 2:
        df_columns = list(df.columns)
        df_columns.remove(date_col)

        linechart = sns.lineplot(x=date_col, y=df_columns[0], data=df)
        linechart.set_title(caption)
        # linechart.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        linechart.tick_params('x', labelrotation=90)
        linechart.get_figure().savefig(chart_path, bbox_inches="tight")
        linechart.get_figure().clf()

    elif chart_type_path == "linechart" and len(df.columns) == 3:
        df_columns = list(df.columns)
        df_columns.remove(date_col)

        linechart = sns.lineplot(data=df)
        linechart.set_title(caption)
        # linechart.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        linechart.tick_params('x', labelrotation=90)
        linechart.get_figure().savefig(chart_path, bbox_inches="tight")
        linechart.get_figure().clf()

    else:
        chart_info = dw.create_chart(title=caption, chart_type=chart_type, data=df)
        dw.update_metadata(chart_info['id'], properties)
        dw.publish_chart(chart_id=chart_info['id'])

        # save chart
        image_file = dw.export_chart(chart_info['id'], output='png', filepath=chart_path, display=True,
                                     width=properties["publish"]["embed-width"], zoom=1)
        if not image_file:
            print(f"Error in creation of file: {filename}.")
            # num_col = list(df.select_dtypes(include=[np.number]))
            # df[num_col] = [int(str(num)[2:]) for num in df[num_col]]
            # chart_info = dw.create_chart(title=caption, chart_type=chart_type, data=df)
            # dw.update_metadata(chart_info['id'], properties)
            # dw.publish_chart(chart_id=chart_info['id'])
            #
            # # save chart
            # dw.export_chart(chart_info['id'], output='png', filepath=chart_path, display=True,
            #                 width=properties["publish"]["embed-width"], zoom=1)


def multiprocess_create_chart(directory: str):
    """
    Given the path to a folder containing subtables, create charts for all of them using multiprocessing
    """
    use_multiprocessing = True

    if use_multiprocessing:
        cores = multiprocessing.cpu_count()
        print(f"{cores} cores used for multiprocessing.")

        files = [subtable_file for subtable_file in os.listdir(directory)]
        print(f"Total number of files is {len(files)}.")

        start_time = time.time()
        pool = multiprocessing.Pool(cores)
        pool.map(create_save_chart, files)

        pool.close()
        pool.join()
        end_time = time.time()
        print(end_time - start_time)
    else:
        for filename in os.listdir(directory):
            # if any([chartfile for chartfile in os.listdir(chart_dir) if filename.split(".html")[0] in chartfile]):
            #     print(f"Chart for {filename} already exists.")
            #     continue
            create_save_chart(filename)


def create_chart_seaborn_multiprocess(directory):
    use_multiprocessing = True

    if use_multiprocessing:
        cores = multiprocessing.cpu_count()
        print(f"{cores} cores used for multiprocessing.")

        files = [subtable_file for subtable_file in os.listdir(directory)]
        print(f"Total number of files is {len(files)}.")

        start_time = time.time()
        pool = multiprocessing.Pool(cores)
        pool.map(create_chart_seaborn, files)

        pool.close()
        pool.join()
        end_time = time.time()
        print(end_time - start_time)
    else:
        for filename in os.listdir(directory):
            create_chart_seaborn(filename)


def create_chart_seaborn(filename):
    assert filename, "A non-empty string must be given as filename."
    directory = r'..\data\subtables'
    chart_dir = r'..\data\charts_seaborn_v5'

    # select variables
    chart_variables = rand_chart_variable_selection()

    date_col = False
    path = os.path.join(directory, filename)
    try:
        # Load subtable into a pd.DataFrame
        df = pd.read_csv(path, sep="#")
    except Exception as e:
        print(f"The following error occurred while creating chart for {filename}: {e}.")
        return None

    # deal with empty cells
    df = df.replace("-", np.nan)
    df = df.replace("n / a", np.nan)
    df = df.dropna()

    regex_1 = r'^(?:(?:31(\/|-|\.| \/ | - | \. | )(?:0?[13578]|1[02]|(?:jan|january|mar|march|may|jul|july|aug|august|oct|october|dec|december)))\1|(?:(?:29|30)(\/|-|\.| \/ | - | \. | )(?:0?[1,3-9]|1[0-2]|(?:jan|january|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.| \/ | - | \. | )(?:0?2|(?:feb|february))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.| \/ | - | \. | )(?:(?:0?[1-9]|(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september))|(?:1[0-2]|(?:oct|october|nov|november|dec|december)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$'
    regex_2 = r'/^(?:(?:1[6-9]|[2-9]\d)?\d{2})(?:(?:(\/|-|\.| \/ | - | \. | )(?:0?[13578]|1[02])\1(?:31))|(?:(\/|-|\.| \/ | - | \. | )(?:0?[13-9]|1[0-2])\2(?:29|30)))|(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00)))(\/|-|\.| \/ | - | \. | )0?2\3(?:29)|(?:(?:1[6-9]|[2-9]\d)?\d{2})(\/|-|\.| \/ | - | \. | )(?:(?:0?[1-9])|(?:1[0-2]))\4(?:0?[1-9]|1\d|2[0-8])$'
    regexes = [regex_1, regex_2]

    # preprocess dataframe
    for col_name, col_type in dict(df.dtypes).items():
        if col_type == object:
            # parse datetimes
            try:
                for regex in regexes:
                    r = re.compile(regex)
                    if not date_col and len(df[col_name]) > 0 and r.match(list(df[col_name])[0]) and list(
                            df.select_dtypes(exclude=[np.number])) == [
                        col_name]:
                        df[col_name] = df[col_name].apply(lambda x: parser.parse(x).strftime("%y-%m-%d"))
                        df = df.sort_values(by=col_name)
                        date_col = col_name
            except Exception as e:
                print(f"The following error occurred while creating chart for {filename}: {e}.")
                continue

    # prepare chart creation
    cols = list(df.columns)
    if len(cols) == 1 and len(list(df.select_dtypes(exclude=[np.number]))) == 1:
        # if subtable only has one categorical column
        d = Counter(df.iloc[:, 0])
        if all(entry == 1 for entry in d.values()):  # don't create chart if all entries in list are equal one
            return None
        df = pd.DataFrame.from_dict(d, orient='index').reset_index()
        df.columns = cols + ["total"]

    if len(cols) == 2 and len(list(df.select_dtypes(include=[np.number]))) == 1:
        # create chart with seaborn
        num_col = df.select_dtypes(include=[np.number]).columns[0]
        text_col = df.select_dtypes(exclude=[np.number]).columns[0]
        plt.clf()
        sns.set(font_scale=chart_variables["font_scale"])
        sns.set_style(chart_variables["grid_style"])

        if chart_variables["orientation"] == 'v':
            ax = sns.barplot(x=df.index, y=num_col, data=df, ci=None, color=chart_variables["color"], orient=chart_variables["orientation"])
            ax.set(xlabel=text_col),
            ax.tick_params(axis='x', rotation=90)
            ax.set_xticklabels(df[text_col])
        else: # orientation is horizontal
            ax = sns.barplot(x=num_col, y=df.index, data=df, ci=None, color=chart_variables["color"], orient=chart_variables["orientation"])
            ax.set(ylabel=text_col)
            ax.tick_params(axis='y', rotation=360)
            ax.set_yticklabels(df[text_col])

        if chart_variables["orientation"] == "v" and max(df[num_col]) >= 9999: # bar label centered in the bar
            ax.bar_label(ax.containers[0], padding=10, fmt='%.0f', rotation=90, label_type='center', fontsize=12)
        else:
            ax.bar_label(ax.containers[0], padding=10, fmt='%.0f', fontsize=12)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        try:
            plt.tight_layout()
        except Warning:
            print(f"Warning occurred for figure {filename}!")
            return None

        chart_path = os.path.join(chart_dir, "{}.png".format(filename.split(".html")[0]))
        fig = ax.get_figure()
        fig.savefig(chart_path)
    else:
        print(f"No chart created for file {filename}.")


def create_dataset_file():
    chart_dir = r'..\data\charts_seaborn'
    dataset_dir = r'..\data\qa'
    with open(r"..\data\preprocessed_data\preprocessed_data_joined.json", encoding="utf-8") as file:
        dataset = json.load(file)

    entry = {
        "answer": "",
        "image_filename": "",
        "question": "",
        "caption": "",
    }
    supports_samples = 0

    new_dataset = []
    for key, value in dataset.items():
        key = key.split(".html.csv")[0]
        for claim_i, claim  in enumerate(value[0]):
            path_chart_file = os.path.join(chart_dir, key+"_"+str(claim_i)+".png")
            if os.path.exists(path_chart_file):
                new_entry = copy.deepcopy(entry)
                new_entry["answer"] = value[1][claim_i]
                new_entry["image_filename"] = key+"_"+str(claim_i)+".png"
                new_entry["question"] = value[4][claim_i]
                new_entry["caption"] = value[3]

                new_dataset.append(new_entry)

    print(f"Size of total dataset is {len(new_dataset)}.")
    # split in train, dev and test
    trainset, dev_test_set = train_test_split(new_dataset, test_size=0.2, random_state=42)
    testset, devset = train_test_split(dev_test_set, test_size=0.5, random_state=42)

    print(f"Size of train is {len(trainset)}; Size of dev is {len(devset)}; Size of test is {len(testset)}.")

    with open(os.path.join(dataset_dir, "train_barplot_seaborn.json"), "w", encoding="utf-8") as file:
        json.dump(trainset, file)

    with open(os.path.join(dataset_dir, "test_barplot_seaborn.json"), "w", encoding="utf-8") as file:
        json.dump(testset, file)

    with open(os.path.join(dataset_dir, "valid_barplot_seaborn.json"), "w", encoding="utf-8") as file:
        json.dump(devset, file)


def analyse_dataset():
    dataset_dir = r'..\data\qa'
    with open(os.path.join(dataset_dir, "train_barplot_seaborn.json"), "r", encoding="utf-8") as file:
        dataset = json.load(file)

    supports_count = sum([1 for entry in dataset if entry["answer"]==1])
    print(f"Trainset has {supports_count} support claims and {len(dataset)-supports_count} refute claims.")
    print(f"Thats a ratio of {round(supports_count/len(dataset), 2)} to "
          f"{round((len(dataset)-supports_count)/len(dataset), 2)} refute claims.")


if __name__ == '__main__':
    # set path to directory with all subtables/charts
    directory = r'..\data\subtables'
    create_chart_seaborn_multiprocess(directory)

    # create_dataset_file()
    # analyse_dataset()
