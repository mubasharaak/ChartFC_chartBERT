import argparse
import json
import multiprocessing
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from imutils.object_detection import non_max_suppression
from string import punctuation

sys.path.insert(0, r"C:\Users\k20116188\PycharmProjects\ChartFC\code\PReFIL")
import configs.config_template as CONFIG  # Allows use of autocomplete, this is overwritten by cmd line argument

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\k20116188\AppData\Local\Tesseract-OCR\tesseract.exe'

parser = argparse.ArgumentParser()
parser.add_argument('--expt_name', required=True, type=str)
parser.add_argument('--data_root', default='data', type=str)

args = parser.parse_args()
exec(f'import configs.config_{args.expt_name} as CONFIG')
CONFIG.root = args.data_root
east = "./frozen_east_text_detection.pb"
net = cv2.dnn.readNet(east)
prep_cropped = CONFIG.transform_cropped


def get_bounding_boxes(img_path, net):
    """
    (C) https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

    :param img_path:
    :return:
    """
    width = 960
    height = 960
    min_confidence = 0.5,

    # load the input image and grab the image dimensions
    image = cv2.imread(img_path)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    max_conf = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # boxes = np.array(rects)
    print(f"Number of boxes are: {len(boxes)}")

    # loop over the bounding boxes
    bboxes = []
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        bboxes.append((startX, startY, endX, endY))

    return bboxes


def crop_image(filename: str, config_text='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789', output="text"):
    """
    Given an image path, the text regions on the image are cropped with EAST, tensors generated and saved.

    :param img_path:
    :return:
    """
    # root_images = os.path.join(args.data_root, args.expt_name, "images")
    root_images = r"C:\Users\k20116188\PycharmProjects\ChartFC\data\charts"
    img_path = os.path.join(root_images, filename)
    img = Image.open(img_path).convert('L')  # set to 'L' if black-white images for FigureQA
    # img = img.point(lambda x: 0 if x < 160 else 255, '1')
    img = img.point(lambda x: 1 if x < 160 else 0, '1')
    image = cv2.imread(img_path)

    # get bounding boxes
    bboxes = get_bounding_boxes(img_path, net)
    img_cropped_list = []
    img_cropped_text_list = []

    # crop image for each bounding box
    for (startX, startY, endX, endY) in bboxes:  # @todo extract white text from colorful background
        # crop original image for each box
        cropped_image = img.crop((startX, startY, endX, endY))

        # extract img_tensor_box for each box
        # tensor_cropped_image = prep_cropped(cropped_image).unsqueeze(0)
        # img_cropped_list.append(tensor_cropped_image)

        # thresh = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # result = 255 - thresh
        result = cropped_image

        text = pytesseract.image_to_string(result, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        print(text)

        text = pytesseract.image_to_string(cropped_image,
                                           config=config_text).strip()
        img_cropped_text_list.append([startX, startY, endX, endY, text])

    # create & save cropped images tensor for img_path image
    save_path = img_path.split("charts")[0]
    save_path = os.path.join(save_path, "images_cropped", filename.split(".png")[0] + ".pt")
    # torch.save(img_cropped_list, save_path)

    print(f"Length bboxes: {len(bboxes)}")
    print(f"Length img_cropped_text_list: {len(img_cropped_text_list)}")

    return img_cropped_text_list


def run_image_cropping():
    # Load EAST test detector
    east = "./frozen_east_text_detection.pb"
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)
    CONFIG.root = args.data_root

    prep_cropped = CONFIG.transform_cropped

    cores = multiprocessing.cpu_count()
    print(f"{cores} cores used for multiprocessing.")

    start_time = time.time()
    pool = multiprocessing.Pool(cores)
    result = pool.map(crop_image, os.listdir(root_images))

    pool.close()
    pool.join()
    end_time = time.time()
    print(end_time - start_time)


def run_text_extraction(qa_file_path: str):
    # load qa file
    prep_cropped = CONFIG.transform_cropped
    east = "./frozen_east_text_detection.pb"
    net = cv2.dnn.readNet(east)

    with open(qa_file_path, "r", encoding="utf-8") as file_path:
        qa_file = json.load(file_path)

    # in parallel: iterate over each entry, get image, crop text regions, extract text
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)
    images = [entry["image_filename"] for entry in qa_file]
    result = pool.map(crop_image, images)
    pool.close()
    pool.join()

    # for each image region save text in the original file
    for i, entry in enumerate(qa_file):
        entry["img_text"] = result[i]

    # save updated file
    updated_path = qa_file_path.split(".json")[0] + "_imgtext_updated.json"
    with open(updated_path, "w", encoding="utf-8") as file_path:
        json.dump(qa_file, file_path)


def get_img_content_tesseract(filename):
    root_images = r"C:\Users\k20116188\PycharmProjects\ChartFC\data\charts_seaborn_v2"
    img_path = os.path.join(root_images, filename)
    img = Image.open(img_path).convert('L')  # set to 'L' if black-white images for FigureQA
    width, height = img.size

    # first extract position of y-axis
    img_x_axis_labels = img.crop((0, height - 55, width, height - 30))  # extract labels of x axis
    ocr_df_x_axis_labels = pytesseract.image_to_data(img_x_axis_labels, config='--psm 6 --oem 3',
                                                     output_type=pytesseract.Output.DATAFRAME)
    ocr_df_x_axis_labels = ocr_df_x_axis_labels.loc[ocr_df_x_axis_labels['conf'] > -1]
    y_axis_pos = list(ocr_df_x_axis_labels.loc[ocr_df_x_axis_labels['conf'] > -1]["left"])[0]

    # crop such that these parts are seperate: x-/y-axis titles, x-/y-axis labels, rest of chart
    img_cropped = img.crop((y_axis_pos, 0, width, height - 60))  # to extract chart content (without labels and axis)
    img_x_label = img.crop((0, height - 32, width, height))  # to extract the x label
    img_y_label = img.rotate(270, expand=True).crop((0, 0, height, 32))  # to extract the y label
    img_y_axis_labels = img.crop((32, 0, y_axis_pos - 2, height))  # extract labels of y axis
    ocr_result = []

    # resize to better recognize text w/ OCR engine
    scale_ratio = 2.5
    img_cropped = img_cropped.resize((int(img_cropped.size[0] * 1.5), int(img_cropped.size[1] * 1.5)))
    img_x_label = img_x_label.resize((int(img_x_label.size[0] * scale_ratio), int(img_x_label.size[1] * scale_ratio)))
    img_y_label = img_y_label.resize((int(img_y_label.size[0] * scale_ratio), int(img_y_label.size[1] * scale_ratio)))
    img_x_axis_labels = img_x_axis_labels.resize(
        (int(img_x_axis_labels.size[0] * scale_ratio), int(img_x_axis_labels.size[1] * scale_ratio)))
    img_y_axis_labels = img_y_axis_labels.resize(
        (int(img_y_axis_labels.size[0] * scale_ratio), int(img_y_axis_labels.size[1] * scale_ratio)))

    # detect text regions w/ tesseract
    ocr_df = pytesseract.image_to_data(img_cropped, config='--psm 6 --oem 3', output_type=pytesseract.Output.DATAFRAME)
    ocr_df_x = pytesseract.image_to_data(img_x_label, config='--psm 6 --oem 3',
                                         output_type=pytesseract.Output.DATAFRAME)
    ocr_df_y = pytesseract.image_to_data(img_y_label, config='--psm 4',
                                         output_type=pytesseract.Output.DATAFRAME)  # rotated
    ocr_df_x_axis_labels = pytesseract.image_to_data(img_x_axis_labels, config='--psm 6 --oem 3',
                                                     output_type=pytesseract.Output.DATAFRAME)
    ocr_df_y_axis_labels = pytesseract.image_to_data(img_y_axis_labels, config='--psm 6 --oem 3',
                                                     output_type=pytesseract.Output.DATAFRAME)

    # to get rid of entries with conf == -1
    conf_threshold = 50
    ocr_df = ocr_df.loc[ocr_df['conf'] > conf_threshold]
    ocr_df_x = ocr_df_x.loc[ocr_df_x['conf'] > conf_threshold]
    ocr_df_y = ocr_df_y.loc[ocr_df_y['conf'] > conf_threshold]
    ocr_df_x_axis_labels = ocr_df_x_axis_labels.loc[ocr_df_x_axis_labels['conf'] > conf_threshold]
    ocr_df_y_axis_labels = ocr_df_y_axis_labels.loc[ocr_df_y_axis_labels['conf'] > conf_threshold]

    # extract position and text for all text elements
    # scale back and add cropped pixels
    for i, row in ocr_df.iterrows():
        # if row['text'].strip():
        (x, y, w, h, text) = (
            row['left'] / 1.5 + y_axis_pos, row['top'] / 1.5, row['width'], row['height'], row['text'])
        ocr_result.append((x, y, w, h, 0, 0, text, "value_label", 0))
    for i, row in ocr_df_x.iterrows():
        # if row['text'].strip():
        (x, y, w, h, text) = (
            row['left'] / scale_ratio, row['top'] / scale_ratio + (height - 32), row['width'], row['height'],
            row['text'])
        ocr_result.append((x, y, w, h, 1, 0, text, "x_axis_title", 1))
    for i, row in ocr_df_y.iterrows():
        (x, y, w, h, text) = (
            row['left'] / scale_ratio, row['top'] / scale_ratio, row['width'], row['height'], row['text'])
        ocr_result.append((x, y, w, h, 0, 1, text, "y_axis_title", 2))
    for i, row in ocr_df_x_axis_labels.iterrows():
        (x, y, w, h, text) = (
            row['left'] / scale_ratio, row['top'] / scale_ratio + (height - 55), row['width'], row['height'],
            row['text'])
        ocr_result.append((x, y, w, h, 0, 0, text, "x_axis_labels", 3))
    for i, row in ocr_df_y_axis_labels.iterrows():
        (x, y, w, h, text) = (
            row['left'] / scale_ratio + 32, row['top'] / scale_ratio, row['width'], row['height'], row['text'])
        ocr_result.append((x, y, w, h, 0, 0, text, "y_axis_labels", 4))

    return ocr_result


def get_img_content_diverse_charts_tesseract(filename):
    root_images = r"C:\Users\k20116188\PycharmProjects\ChartFC\data\charts_seaborn_v5"
    img_path = os.path.join(root_images, filename)
    try:
        img_color = Image.open(img_path)
    except Exception as e:
        print(f"Error {e} occurred while opening image {img_path}.")
        return []

    ocr_result = []

    # resize to better recognize text w/ OCR engine
    scale_ratio = 2.5
    img_c_resized = img_color.resize((int(img_color.size[0] * scale_ratio), int(img_color.size[1] * scale_ratio)))
    img_c_rotated = img_color.rotate(270, expand=True).resize(
        (int(img_color.size[0] * scale_ratio), int(img_color.size[1] * scale_ratio)))

    ocr_df = pytesseract.image_to_data(img_c_resized, config='--psm 6 --oem 3',
                                       output_type=pytesseract.Output.DATAFRAME)
    ocr_df_2 = pytesseract.image_to_data(img_c_resized, config='--psm 12 --oem 3',
                                         output_type=pytesseract.Output.DATAFRAME)
    ocr_df_3 = pytesseract.image_to_data(img_c_rotated, config='--psm 6 --oem 3',
                                         output_type=pytesseract.Output.DATAFRAME)
    ocr_df_4 = pytesseract.image_to_data(img_c_rotated, config='--psm 12 --oem 3',
                                         output_type=pytesseract.Output.DATAFRAME)

    # to get rid of entries with conf == -1
    conf_threshold = 50

    # extract position and text for all text elements and scale back
    text_elem_dict = {}
    for df in [ocr_df, ocr_df_2, ocr_df_3, ocr_df_4]:
        df = df.loc[df['conf'] > conf_threshold]
        for i, row in df.iterrows():
            (x, y, w, h, text) = (
                row['left'] / scale_ratio, row['top'] / scale_ratio, row['width'], row['height'], row['text'])
            if pd.isna(text) or (len(text) == 1 and text in punctuation):  # filter entries == spec. character e.g. "$"
                continue
            elif text in text_elem_dict.keys(): # filter out duplicate entries detected with OCR
                if any([
                    (x_saved - 10 < x < x_saved + 10) and (y_saved - 10 < y < y_saved + 10) for (x_saved, y_saved) in
                    text_elem_dict[text]]):  # if same text at similar position has been added, skip
                    continue
                else:
                    text_elem_dict[text].append((x, y))
            else:
                text_elem_dict[text] = [(x, y)]

            ocr_result.append((x, y, w, h, 0, 0, text))

    return ocr_result


def run_text_extraction_tesseract(qa_file_path: str):
    """
    Extracts text and coordinates of boundig boxes for chart images
    :param qa_file_path:
    :return:
    """

    # load qa file
    with open(qa_file_path, "r", encoding="utf-8") as file_path:
        qa_file = json.load(file_path)

    # in parallel: iterate over each entry, get image, extract text & bounding box
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)
    images = [entry["image_filename"] for entry in qa_file]
    result = pool.map(get_img_content_diverse_charts_tesseract, images)
    pool.close()
    pool.join()

    # for each image region save text in the original file
    for i, entry in enumerate(qa_file):
        entry["img_text"] = result[i]

    # save updated file
    updated_path = qa_file_path.split(".json")[0] + "_diverse_charts.json"
    with open(updated_path, "w", encoding="utf-8") as file_path:
        json.dump(qa_file, file_path)

    # save result file
    # path = r"C:\Users\k20116188\PycharmProjects\ChartFC\data\text_elements\train_chart_text_labelled.csv"
    # df = pd.DataFrame([i for entry in result for i in entry], columns=["x", "y", "w", "h", "is_x_title", "is_y_title", "text", "label_text", "label_num"])
    # df.to_csv(path, index=False)

    print(f"After text extraction, updated file saved in {updated_path}.")


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


if __name__ == '__main__':
    iter_all_images = True
    qa_files_list = [
        r"C:\Users\k20116188\PycharmProjects\ChartFC\data\qa\valid_barplot_seaborn.json",
        r"C:\Users\k20116188\PycharmProjects\ChartFC\data\qa\test_barplot_seaborn.json",
        r"C:\Users\k20116188\PycharmProjects\ChartFC\data\qa\train_barplot_seaborn.json",
    ]
    if iter_all_images:
        for qa_file_path in qa_files_list:
            run_text_extraction_tesseract(qa_file_path)

    else:
        for filename in os.listdir(r"C:\Users\k20116188\PycharmProjects\ChartFC\data\charts_seaborn_v5"):
            # filename = "1-14532-1_4.png"
            print(get_img_content_diverse_charts_tesseract(filename))
