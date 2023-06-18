"""
@TODO give credit to TabFact authors and copy their LICENSE file
"""

# encoding=utf8
import json
import multiprocessing
import nltk
import os
import pandas as pd
import re
import spacy

from multiprocessing import Pool
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode

with open('../data/freq_list.json') as f:
    vocab = json.load(f)

with open('../data/stop_words_adjusted.json') as f:
    stop_words = json.load(f)

months_a = ['january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december']
months_b = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
a2b = {a: b for a, b in zip(months_a, months_b)}
b2a = {b: a for a, b in zip(months_a, months_b)}

numbers_text = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
}
spacy_named_entities = spacy.load("en_core_web_sm")


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def calc_levenshtein_dist(s1="", s2=""):
    """
    Used recommended implementation from here:
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

    If multiple candidates, use levenshtein distance to decide for column linking candidate

    :param s1: first input string for calculation of edit distance
    :param s2: second input string for calculation of edit distance
    :return: minimum edit distance between both input strings
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    if '(' in s1 and ')' in s1:
        s1 = s1.replace('(', '').replace(')', '')
    if '(' in s2 and ')' in s2:
        s2 = s2.replace('(', '').replace(')', '')

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                min_dist = min((distances[i1], distances[i1 + 1], distances_[-1]))
                # if min_dist > threshold:
                #     distances_ = distances_ + [threshold] * (len(s1) - len(distances_)+1)
                #     break
                distances_.append(1 + min_dist)
        distances = distances_
    return distances[-1]


def calculate_min_edit_distance(inp_1: str, inp_2: str):
    """
    Preprocessing for calculation of min edit distance
    :param inp_1:
    :param inp_2:
    :return: minimum edit distance (calculated using Levenshtein distance)
    """
    if len(inp_1.strip().split(" ")) == len(inp_2.strip().split(" ")):
        return calc_levenshtein_dist(inp_1.strip(), inp_2.strip())
    elif len(inp_1.strip().split(" ")) < len(inp_2.strip().split(" ")):
        short_str = inp_1.strip()
        long_str = inp_2.strip()
    else:
        short_str = inp_2.strip()
        long_str = inp_1.strip()

    len_short_str = len(short_str.strip().split(" "))
    len_long_str = len(long_str.strip().split(" "))
    distance_list = []
    for i in range(len_long_str - len_short_str + 1):
        dist = calc_levenshtein_dist(short_str, " ".join(long_str.strip().split(" ")[i:(i + len_short_str)]))
        distance_list.append(dist)
    return min(distance_list)


def augment(s):
    """
    Function to augment table content,
    e.g. date column in table, add "before", "latest", "year" tokens as indicator for date column

    :param s:
    :return:
    """
    recover_dict = {}
    if 'first' in s:
        s.append("1st")
        recover_dict[s[-1]] = 'first'  # adding to recover_dict "'1st': 'first'"
    elif 'second' in s:
        s.append("2nd")
        recover_dict[s[-1]] = 'second'
    elif 'third' in s:
        s.append("3rd")
        recover_dict[s[-1]] = 'third'
    elif 'fourth' in s:
        s.append("4th")
        recover_dict[s[-1]] = 'fourth'
    elif 'fifth' in s:
        s.append("5th")
        recover_dict[s[-1]] = 'fifth'
    elif 'sixth' in s:
        s.append("6th")
        recover_dict[s[-1]] = 'sixth'

    for i in range(1, 10):  # replacing e.g. '09' by '9'
        if "0" + str(i) in s:
            s.append(str(i))
            recover_dict[s[-1]] = "0" + str(i)
        if str(i) in s:
            s.append(numbers_text[str(i)])
            recover_dict[s[-1]] = numbers_text[str(i)]

    if 'crowd' in s or 'attendance' in s:  # replacing crowd -> people and crowd -> audience
        s.append("people")
        recover_dict[s[-1]] = 'crowd'
        s.append("audience")
        recover_dict[s[-1]] = 'crowd'

    # some tables include "w", "l", ... which stands for win/loss(es)
    for abbrev in ["w", "(w)", " w ", " (w) "]:
        if abbrev in s:
            s.append("win")
            recover_dict[s[-1]] = abbrev
            s.append("wins")
            recover_dict[s[-1]] = abbrev

    for abbrev in ["l", "(l)", " l ", " (l) "]:
        if abbrev in s:
            s.append("lose")
            recover_dict[s[-1]] = abbrev
            s.append("loss")
            recover_dict[s[-1]] = abbrev
            s.append("losses")
            recover_dict[s[-1]] = abbrev

    for abbrev in ["td's", "td ' s", "td 's"]:
        if abbrev in s:
            s.append("touchdown")
            recover_dict[s[-1]] = abbrev
            s.append("touchdowns")
            recover_dict[s[-1]] = abbrev

    if "avg" in s:
        s.append("average")
        recover_dict[s[-1]] = "avg"
    if "min" in s:
        s.append("minimum")
        recover_dict[s[-1]] = "min"
    if "max" in s:
        s.append("maximum")
        recover_dict[s[-1]] = "max"
    if "att" in s:
        s.append("attempt")
        recover_dict[s[-1]] = "att"
    if "%" in s:
        s.append("percentage")
        recover_dict[s[-1]] = "%"

    # one of the following time/date tokens in table, add tokens like "later" to recovery dict
    for time_tok in ["date", "time", "year", "month"]:
        if time_tok in s:  #
            s.append("before")
            recover_dict[s[-1]] = time_tok
            s.append("after")
            recover_dict[s[-1]] = time_tok
            s.append("earlier")
            recover_dict[s[-1]] = time_tok
            s.append("later")
            recover_dict[s[-1]] = time_tok
            s.append("recent")
            recover_dict[s[-1]] = time_tok
            s.append("recently")
            recover_dict[s[-1]] = time_tok
            s.append("earliest")
            recover_dict[s[-1]] = time_tok
            s.append("newest")
            recover_dict[s[-1]] = time_tok
            s.append("oldest")
            recover_dict[s[-1]] = time_tok
            s.append("older")
            recover_dict[s[-1]] = time_tok
            s.append("latest")
            recover_dict[s[-1]] = time_tok
            s.append("new years")
            recover_dict[s[-1]] = time_tok
            s.append("new year")
            recover_dict[s[-1]] = time_tok

            if "day" not in s and time_tok != "time":
                s.append("day")
                recover_dict[s[-1]] = time_tok
            if "date" not in s and time_tok != "time":
                s.append("date")
                recover_dict[s[-1]] = time_tok
            if "month" not in s and time_tok != "time":
                s.append("month")
                recover_dict[s[-1]] = time_tok
            if "year" not in s and time_tok != "time":
                s.append("year")
                recover_dict[s[-1]] = time_tok

    for entry in s:  # if '(days)' in table, add 'day' and 'days' to recovery dictionary
        if '(' in entry and ')' in entry:
            entry_no_brackets = entry.replace('(', '').replace(')', '')
            s.append(entry_no_brackets)
            recover_dict[s[-1]] = entry

            # add lemmatized version as well
            entry_lemma_list = get_lemmatize(entry_no_brackets, False)
            for entry_lemma in entry_lemma_list[0]:
                if entry_lemma != entry_no_brackets:
                    s.append(entry_lemma)
                    recover_dict[s[-1]] = entry

        for time_tok in entry.strip().split(" "):
            if time_tok in months_a + months_b and 'date' not in s:
                s.append("before")
                recover_dict[s[-1]] = time_tok
                s.append("after")
                recover_dict[s[-1]] = time_tok
                s.append("earlier")
                recover_dict[s[-1]] = time_tok
                s.append("later")
                recover_dict[s[-1]] = time_tok
                s.append("recent")
                recover_dict[s[-1]] = time_tok
                s.append("recently")
                recover_dict[s[-1]] = time_tok
                s.append("earliest")
                recover_dict[s[-1]] = time_tok
                s.append("newest")
                recover_dict[s[-1]] = time_tok
                s.append("oldest")
                recover_dict[s[-1]] = time_tok
                s.append("older")
                recover_dict[s[-1]] = time_tok
                s.append("latest")
                recover_dict[s[-1]] = time_tok
                s.append("new years")
                recover_dict[s[-1]] = time_tok
                s.append("new year")
                recover_dict[s[-1]] = time_tok

                if "day" not in s and "days" not in s:
                    s.append("day")
                    recover_dict[s[-1]] = time_tok
                if "date" not in s:
                    s.append("date")
                    recover_dict[s[-1]] = time_tok
                if "month" not in s:
                    s.append("month")
                    recover_dict[s[-1]] = time_tok
                if "year" not in s:
                    s.append("year")
                    recover_dict[s[-1]] = time_tok

    if any([_ in months_a + months_b for _ in s]):  # if a month string (from months_a and months_b) in text
        for i in range(1, 32):
            if str(i) in s:
                if i % 10 == 1:
                    s.append(str(i) + "st")
                elif i % 10 == 2:
                    s.append(str(i) + "nd")
                elif i % 10 == 3:
                    s.append(str(i) + "rd")
                else:
                    s.append(str(i) + "th")
                recover_dict[s[-1]] = str(i)

        for k in a2b:
            if k in s:
                s.append(a2b[k])
                recover_dict[s[-1]] = k

        for k in b2a:  # adds 'september' to s although was initially in list, because 'sep' has been added in
            # previous loop
            if k in s:
                s.append(b2a[k])
                recover_dict[s[-1]] = k

    # augmenting time cells in tables
    r = re.compile("^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$")
    for entry in list(filter(r.match, [s_entry.strip() for s_entry in s])):
        s.append("hour")
        recover_dict[s[-1]] = str(entry)
        s.append("minute")
        recover_dict[s[-1]] = str(entry)
        s.append("min")
        recover_dict[s[-1]] = str(entry)
        s.append("second")
        recover_dict[s[-1]] = str(entry)
        s.append("sec")
        recover_dict[s[-1]] = str(entry)
        s.append("before")
        recover_dict[s[-1]] = str(entry)
        s.append("after")
        recover_dict[s[-1]] = str(entry)
        s.append("earlier")
        recover_dict[s[-1]] = str(entry)
        s.append("later")
        recover_dict[s[-1]] = str(entry)
        s.append("recent")
        recover_dict[s[-1]] = str(entry)
        s.append("recently")
        recover_dict[s[-1]] = str(entry)
        s.append("earliest")
        recover_dict[s[-1]] = str(entry)
        s.append("latest")
        recover_dict[s[-1]] = str(entry)
        s.append("time")
        recover_dict[s[-1]] = str(entry)

        first_num = re.match("(\d+):(\d+)", entry).group(1)
        s.append(str(first_num))
        recover_dict[s[-1]] = str(entry)

        second_num = re.match("(\d+):(\d+)", entry).group(2)
        s.append(str(second_num))
        recover_dict[s[-1]] = str(entry)

        third_num = re.match("(\d+):(\d+):(\d+)", entry).group(3)
        s.append(str(third_num))
        recover_dict[s[-1]] = str(entry)

    r = re.compile("^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$")
    for entry in list(filter(r.match, s)):
        s.append("hour")
        recover_dict[s[-1]] = str(entry)
        s.append("minute")
        recover_dict[s[-1]] = str(entry)
        s.append("min")
        recover_dict[s[-1]] = str(entry)
        s.append("second")
        recover_dict[s[-1]] = str(entry)
        s.append("sec")
        recover_dict[s[-1]] = str(entry)
        s.append("before")
        recover_dict[s[-1]] = str(entry)
        s.append("after")
        recover_dict[s[-1]] = str(entry)
        s.append("earlier")
        recover_dict[s[-1]] = str(entry)
        s.append("later")
        recover_dict[s[-1]] = str(entry)
        s.append("recent")
        recover_dict[s[-1]] = str(entry)
        s.append("recently")
        recover_dict[s[-1]] = str(entry)
        s.append("earliest")
        recover_dict[s[-1]] = str(entry)
        s.append("latest")
        recover_dict[s[-1]] = str(entry)
        s.append("time")
        recover_dict[s[-1]] = str(entry)

        first_num = re.match("(\d+):(\d+)", entry).group(1)
        s.append(str(first_num))
        recover_dict[s[-1]] = str(entry)

        second_num = re.match("(\d+):(\d+)", entry).group(2)
        s.append(str(second_num))
        recover_dict[s[-1]] = str(entry)

    # augmenting date fields recognised in tables
    regex_1 = r'^(?:(?:31(\/|-|\.| \/ | - | \. | )(?:0?[13578]|1[02]|(?:jan|january|mar|march|may|jul|july|aug|august|oct|october|dec|december)))\1|(?:(?:29|30)(\/|-|\.| \/ | - | \. | )(?:0?[1,3-9]|1[0-2]|(?:jan|january|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.| \/ | - | \. | )(?:0?2|(?:feb|february))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.| \/ | - | \. | )(?:(?:0?[1-9]|(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september))|(?:1[0-2]|(?:oct|october|nov|november|dec|december)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$'
    regex_2 = r'/^(?:(?:1[6-9]|[2-9]\d)?\d{2})(?:(?:(\/|-|\.| \/ | - | \. | )(?:0?[13578]|1[02])\1(?:31))|(?:(\/|-|\.| \/ | - | \. | )(?:0?[13-9]|1[0-2])\2(?:29|30)))|(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00)))(\/|-|\.| \/ | - | \. | )0?2\3(?:29)|(?:(?:1[6-9]|[2-9]\d)?\d{2})(\/|-|\.| \/ | - | \. | )(?:(?:0?[1-9])|(?:1[0-2]))\4(?:0?[1-9]|1\d|2[0-8])$'
    regexes = [regex_1, regex_2]
    for regex in regexes:
        r = re.compile(regex)
        for entry in list(filter(r.match, s)):
            s.append("before")
            recover_dict[s[-1]] = str(entry)
            s.append("after")
            recover_dict[s[-1]] = str(entry)
            s.append("earlier")
            recover_dict[s[-1]] = str(entry)
            s.append("later")
            recover_dict[s[-1]] = str(entry)
            s.append("recent")
            recover_dict[s[-1]] = str(entry)
            s.append("recently")
            recover_dict[s[-1]] = str(entry)
            s.append("earliest")
            recover_dict[s[-1]] = str(entry)
            s.append("newest")
            recover_dict[s[-1]] = str(entry)
            s.append("oldest")
            recover_dict[s[-1]] = str(entry)
            s.append("older")
            recover_dict[s[-1]] = str(entry)
            s.append("latest")
            recover_dict[s[-1]] = str(entry)
            s.append("new years")
            recover_dict[s[-1]] = str(entry)
            s.append("new year")
            recover_dict[s[-1]] = str(entry)

            if "day" not in s:
                s.append("day")
                recover_dict[s[-1]] = str(entry)
            if "date" not in s:
                s.append("date")
                recover_dict[s[-1]] = str(entry)
            if "month" not in s:
                s.append("month")
                recover_dict[s[-1]] = str(entry)
            if "year" not in s:
                s.append("year")
                recover_dict[s[-1]] = str(entry)

            # add month
            group = re.match(
                ".*(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec).*",
                entry)
            if group:
                s.append(group.group(1))
                recover_dict[s[-1]] = str(entry)

            group = re.match("(\d+)(\/|-|\.)(\d+)(\/|-|\.)(\d{4})", entry)
            if group:
                # add 12th if date day is '12'
                i = int(group.group(1))
                if i % 10 == 1:
                    s.append(str(i) + "st")
                    recover_dict[s[-1]] = str(entry)
                elif i % 10 == 2:
                    s.append(str(i) + "nd")
                    recover_dict[s[-1]] = str(entry)
                elif i % 10 == 3:
                    s.append(str(i) + "rd")
                    recover_dict[s[-1]] = str(entry)
                else:
                    s.append(str(i) + "th")
                    recover_dict[s[-1]] = str(entry)

                # add month, e.g. "2" -> "february"
                month = int(group.group(3))
                s.append(months_a[month - 1])
                recover_dict[s[-1]] = str(entry)
                s.append(months_b[month - 1])
                recover_dict[s[-1]] = str(entry)

                year = group.group(5)
                s.append(year)
                recover_dict[s[-1]] = str(entry)

    return s, recover_dict


def replace_useless(s):
    s = s.replace(',', '')
    s = s.replace('.', '')
    s = s.replace('/', '')
    return s


def get_closest(inp, string, indexes, tabs,
                threshold):  # BP: inp=="2 players were picked during the 2nd round of the draft"
    if string in stop_words:  # stopwords are ignored in linking
        return None

    dist = 10000
    rep_string = replace_useless(string)  # replaces: , . / by blank space
    len_string = len(rep_string.split())

    # minimum = []
    # for index in indexes: # @todo dont prefilter
    #     entity = replace_useless(tabs[index[0]][index[1]])
    #     len_tab = len(entity.split())
    #     if abs(len_tab - len_string) < dist:
    #         minimum = [index]
    #         dist = abs(len_tab - len_string)
    #     elif abs(len_tab - len_string) == dist:
    #         minimum.append(index)
    minimum = indexes
    vocabs = []
    for s in rep_string.split(' '):
        vocabs.append(vocab.get(s, 10000))  # get index for s from a frequency list?

    own_implementation = True
    if own_implementation:
        indexes_list = list(indexes)
        min_distance_index = []
        columns = []
        min_dist = 10000
        for i, index in enumerate(indexes_list):
            cand = replace_useless(tabs[index[0]][index[1]])
            levenst_dist = calculate_min_edit_distance(rep_string, cand)
            if levenst_dist <= min_dist:
                min_dist = levenst_dist
                min_distance_index.append(i)
                columns.append(index[1])

        # if multiple candidates with same edit distance
        # check if one of their column headers mentioned in claim text
        columns = list(set(columns))
        if len(columns) == 1:
            return indexes_list[min_distance_index[0]]
        elif len(columns) == 0:
            return None
        else:  # multiple candidates
            min_dist_col = 10000
            col_match = -1
            for tok in inp:
                for col_i in columns:
                    col = tabs[0][col_i]
                    levenst_dist_col = calculate_min_edit_distance(tok, col)
                    if levenst_dist_col <= min_dist_col:
                        min_dist_col = levenst_dist_col
                        col_match = col_i

            cand_match = [ind for ind in indexes_list if ind[1] == col_match][0]  # one candidate of the matched column
            return cand_match

    # ignore this part of previous implementation, not understandable and does not align with what they say in paper
    else:
        # Whether contain rare words
        if dist == 0:
            return minimum[0]

        feature = [len_string]
        # Proportion
        feature.append(-dist / (len_string + dist + 0.) * 4)  # small distance is in proportion to len_string (+)
        if any([(s.isdigit() and int(s) < 100) for s in rep_string.split()]):
            feature.extend([0, 0])
        else:
            # Quite rare words (+)
            if max(vocabs) > 1000:
                feature.append(1)
            else:
                feature.append(-1)
            # Whether contain super rare words (+)
            if max(vocabs) > 5000:
                feature.append(3)
            else:
                feature.append(0)
        # Whether it is only a word (+)
        if len_string > 1:
            feature.append(1)
        else:
            feature.append(0)
        # Only one candidate in indexes (+)
        if len(indexes) == 1:
            feature.append(1)
        else:
            feature.append(0)
        # length of string matched to cell is bigger than distance (defacto non-empty) (+)
        if len_string > dist:
            feature.append(1)
        else:
            feature.append(0)

        # Only first entry of minimum considered
        cand = replace_useless(tabs[minimum[0][0]][minimum[0][1]])
        if '(' in cand and ')' in cand:
            feature.append(2)
        else:
            feature.append(0)
        # Is a header cell (+)
        if minimum[0][0] == 0:
            feature.append(2)
        else:
            feature.append(0)
        # Contains month (+)
        if any([" " + _ + " " in " " + rep_string + " " for _ in months_a + months_b]):
            feature.append(5)
        else:
            feature.append(0)

        # String found in cell text (BUT ignores that rep_string is lemmatizes and cand not) (+)
        if rep_string in cand:
            feature.append(0)
        else:
            feature.append(-5)

        if sum(feature) > threshold:
            if len(minimum) > 1:  # if two candidate cells, automatically takes one of them => adjust
                if minimum[0][0] > 0:  # cell is non-header => only column relevant
                    return [-2, minimum[0][1]]
                else:
                    return minimum[0]
            else:
                return minimum[0]
        else:  # features arbitrary calculated and ignored if bigger than 1.0 which is threshold...
            return None


def replace_number(string):
    string = re.sub(r'(\b)one(\b)', r'\g<1>1\g<2>', string)
    string = re.sub(r'(\b)two(\b)', '\g<1>2\g<2>', string)
    string = re.sub(r'(\b)three(\b)', '\g<1>3\g<2>', string)
    string = re.sub(r'(\b)four(\b)', '\g<1>4\g<2>', string)
    string = re.sub(r'(\b)five(\b)', '\g<1>5\g<2>', string)
    string = re.sub(r'(\b)six(\b)', '\g<1>6\g<2>', string)
    string = re.sub(r'(\b)seven(\b)', '\g<1>7\g<2>', string)
    string = re.sub(r'(\b)eight(\b)', '\g<1>8\g<2>', string)
    string = re.sub(r'(\b)nine(\b)', '\g<1>9\g<2>', string)
    string = re.sub(r'(\b)ten(\b)', '\g<1>10\g<2>', string)
    string = re.sub(r'(\b)eleven(\b)', '\g<1>11\g<2>', string)
    string = re.sub(r'(\b)twelve(\b)', '\g<1>12\g<2>', string)
    string = re.sub(r'(\b)thirteen(\b)', '\g<1>13\g<2>', string)
    string = re.sub(r'(\b)fourteen(\b)', '\g<1>14\g<2>', string)
    string = re.sub(r'(\b)fifteen(\b)', '\g<1>15\g<2>', string)
    string = re.sub(r'(\b)sixteen(\b)', '\g<1>16\g<2>', string)
    string = re.sub(r'(\b)seventeen(\b)', '\g<1>17\g<2>', string)
    string = re.sub(r'(\b)eighteen(\b)', '\g<1>18\g<2>', string)
    string = re.sub(r'(\b)nineteen(\b)', '\g<1>19\g<2>', string)
    string = re.sub(r'(\b)twenty(\b)', '\g<1>20\g<2>', string)
    return string


def replace(w, transliterate):
    if w in transliterate:
        return transliterate[w]
    else:
        return w


def intersect(w_new, w_old):
    new_set = []
    for w_1 in w_new:
        for w_2 in w_old:
            if w_1[:2] == w_2[:2] and w_1[2] > w_2[2]:
                new_set.append(w_2)
    return new_set


def recover(buf, recover_dict, content):
    if len(recover_dict) == 0:
        return buf
    else:
        new_buf = []
        for w in buf.split(' '):
            if w not in content:
                new_buf.append(recover_dict.get(w, w))
            else:
                new_buf.append(w)
        return ' '.join(new_buf)


def postprocess(inp, backbone, trans_backbone, transliterate, tabs, recover_dicts, repeat, threshold=1.0):
    new_str = []
    new_tags = []
    buf = ""  # words from inp if occur in backbone, i.e. match in table
    pos_buf = []  # POS tag of words added to 'buf'
    last = set()
    prev_closest = []
    inp, _, pos_tags = get_lemmatize(inp, True)
    for w, p in zip(inp, pos_tags):
        if (w in backbone) and (
                (" " + w + " " in " " + buf + " " and w in repeat) or (" " + w + " " not in " " + buf + " ")):
            if buf == "":
                last = set(
                    backbone[w])  # get cell position from backbone if word 'w' in backbone (=> occurs in table cell)
                buf = w
                pos_buf.append(p)
            else:
                proposed = set(backbone[w]) & last  # last word(s) and current w link to the same cell?
                if not proposed:  # no itersection
                    closest = get_closest(inp, buf, last, tabs,
                                          threshold)  # multiple candidates -> closest according to levenshtein edit distance
                    if closest:
                        buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                          tabs[closest[0]][closest[1]]), closest[0], closest[1])

                    new_str.append(buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = w
                    last = set(backbone[w])
                    pos_buf.append(p)
                else:  # intersect -> current and previous 'w'-> same cell in table?
                    last = proposed
                    buf += " " + w
                    pos_buf.append(p)
        # special characters
        elif w in trans_backbone and (
                (" " + w + " " in " " + buf + " " and w in repeat) or (" " + w + " " not in " " + buf + " ")):
            if buf == "":
                last = set(trans_backbone[w])
                buf = transliterate[w]
                pos_buf.append(p)
            else:
                proposed = set(trans_backbone[w]) & last
                if not proposed:
                    closest = get_closest(inp, buf, last, tabs, threshold)
                    if closest:
                        buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                          tabs[closest[0]][closest[1]]), closest[0], closest[1])
                    new_str.append(buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = transliterate[w]
                    last = set(trans_backbone[w])
                    pos_buf.append(p)
                else:
                    buf += " " + transliterate[w]
                    last = proposed
                    pos_buf.append(p)

        else:
            if buf != "":  # current word not relevant, e.g. stopword, no linking, etc. => before moving, fix previously linked words
                closest = get_closest(inp, buf, last, tabs, threshold)
                if closest:
                    buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                      tabs[closest[0]][closest[1]]), closest[0], closest[1])
                new_str.append(buf)
                if buf.startswith("#"):
                    new_tags.append('ENT')
                else:
                    new_tags.extend(pos_buf)
                pos_buf = []
                new_str.append(replace_number(w))
                new_tags.append(p)

            # if any claim token matched with NE 'LOC' or 'GPE', check if any column with this type
            # column exists, link
            elif spacy_named_entities(w).ents and any(
                    [ent for ent in spacy_named_entities(w).ents if ent.label_ in ["LOC", "GPE"]]):
                for i, cell in enumerate(tabs[1]):
                    if spacy_named_entities(cell).ents and any(
                            [ent for ent in spacy_named_entities(cell).ents if ent.label_ in ["LOC", "GPE"]]):
                        buf = '#{};{},{}#'.format(w, 0, i)
                        new_str.append(buf)
                        new_tags.append("ENT")
                        # pos_buf = []
                        break
            else:
                new_str.append(replace_number(w))
                new_tags.append(p)
            buf = ""
            last = set()

    if buf != "":
        closest = get_closest(inp, buf, last, tabs, threshold)
        if closest:
            buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                              tabs[closest[0]][closest[1]]), closest[0], closest[1])
        new_str.append(buf)
        if buf.startswith("#"):
            new_tags.append('ENT')
        else:
            new_tags.extend(pos_buf)
        pos_buf = []

    # print("LINKED FACT: {}".format(" ".join(new_str), " ".join(new_tags)))
    # print("______________________________________________________________")
    return " ".join(new_str), " ".join(new_tags)


def get_lemmatize(words, return_pos):
    """
    Lemmatizes the input of each cell, by iterating over each word, getting POS tag, applying wordnet lemmatizer
    :param words:
    :param return_pos:
    :return: (i) words after lemmatization; (ii) if change applied, a dict mapping lemma version of word to original
    """
    # words = nltk.word_tokenize(words)
    recover_dict = {}
    words = words.strip().split(' ')  # split and put words into a list
    pos_tags = [_[1] for _ in nltk.pos_tag(words)]  # decide POS tags for words in 'words'
    word_roots = []

    if "ranked" in words:
        pos_tags[words.index("ranked")] = "VBP"

    for w, p in zip(words, pos_tags):
        if is_ascii(w) and p in tag_dict:
            lemm = lemmatizer.lemmatize(w, tag_dict[p])  # wordnet lemmatizer needs POS to create correct result
            if lemm != w:
                recover_dict[lemm] = w  # if lemma same as 'w' -> recover_dict
            word_roots.append(lemm)  # else -> word_roots
        else:
            word_roots.append(w)
    if return_pos:
        return word_roots, recover_dict, pos_tags
    else:
        return word_roots, recover_dict


tag_dict = {"JJ": wordnet.ADJ,
            "NN": wordnet.NOUN,
            "NNS": wordnet.NOUN,
            "NNP": wordnet.NOUN,
            "NNPS": wordnet.NOUN,
            "VB": wordnet.VERB,
            "VBD": wordnet.VERB,
            "VBG": wordnet.VERB,
            "VBN": wordnet.VERB,
            "VBP": wordnet.VERB,
            "VBZ": wordnet.VERB,
            "RB": wordnet.ADV,
            "RP": wordnet.ADV}

lemmatizer = WordNetLemmatizer()


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def merge_strings(name, string, tags=None):
    buff = ""
    inside = False
    words = []

    for c in string:
        if c == "#" and not inside:
            inside = True
            buff += c
        elif c == "#" and inside:
            inside = False
            buff += c
            words.append(buff)
            buff = ""
        elif c == " " and not inside:
            if buff:
                words.append(buff)
            buff = ""
        elif c == " " and inside:
            buff += c
        else:
            buff += c

    if buff:
        words.append(buff)

    tags = tags.split(' ')
    assert len(words) == len(tags), "{} and {}".format(words, tags)

    i = 0
    while i < len(words):
        if i < 2:
            i += 1
        elif words[i].startswith('#') and (not words[i - 1].startswith('#')) and words[i - 2].startswith('#'):
            if is_number(words[i].split(';')[0][1:]) and is_number(words[i - 2].split(';')[0][1:]):
                i += 1
            else:
                prev_idx = words[i - 2].split(';')[1][:-1].split(',')
                cur_idx = words[i].split(';')[1][:-1].split(',')
                if cur_idx == prev_idx or (prev_idx[0] == '-2' and prev_idx[1] == cur_idx[1]):
                    position = "{},{}".format(cur_idx[0], cur_idx[1])
                    candidate = words[i - 2].split(';')[0] + " " + words[i].split(';')[0][1:] + ";" + position + "#"
                    words[i] = candidate
                    del words[i - 1]
                    del tags[i - 1]
                    i -= 1
                    del words[i - 1]
                    del tags[i - 1]
                    i -= 1
                else:
                    i += 1
        else:
            i += 1

    return " ".join(words), " ".join(tags)


def sub_func(inputs):
    name, entry = inputs
    # print("TABLE FILE: {}".format(name))
    # print("______________________________________________________________")

    # for testing: load and print table
    # try:
    #     with open(os.path.join(r"..\data\subtables\all_csv", name), encoding='UTF8') as file:
    #         print(pd.read_csv(file, sep="#"))
    # except Exception as e:
    #     print(f"Following exception was thrown for table file {name}: {e}.")
    #     return None

    backbone = {}  # dictionary mapping text to table cells
    trans_backbone = {}
    transliterate = {}
    tabs = []  # is a list of lists: contains each row of table and caption as a list
    recover_dicts = []  # dicts to get from augemented/lemmatized word to original
    repeat = set()
    with open('../data/tabfact_collected_data/all_csv/' + name, 'r', encoding='UTF8') as f:  # reading in the table
        # iterating over each line of file (= table)
        for k, _ in enumerate(f.readlines()):
            # _ = _.decode('utf8')
            tabs.append([])
            recover_dicts.append([])  # list of recover_dict for each table cell

            # per line: iterating over each cell
            for l, w in enumerate(_.strip().split('#')):  # iterating over each cell of the line
                # w = w.replace(',', '').replace('  ', ' ')
                tabs[-1].append(w)
                if len(w) > 0:  # if cell not empty
                    # lemmatize cell content
                    lemmatized_w, recover_dict = get_lemmatize(w, False)
                    lemmatized_w, new_dict = augment(lemmatized_w)  # some augmentation e.g. "first" -> "1st"
                    recover_dict.update(new_dict)  # recover_dict from both two previous steps added
                    recover_dicts[-1].append(recover_dict)

                    for i, sub in enumerate(lemmatized_w):  # all agumented/lemma.  version -> add cell link to backbone
                        if sub not in backbone:
                            backbone[sub] = [(k, l)]  # adding the row, column index for each word in table
                            if not is_ascii(sub):  # for special characters
                                trans_backbone[unidecode(sub)] = [(k, l)]
                                transliterate[unidecode(sub)] = sub
                        else:
                            if (k, l) not in backbone[sub]:
                                backbone[sub].append((k, l))
                            else:
                                if sub not in months_a + months_b:
                                    repeat.add(sub)
                            if not is_ascii(sub):  # dealing with special characters
                                trans_backbone[unidecode(sub)].append((k, l))
                                transliterate[unidecode(sub)] = sub

                    for i, sub in enumerate(w.split(' ')):
                        if sub not in backbone:
                            backbone[sub] = [(k, l)]
                            if not is_ascii(sub):
                                trans_backbone[unidecode(sub)] = [(k, l)]
                                transliterate[unidecode(sub)] = sub
                        else:
                            if (k, l) not in backbone[sub]:
                                backbone[sub].append((k, l))
                            if not is_ascii(sub):
                                trans_backbone[unidecode(sub)].append((k, l))
                                transliterate[unidecode(sub)] = sub
                else:
                    recover_dicts[-1].append({})
                    # raise ValueError("Empty Cell")

    # Adding content of table cells to backbone and tabs as well (for linking fact content to caption)
    ignore_caption = True
    captions, _ = get_lemmatize(entry[2].strip(), False)  # after processing table content, continue with caption
    # print("CAPTION: {}".format(entry[2].strip()))
    # print("______________________________________________________________")

    if not ignore_caption:
        for i, w in enumerate(captions):
            if w not in backbone:
                backbone[w] = [(-1, -1)]
            else:
                backbone[w].append((-1, -1))
        tabs.append([" ".join(captions)])

    results = []
    for i in range(len(entry[0])):  # iterating over each claim in the list of claims for corresponding table
        orig_sent = entry[0][i]
        # print(orig_sent)

        if "=" not in orig_sent:
            # linking with threshold=1.0
            sent, tags = postprocess(orig_sent, backbone, trans_backbone,
                                     transliterate, tabs, recover_dicts, repeat, threshold=1.0)
            if "#" not in sent:  # no table cell linked to fact
                sent, tags = postprocess(orig_sent, backbone, trans_backbone,
                                         transliterate, tabs, recover_dicts, repeat, threshold=0.0)  # linking again
            sent, tags = merge_strings(name, sent, tags)  # merge both results
            if not results:
                results = [[sent], [entry[1][i]], [tags], entry[2]]
            else:
                results[0].append(sent)
                results[1].append(entry[1][i])
                results[2].append(tags)
        else:  # @todo set breakpoint and examine which facts fall into this category
            # print("drop sentence: {}".format(orig_sent))
            continue

    return name, results


def get_func(filename):
    with open(filename) as f:
        data = json.load(f)
    r1_results = {}
    names = []
    entries = []
    use_multiprocessing = True

    for name in data:  # name contains all table file names; entries the fact, label and caption
        names.append(name)
        entries.append(data[name])

    # s_time = time.time()
    # for i in range(100):
    #    r = [sub_func((names[i], entries[i]))]
    # r = sub_func(('1-12221135-3.html.csv', data['1-12221135-3.html.csv']))
    # print("spent {}".format(time.time() - s_time))

    if use_multiprocessing:
        cores = multiprocessing.cpu_count()
        print(f"Number of cores is {cores}.")
        pool = Pool(cores)

        r = pool.map(sub_func, zip(names, entries))
        print("there are {} tables".format(len(r)))
        pool.close()
        pool.join()
    else:
        r = [sub_func((names[i], entries[i])) for i in range(len(names))]
        # r = [sub_func((names[i], entries[i])) for i in range(100)]

    return dict(r)


if __name__ == '__main__':
    debug = True

    if debug:
        results1 = get_func('../data/tabfact_collected_data/test.json')
        with open('../data/preprocessed_data/preprocessed_data_test.json', 'w', encoding="utf-8") as f:
            json.dump(results1, f, indent=4)

    else:
        results1 = get_func('../data/tabfact_collected_data/r1_training_all.json')
        print("finished part 1")
        results2 = get_func('../data/tabfact_collected_data/r2_training_all.json')
        print("finished part 2")
        # results2 = get_func('../collected_data/training_sample.json', '../tokenized_data/r2_training_cleaned.json')

        results2.update(results1)

        with open('../data/preprocessed_data/preprocessed_data_sample.json', 'w', encoding="utf-8") as f:
            json.dump(results2, f, indent=4)
