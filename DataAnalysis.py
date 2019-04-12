# -*- coding: utf-8 -*-
"""
@File  : DataAnalysis.py
@Author: SangYu
@Date  : 2019/4/10 15:57
@Desc  : 数据分析
"""
from DataPretreatment import segment_hanlp


def count_letter_frequency():
    """
    字频分析
    :return:
    """
    labelled_data_path = "Data/sentiment-data/pnn_annotated.txt"
    lines = [line.strip().split("\t") for line in open(labelled_data_path, "r", encoding="utf-8").readlines()]
    letter_frequency_dict = {}
    letter_count = 0
    for line in lines:
        letter_count += len(line[1])
        for word in line[1]:
            if word not in letter_frequency_dict:
                letter_frequency_dict[word] = 1
            else:
                letter_frequency_dict[word] += 1
    print("总字数：%d" % letter_count)
    with open("Analysis/letter_frequency", "w", encoding="utf-8") as f:
        f.write("字\t频数\t百分比\t累计\n")
        add_temp = 0
        for item in sorted(letter_frequency_dict.items(), key=lambda d: d[1], reverse=True):
            add_temp += int(item[1])
            f.write("%s\t%4.d\t%4.2f\t%4.2f\n"
                    % (item[0], int(item[1]), (int(item[1]) / letter_count * 100), (add_temp / letter_count * 100)))


def count_word_frequency():
    """
    词频分析
    :return:
    """
    labelled_data_path = "Data/sentiment-data/pnn_annotated.txt"
    lines = [line.strip().split("\t") for line in open(labelled_data_path, "r", encoding="utf-8").readlines()]
    word_frequency_dict = {}
    word_count = 0
    for line in lines:
        seg_line = segment_hanlp(line[1])
        word_count += len(seg_line)
        for word in seg_line:
            if word not in word_frequency_dict:
                word_frequency_dict[word] = 1
            else:
                word_frequency_dict[word] += 1
    print("总词数：%d" % word_count)
    with open("Analysis/word_frequency", "w", encoding="utf-8") as f:
        f.write("词\t频数\t百分比\t累计\n")
        add_temp = 0
        for item in sorted(word_frequency_dict.items(), key=lambda d: d[1], reverse=True):
            add_temp += item[1]
            f.write("%s\t%4.d\t%4.2f\t%4.2f\n" % (
                item[0], item[1], (item[1] / word_count * 100), (add_temp / word_count * 100)))


def count_sentence_length_frequency():
    """
    句子长度分析
    :return:
    """
    labelled_data_path = "Data/sentiment-data/pnn_annotated.txt"
    lines = [line.strip().split("\t") for line in open(labelled_data_path, "r", encoding="utf-8").readlines()]
    sentence_count = len(lines)
    sentence_length_dict = {}
    for line in lines:
        if len(line[1]) not in sentence_length_dict:
            sentence_length_dict[len(line[1])] = 1
        else:
            sentence_length_dict[len(line[1])] += 1
    sorted_dict = sorted(sentence_length_dict.items(), key=lambda d: d[0], reverse=False)
    print("总句子数：%d" % sentence_count)
    with open("Analysis/sentence_length_frequency", "w", encoding="utf-8") as f:
        f.write("长度\t频数\t百分比\t累计\n")
        add_temp = 0
        for record in sorted_dict:
            add_temp += record[1]
            f.write("%4.d\t%4.d\t%4.2f\t%4.2f\n" % (
                record[0], record[1], (record[1] / sentence_count * 100), (add_temp / sentence_count * 100)))


if __name__ == '__main__':
    count_letter_frequency()
    count_word_frequency()
    count_sentence_length_frequency()
