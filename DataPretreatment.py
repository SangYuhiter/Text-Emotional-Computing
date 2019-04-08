# -*- coding: utf-8 -*-
"""
@File  : DataPretreatment.py
@Author: SangYu
@Date  : 2019/4/8 16:48
@Desc  : 数据预处理
"""
import random
import os


def load_stop_words():
    """
    加载停用词典
    :return: 停用词表
    """
    stop_words_path = "Data/sentiment-dict/stopwords.dic"
    return [line.strip() for line in open(stop_words_path, "r", encoding="utf-8").readlines()]


def load_negative_words():
    """
    加载消极词典
    :return: 消极词表
    """
    negative_words_path = "Data/sentiment-dict/sentiment-words/negative.txt"
    return [line.strip() for line in open(negative_words_path, "r", encoding="utf-8").readlines()]


def load_positive_words():
    """
    加载积极词典
    :return: 积极词表
    """
    positive_words_path = "Data/sentiment-dict/sentiment-words/positive.txt"
    return [line.strip() for line in open(positive_words_path, "r", encoding="utf-8").readlines()]


def split_labelled_data():
    """
    切分标记数据，获得训练集和测试集
    :return: 训练集和测试集
    """
    labelled_data_path = "Data/sentiment-data/pnn_annotated.txt"
    lines = [line for line in open(labelled_data_path, "r", encoding="utf-8").readlines()]
    lines_negative = [lines[i_line] for i_line in range(0, len(lines), 3)]
    lines_positive = [lines[i_line] for i_line in range(1, len(lines), 3)]
    lines_neutral = [lines[i_line] for i_line in range(2, len(lines), 3)]
    labelled_data_dic = {"negative": lines_negative, "positive": lines_positive, "neutral": lines_neutral}
    # 随机从10个标记数据中取出一个作为测试集
    group_size = 10
    for label, line_data in labelled_data_dic.items():
        data_count = len(line_data)
        random_index_list = []
        for i_line in range(0, data_count, group_size):
            random_end = min(group_size, data_count - i_line)
            random_index = random.randrange(0, random_end)
            random_index_list.append(i_line + random_index)
        # 测试集
        test_data = [line_data[i_line] for i_line in random_index_list]
        # 训练集
        train_data = line_data
        for test in test_data:
            train_data.remove(test)
        # 写入文件
        train_data_dir = "Data/train-data"
        test_data_dir = "Data/test-data"
        with open(test_data_dir + "/" + label + ".test", "w", encoding="utf-8") as f_test:
            f_test.writelines(test_data)
        with open(train_data_dir + "/" + label + ".train", "w", encoding="utf-8") as f_train:
            f_train.writelines(train_data)


def merge_labelled_data():
    """
    分别合并已标记的训练集和数据集
    :return: train.data、test.data
    """
    data_dir_list = ["Data/train-data", "Data/test-data"]
    for data_dir in data_dir_list:
        with open(data_dir + "/" + data_dir.split("/")[-1], "w", encoding="utf-8")as f:
            f.truncate()
            file_list = os.listdir(data_dir)
            for file in file_list:
                if file[-4:] != "data":
                    lines = [line.strip() + "\n" for line in
                             open(data_dir + "/" + file, "r", encoding="utf-8").readlines()]
                    print(len(lines))
                    f.writelines(lines)


if __name__ == '__main__':
    # stop_words = load_stop_words()
    # print(stop_words)
    # negative_words = load_negative_words()
    # print(negative_words)
    # positive_words = load_positive_words()
    # print(positive_words)
    # split_labelled_data()
    merge_labelled_data()
