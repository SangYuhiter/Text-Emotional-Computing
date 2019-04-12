# -*- coding: utf-8 -*-
"""
@File  : DataPretreatment.py
@Author: SangYu
@Date  : 2019/4/8 16:48
@Desc  : 数据预处理
"""
import random
import time
from pyhanlp import *
import jieba

web_punctuation = "。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$" \
                  "﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」" \
                  "【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼"


def add_punctuation_words():
    """
    添加标点符号
    :return:
    """
    punctuation_words_path = "Data/sentiment-dict/punctuations.dic"
    old_punctuation = [line.strip() for line in open(punctuation_words_path, "r", encoding="utf-8").readlines()]
    new_punctuation = []
    for op in old_punctuation:
        if op not in new_punctuation:
            new_punctuation.append(op)
    for wp in web_punctuation:
        if wp not in new_punctuation:
            new_punctuation.append(wp)
    with open(punctuation_words_path, "w", encoding="utf-8") as f_p:
        f_p.truncate()
        for p_word in new_punctuation:
            f_p.write(p_word + "\n")


def segment_hanlp(sentence):
    nlp_tokenizer = JClass("com.hankcs.hanlp.tokenizer.BasicTokenizer")
    return [seg.word for seg in nlp_tokenizer.segment(sentence)]


def load_stop_words():
    """
    加载停用词典
    :return: 停用词表
    """
    stop_words_path = "Data/sentiment-dict/stopwords.dic"
    return set([line.strip() for line in open(stop_words_path, "r", encoding="utf-8").readlines()])


def load_punctuation_words():
    """
    加载标点符号词典
    :return: 标点符号词表
    """
    punctuation_words_path = "Data/sentiment-dict/punctuations.dic"
    return set([line.strip() for line in open(punctuation_words_path, "r", encoding="utf-8").readlines()])


def load_negative_words():
    """
    加载消极词典
    :return: 消极词表
    """
    negative_words_path = "Data/sentiment-dict/sentiment-words/negative_word.txt"
    return set([line.strip() for line in open(negative_words_path, "r", encoding="utf-8").readlines()])


def load_negative_emoticons():
    """
    加载消极表情符
    :return: 消极表情符
    """
    negative_emoticons_path = "Data/sentiment-dict/sentiment-words/negative_emoticon.txt"
    return set([line.strip() for line in open(negative_emoticons_path, "r", encoding="utf-8").readlines()])


def load_positive_words():
    """
    加载积极词典
    :return: 积极词表
    """
    positive_words_path = "Data/sentiment-dict/sentiment-words/positive_word.txt"
    return set([line.strip() for line in open(positive_words_path, "r", encoding="utf-8").readlines()])


def load_positive_emoticons():
    """
    加载积极表情符
    :return: 积极表情符
    """
    positive_emoticons_path = "Data/sentiment-dict/sentiment-words/positive_emoticon.txt"
    return set([line.strip() for line in open(positive_emoticons_path, "r", encoding="utf-8").readlines()])


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
                    f.writelines(lines)


def pretreat_data():
    f_train = open("fastText/train_data", "w", encoding="utf-8")
    f_test = open("fastText/test_data", "w", encoding="utf-8")

    train_doc = [line.strip().split("\t") for line in
                 open("Data/train-data/train-data", "r", encoding="utf-8").readlines()]
    test_doc = [line.strip().split("\t") for line in
                open("Data/test-data/test-data", "r", encoding="utf-8").readlines()]

    positive_emoticons = load_positive_emoticons()
    negative_emoticons = load_negative_emoticons()
    punctuation_words = load_punctuation_words()
    stop_words = load_stop_words()
    positive_words = load_positive_words()
    negative_words = load_negative_words()
    train_sentence = []
    for i_line in range(len(train_doc)):
        print(train_doc[i_line])
        # 识别其中的表情符，并替换成相应的字符
        for p_word in positive_emoticons:
            if p_word in train_doc[i_line][1]:
                train_doc[i_line][1] = train_doc[i_line][1].replace(p_word, " 积极 ")
        for p_word in negative_emoticons:
            if p_word in train_doc[i_line][1]:
                train_doc[i_line][1] = train_doc[i_line][1].replace(p_word, " 消极 ")
        print(train_doc[i_line][1])
        # 识别其中的标点符号，替换成空格
        for p_word in punctuation_words:
            if p_word in train_doc[i_line][1]:
                train_doc[i_line][1] = train_doc[i_line][1].replace(p_word, " ")
        print(train_doc[i_line][1])
        # 对句子进行分词,去除停用词，识别情感词，并为情感词打上特殊标记
        line = train_doc[i_line][1].replace("蒙牛", "")
        seg_line = segment_hanlp(line)
        print(seg_line)
        add_str = ""
        for word in seg_line:
            if word not in stop_words and word != "" and " " not in word:
                if word in positive_words:
                    add_str += "# " + word + " # "
                elif word in negative_words:
                    add_str += "* " + word + " * "
                else:
                    add_str += word + " "
        print(add_str.strip())
        train_sentence.append("__label__" + train_doc[i_line][0] + " , " + add_str.strip() + "\n")
    f_train.writelines(train_sentence)
    f_train.close()

    test_sentence = []
    for i_line in range(len(test_doc)):
        print(test_doc[i_line])
        # 识别其中的表情符，并替换成相应的字符
        for p_word in positive_emoticons:
            if p_word in test_doc[i_line][1]:
                test_doc[i_line][1] = test_doc[i_line][1].replace(p_word, " 积极 ")
        for p_word in negative_emoticons:
            if p_word in test_doc[i_line][1]:
                test_doc[i_line][1] = test_doc[i_line][1].replace(p_word, " 消极 ")
        print(test_doc[i_line][1])
        # 识别其中的标点符号，替换成空格
        for p_word in punctuation_words:
            if p_word in test_doc[i_line][1]:
                test_doc[i_line][1] = test_doc[i_line][1].replace(p_word, " ")
        print(test_doc[i_line][1])
        # 对句子进行分词,去除停用词，识别情感词，并为情感词打上特殊标记
        line = test_doc[i_line][1].replace("蒙牛", "")
        seg_line = segment_hanlp(line)
        print(seg_line)
        add_str = ""
        for word in seg_line:
            if word not in stop_words and word != "" and " " not in word:
                if word in positive_words:
                    add_str += "# " + word + " # "
                elif word in negative_words:
                    add_str += "* " + word + " * "
                else:
                    add_str += word + " "
        print(add_str.strip())
        test_sentence.append("__label__" + test_doc[i_line][0] + " , " + add_str.strip() + "\n")
    f_test.writelines(test_sentence)
    f_test.close()


if __name__ == '__main__':
    # stop_words = load_stop_words()
    # print(stop_words)
    # negative_words = load_negative_words()
    # print(negative_words)
    # positive_words = load_positive_words()
    # print(positive_words)
    # positive_emoticons = load_positive_emoticons()
    # negative_emoticons = load_negative_emoticons()
    # print(positive_emoticons)
    # print(negative_emoticons)
    # split_labelled_data()
    # merge_labelled_data()
    # pretreat_data()
    # add_punctuation_words()
    print(segment_hanlp("我不是蒙牛、没你想象那么纯。--要不要这么讽刺啊？蒙牛好尴尬。。。"))
    print([seg for seg in jieba.cut("我不是蒙牛、没你想象那么纯。--要不要这么讽刺啊？蒙牛好尴尬。。。")])
