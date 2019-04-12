# -*- coding: utf-8 -*-
"""
@File  : FastTextModel.py
@Author: SangYu
@Date  : 2019/4/9 19:05
@Desc  : FastText方法
"""
import time
import fastText.FastText as ff
from DataPretreatment import load_stop_words,load_punctuation_words,load_negative_emoticons,load_positive_emoticons,load_negative_words,load_positive_words
from pyhanlp import *
import os


def segment_hanlp(sentence):
    nlp_tokenizer = JClass("com.hankcs.hanlp.tokenizer.BasicTokenizer")
    return [seg.word for seg in nlp_tokenizer.segment(sentence)]


def train_model():
    start_time = time.time()
    for i in range(5, 50):
        classifier = ff.train_supervised("fastText/train_data",epoch=i)
        classifier.save_model("fastText/model/train")
        print(time.time() - start_time)
        test = classifier.test('fastText/test_data')
        print(test)


def sentence_input(sentence):
    classifier = ff.load_model("fastText/model/train")
    stop_words = load_stop_words()
    seg_line = segment_hanlp(sentence)
    positive_emoticons = load_positive_emoticons()
    negative_emoticons = load_negative_emoticons()
    punctuation_words = load_punctuation_words()
    stop_words = load_stop_words()
    positive_words = load_positive_words()
    negative_words = load_negative_words()
    print(sentence)
    # 识别其中的表情符，并替换成相应的字符
    for p_word in positive_emoticons:
        if p_word in sentence:
            sentence = sentence.replace(p_word, " 积极 ")
    for p_word in negative_emoticons:
        if p_word in sentence:
            sentence = sentence.replace(p_word, " 消极 ")
    print(sentence)
    # 识别其中的标点符号，替换成空格
    for p_word in punctuation_words:
        if p_word in sentence:
            sentence = sentence.replace(p_word, " ")
    print(sentence)
    # 对句子进行分词,去除停用词，识别情感词，并为情感词打上特殊标记
    line = sentence.replace("蒙牛", "")
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
    label = classifier.predict(add_str.strip())
    print(label)


if __name__ == '__main__':
    # train_model()
    sentence_input("真是个良心企业")
    # print(segment_hanlp("蒙牛的加工场#爱国主义者#   ".strip()))
