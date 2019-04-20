# -*- coding: utf-8 -*-
"""
@File  : FastTextModel.py
@Author: SangYu
@Date  : 2019/4/9 19:05
@Desc  : FastText方法
"""
import time
import fastText.FastText as ff
from DataPretreatment import load_stop_words, load_punctuation_words, load_negative_emoticons, load_positive_emoticons, \
    load_negative_words, load_positive_words
from pyhanlp import *
import os
import matplotlib.pyplot as plt
from pylab import *
import jieba

mpl.rcParams['font.sans-serif'] = ['SimHei']


def segment_hanlp(sentence):
    nlp_tokenizer = JClass("com.hankcs.hanlp.tokenizer.BasicTokenizer")
    return [seg.word for seg in nlp_tokenizer.segment(sentence)]


def train_model():
    start_time = time.time()
    all_marco_precision = []
    all_marco_recall = []
    all_marco_f1 = []
    all_micro_precision = []
    all_micro_recall = []
    all_micro_f1 = []
    for i in range(5, 51):
        classifier = ff.train_supervised("fastText/train_data", epoch=i, lr=0.5)
        classifier.save_model("fastText/model/train")
        print("模型构建时间：%s s" % str(time.time() - start_time))

        # 因为fasttext中设计的是针对多标签的精确率与召回率，对于单标签，计算结果一致，不具有参考价值
        # print("积极数据测试：")
        # test = classifier.test('fastText/test_data_positive')
        # print("测试数据数量：%d\t准确率：%f\t召回率：%f" % (test[0], test[1], test[2]))
        # print("中立数据测试：")
        # test = classifier.test('fastText/test_data_neutral')
        # print("测试数据数量：%d\t准确率：%f\t召回率：%f" % (test[0], test[1], test[2]))
        # print("消极数据测试：")
        # test = classifier.test('fastText/test_data_negative')
        # print("测试数据数量：%d\t准确率：%f\t召回率：%f" % (test[0], test[1], test[2]))

        correct_labels = [line.strip().split(" , ")[0] for line in
                          open('fastText/test_data', "r", encoding="utf-8").readlines()]
        texts = [line.strip().split(" , ")[1] for line in open('fastText/test_data', "r", encoding="utf-8").readlines()]
        predict_labels = classifier.predict(texts)[0]
        true_positive = 0
        false_positive = 0
        false_negative = 0
        evaluation_parameters = []
        labels = {"__label__-1": "消极", "__label__0": "中立", "__label__1": "积极"}
        for label, name in labels.items():
            evaluate_p = {}
            print("%s标签测试结果：" % name)
            evaluate_p["name"] = name
            evaluate_p["nexample"] = len(texts)
            for i in range(len(texts)):
                # 预测属于该类，实际属于该类
                if predict_labels[i] == label and correct_labels[i] == label:
                    true_positive += 1
                # 预测属于该类，实际不属于该类
                elif predict_labels[i] == label and correct_labels[i] != label:
                    false_positive += 1
                # 预测不属于该类，实际属于该类
                elif predict_labels[i] != label and correct_labels[i] == label:
                    false_negative += 1
            evaluate_p["true_positive"] = true_positive
            evaluate_p["false_positive"] = false_positive
            evaluate_p["false_negative"] = false_negative
            # 计算精确率、召回率、F值
            precision = true_positive / (true_positive + false_positive)
            evaluate_p["precision"] = precision
            recall = true_positive / (true_positive + false_negative)
            evaluate_p["recall"] = recall
            f1 = 2 * precision * recall / (precision + recall)
            evaluate_p["f1"] = f1
            evaluation_parameters.append(evaluate_p)
            print("测试集大小：%d\t精确率：%f\t召回率：%f\tF_1：%f" % (len(texts), precision, recall, f1))
        # 计算宏平均和微平均
        sum_precision = 0
        sum_recall = 0
        sum_true_positive = 0
        sum_false_positive = 0
        sum_false_negative = 0
        for p in evaluation_parameters:
            sum_precision += p["precision"]
            sum_recall += p["recall"]
            sum_true_positive += p["true_positive"]
            sum_false_positive += p["false_positive"]
            sum_false_negative += p["false_negative"]
        n = len(evaluation_parameters)
        marco_precision = sum_precision / n
        all_marco_precision.append(marco_precision)
        marco_recall = sum_recall / n
        all_marco_recall.append(marco_recall)
        marco_f1 = 2 * marco_precision * marco_recall / (marco_precision + marco_recall)
        all_marco_f1.append(marco_f1)
        print("宏平均----测试集大小：%d\t精确率：%f\t召回率：%f\tF_1：%f" % (len(texts), marco_precision, marco_recall, marco_f1))
        micro_true_positive = sum_true_positive / n
        micro_false_positive = sum_false_positive / n
        micro_false_negative = sum_false_negative / n
        micro_precision = micro_true_positive / (micro_true_positive + micro_false_positive)
        all_micro_precision.append(micro_precision)
        micro_recall = micro_true_positive / (micro_true_positive + micro_false_negative)
        all_micro_recall.append(micro_recall)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        all_micro_f1.append(micro_f1)
        print("微平均----测试集大小：%d\t精确率：%f\t召回率：%f\tF_1：%f" % (len(texts), micro_precision, micro_recall, micro_f1))

    names = [i for i in range(5, 51)]
    ax1 = plt.subplot(311)
    plt.plot(names, all_marco_precision, label='marco-P')
    plt.plot(names, all_micro_precision, label='micro-P')
    plt.legend(loc='upper left')
    ax2 = plt.subplot(312, sharey=ax1)
    plt.plot(names, all_marco_recall, label='marco-P')
    plt.plot(names, all_micro_recall, label='micro-R')
    plt.legend(loc='upper left')
    plt.subplot(313, sharey=ax1)
    plt.plot(names, all_marco_f1, label='marco-F1')
    plt.plot(names, all_micro_f1, label='micro-F1')
    plt.legend(loc='upper left')
    plt.xlabel(u"训练轮数(ngram=1)")
    plt.savefig('./ngram1.png')
    plt.show()


def sentence_input(sentence):
    classifier = ff.load_model("fastText/model/train")
    stop_words = load_stop_words()
    seg_line = jieba.cut(sentence)
    positive_emoticons = load_positive_emoticons()
    negative_emoticons = load_negative_emoticons()
    punctuation_words = load_punctuation_words()
    stop_words = load_stop_words()
    positive_words = load_positive_words()
    negative_words = load_negative_words()
    # print(sentence)
    # 识别其中的表情符，并替换成相应的字符
    for p_word in positive_emoticons:
        if p_word in sentence:
            sentence = sentence.replace(p_word, " 积极 ")
    for p_word in negative_emoticons:
        if p_word in sentence:
            sentence = sentence.replace(p_word, " 消极 ")
    # print(sentence)
    # 识别其中的标点符号，替换成空格
    for p_word in punctuation_words:
        if p_word in sentence:
            sentence = sentence.replace(p_word, " ")
    # print(sentence)
    # 对句子进行分词,去除停用词，识别情感词，并为情感词打上特殊标记
    line = sentence.replace("蒙牛", "")
    seg_line = jieba.cut(line)
    # print(seg_line)
    add_str = ""
    for word in seg_line:
        if word not in stop_words and word != "" and " " not in word:
            if word in positive_words:
                add_str += "# " + word + " # "
            elif word in negative_words:
                add_str += "* " + word + " * "
            else:
                add_str += word + " "
    # print(add_str.strip())
    label = classifier.predict([add_str.strip()], k=3)
    result = [(str(i), j) for i, j in zip(label[0], label[1])]
    print(result)
    return result
    # label_dict = {"__label__1": "正向", "__label__0": "中立", "__label__-1": "负向"}
    # label_probs = []
    # for label in label:
    #     label_probs.append([label_dict[label[0]], label[1]])
    # print(label_probs)


def sentence(sentence):
    return [('__label__1', 0.9969652891159058), ('__label__-1', 0.0030545589979737997), ('__label__0', 1.0085684152727481e-05)]

if __name__ == '__main__':
    # train_model()
    print(sentence_input("真是个良心企业"))
    # print(segment_hanlp("蒙牛的加工场#爱国主义者#   ".strip()))
