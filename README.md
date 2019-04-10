# Text-Emotional-Computing
文本情感计算--文本分类任务

### DataPretreatment.py
#### 数据预处理：加载停用词词典、正向词词典、负向词词典、训练集与测试集的划分

### Data:数据集
#### sentiment-data:已标记的原始数据集（正向、负向、中立各1000条）
#### sentiment-dict:停用词典、正向词词典、负向词词典
#### train-data:训练集，分别标记的与汇总的，占原数据量的90%
#### test-data:测试集，分别标记的与汇总的，占原数据量的10%，使用随机数的方法分别从每类中每10个抽取1个

### DataAnalysis.py
#### 数据分析:字频统计、词频统计、句子长度统计

### Analysis:分析结果
#### letter_frequency:字频统计结果
#### word_frequency:词频统计结果
#### sentence_length_frequency:句子长度统计结果