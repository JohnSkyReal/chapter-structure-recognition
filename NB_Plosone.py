import csv
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import random
import time
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


def readcsv(path):
    csv_reader = csv.reader(open(path, 'r', encoding='utf-8', errors='ignore'))
    # for row in csv_reader:
    #     print(row)
    return csv_reader


def get_text_class(data, type_list):
    label_freq = {}  # {M:[{},{},{}], D:[{},{},{}], }
    text_type = {}
    for label in type_list:
        label_freq[label] = []
    for label in type_list:
        text_type[label] = []
    for each in data:
        line = each[5].strip().split(' ')
        label = each[4].strip()
        freq = dict(Counter(line))
        # print(freq)
        all_word_num = len(line)
        for word, num in freq.items():
            freq[word] = num/all_word_num
        # print(freq)
        label_freq[label].append(freq)
        # for calculate tfidf
        text = each[5].strip()
        text_type[label].append(text)

    # for k,v in label_freq.items():
    #     print(k,v)

    word_freq = []  # 顺序如下：['I', 'M', 'R', 'D', 'RD']
    word_freq.append(label_freq['I'])
    word_freq.append(label_freq['M'])
    word_freq.append(label_freq['R'])
    word_freq.append(label_freq['D'])
    word_freq.append(label_freq['RD'])
    # for i in word_freq:
    #     print(i)

    corpus = []  # 顺序如下：['I', 'M', 'R', 'D', 'RD']
    for label, text_list in text_type.items():
        text_type[label] = ' '.join(text_list)
    corpus.append(text_type['I'])
    corpus.append(text_type['M'])
    corpus.append(text_type['R'])
    corpus.append(text_type['D'])
    corpus.append(text_type['RD'])
    # for i in corpus:
    #     print(i)

    return word_freq, corpus


def calc_tfidf(corpus):
    # corpus = ["I come to China to travel",
    #           "This is a car polupar in China",
    #           "I love tea and Apple ",
    #           "The work is to write some papers in science"]
    vectorizer = TfidfVectorizer(stop_words='english')  # 去停用词
    tfidf = vectorizer.fit_transform(corpus)
    print(tfidf.shape)
    keywords = [{} for i in range(5)]  # 顺序如下：['I', 'M', 'R', 'D', 'RD']
    words = vectorizer.get_feature_names()
    for i in range(len(corpus)):
        # print('----Document %d----' % (i))
        for j in range(len(words)):
            if tfidf[i, j] > 1e-5:
                keywords[i][words[j]] = tfidf[i, j]
                # print(words[j], tfidf[i, j])
    for i in range(len(keywords)):
        keywords[i] = sorted(keywords[i].items(), key=lambda d: d[1], reverse=True)[:500]
        # print(keywords[i])
    return keywords


def creat_metric_tf(word_freq, keywords):
    # word_freq, keywords 顺序如下：['I', 'M', 'R', 'D', 'RD']
    # x = []
    # y = []
    for_fit = []
    label_dict = {0: 'I', 1: 'M', 2: 'R', 3: 'D', 4: 'RD'}
    for i in range(5):
        for each_dict in word_freq[i]:
            vector = []
            for word in keywords[i]:
                if word[0] in each_dict.keys():
                    vector.append(each_dict[word[0]])
                else:
                    vector.append(float(0))
            if len(vector) < 500:
                vector.extend([float(0) for i in range(500-len(vector))])
            # print(len(vector))
            # x.append(vector)
            # y.append(label_dict[i])
            for_fit.append([vector, label_dict[i]])
    return for_fit


def creat_metric_tfidf(word_freq, keywords):
    # word_freq, keywords 顺序如下：['I', 'M', 'R', 'D', 'RD']
    # x = []
    # y = []
    for_fit = []
    label_dict = {0: 'I', 1: 'M', 2: 'R', 3: 'D', 4: 'RD'}
    for i in range(5):
        for each_dict in word_freq[i]:
            vector = []
            for word in keywords[i]:
                if word[0] in each_dict.keys():
                    vector.append(word[1])
                else:
                    vector.append(float(0))
            if len(vector) < 500:
                vector.extend([float(0) for i in range(500-len(vector))])
            # print(len(vector))
            # x.append(vector)
            # y.append(label_dict[i])
            for_fit.append([vector, label_dict[i]])
    return for_fit


def main():
    path = r'../SVM/plosone_data.csv'
    new_data = []
    print("读取csv数据")
    data = readcsv(path)
    for i in data:
        new_data.append(i)  # csv生成器无法切片
    type_list = ['I', 'M', 'R', 'D', 'RD']
    print("生成词频矩阵、tfidf语料库")
    word_freq, corpus = get_text_class(new_data[1:], type_list)
    print("计算tfidf")
    keywords = calc_tfidf(corpus)
    print("构建训练与测试语料")

    # 使用tf特征构建向量空间
    # for_fit = creat_metric_tf(word_freq, keywords)
    # print("TF作为权重")

    # 使用tf-idf特征构建向量空间
    for_fit = creat_metric_tfidf(word_freq, keywords)
    print("TFIDF作为权重")

    random.shuffle(for_fit)
    length = len(for_fit)
    for i in range(5):
        x_train = []
        y_train = []
        left = int(0.1*i*length)
        right = int(0.1*(i+1)*length)
        for pair in for_fit[:left]:
            x_train.append(pair[0])
            y_train.append(pair[1])
        for pair in for_fit[right:]:
            x_train.append(pair[0])
            y_train.append(pair[1])
        x_test = []
        y_test = []
        for pair in for_fit[left:right]:
            x_test.append(pair[0])
            y_test.append(pair[1])

        # print(x)
        # print(y)
        print("开始进行第", i+1, "次交叉验证")
        print("开始训练高斯朴素贝叶斯")
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        print("训练结束，开始测试")
        y_predict = gnb.predict(x_test)
        print("开始评价")
        evalua = classification_report(y_test, y_predict, digits=4)
        print(evalua)
        pathGNB = r'GNBevaluationTFIDF'+ str(i+1) + '.txt'
        with open(pathGNB, 'w', encoding='utf-8')as f:
            f.write(str(evalua))

        print("开始训练多项朴素贝叶斯")
        mnb = MultinomialNB()
        mnb.fit(x_train, y_train)
        print("训练结束，开始测试")
        y_predict = mnb.predict(x_test)
        print("开始评价")
        evalua = classification_report(y_test, y_predict, digits=4)
        print(evalua)
        pathMNB = r'MNBevaluationTFIDF'+ str(i+1) + '.txt'
        with open(pathMNB, 'w', encoding='utf-8')as f:
            f.write(str(evalua))

        print("开始训练伯努利朴素贝叶斯")
        bnb = BernoulliNB()
        bnb.fit(x_train, y_train)
        print("训练结束，开始测试")
        y_predict = bnb.predict(x_test)
        print("开始评价")
        evalua = classification_report(y_test, y_predict, digits=4)
        print(evalua)
        pathBNB = r'BNBevaluationTFIDF' + str(i+1) + '.txt'
        with open(pathBNB, 'w', encoding='utf-8')as f:
            f.write(str(evalua))

    # len_list = []
    # for each in data_label:
    #     len_list.append(len(each[0]))
    # len_list.sort()
    # print(len_list[:100])
    # print(len_list[-100:])
    # print(len_list[int(len(len_list)/2-10):int(len(len_list)/2+10)])
    # print(sum(len_list)/len(len_list))


if __name__ == '__main__':
    start = time.clock()
    main()
    print('time used: ', time.clock() - start)
    # calc_tfidf([])
