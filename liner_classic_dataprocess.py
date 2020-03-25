import csv
import json
import random
import time
# from gensim.models import word2vec
# import codecs


def readcsv(path):
    csv_reader = csv.reader(open(path, 'r', encoding='utf-8', errors='ignore'))
    # for row in csv_reader:
    #     print(row)
    return csv_reader


def get_text_class(data):
    out_data = []
    all_text = []
    for each in data:
        temp = [each[5].strip(), each[4].strip()]
        out_data.append(temp)
        all_text.append(each[5].split(' '))
    return out_data[1:], all_text[1:]


def modify(data):
    out_list = []
    for each in data:
        temp = []
        line = each[0].split(' ')
        label = each[1]
        if len(line) == 1:
            temp.append(line[0] + '\t' + 'S-' + label)
        elif len(line) > 1:
            temp.append(line[0] + '\t' + 'B-' + label)
            for i in range(1,len(line)-1):
                temp.append(line[i] + '\t' + 'I-' + label)
            temp.append(line[-1] + '\t' + 'E-' + label)
        else:
            print(line)
        out_list.append(temp)
    return out_list


def out_put(path, data):
    with open(path, 'w', encoding='utf-8')as f:
        for each in data:
            for word in each:
                f.write(word + '\n')
            f.write('\n')


def main():
    path = r'data/plosone_data.csv'
    data = readcsv(path)
    json_data, all_text = get_text_class(data)
    random.shuffle(json_data)
    length = len(json_data)
    print(length)
    out_list = modify(json_data)
    print(len(out_list))
    for i in range(5):
        print(out_list[i])
    random.shuffle(out_list)


    #只取十分之一的数据来测试一下
    out_list = out_list[:int(0.1*len(out_list))]
    length = len(out_list)
    for i in range(10):
        left = int(0.1*i*length)
        right = int(0.1*(i+1)*length)
        train_data = out_list[: left] + out_list[right:]
        test_data = out_list[left:right]
        train_path = r'BERT\Data\train' + str(i+1) + '.txt'
        test_path = r'BERT\Data\test' + str(i+1) + '.txt'
        out_put(train_path, train_data)
        out_put(test_path, test_data)

    # print('-----------train vec------------')
    # model = word2vec.Word2Vec(all_text, min_count=1, size=100)
    # fw = codecs.open("wordvec.txt", "w", "utf-8")
    # fw.write(str(len(model.wv.vocab.keys())) + " " + "100")
    # fw.write("\n")
    # for k in model.wv.vocab.keys():
    #     fw.write(k + " " + ' '.join([str(wxs) for wxs in model[k].tolist()]))
    #     fw.write("\n")


if __name__ == '__main__':
    begin = time.clock()
    main()
    end = time.clock()
    print('用时为：' + str(end - begin))
