import os
import re
import j_baseio
from setting import LABELS

def main(path):
    token = j_baseio.readtxt_list_all_strip(path + "token_test.txt")
    raw = j_baseio.readtxt_list_all_strip(path + "raw_label_test.txt")
    pre = j_baseio.readtxt_list_all_strip(path + "label_test.txt")
    output = []
    label_map = dict(zip(range(1, len(LABELS)+1), LABELS))
    for word, label, label_predict in zip(token, raw, pre):
        label = int(label)
        output.append([word, label_map[label], label_predict])

    with open("labed_1575186054.txt", "w", encoding='utf-8') as fw:
        for o in output:
            if o[1] == '[SEP]':
                fw.write("\n")
            elif o[2] == '[CLS]':
                continue
            else:
                fw.write("\t".join(o) + '\n')
    with open("raw_label_test_label.txt", 'w', encoding='utf-8')as f:
        for o in output:
            f.write(o[1]+'\n')


if __name__ == '__main__':
    path = "output/"
    main(path)
