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

    with open("labed_cased_5e5_epoch10_1574938019.txt", "w", encoding='utf-8') as fw:
        for o in output:
            if o[1] == '[SEP]':
                fw.write("\n")
            elif o[2] == '[CLS]':
                continue
            else:
                fw.write("\t".join(o) + '\n')

if __name__ == '__main__':
    path = "output/cased_5e5_epoch10_1574938019/"
    main(path)
