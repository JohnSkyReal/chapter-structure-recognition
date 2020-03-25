import numpy as np
import collections


def find_entity_index(label, entity_cls):
    get_entity_tag = lambda l: l.split('-')[0]
    get_entity_cls = lambda l: l.split('-')[1]
    get_entity_cls_set = \
        lambda start, end: list(set([get_entity_cls(l_item) for l_item in l[index: end_index]]))
    l = label
    index = 0
    entity_index = {}
    for e in entity_cls:
        entity_index[e] = []
    entity_index['all'] = []

    while index < len(l):
        if get_entity_tag(l[index]) == 'B':
            # 两字
            if get_entity_tag(l[index + len('B')]) == 'E':
                end_index = index + len("BE")
                cls_l_set = get_entity_cls_set(index, end_index)
                if len(cls_l_set) == 1:
                    entity_index[cls_l_set[0]].append((index, end_index))
                    entity_index['all'].append((index, end_index))
                index = end_index
            # 多字
            elif get_entity_tag(l[index + len('B')]) == 'I':
                find_index = index + len('I')
                while get_entity_tag(l[find_index]) == 'I':
                    find_index += len('I')
                if get_entity_tag(l[find_index]) == 'E':
                    end_index = find_index + len('E')
                    cls_l_set = get_entity_cls_set(index, end_index)
                    if len(cls_l_set) == 1:
                        entity_index[cls_l_set[0]].append((index, end_index))
                        entity_index['all'].append((index, end_index))
                    index = end_index
                # 无关字符
                else:
                    end_index = find_index + len('E')
                    index = end_index
            # 无关字符
            else:
                end_index = index + len("B")
                index = end_index
        # 单字
        elif get_entity_tag(l[index]) == 'S':
            end_index = index + len("S")
            cls_l_set = get_entity_cls_set(index, end_index)
            if len(cls_l_set) == 1:
                entity_index[cls_l_set[0]].append((index, end_index))
                entity_index['all'].append((index, end_index))
            index = end_index
        # 无关字符
        else:
            end_index = index + len("O")
            index = end_index
    return entity_index


def calc_total_p_r_f(lab, lab_pred, label_dict, predict_dict, entity_cls):
    lab_index = find_entity_index(lab, entity_cls)
    lab_pred_index = find_entity_index(lab_pred, entity_cls)
    for l in lab_index:
        for (s_i, e_i) in lab_index[l]:
            label_dict[l] += [lab[s_i: e_i] == lab_pred[s_i: e_i]]
        for (s_i, e_i) in lab_pred_index[l]:
            predict_dict[l] += [lab_pred[s_i: e_i] == lab[s_i: e_i]]

def get_labels():
    import j_baseio
    from setting import LABELS
    content = j_baseio.readtxt_list_all_strip("test2.txt")
    lab = [i.split("\t")[1] for i in content if len(i.split("\t"))>1]
    lab_pred = [i.split("\t")[2] for i in content if len(i.split("\t"))>1]
    print(len(lab))
    print(len(lab_pred))
    label_set = LABELS[:-3]
    print(label_set)
    return label_set, lab, lab_pred

def main():
    label_set = ['B-s', 'I-s', 'E-s', 'B-a', 'E-a', 'O', 'S-c']
    label_set, lab, lab_pred = get_labels()
    # entity_cls = ['s', 'a', 'c']
    entity_cls = ['B','P','M',"R",'C']
    label_dict = collections.OrderedDict()
    predict_dict = collections.OrderedDict()
    total_acc = []

    predict_dict['all'] = []
    label_dict['all'] = []
    for label in entity_cls + label_set:
        label_dict[label] = []
        predict_dict[label] = []

    # lab = ['B-s', 'I-s', 'E-s', 'B-a', 'E-a', 'S-c', 'O']
    # lab_pred = ['B-s', 'E-s', 'B-s', 'B-a', 'E-a', 'S-c', 'O']


    for (l, p) in zip(lab, lab_pred):
        label_dict[l] += [l == p]
        predict_dict[p] += [l == p]
        total_acc += [l == p]

    calc_total_p_r_f(lab, lab_pred, label_dict, predict_dict, entity_cls)
    acc = np.mean(total_acc) if total_acc else 0
    print("total acc: {}".format(acc))

    for label in label_dict:
        p = np.mean(predict_dict[label]) if predict_dict[label] else 0
        r = np.mean(label_dict[label]) if label_dict[label] else 0
        f = (2 * p * r) / (p + r) if p + r != 0 else 0
        print("label {} p {:.3f} r {:.3f}, f {:.3f}".format(label, p, r, f))

if __name__ == '__main__':
    main()
