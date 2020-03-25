import codecs
import os
import random
def eachFile(filepath):
    pathls=[]
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        pathls.append(child)
    return pathls
def gettrainval():
    pathls=eachFile('/mnt/中古语料/')
    trainsents=[]
    valsents=[]
    for path in pathls:
        fr=codecs.open(path,'r','utf-8')
        sents=[]
        sent=[]
        for line in fr:
            word=line.split()[0]
            if word in u"。;!?":
                sent.append(line)
                sents.append(sent)
                sent=[]
            else:
                sent.append(line)
        random.shuffle(sents)
        trainsents.append(sents[int(len(sents)/10):])
        valsents.append(sents[:int(len(sents)/10)])
    fw1=codecs.open('/mnt/zhonggufordl/zhonggutrain.txt','w','utf-8')
    for sents in trainsents:
        for sent in sents:
            fw1.write(''.join(sent))
    fw2=codecs.open('/mnt/zhonggufordl/zhongguval.txt','w','utf-8')
    for sents in valsents:
        for sent in sents:
            fw2.write(''.join(sent))
def pretrain(path):
    fr = codecs.open(path, 'r', 'utf-8')
    fw = codecs.open(path.split('.')[0]+'ed.txt', 'w', 'utf-8')
    for line in fr:
        word=line.split()[0]
        tag=line.split()[1]
        if len(word)>1:
            fw.write(word[0]+' '+'B-'+tag+'\n')
            # 词中间的字
            for char in word[1:(len(word) - 1)]:
                fw.write(char + ' ' + 'I-' + tag+'\n')
            # 词的尾字
            fw.write(word[-1] + ' ' + 'E-' + tag+'\n')
            # 单字词
        else:
            fw.write(word + ' ' + 'S-' + tag+'\n')
        if (word == '。' or word == '？' or word == '！'):
            fw.write('\n')
    print('done')
if __name__ == "__main__":
    #gettrainval()
    pretrain('/mnt/zhonggufordl/zhonggutrain.txt')