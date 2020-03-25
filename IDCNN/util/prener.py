import codecs

fr = codecs.open("/mnt/corpus/cclnerfortrain.txt", "r", "utf-8")
fw = codecs.open("/mnt/cclnerfortraincnn.txt", "w", "utf-8")
cixingdict = {"ns": "LOC", "nr": "LOC"}
weizhidict ={"b": "B", "m": "I", "e": "E", "s": "S"}
num=0
for line in fr:
    wordtags = line.split()
    for wordtag in wordtags:
        word, tag = wordtag.split('/')
        cixing, weizhi = tag.split('-')
        if cixing=='nt' or cixing=='nr' or cixing=='ns':
            fw.write(word+" "+weizhidict[weizhi]+"-"+cixing+"\n")
        else:
            fw.write(word + " " + "O" + "\n")
        if(word=='。' or word=='？' or word == '！'):
            num += 1
            fw.write('\n')
            if num == 2:
                num=0
print("done")