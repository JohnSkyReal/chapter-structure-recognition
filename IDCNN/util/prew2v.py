import gensim
import codecs
model = gensim.models.Word2Vec.load("/mnt/model/siku.model")
fw = codecs.open("/mnt/sikuvec.txt","w","utf-8")
fw.write(str(len(model.wv.vocab.keys()))+" "+"128")
fw.write("\n")
for k in model.wv.vocab.keys():
    fw.write(k+" "+' '.join([str(wxs) for wxs in model[k].tolist()]))
    fw.write("\n")
print('done')