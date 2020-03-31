### 执行命令
```cmd
python bert_lstm_ner.py \
--task_name="NER" \ # 任务名
--do_train=True \ # 是否训练
--do_eval=True \  # 是否执行验证
--do_predict=True \ # 是否执行预测
--train_path=Abs_data/train.txt \ # 训练文件路径
--test_path=Abs_data/test.txt \ # 测试文件路径
--vocab_file=pre_models/uncased_L-12_H-768_A-12/vocab.txt \ # 选择的预训练模型,(自己去下,放pre_models文件夹里)具体介绍在下面
--bert_config_file=pre_models/uncased_L-12_H-768_A-12/bert_config.json \  # 同上
--init_checkpoint=pre_models/uncased_L-12_H-768_A-12/bert_model.ckpt \ # 同上
--max_seq_length=128 \  # 最大截断长度,i.e.每个序列超过该值长度的字符不进入训练
--train_batch_size=32 \ # 批次大小
--learning_rate=2e-5 \  # 学习率
--num_train_epochs=10.0 \ # 迭代次数 一般是3
--output_dir=./output/result_dir_1/ \ # 输出文件目录,如果存在,则会新建一个result_dir_1 + 时间戳的文件名,训练结果保存在那里
```

### 注意

1. 修改setting.py里的类别,最后三个类别别删.
2. 执行exc里的命令,修改train_path test_path 和 output_dir  // 如果是优化模型效果 修改 max_seq_length train_batch_size learning_rate 和num_train_epochs参数
3. 训练完成后会在output_dir生成 label1_test.txt token_test.txt 和 eval_result.txt三个文件
  运行python readlog.py(先打开readlog.py这个文件看看里面的路径和生成的文件名。默认是"output/result_dir_1/"和"labed_1.txt",自己修改这个文件) 生成label_1.txt 文件,这个文件就是和crf_test的结果文件一样格式。
4. 运行python conlleval.py labed_1.txt 得到prf值


### 预训练模型

预训练模型:https://github.com/google-research/bert 如图![](premodels.png)

英文:只有下面两个模型实验室电脑能跑动,分别为大小写不敏感,大小写敏感.大小写敏感的模型可以设置参数do_lower_case=flase使其生效.大小写不敏感的模型设置该参数没用

|`BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters`|`BERT-Base, Cased: 12-layer, 768-hidden, 12-heads , 110M parameters`|
|----|----|

中文:

|BERT-Base, Chinese: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters|
|----|

以及没有出现在这张图的(多语言):
|BERT-Base, Multilingual Cased: 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters|
|----|


