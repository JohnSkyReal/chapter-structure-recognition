import yaml,time,os
from datapreprocess import vocab_build,read_dictionary,embedding_build,init_data
from model import MF_SequenceLabelingModel
# 1.加载配置文件
with open('./config.yml',encoding='utf-8') as file_config:
    config = yaml.load(file_config)
print('配置文件加载成功')
# 2.生成特征词典
print('生成特征词典')
vocab_build(config)

# 3.加载预训练词向量或生成随机初始化词向量
print('加载embedding')
fea2id_list,feature_embedding_list=embedding_build(config)


feature_num=config['model_params']['feature_nums']
feature_weight_dropout_list=[]
for i in range(feature_num):
    feature_weight_dropout_list.append(config['model_params']['embed_params'][i]['dropout_rate'])

# 4.加载标签2id
label2id = config['data_params']['label2id']
num_class = len(label2id)
print(label2id)

# 5.读取模型参数
batch_size = config['model_params']['batch_size']
epoch_num = config['model_params']['epoch_num']
max_patience = config['model_params']['max_patience'] #early stop

num_layers= config['model_params']['num_layers']
rnn_unit=config['model_params']['rnn_unit']#rnn 类型
hidden_dim = config['model_params']['hidden_dim']#rnn 单元数

dropout = config['model_params']['dropout_rate']
optimizer = config['model_params']['optimizer']
lr = config['model_params']['learning_rate']

clip = config['model_params']['clip']

use_crf=config['model_params']['use_crf']
is_attention = config['model_params']['is_attention']

timestamp = str(int(time.time()))
output_path = os.path.join('.', config['model_params']['path_save'], timestamp)
# 6.数据初始化

print('数据初始化')
train_data = init_data(feature_num,config['data_params']['path_train'],fea2id_list,label2id)
test_data = init_data(feature_num,config['data_params']['path_test'],fea2id_list,label2id)
# 7.模型初始化
print('创建模型')
model = MF_SequenceLabelingModel(feature_embedding_list,feature_num,feature_weight_dropout_list, label2id,num_class,
                 batch_size,epoch_num,max_patience,num_layers,rnn_unit,hidden_dim,
                 dropout,optimizer,lr,clip,use_crf,output_path, is_attention, config)

# 8.模型训练
print('训练开始')
model.train(train_data, test_data)



