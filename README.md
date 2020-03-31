# chapter-structure-recognition
data and codes of chapter structure recognition on PLOS ONE

中文：   
1. PLOS ONE原始语料：plosone_data.csv（Part1-3）   
2. 用于线性分类模型的tokens形式测试语料：test_data_for_linear_model文件夹   
3. 线性分类模型数据预处理及word2vec训练代码：liner_classic_dataprocess.py   
4. NB模型：NB_Plosone.py   
5. SVM模型：SVM_Plosone.py   
6. CRF模型：crf++ tools文件夹   
7. RNN模型组、Bi-LSTM模型组：RNNBiLSTMCRFATTENTION文件夹，需自行新建data、modeloutput、voc文件夹   
8. BERT模型：bert_ner文件夹，需自行新建Abs_data、output、pre_models文件夹   
9. BERT-Bi-LSTM-CRF模型：bert_lstm_crf文件夹，需自行新建Abs_data、output、pre_models文件夹   
10. IDCNN模型：IDCNN文件夹，需自行新建data、ckpt_IDCNN、log、result文件夹   
11. RNN、Bi-LSTM、IDCNN模型 tensorflow <= 1.8.0    
12. BERT、BERT-Bi-LSTM-CRF模型 tensorflow >= 1.12.0   

*****

English：   
1. PLOS ONE original data: plosone_data.csv（Part1-3）   
2. Test corpus of tokens for linear classification models: test_data_for_linear_model folder   
3. Linear classification model data preprocessing and word2vec training code: liner_classic_dataprocess.py   
4. NB model: NB_Plosone.py   
5. SVM model: SVM_Plosone.py   
6. CRF model: crf ++ tools folder   
7. RNN model group, Bi-LSTM model group: RNBBiLSTMCRFATTENTION folder, you need to create new data, modeloutput, voc folders by yourself   
8. BERT model: bert_ner folder, you need to create new Abs_data, output, pre_models folders by yourself   
9. BERT-Bi-LSTM-CRF model: bert_lstm_crf folder, you need to create new Abs_data, output, pre_models folders by yourself   
10. IDCNN model: IDCNN folder, you need to create new data, ckpt_IDCNN, log, result folders by yourself   
11. RNN, Bi-LSTM, IDCNN models tensorflow <= 1.8.0   
12. BERT, BERT-Bi-LSTM-CRF model tensorflow> = 1.12.0   
