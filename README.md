# seq2seq_attention_1

# 1.预处理
# 2.构建模型：encoder、attention、decoder、 损失和优化器、训练
# Encoder通过学习输入，将其编码成一个固定大小的状态向量(语义编码向量.)S，继而将S传给Decoder，
# Decoder再通过对状态向量S的学习来进行输出。
# 3.评估：给句子返回结果、 结果中的词在原句子中的权重
