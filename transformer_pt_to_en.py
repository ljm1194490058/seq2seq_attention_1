import matplotlib.pyplot as plt
import tensorflow as tf
import time
import numpy as np
import unicodedata
import re
from tensorflow import keras
from sklearn.model_selection import train_test_split
import sys
import warnings
warnings.filterwarnings("ignore")

#1.加载数据， 预处理-> dateset
#2.tools
# 2.1 位置编码
# 2.2 创建mask （a.padding  b.decoder）
# 2.3 缩放点积注意力创建
# 3.模型
# 3.1 多头注意力
# 3.2 encoder_layer
# 3.3 decoder_layer
# 3.4 encoder_model
# 3.5 decoder_model
# 3.6 transformer 串联起来
# 4. 优化器和损失函数
# 5. train step -> train
# 6. 评估和可视化


#######################加载数据及预处理###############
##############使用subword进行分词和文本id化############
#tfds.list_builders()z`
import tensorflow_datasets as tfds
examples, info = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                           as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']
# print(info)
#test': 1803,'train': 51785,'validation': 1193,

# 查看样本
# for pt, en in train_examples.take(5):
#     print(pt.numpy())   #.numpy 获取值
#     print(en.numpy())
# b"`` `` '' podem usar tudo sobre a mesa no meu corpo . ''"
# b'you can use everything on the table on me .'


# 取出英语的  使用了subword， 功能等同于tokenizer， 生成词汇表
# 文本做成类似字典的结构，既每个字都有对应的唯一数字
en_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples),
    target_vocab_size=2 ** 13)   #词汇表大小：8192个词汇量

#取葡萄牙语
pt_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
(pt.numpy() for pt, en in train_examples),
    target_vocab_size=2 ** 13)   #

###################测试是否tokenizer 成功###################
sample_string = "Transformer is awesome."
tokenized_string = en_tokenizer.encode(sample_string)    #encode 后是个列表
print('Tokenized string is {}'.format(tokenized_string))

origin_string = en_tokenizer.decode(tokenized_string)    #传入decode时必须是个列表
print('The original string is {}'.format(origin_string))

assert origin_string == sample_string

for token in tokenized_string:
    print('{} --> "{}"'.format(token, en_tokenizer.decode([token])))
##########################################################


buffer_size =20000   #shuffle用
batch_size = 64
max_length = 40

#将输入的两个句子转换为subword
def encode_to_subword(pt_sentence, en_sentence):
    #最前面和最后面添加star和end的id  方便模型知道什么时候开始什么时候结束
    pt_sentence = [pt_tokenizer.vocab_size] \
        + pt_tokenizer.encode(pt_sentence.numpy()) \
        +[pt_tokenizer.vocab_size + 1]

    en_sentence = [en_tokenizer.vocab_size] \
        + en_tokenizer.encode(en_sentence.numpy()) \
        +[en_tokenizer.vocab_size + 1]

    return pt_sentence, en_sentence

#低于max_length的样本过滤出来   进入这个函数的输入可能是tensor，用tf的方法
def filter_by_max_length(pt, en):
    #返回true false
    return tf.logical_and(tf.size(pt) <= max_length,
                          tf.size(en) <= max_length)

#使用dataset.map方法时不能直接调用python函数， 需要py_function封装起来python函数
def tf_encode_to_subword(pt_sentence, en_sentence):
    return tf.py_function(encode_to_subword,
                          [pt_sentence, en_sentence],
                          [tf.int64, tf.int64])

#######处理训练集
#将train_examples中的葡萄牙语和英语都转化为id
train_dateset = train_examples.map(tf_encode_to_subword)
#filter一下
train_dateset = train_dateset.filter(filter_by_max_length)
train_dateset = train_dateset.shuffle(
    #shuffle：打乱前buffer_size个元素
    #padded_batch: 对一个变长序列，通过padding操作将每个序列补成一样的长度
    buffer_size).padded_batch(
    #-1，-1代表数据当前有两个维度且都是列表，每个维度都在当前维度下拓展到其最大值
    batch_size, padded_shapes=([-1], [-1]))

########处理验证集
valid_dataset = val_examples.map(tf_encode_to_subword)
valid_dataset = valid_dataset.filter(
    filter_by_max_length).padded_batch(
    batch_size, padded_shapes=([-1], [-1]))

# for pt_batch,en_batch in valid_dataset.take(5):
#     print(pt_batch.shape, en_batch.shape)   #(64, 38) (64, 40) 对照([-1], [-1])
    # print(pt_batch, en_batch)    #这里的8214 和8087感觉就是起start的作用
    # pt_batch:  [[8214  342 3032...    0    0    0]
    #             [8214   95  198...    0    0    0]]
    # en_batch:  [[8087    7  618...    0    0    0]
    #             [8087 6450    1...    0    0    0]]





########################位置编码##################
#计算公式：
#PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
#PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

#pos.shape:  [sentence_length , 1]
#i.shape:  [1, d_model]
#result.shape:  [sentence_length, d_model]
#这个函数用于完成sin或cos里面的部分 ：pos /10000^(2i/d_model)
# pos:词语在句子中的位置   i：在embedding位置， d_model:embedding大小
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000,
                               (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def get_position_embedding(sentence_length, d_model):
    #别忘记增加维度 pos.shape:  [sentence_length , 1]；  i.shape:  [1, d_model]
    angle_rads = get_angles(np.arange(sentence_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    #sines.shape: [sentence_length, d_model / 2]
    #cosines.shape:  [sentence_length, d_model / 2]
    #0::2取出所有偶数， 第一个冒号取位置，第二个冒号走两步
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    #position_embedding.shape:  [sentence_length, d_model]
    #拼接最后一维
    position_embedding = np.concatenate([sines, cosines], axis=-1)

    #position_embedding.shape:  [1, sentence_length, d_model]
    #拓展一下维度
    position_embedding = position_embedding[np.newaxis, ...]

    #cast：转类型
    return tf.cast(position_embedding, dtype=tf.float32)

position_embedding = get_position_embedding(50, 512)
print(position_embedding.shape)   #(1, 50， 512)

#可视化出来
def plot_position_embedding(position_embedding):
    #RdBu配色格式，  pcolormesh：矩阵显示
    plt.pcolormesh(position_embedding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim(0, 512)
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()

plot_position_embedding(position_embedding)



##############################mask#########################
# 1.padding mask(补零) 2.look ahead mask (当前词只能与它之前词语发生attention)

# batch_data.shape:[batch_size, seq_len]
def create_padding_mask(batch_data):
    #判断batch_data是否为0， 是的话返回true并转换类型
    padding_mask = tf.cast(tf.math.equal(batch_data, 0) , tf.float32)

    #在中间添加维度  [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

x = tf.constant([[7,6,0,0,1], [1,2,3,0,0], [0,0,0,4,5]])
# print(create_padding_mask(x))
# 结果：得到4维矩阵  [[[[0., 0., 1., 1., 0.]]],     0部分返回1，非0返回0
                   # [[[0., 0., 0., 1., 1.]]]]

# attention_weight.shape:[3,3]
# [[1,0,0],    #这个二维矩阵1代表第一个词与第一个词的self-attention
#  [4,5,0],    #4代表第二个词与第一个词的self-attention
#  [7,8,9]]    #0的部分针对当前词只能与它之前词语发生attention  所以为0，需要mask掉
def create_look_ahead_mask(size):
    #tf.linalg.band_part：保留非主对角线的元素，其余位置元素为0，
    # 1-（）指 矩阵下方元素为0，矩阵左上角元素改为1  -->下三角为0，上三角为1（数值是0会返回1）
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

# print(create_look_ahead_mask(3))
#  [[0., 1., 1.],
#   [0., 0., 1.],
#   [0., 0., 0.]]



################缩放点积注意力####################
def scaled_dot_product_attention(q, k, v, mask):
    """
    Args:
    ...代表前面可能还有多个维度
    :param q: shape == (..., seq_len_q, depth)
    :param k: shape == (..., seq_len_k, depth)
    :param v: shape == (..., seq_len_v, depth_v)
       seq_len_k == seq_len_v   （q，k一一对应）
    :param mask: shape == (..., seq_len_q, seq_len_k)
    Returns:
    - output: weighted sum  （与v相乘后的加权总和）
    - attention_weights: weights of attention
    """

    #qk相乘  让第二个矩阵即k做转置才能相乘 transpose_b：转置第二个
    #matmul_qk.shape:  (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    #dk: k的维度，且是最后一维(depth这一维)  且转换下类型以便做后面的开方
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # 得到divided by 根号dk后的结果  就是缩放
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)


    if mask is not None:
        #mask * -1e9： 使得在softmax后值趋近于0  取消掉数值为0的部分
        scaled_attention_logits += (mask * -1e9)

    #attention_weights.shape :   (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis= -1)


    #去与v做相乘
    #output.shape: (..., seq_len_q, seq_len_v)    （1,4）*（4,2）
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

#打印结果进行调试
def print_scaled_dot_product_attention(q, k, v):
    temp_out, temp_att = scaled_dot_product_attention(q, k, v, None)
    print("Attention weights are: ")
    print(temp_att)
    print("Output is: ")
    print(temp_out)

##########定义临时矩阵测试缩放点积代码########
temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype=tf.float32)   #(4, 3)

temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype=tf.float32)   #(4, 2)

#q1与k转置相乘， 应该得到[0,1,0,0]
temp_q1 = tf.constant([[0, 10, 0]], dtype=tf.float32)  #(1,3)
np.set_printoptions(suppress=True)   #四舍五入去小数点
print_scaled_dot_product_attention(temp_q1, temp_k, temp_v)
# q(1, 3)  k(4, 3) v(4, 2)   dk = k[-1] 为3
# Attention weights are:  tf.Tensor([[0. 1. 0. 0.]], shape=(1, 4), dtype=float32)
# Output is:  tf.Tensor([[10.  0.]], shape=(1, 2), dtype=float32)


# temp_q2 = tf.constant([[0, 0, 10]], dtype=tf.float32)   #(1, 3)
# print_scaled_dot_product_attention(temp_q2, temp_k, temp_v)
# # Attention weights are: tf.Tensor([[0.  0.  0.5 0.5]], shape=(1, 4), dtype=float32)
# # Output is: tf.Tensor([[550.  5.5]], shape=(1, 2), dtype=float32)
#
# temp_q3 = tf.constant([[10, 10, 0]], dtype=tf.float32)   #(1, 3)
# print_scaled_dot_product_attention(temp_q3, temp_k, temp_v)
#
# #拼起来q1q2q3
# temp_q4 = tf.constant([[0, 10, 0],
#                        [0, 0, 10],
#                        [10, 10, 0]], dtype=tf.float32)   #(3, 3)
# print_scaled_dot_product_attention(temp_q4, temp_k, temp_v)
# Attention weights are: tf.Tensor(
# [[0.  1.  0.  0. ]
#  [0.  0.  0.5 0.5]
#  [0.5 0.5 0.  0. ]], shape=(3, 4), dtype=float32)
# Output is: tf.Tensor(
# [[ 10.    0. ]
#  [550.    5.5]
#  [  5.5   0. ]], shape=(3, 2), dtype=float32)



########################多头注意力###############
class MultiHeadAttention(keras.layers.Layer):
    """
    理论上：
    x -> Wq0 -> q0
    x -> Wk0 -> k0
    x -> Wv0 -> v0

    实战中：
    q -> Wq0 -> q0
    k -> Wk0 -> k0
    v -> Wv0 -> v0
    实战技巧：
    q -> Wq -> Q -> split -> q0, q1, q2...
    k,v类似
    """
    #这里的d_model 其实就是Wq的输出；  然后再split
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)

        #在多头注意力之后需要用全连接层进行拼接
        self.dense = keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        # x.shape: (batch_size, seq_len, d_model)
        # d_model = num_heads * depth
        # 期望将 x -> (batch_size, num_heads, seq_len, depth)
        #-1代表不知道填什么数字合适的情况下，可以选择， 这里就是seq_len
        x = tf.reshape(x,
                       (batch_size, -1,  self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])   #进行维度重排列

    def call(self, q, k ,v, mask):
        batch_size = tf.shape(q)[0]
        #eg.(1, 60, 256)
        q = self.WQ(q)   #q.shape :  (batch_size， seq_len_q, d_model)
        k = self.WK(k)   #k.shape :  (batch_size， seq_len_k, d_model)
        v = self.WV(v)   #v.shape :  (batch_size， seq_len_v, d_model)

        #q.shape: (batch_size，num_head, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        #k.shape: (batch_size，num_head, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        #v.shape: (batch_size，num_head, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        #scale_attention_outputs.shape :(batch_size，num_head, seq_len_q, depth) 1，8, 60，64
        #attention_weights.shape: (batch_size，num_head, seq_len_q, seq_len_k)  1，8，60，60
        scaled_attention_outputs, attention_weights = \
        scaled_dot_product_attention(q, k ,v, mask)

        #scale_attention_outputs.shape: :(batch_size，, seq_len_q, num_head， depth)
        # 计算多头注意力需要拼接depth和num_heads， 所以进行transpose，
        # 但在scaled_dot_product_attention中用的不是这个顺序
        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs, perm=[0, 2, 1, 3])

        # 拼接depth和num_head， 得到多头注意力结果
        #concat_attention.shape:  (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention_outputs,
                                      (batch_size, -1, self.d_model))

        #output.shape:  (batch_size，, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 256))  #(batch_size，, seq_len_q, dim)
output, attn = temp_mha.call(y, y, y, mask = None)
print(output.shape)   #(1, 60, 512)
print(attn.shape)     #(1, 8, 60, 60)


###################feed_forward_work###############3
def feed_forward_network(d_model, dff):
    #dff: dim of feed forward network
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])

# sample_ffn = feed_forward_network(512, 2048)
# sample_ffn(tf.random.uniform((64, 50, 512))).shape
# Output: TensorShape([64, 50, 512])

##################Encoder layer####################
class EncoderLayer(keras.layers.Layer):
    """
    encoderlayer的过程：
    x ->self attention（多头注意力） -> add & normalize & dropout
      -> feed_forward  -> add & normalize & dropout
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, training, encoder_padding_mask):
        #x.shape : (batch_size, seq_len, dim = d_model)  dim = d_model才能做残差连接
        #attn_output.shape:  (batch_size, seq_len, d_model)
        #out1.shape :   (batch_size, seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, encoder_padding_mask)   #先做self_sttention
        attn_output = self.dropout1(attn_output, training= training)  #training：告诉dropout此时在训练
        #同时做了残差连接和normalize
        out1 = self.layer_norm1(x + attn_output)

        #ffn_output.shape : (batch_size, seq_len, d_model)
        #out2.shape :  (batch_size, seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2

#注意512 得相等
sample_encoder_layer = EncoderLayer(512, 8, 2048)
sample_input = tf.random.uniform((64, 50, 512))
sample_output = sample_encoder_layer.call(sample_input, False, None)
print("EncoderLayer sample output.shape :  ", sample_output.shape)     #(64, 50, 512)


#######################decoder layer##########################
class DecoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalize & dropout -> out1
    out1, encoding_outputs -> attention -> add & normalize & dropout ->out2
    out2 -> ffn -> add & normalize & dropout
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        #mha2是编码解码器之间的注意力
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = feed_forward_network(d_model, dff)

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, encoding_outputs, training, decoder_mask, encoder_decoder_padding_mask):
        #target_seq_len： 目标语言的长度
        #######decoder_mask:是由look_ahead_mask和decoder_padding_mask合并来的

        #x.shape: (batch_size, target_seq_len, d_model) 两个d_model分别来自编解码器，所以需要值相同
        #encoding_outputs. shape:  (batch_size, iput_seq_len, d_model)

        #attn1, out1.shape:  (batch_size, target_seq_len, d_model)
        attn1, attn_weights1 = self.mha1(x, x, x, decoder_mask)  # 先做self_sttention  告诉dropout此时在训练
        attn1 = self.dropout1(attn1, training=training)
        # 同时做了残差连接和normalize
        out1 = self.layer_norm1(x + attn1)

        #attn2是编码解码器之间的attention
        #attn2, out2.shape:  (batch_size, target_seq_len, d_model)
        attn2, attn_weights2 = self.mha2(out1, encoding_outputs,
                encoding_outputs, encoder_decoder_padding_mask)  # 先做self_sttention
        attn2 = self.dropout2(attn2, training=training)
        # 同时做了残差连接和normalize
        out2 = self.layer_norm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layer_norm3(out2 + ffn_output)

        return out3, attn_weights1, attn_weights2

sample_decoder_layer = DecoderLayer(512, 8, 2048)
sample_decoder_input = tf.random.uniform((64, 60, 512))
sample_decoder_output, sample_decoder_attn_weights1, sample_decoder_attn_weights2 = sample_decoder_layer.call(
    sample_decoder_input, sample_output, False, None, None)

print("sample_decoder_output.shape: ", sample_decoder_output.shape)
print("sample_decoder_attn_weights1.shape: ", sample_decoder_attn_weights1.shape)
print(sample_decoder_attn_weights2.shape)
# (64, 60, 512)
# (64, 8, 60, 60)
# (64, 8, 60, 50)


#堆叠多个encoder layer构建encoder model
####################encoder model################
class EncoderModel(keras.layers.Layer):
    #num_layers:layer个数   input_vocab_size:输入词表大小做embedding
    def __init__(self, num_layers, input_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = keras.layers.Embedding(input_vocab_size,
                                                self.d_model)

        #position_embedding。shape ： (1, max_length, d_model)
        self.position_embedding = get_position_embedding(max_length,
                                                         self.d_model)
        #最后embedding上dropout
        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.num_layers)]

    def call(self, x, training, encoder_padding_mask):
        #x.shape: (batch_size, input_seq_len)
        input_seq_len = tf.shape(x)[1]
        #使用tf自带的断言，以免异常发生
        tf.debugging.assert_less_equal(input_seq_len , self.max_length,
                                       "input_seq_len should be less or equal to max_length")

        #x.shape: (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        #缩放， x默认embedding初始化从(0,1)的均匀分布取到，做缩放达到(0，d_model)
        #这样一来，当x与位置编码相加时，位置编码对运算影响极小
        #x的input_seq_len可能小于位置编码的max_length，需要切片
        #同时对应x.shape和位置编码shape， 发现位置编码的第一个维度为1， 但在相加时tf会自动复制batch_size份
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :input_seq_len, :]

        x = self.dropout(x, training = training)

        #经过所有的encoder layer
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, encoder_padding_mask)

        #x.shape: (batch_size, input_seq_len, d_model)
        return x
#num_layers, input_vocab_size, max_length(40),
# d_model, num_heads, dff, rate=0.1
sample_encoder_model = EncoderModel(2, 8500, max_length, 512, 8, 2048)
sample_encoder_model_input = tf.random.uniform((64, 37))
sample_encoder_model_output = sample_encoder_model.call(
    sample_encoder_model_input, training = False, encoder_padding_mask = None)
print("sample_encoder_model_output.shape： ",sample_encoder_model_output.shape)
#output:  (batch_size, input_seq_len, d_model) <=> (64, 37, 512)


######################decoder model#####################
class DecoderModel(keras.layers.Layer):
    #num_layers:layer个数   tatget_vocab_size:目标词表大小
    def __init__(self, num_layers, target_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(DecoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = keras.layers.Embedding(target_vocab_size,
                                                self.d_model)

        #position_embedding。shape ： (1, max_length, d_model)
        self.position_embedding = get_position_embedding(max_length,
                                                         self.d_model)
        #最后embedding上dropout
        self.dropout = keras.layers.Dropout(rate)

        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.num_layers)]

    def call(self, x, encoding_outputs,training, decoder_mask, encoder_decoder_padding_mask):
        #x.shape: (batch_size, output_seq_len)
        output_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(output_seq_len , self.max_length,
                                       "output_seq_len should be less or equal to max_length")

        attention_weights = {}

        #x.shape: (batch_size, output_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :output_seq_len, :]

        x = self.dropout(x, training = training)

        #经过所有的decoder layer   每一次得到的decoder layer再传给下一次做输入
        for i in range(self.num_layers):
            x , attn1, attn2 = self.decoder_layers[i](
                x,  encoding_outputs, training,
                decoder_mask, encoder_decoder_padding_mask)
            attention_weights['decoder_layer{}_att1'.format(i+1)] = attn1
            attention_weights['decoder_layer{}_att2'.format(i+1)] = attn2

        #x.shape: (batch_size, output_seq_len, d_model)

        return x, attention_weights

# num_layers, target_vocab_size, max_length,
# d_model, num_heads, dff, rate=0.1):
sample_decoder_model = DecoderModel(2, 8000, max_length,
                                    512, 8, 2048)
sample_decoder_model_input = tf.random.uniform((64, 35))
sample_decoder_model_output, sample_decoder_model_att \
    =sample_decoder_model.call(sample_decoder_model_input,
                          sample_encoder_model_output,
                          training = False,
                          decoder_mask = None,
                          encoder_decoder_padding_mask = None
                          )

print("sample_decoder_model_output.shape: ", sample_decoder_model_output.shape)
for key in sample_decoder_model_att:
    print(sample_decoder_model_att[key].shape)

# sample_decoder_model_output.shape:  (64,35, 512)
# (64, 8, 35, 35)
# (64, 8, 35, 37)
# (64, 8, 35, 35)
# (64, 8, 35, 37)




###########################串联编码解码器形成transformer#################
#最后model时才继承keras.model 其他时候继承layer
class Transformer(keras.Model):
    def __init__(self, num_layers, input_vocab_size, target_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder_model = EncoderModel(
            num_layers, input_vocab_size, max_length,
            d_model, num_heads, dff, rate )

        self.decoder_model = DecoderModel(
            num_layers, target_vocab_size, max_length,
            d_model, num_heads, dff, rate)

        #接收decoder的输出，映射到词表大小空间中
        self.final_layer = keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, encoder_padding_mask,
             decoder_mask, encoder_decoder_padding_mask):
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)
        encoding_outputs = self.encoder_model(
            inp, training, encoder_padding_mask)

        #decoding_outputs.shape:  (batch_size, output_seq_len, d_model)
        decoding_outputs, attention_weights = self.decoder_model(
            tar, encoding_outputs, training, decoder_mask,encoder_decoder_padding_mask)

        #predictions.shape:  (batch_size, output_seq_len, target_vocab_size)
        predictions = self.final_layer(decoding_outputs)

        return predictions, attention_weights

# num_layers, input_vocab_size, target_vocab_size, max_length,
#  d_model, num_heads, dff, rate
sample_transformer = Transformer(2, 8500, 8000, max_length,
                                 512, 8, 2048, rate = 0.1)
temp_input = tf.random.uniform((64, 26))
temp_target = tf.random.uniform((64, 31))

predictions, attention_weights = sample_transformer(
    temp_input, temp_target, training = False,
    encoder_padding_mask = None,
    decoder_mask = None,
    encoder_decoder_padding_mask = None)
print("predictions.shape: ", predictions.shape)  #(64, 31, 8000)
# print(predictions)
#[[[ 0.21594965 -0.43193454  0.16697396 ... -0.18922876 -0.33537954-0.48972318]]]

for key in attention_weights:
    print(key, attention_weights[key].shape)
# decoder_layer1_att1 (64, 8, 31, 31)
# decoder_layer1_att2 (64, 8, 31, 26)
# decoder_layer2_att1 (64, 8, 31, 31)
# decoder_layer2_att2 (64, 8, 31, 26)




############################ transformer连接后#########################
# 1.初始化模型
# 2.定义损失，优化器，learning_rate schedule（它能动态调整learning_rate）
# 3.训练

num_layers = 4
d_model = 128   #越大，transformer模型size越大 此时learning rate不应过高
dff = 512
num_heads = 8

# +2: 代表加入的start和end
input_vocab_size = pt_tokenizer.vocab_size + 2
target_vocab_size = en_tokenizer.vocab_size + 2

dropout_rate = 0.1

transformer = Transformer(num_layers,
                          input_vocab_size,
                          target_vocab_size,
                          max_length,
                          d_model, num_heads, dff, dropout_rate)


############################自动调节learning rate方法 自动学习率##########################
#公式：  lrate = (d_model ** -0.5) * min(step_num ** (-0.5),
#                                   step_num * warm_up_steps ** (-1.5))
#先增后减

class CustomizedSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomizedSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)  #根号下x分之1 也就是  ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))  #对应第三步

        arg3 = tf.math.rsqrt(self.d_model)   #对应第一步

        return arg3 * tf.math.minimum(arg1, arg2)

learning_rate = CustomizedSchedule(d_model)
optimizer = keras.optimizers.Adam(learning_rate,
                                  beta_1=0.9,
                                  beta_2=0.98,
                                  epsilon=1e-9)
# beta1：一阶矩估计的指数衰减率
# beta2：二阶矩估计的指数衰减率
# epsilon：一个非常小的数，防止除以零


temp_learning_rate_schedule = CustomizedSchedule(d_model)
plt.plot(
    temp_learning_rate_schedule(
        tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning rate")
plt.xlabel("Train step")
plt.show()


####损失函数
loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits= True, reduction= 'none')

#参照seq2中，  有padding部分都是0，不参与损失计算
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


######对于每组输入都创建新的mask
def create_masks(inp, tar):
    """
    Encoder:
      - encoder_padding_mask  (self attention of EncoderLayer)
    Decoder:
        look_ahead_mask (self attention of DecoderLayer)

        # encoder输出传入decoder时，应当将输出里的padding部分mask掉
        encoder_decoder_padding_mask (encoder_decoder attention of DecoderLayer)
        decoder_padding_mask (self attention of DecoderLayer)
    """
    encoder_padding_mask = create_padding_mask(inp)
    encoder_decoder_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decoder_padding_mask = create_padding_mask(tar)

    #将这两种mask合并， 因为这两种mask都在decoder layer的第一层作用， 只要有一个地方为0就mask(与运算)
    #采用取最大值方式来合并，  既不关注词它后面的单词，也不关注前面带padding的单词
    decoder_mask = tf.maximum(decoder_padding_mask,
                              look_ahead_mask)

    # print('**'*20)
    # print(encoder_padding_mask.shape)                #(64, 1, 1, 38)
    # print(encoder_decoder_padding_mask.shape)       #(64, 1, 1, 38)
    # print(look_ahead_mask.shape)                    #(34, 34)  tf会自动扩充变成--> (64, 1, 34, 34)
    # print(decoder_padding_mask.shape)               #(64, 1, 1, 34)   第三个维度1小于上方的第三个维度34 变成34
    # print(decoder_mask.shape)                       #(64, 1, 34, 34)

    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask

#迭代器，
temp_inp, temp_tar = iter(train_dateset.take(1)).next()
print(temp_inp.shape)   #(64, 38)   64个样本 长度38
print(temp_tar.shape)   #(64, 34)

create_masks(temp_inp, temp_tar)


######################训练##########################
#累计平均loss的值
train_loss = keras.metrics.Mean(name='train_loss')
#遍历时累计accuracy的值
train_accuracy = keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

@tf.function
def train_step(inp, tar):

    # 切分 把tar_inp输入给decoder，预测tar_real是否正确
    # inp： 从0到倒数第二个数    real： 从1到最后一个数
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask \
        =create_masks(inp, tar_inp)

    #求梯度及损失函数
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     encoder_padding_mask,
                                     decoder_mask,
                                     encoder_decoder_padding_mask)
        loss = loss_function(tar_real, predictions)


    gradients = tape.gradient(loss, transformer.trainable_variables)
    #梯度添加到变量上
    optimizer.apply_gradients(
        zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)

epochs = 20
for epoch in range(epochs):
    start = time.time()

    # reset_states: 从0开始累计
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dateset):
        train_step(inp, tar)
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:4f} Accuracy {:4f}'.format(
                epoch + 1, batch, train_loss.result(),
                train_accuracy.result()))

    print('Epoch {} Loss {:4f} Accuracy {:4f}'.format(
        epoch + 1, train_loss.result(), train_accuracy.result()))
    print('Time take for 1 epoch: {} secs\n'.format(time.time() - start))



#############evaluate################
"""
eg.  A B C D -> E F G H.
Train: A B C D , E F G -> F G H
Eval:  A B C D -> E
       A B C D, E -> F
       A B C D, E F -> G
       A B C D, E F G -> H
"""
def evaluate(inp_sentence):
    #先转id
    input_id_sentence = [pt_tokenizer.vocab_size] \
      + pt_tokenizer.encode(inp_sentence) + [pt_tokenizer.vocab_size + 1]

    #transformer输入都是二维的 batch_size, inp_sentence_length
    #encoder_input.shape:  (1, input_sentence_length)
    encoder_input = tf.expand_dims(input_id_sentence, 0)

    #decoder_input.shape:  (1, 1)
    decoder_input = tf.expand_dims([en_tokenizer.vocab_size], 0)

    for i in range(max_length):
        #类似train_step部分
        encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask \
            = create_masks(encoder_input, decoder_input)

        #prediction.shape: (batch_size, output_target_len, target_vocab_size)
        predictions, attention_weights = transformer(
            encoder_input,
            decoder_input,
            False,
            encoder_padding_mask,
            decoder_mask,
            encoder_decoder_padding_mask)

        #############注意！！！！  根据decoder_input中第一个值确认输出的个数，
        #############若为1则输出1个，为2输出2个预测值，区分seq2seq
        #使用切片方式取出来predictions的最后一个预测值
        #predictions.shape :   (batch_size, target_vocab_size)
        #注意这个-1， 就是取的值   注意比较两个predictions.shape
        predictions = predictions[:, -1, :]

        #预测值就是概率最大的那个索引
        predictions_id = tf.cast(tf.argmax(predictions, axis = -1), tf.int32)

        if tf.equal(predictions_id, en_tokenizer.vocab_size + 1):
            #维度缩减： (1, target_len) --> (target_len)
            return tf.squeeze(decoder_input, axis=0), attention_weights



        #若不相同，则添加到decoder_input中继续循环
        #decoder_input.shape :  (1, sentence_length)
        decoder_input = tf.concat([decoder_input, [predictions_id]],
                                  axis=-1)

    return tf.squeeze(decoder_input, axis=0), attention_weights


##########可视化multihead_attention#########################
def plot_encoder_decoder_attention(attention, input_sentence,
                                   result, layer_name):
    fig = plt.figure(figsize= (16, 8))

    input_id_sentence = pt_tokenizer.encode(input_sentence)

    #attention.shape:  (num_heads, tar_len, input_len)
    attention = tf.squeeze(attention[layer_name], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        ax.matshow(attention.shape[0])

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(input_id_sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xtickslabels(
            ['<start>'] + [pt_tokenizer.decode([i]) for i in input_id_sentence] + ['<end>'],
            fontdict = fontdict, rotation = 90   )

        ax.set_ytickslabels(
            [en_tokenizer.decode([i]) for i in result if i < en_tokenizer.vocab_size],
        fontdict = fontdict)

        ax.set_labels('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(input_sentence, layer_name = ''):
    result, attention_weights = evaluate(input_sentence)

    predicted_sentence = en_tokenizer(
        [i for i in result if i < en_tokenizer.vocab_size])

    print('Input: {}'.format(input_sentence))
    print("Predicted translation: {}".format(predicted_sentence))

    if layer_name:
        plot_encoder_decoder_attention(attention_weights, input_sentence,
                                       result, layer_name)


translate('está muito frio aqui.')
translate('esta é a minha vida')
translate('v ocê ainda está em casa')
translate('este é o primeiro livro que eu li',
          layer_name='decoder_layer4_att2')


#############模型问题
# 1.数据量少
# 2.训练次数少(20次一个小时仍然不够)


