import matplotlib.pyplot as plt
import sys
import tensorflow as tf        #2.1.0
import time
import numpy as np
import sklearn
import unicodedata
import re
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

# print(tf.__version__)
# print(sys.version_info)
# for module in np, pd, sklearn, tf, keras:
#     print(module.__name__, module.__version__)

#运用seq2seq+注意力机制做西班牙语和英语的机器翻译
# 1.预处理
# 2.构建模型：encoder、attention、decoder、 损失和优化器、训练
# Encoder通过学习输入，将其编码成一个固定大小的状态向量(语义编码向量.)S，继而将S传给Decoder，
# Decoder再通过对状态向量S的学习来进行输出。
# 3.评估：给句子返回结果、 结果中的词在原句子中的权重

#西班牙语和英语翻译语料 \t分割
file_path = r'C:\Users\独为我唱\Desktop\spa-eng\spa.txt'


test_eng = 'Duck!'
test_spa = '¡Inclínate!'

#由unicode转ascii 主要为了降低词表  毕竟ascii只有两百多   这里按数据集可用可不用
def unicode_to_ascii(s):
    #NFD ： 如果一个ascii由多个unicode组成， 则拆开
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c) != 'Mn')


#对每个句子进行预处理
def preprocess_sentence(s):
    s = unicode_to_ascii(s.lower().strip())   #小写+去多余空格
    #标点符号前后加空格
    s = re.sub(r"([?.!,¡¿])", r" \1 ", s)
    #多余的空格变成一个空格
    s = re.sub(r'[" "]+', " ", s)
    #除了标点符号和字母外都是空格
    s = re.sub(r'[^a-zA-Z?.!,¡¿]', " ", s)
    #去掉前后空格
    s = s.rstrip().strip()
    s = '<start> ' + s + ' <end>'
    return s
# print(preprocess_sentence(test_eng))


#分隔开西班牙与和英语， 将西班牙与和英语分别放在两个dataset中
def parse_data(filename):
    lines = open(file_path, encoding='UTF-8').read().strip().split('\n')
    sentence_pairs = [line.strip().split('\t',1) for line in lines]    #英文和西班牙语\t分隔开的，且每句结尾有\t，故分割一次

    preprocess_sentence_pairs = [
        (preprocess_sentence(en), preprocess_sentence(sp)) for en, sp in sentence_pairs
    ]
    return zip(*preprocess_sentence_pairs)    #zip(*[(1,2), (3,4)])  会变成(1,3) (2,4)
en_dataset, sp_dataset = parse_data(file_path)
# print(preprocess_sentence(en_dataset[-1]))


#数据id化，生成词典  使用tokenizer
def tokenizer(lang):
    lang_tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=None, filters='', split=' '     #对词语数目无限制， 不过滤， 以空格为分割
    )
    lang_tokenizer.fit_on_texts(lang)  #统计词频，生成词表
    tensor = lang_tokenizer.texts_to_sequences(lang)    #文本转id
    tensor = keras.preprocessing.sequence.pad_sequences(       #在后面padding
        tensor, padding='post'
    )
    return tensor, lang_tokenizer

######这里将西班牙语设定为输入， 英语设定为输出###########
input_tensor, input_tokenizer = tokenizer(sp_dataset[0:30000])   #input_tensor类似于二维数组. 全是文本的id
output_tensor, output_tokenizer = tokenizer(en_dataset[0:30000])   #[[   1   25  729 ...    0    0    0]
                                                                    # [   1   25 4828 ...    0    0    0]]

def max_length(tensor):
    return max(len(t) for t in tensor)

max_length_input = max_length(input_tensor)
max_length_output = max_length(output_tensor)
# print(max_length_input, max_length_output)   #10, 17


################## 做dataset 训练集测试集###################
input_train, input_eval, output_train, output_eval = train_test_split(input_tensor, output_tensor, test_size=0.2)


#验证tokenizer是否正常转换  只是验证用
def convert(example, tokenizer):
    for t in example:
        if t != 0:
            print('%d --> %s' % (t, tokenizer.index_word[t]))
# convert(input_train[0], input_tokenizer)   #22 --> estoy
# print()
# convert(output_train[0], output_tokenizer) #4 --> i


#制作dataset, 返回的dataset包含特征和标签两部分，这里是英文和西班牙语， tensor类型
def make_dataset(input_tensor, output_tensor,
                 batch_size, epochs, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor, output_tensor))
    if shuffle:
        dataset = dataset.shuffle(30000)    #打乱训练集中dataset的元素，30000表示打乱时使用的buffer的大小：
    dataset = dataset.repeat(epochs).batch(
        batch_size, drop_remainder = True    #将序列进行重复
    )
    return dataset

batch_size = 64
epochs = 20
train_dataset = make_dataset(
    input_train, output_train, batch_size, epochs, True)
eval_dataset = make_dataset(
    input_eval, output_eval, batch_size, 1, False)
for x, y in train_dataset.take(1):
    print("train_dataset_x_shape: ", x.shape)   #(64, 17)
    # print(y.shape)  #(64, 10)



##########################模型定义￥#############################
#超参定义  encoder， decoder， 注意力机制
embedding_units = 256    #每个单词转为embedding是多少维
units = 1024     #神经网络用的units， encoder与decoder的一样
input_vocab_size = len(input_tokenizer.word_index) + 1     #9326 tokenizer字典长度+1
output_vocab_size = len(output_tokenizer.word_index) + 1   #4744

class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_units, encoding_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoding_units = encoding_units
        self.embedding = keras.layers.Embedding(vocab_size,    #输入数目(词汇表大小)，文本数据中词汇取值的可能数，在字典内
                                                embedding_units)   #输出向量大小
        #初始化，基于方差缩放  参考：https://blog.csdn.net/lygeneral/article/details/106733877
        self.gru = keras.layers.GRU(self.encoding_units,
                                    return_sequences=True,      #因为要加注意力， 需拿到每一步的输出
                                    return_state=True,      #同上一行
                                    recurrent_initializer='glorot_uniform')   #glorot_uniform，根据每层神经元数量调整参数方差。

    def call(self, x, hidden):
        x = self.embedding(x)    #因为上面的embedding确定了前两个超参，
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):   #初始化的隐藏状态，
        return tf.zeros((self.batch_size, self.encoding_units))   #生成全为0的shape为(64, 1024)的隐藏状态


#调用后，输出encoder的隐藏状态以及每一步的输出   call函数
encoder = Encoder(input_vocab_size, embedding_units, units, batch_size)   #units对应encoding_units
sample_hidden = encoder.initialize_hidden_state()     #获取初始化的hidden
sample_output, sample_hidden = encoder.call(x, sample_hidden)
# print(sample_output)    #(64, 17, 1024)
print("sample_output.shape:" , sample_output.shape) #(64, 17, 1024)
print("sample_hidden.shape:" , sample_hidden.shape) #(64, 1024)  batch_size, encoding_units


#参照BahdanauAttention的计算过程  区别W1W2和V
class BahdanauAttention(keras.Model):
    def __init__(self, units):   #区别这个units（全连接层的）和call中units（编解码器中的）
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_outputs):
        #decoder_hidden.shape: (batch_size, units)            (64,  1024)
        #encoder_outputs.shape: (batch_size, leng-th, units)   (64, 17, 1024)
        decoder_hidden_with_time_axis = tf.expand_dims(
            decoder_hidden, 1)      #(64,  1024)进行维度拓展，在第一维拓，变成了 (64, 1, 1024)

        #before V: (batch_size, length, units)
        #after V: (batch_size, length, 1)  #score可理解为相似度
        score = self.V(
            tf.nn.tanh(
                self.W1(encoder_outputs) + self.W2(decoder_hidden_with_time_axis)))

        #shape: (batch_size, length, 1)    (64, 17, 1)     #利用softmax将scores转化为概率分布。
        attention_weights = tf.nn.softmax(score, axis=1)   #在第一维(length)上算attention权重，

        #context_vector.shape: (batch_size, length, units)   (64, 17, 1024)
        context_vector = attention_weights * encoder_outputs   #(64, 17, 1024)*(64, 17, 1) tensorflow自己做拓展

        #context_vector.shape: (batch_size, units)   (64, 1024)
        # #decoder的第t时刻的注意力向量，可理解为上下文向量
        context_vector = tf.reduce_sum(context_vector, axis=1)  #在length上求和，即将length个vector加在一起

        return context_vector, attention_weights

attention_model = BahdanauAttention(units=10)
attention_results, attention_weights = attention_model.call(
    sample_hidden, sample_output)

print("attention_results.shape: ", attention_results.shape)    #(64, 1024)
print("attention_weights.shape: ", attention_weights.shape)    #(64, 17, 1)


#############decoder###############
class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_units, decoding_units, batch_size):
        super(Decoder,self).__init__()
        self.batch_size = batch_size
        self.decoding_units = decoding_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_units)
        self.gru = keras.layers.GRU(self.decoding_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer= 'glorot_uniform')
        self.fc = keras.layers.Dense(vocab_size)   #输出成某个词
        self.attention = BahdanauAttention(self.decoding_units)   #每一层都添加注意力，每一步都调用

    def call(self, x, hidden, encoding_outputs):

        #context_vector.shape:(batch_size, units)
        context_vector, attention_weights = self.attention.call(
            hidden, encoding_outputs)

        #before embedding: x.shape: (batch_size, 1)
        #after embedding: x.shape: (batch_size, 1, embedding_units)
        x = self.embedding(x)

        # 此时combined_x的shape：  (64, 1, 1280)
        combined_x = tf.concat(      #拓展(batch_size, units)变为(batch_size,1， units)
            [tf.expand_dims(context_vector, 1), x] , axis = -1)    #设置-1即拼接最后一个维度


        #output.shape :  [batch_size, 1, decoding_units]
        #state.shape:  [batch_size, decoding_units]
        #输出decoder的隐藏状态以及每一步的输出
        output, state = self.gru(combined_x)

        #output.shape:  [batch_size, decoding_units]
        output = tf.reshape(output, (-1, output.shape[2]))   #[batch_size, 1, decoding_units]降为 [batch_size, decoding_units]

        #output.shape:  [batch_size, vocab_size]
        output = self.fc(output)

        return output, state, attention_weights

decoder = Decoder(output_vocab_size, embedding_units, units, batch_size)
outputs = decoder.call(tf.random.uniform((batch_size, 1)),   #x使用随机数据去模拟
                  sample_hidden,
                  sample_output)

decoder_output, decoder_hidden, decoder_aw = outputs
print("decoder_output.shape: ", decoder_output.shape)    #(64, 4744)
print("decoder_hidden.shape: ", decoder_hidden.shape)    #(64, 1024)
print("decoder_attention_weights.shape: ", decoder_aw.shape)   #(64, 17, 1)



###############损失函数##############
optimizer = keras.optimizers.Adam()

loss_object = keras.losses.SparseCategoricalCrossentropy(   #分类问题用SparseCategoricalCrossentropy
    from_logits=True,  reduction='none')  #from_logits: 没有经过softmax或激活函数就设置为true   reduction分布式

def loss_function(real, pred):
    # mask：将向量中0的部分去掉，防止其参与到损失函数的计算中
    #同时equal(real, 0)会将padding部 分变1， 不padding为0，  所以需要用logical_not取反
    mask =  tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)    #计算出具体的损失函数

    mask = tf.cast(mask, dtype=loss_.dtype)   #mask类型类型转换：bool->float
    # mask = tf.where(real > 0, 1.0, 0.0)   #mask类型类型转换：bool->float
    loss_ *= mask

    return tf.reduce_mean(loss_)   #取平均



###################计算多步损失函数进而可以做梯度下降#############
@tf.function
def train_step(inp, targ, encoding_hidden):
    loss = 0  #初始
    with tf.GradientTape() as tape:   #
        encoding_outputs, encoding_hidden = encoder.call(
            inp, encoding_hidden)
        decoding_hidden = encoding_hidden

        #eg. <start> I am here <end>   过程
        # 1.<start> ->I
        # 2.I -> am    其实不仅是I，<start>也参与了预测am，记忆力
        # 3.am -> here
        # 4.here -> <end>
        for t in range(0, targ.shape[1] - 1):
            decoding_input = tf.expand_dims(targ[:, t], 1)    #向量拓展

            predictions, decoding_hidden, _ = decoder.call(
                decoding_input, decoding_hidden, encoding_outputs)
            loss += loss_function(targ[:, t+1], predictions)    #得到多步损失函数  64, 1

    #平均每个batch，以免batch数不同，loss值不同
    batch_loss = loss / int(targ.shape[0])
    variables = encoder.trainable_variables + decoder.trainable_variables  #列表
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss



####################训练模型#################
epochs = 10
steps_per_epoch = len(input_tensor) // batch_size

for epoch in range(epochs):
    start = time.time()

    #初始化隐状态和loss
    encoding_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(
        train_dataset.take(steps_per_epoch)):    #对训练集取
        batch_loss = train_step(inp, targ, encoding_hidden)
        total_loss += batch_loss

        if batch %100 == 0:
            print('Epoch {} Batch {} Loss {:4f}'.format(
                epoch + 1, batch, batch_loss.numpy()))

    print('ALL Epoch {} Loss {:4f}'.format(
        epoch +1, total_loss / steps_per_epoch))

    print('Time take for 1 epoch {} sec\n'.format(time.time() - start))



###################评估############
def evaluate(input_sentence, tokenizer_in, tokenizer_out):
    attention_matrix = np.zeros((max_length_output, max_length_input))
    input_sentence = preprocess_sentence(input_sentence)

    inputs = [tokenizer_in.word_index[token] for token in input_sentence.split(' ')]   #转id
    inputs = keras.preprocessing.sequence.pad_sequences(
        #[inputs]  因为需要是二维的
        [inputs], maxlen = max_length_input, padding = 'post')
    inputs = tf.convert_to_tensor(inputs)   #转化为tensor

    results = ''
    # encoding_hidden = encoder.initialize_hidden_state()
    encoding_hidden = tf.zeros((1, units))

    encoding_outputs, encoding_hidden = encoder.call(inputs, encoding_hidden)
    decoding_hidden = encoding_hidden

    #eg:<start> ->A
    #A --> B --> C --> D

    #因为输入是二维， [1,1]
    decoding_input = tf.expand_dims([tokenizer_out.word_index['<start>']], 0)

    #这个for循环会将上一次循环的输出作为本次循环的输入
    #同时保存好attention_matrix， attention_matrix代表了输入和输出的注意力关系
    for t in range(max_length_output):
        predictions, decoding_hidden, attention_weights = decoder.call(
            decoding_input, decoding_hidden, encoding_outputs)

        # attention_weights.shape:  (batch_size, input_length, 1)   --> (1,17,1)
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_matrix[t] = attention_weights.numpy()

        # predictions.shape: (batch_size, vocab_size)    (1, 4744)
        # 取出最大可能的word id
        predicted_id = tf.argmax(predictions[0]).numpy()

        results += tokenizer_out.index_word[predicted_id] + ' '

        if tokenizer_out.index_word[predicted_id] == '<end>':
            return results, input_sentence, attention_matrix

        decoding_input = tf.expand_dims([predicted_id], 0)
    return results, input_sentence, attention_matrix


def evaluate_diff_language(convert_language_name, input_sentence):
    if convert_language_name == '英语':
        results, input_sentence, attention_matrix = evaluate(input_sentence, input_tokenizer, output_tokenizer)

    elif convert_language_name == '西班牙语':
        results, input_sentence, attention_matrix = evaluate(input_sentence, output_tokenizer, input_tokenizer)

    return results, input_sentence, attention_matrix


#可视化attention_matrix
def plot_attention(attention_matrix, input_sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)    #因为只有一个子图，设置3个1
    ax.matshow(attention_matrix, cmap='viridis')   #用不同颜色进行表示

    font_dict = {'fontsize': 14}
    ax.set_xticklabels([''] + input_sentence, fontdict=font_dict, rotation = 90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=font_dict)
    plt.show()



#集成plot_attention 和evaluate
def translate(convert_language_name, input_sentence):
    results, input_sentence, attention_matrix = evaluate_diff_language(convert_language_name,input_sentence)

    print("Input: %s" % (input_sentence))
    print("Predicted  translation: %s " % (results))

    attention_matrix = attention_matrix[:len(results.split(' ')),
                       :len(input_sentence.split(' '))]
    plot_attention(attention_matrix, input_sentence.split(' '),
                   results.split(' '))



translate('英语','Hace mucho frío aquí.')   #it’s really cold here
translate('英语','¿Sigues en casa?')   #are you still at home?
translate('西班牙语', u'it s very cold here.')   #la me conoces de dijo . <end>




