from keras_bert import Tokenizer

# #字典
# token_dict = {
#     '[CLS]': 0,
#     '[SEP]': 1,
#     'un': 2,
#     '##aff': 3,
#     '##able': 4,
#     '[UNK]': 5,
# }
#
# # 拆分单词实例
# tokenizer = Tokenizer(token_dict)
# print(tokenizer.tokenize('unaffable'))  # ['[CLS]', 'un', '##aff', '##able', '[SEP]']
#
# # indices是字对应索引
# # segments表示索引对应位置上的字属于第一句话还是第二句话
# # 这里只有一句话 unaffable，所以segments都是0
# indices, segments = tokenizer.encode('unaffable')
# print(indices)  # [0, 2, 3, 4, 1]
# print(segments)  # [0, 0, 0, 0, 0]
#



import codecs
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
import numpy as np

# 预训练好的模型 bert base
config_path = r'C:\Users\独为我唱\Desktop\bert文本分类测试\bert\bert_config.json'  # 加载配置文件， 模型参数
checkpoint_path = r'C:\Users\独为我唱\Desktop\bert文本分类测试\bert\bert_model.ckpt'   #模型权重
vocab_path = r'C:\Users\独为我唱\Desktop\bert文本分类测试\bert\vocab.txt'     #词表


# 构建字典
# 也可以用 keras_bert 中的 load_vocabulary() 函数
# 传入 vocab_path 即可
# from keras_bert import load_vocabulary
# token_dict = load_vocabulary(vocab_path)

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# 加载预训练模型
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)



tokenizer = Tokenizer(token_dict)
text = '语言模型'
tokens = tokenizer.tokenize(text)
# ['[CLS]', '语', '言', '模', '型', '[SEP]']

indices, segments = tokenizer.encode(first=text, max_len=512)
print(indices[:10])
# [101, 6427, 6241, 3563, 1798, 102, 0, 0, 0, 0]
print(segments[:10])
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])

