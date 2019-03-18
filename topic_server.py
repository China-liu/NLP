import flask
import json
import re
import urllib.parse
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import tensorflow.contrib.keras as kr
from flask import request
from pymongo import MongoClient
import tensorflow.contrib.keras as kr
from sklearn.preprocessing import LabelEncoder


# 创建一个服务，把当前这个python文件当做一个服务
server = flask.Flask(__name__)

# server.config['JSON_AS_ASCII'] = False
# @server.route()可以将普通函数转变为服务 登录接口的路径、请求方式
@server.route('/topic', methods=['get', 'post'])
def topic():
    # 获取通过url请求传参的数据
    blogid = request.values.get('blogid')

    # 根据获取的blogid，获取mongo中的数据
    mongo_client = MongoClient("mongodb://xhql:" + urllib.parse.quote_plus(
        "xhql_190228_snv738J72*fjVNv8220aiVK9V820@_") + "@47.92.174.37:20388/webpage")
    mongo_db = mongo_client["webpage"]
    mongo_col = mongo_db["web_detail"]
    mongo_data = mongo_col.find_one({"id": "%(blogid)s" % {"blogid": blogid}})
    content = mongo_data.get("content_np")
    mongo_client.close()

    if content:
        # 加载本地字典，用于将文本的字转换成相应汉字对应字典的索引
        with open('./cnews/cnews.vocab.txt', encoding='utf8') as file:
            vocabulary_list = [k.strip() for k in file.readlines()]
        # 生成字典{汉字：索引}
        word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
        # 将汉字转化为索引的方法
        content2idList = lambda content: [word2id_dict[word] for word in content if word in word2id_dict]
        vocab_size = 5000  # 词汇表大小
        seq_length = 1000  # 序列长度
        embedding_dim = 64  # 词向量维度
        num_classes = 10  # 类别数
        num_filters = 256  # 卷积核数目
        kernel_size = 5  # 卷积核尺寸
        hidden_dim = 128  # 全连接层神经元
        dropout_keep_prob = 0.5  # dropout保留比例
        learning_rate = 1e-3  # 学习率
        batch_size = 64  # 每批训练大小
        # # 标准化标签，将标签值统一转换成range(标签值个数-1)范围内
        labelEncoder = LabelEncoder()
        # 标签列表
        train_label_set_list = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
        train_label_list = []
        # 将标签列表转换成模型需要的形式
        for i in range(len(train_label_set_list)):
            train_label_list += train_label_set_list[i:i + 1] * 5000
        # 调用LabelEncoder对象的fit_transform方法做标签编码
        labelEncoder.fit_transform(train_label_list)
        # 重置tensorflow图，加强代码的健壮性
        tf.reset_default_graph()
        # 将每次训练的特征矩阵X和预测目标值Y赋值给变量X_holder和Y_holder
        X_holder = tf.placeholder(tf.int32, [None, seq_length])
        Y_holder = tf.placeholder(tf.float32, [None, num_classes])
        # 调用tf库的get_variable方法实例化可以更新的模型参数embedding，矩阵形状为vocab_size*embedding_dim，即5000*64
        embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
        # 调用tf.nn库的embedding_lookup方法将输入数据做词嵌入，得到新变量embedding_inputs的形状为batch_size*seq_length*embedding_dim，即64*600*64
        embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)
        # 赋值给变量conv，形状为batch_size*596*num_filters，596是600-5+1
        conv = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size)
        # 调用tf.reduce_max方法对变量conv的第1个维度做求最大值操作。方法结果赋值给变量max_pooling，形状为batch_size*256
        max_pooling = tf.reduce_max(conv, reduction_indices=[1])
        # 添加全连接层
        full_connect = tf.layers.dense(max_pooling, hidden_dim)
        # 调用tf.contrib.layers.dropout方法，方法需要2个参数，第1个参数是输入数据，第2个参数是保留比例；
        full_connect_dropout = tf.contrib.layers.dropout(full_connect, keep_prob=0.75)
        # 调用tf.nn.relu方法，即激活函数；
        full_connect_activate = tf.nn.relu(full_connect_dropout)
        # 添加全连接层，tf.layers.dense方法结果赋值给变量softmax_before，形状为batch_size*num_classes
        softmax_before = tf.layers.dense(full_connect_activate, num_classes)
        # 调用tf.nn.softmax方法，方法结果是预测概率值
        predict_Y = tf.nn.softmax(softmax_before)
        # 调用tf.global_variables_initializer实例化tensorflow中的Operation对象
        init = tf.global_variables_initializer()
        # 调用tf.Session方法实例化会话对象
        session = tf.Session()
        # 调用tf.Session对象的run方法做变量初始化
        session.run(init)
        # 调用tf.train.Saver()方法来加载模型
        server = tf.train.Saver()
        # 关闭session
        session.close()

        # 调用tf.Session对象的run方法做变量初始化
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        # 加载模型
        server.restore(session, "./ckpt/topic_extract.ckpt")

        # 对传入的文本，利用加载的模型进行文本分类
        def predictAll(test_X, batch_size=100):
            predict_value_list = []
            for i in range(0, len(test_X), batch_size):
                selected_X = test_X[i: i + batch_size]
                predict_value = session.run(predict_Y, {X_holder: selected_X})
                predict_value_list.extend(predict_value)
            return np.array(predict_value_list)

        # 将传入的文本转换成模型需要的形式，列表
        news_text_list = [content, ]
        # 将文本转换成文本中汉字对应的索引的形式
        test_idlist_list = [content2idList(content) for content in news_text_list]
        test_X = kr.preprocessing.sequence.pad_sequences(test_idlist_list, seq_length)
        # 调用predictAll进行文本分类
        Y = predictAll(test_X)
        y = np.argmax(Y, axis=1)
        # 文本分类结果
        predict_label_list = labelEncoder.inverse_transform(y)

        if predict_label_list:
            topic = predict_label_list[0]
            resu = {"id": str(blogid), "content": topic, "source": "NLP"}
        else:
            resu = str({"id": blogid, "content": "", "source": "NLP"})

        return json.dumps(resu, ensure_ascii=False)  # 将字典转换为json串, json是字符串

    else:
        resu = {"id": str(blogid), "content": "", "source": "NLP"}
        return json.dumps(resu, ensure_ascii=False)  # 将字典转换为json串, json是字符串



if __name__ == '__main__':
    # host = '172.26.26.131', port = '8992'
    server.run(threaded=True, debug=True, port=8889, host='0.0.0.0')  # 指定端口、host,0.0.0.0代表不管几个网卡，任何ip都可以访问