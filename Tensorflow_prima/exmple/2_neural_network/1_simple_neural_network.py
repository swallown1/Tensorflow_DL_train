#引入minst字体
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/data',one_hot=True)

import tensorflow as tf

#定义参数
learning_rate = 0.1
batch_size = 128
num_steps = 500
display_steps = 100

#model的参数
num_inputs = 784
num_classes = 10
hidden_1 = 256
hidden_2 = 256


X = tf.placeholder('float',[None,num_inputs])
Y = tf.placeholder("float",[None,num_classes])

#设置权重
weight = {
    'w1':tf.Variable(tf.random_normal([num_inputs,hidden_1])),
    'w2':tf.Variable(tf.random_normal([hidden_1,hidden_2])),
    'wo':tf.Variable(tf.random_normal([hidden_2,num_classes]))
}
biases = {
    "b1":tf.Variable(tf.random_normal([hidden_1])),
    "b2":tf.Variable(tf.random_normal([hidden_2])),
    "bo":tf.Variable(tf.random_normal([num_classes])),
}

#定义模型
def Neural_Network(x):
    layer1 = tf.add(tf.matmul(x,weight['w1']),biases['b1'])
    layer2 = tf.add(tf.matmul(layer1,weight['w2']),biases['b2'])
    output = tf.matmul(layer2,weight['wo'])+biases['bo']
    return output

#构造模型
logits = Neural_Network(X)
prediction = tf.nn.softmax(logits)

#定义损失 和优化
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#模型评估
correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#初始化所有参数
init = tf.global_variables_initializer()

with tf.Session() as sess:
    #初始化参数
    sess.run(init)

    for step in range(1,num_steps+1):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        #初始化训练运算
        sess.run(train_op,feed_dict={X:batch_x,Y:batch_y})
        if step % display_steps ==0 or step == 1:
            #计算batch 的损失
            loss , acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("优化结束")

#在test上测试数据集
    print("在测试数据集上的正确率:",
          sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
