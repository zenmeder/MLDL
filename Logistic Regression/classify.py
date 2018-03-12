import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
data = pd.read_csv("./data/datatraining.txt")
# print(data[:10])

X_train, X_test, y_train, y_test = train_test_split(data[['Temperature','Humidity','Light','CO2','HumidityRatio']].values, data['Occupancy'].values.reshape(-1,1), random_state=26)
# print(X_train.shape,y_train.shape)
# print()
# y_train = [_[0] for _ in y_train]
# print(y_train[:10])
# print(y_train.shape)
y_train = OneHotEncoder().fit_transform(y_train).toarray()
y_test = OneHotEncoder().fit_transform(y_test).toarray()
# print(y_train.shape)
# print(y_train.reshape(-1,)[:10])
learning_rate = 0.001
training_epochs = 50
batch_size = 100
display_step = 1
# print(X_train.shape)
n_samples = X_train.shape[0]
n_features = X_train.shape[1]
n_classes = 2

x = tf.placeholder(tf.float32,[None, n_features])
y = tf.placeholder(tf.int32, [None, n_classes])
W = tf.Variable(tf.zeros([n_features, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))
pred = tf.matmul(x, W)+b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# cost = tf.nn.sigmoid_cross_entropy_with_logits(pred, y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_predition = tf.equal(tf.arg_max(pred, 1),tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
init = tf.initialize_all_variables()
# print(y_train[:10])
# print(tf.one_hot(y_train,1).eval())
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = n_samples//batch_size
        for i in range(total_batch):
            _, c = sess.run([optimizer, cost],feed_dict={x:X_train[i*batch_size:(i+1)*batch_size], y:y_train[i*batch_size:(i+1)*batch_size]})
            avg_cost = c/total_batch
        plt.plot(epoch+1, avg_cost,"co")
        if (epoch+1) % display_step == 0:
                print("Epoch ", epoch+1, " cost= ",avg_cost)
    print('finished')
    print("Testing Accuracy:", accuracy.eval({x: X_test, y:y_test}))
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()
    # sess.run(print(tf.one_hot(indices, depth)))