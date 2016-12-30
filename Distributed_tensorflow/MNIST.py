import tensorflow as tf
import rztutil
import time

# cluster specification
parameter_servers = ["192.168.51.24:8080"]
workers = ["192.168.51.24:8081",
           "192.168.33.10:8080"]
cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

train_data, train_label, test_data, test_label = rztutil.read_csv('mnist.csv', split_ratio=80, delimiter=";",
                                                                  output_label=True, label_vector=True)


# config
learning_rate = 0.01
training_epochs = 20
logs_path = "graph"
display_step = 1
batch_size = 128
saver_session = 1
examples_to_show = 10

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    if FLAGS.task_index == 0:
        train_data = train_data[:400]
    if FLAGS.task_index == 1:
        train_data = train_data[400:799]
    print(len(train_data))
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        # input images
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            y = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            weights = {
                'weight1': tf.Variable(tf.random_normal([784, 512])),
                'weight2': tf.Variable(tf.random_normal([512, 256])),
                'weight3': tf.Variable(tf.random_normal([256, 128])),
                'weight4': tf.Variable(tf.random_normal([128, 64])),
                'weight5': tf.Variable(tf.random_normal([64, 32])),
                'weight6': tf.Variable(tf.random_normal([32, 10]))
            }

        # bias
        with tf.name_scope("biases"):
            bias = {
                'bias1': tf.Variable(tf.random_normal([512])),
                'bias2': tf.Variable(tf.random_normal([256])),
                'bias3': tf.Variable(tf.random_normal([128])),
                'bias4': tf.Variable(tf.random_normal([64])),
                'bias5': tf.Variable(tf.random_normal([32])),
                'bias6': tf.Variable(tf.random_normal([10])),
            }

        # implement model
        with tf.name_scope("model"):
            def model(x):
                layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['weight1']), bias['bias1']))
                layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['weight2']), bias['bias2']))
                layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, weights['weight3']), bias['bias3']))
                layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, weights['weight4']), bias['bias4']))
                layer5 = tf.nn.relu(tf.add(tf.matmul(layer4, weights['weight5']), bias['bias5']))
                layer6 = tf.add(tf.matmul(layer5, weights['weight6']), bias['bias6'])
                return layer6

        with tf.name_scope("pred"):
            pred = model(x)

        # specify cost function
        with tf.name_scope('loss'):
            # this is our cost
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.mul(100.0, tf.reduce_mean(tf.cast(correct_pred, tf.float32)))

        # specify optimizer
        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", loss)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             global_step=global_step,
                             init_op=init_op)
    print("Before session")
    with sv.managed_session(server.target) as sess:
        start = time.time()
        print("Session started at ", start, "time")
        saver.restore(sess, 'save_session/mnist.model')
        while not sv.should_stop():
            print("After session")
            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            for i in range(training_epochs):
                if i % display_step == 0:
                    print("Epoch :", i)
                step, batch = 1, 0
                while step < len(train_data) / batch_size:
                    batch_x, batch_y = train_data[batch: batch + batch_size], train_label[batch:batch + batch_size]
                    batch += batch_size
                    sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
                    if i % display_step == 0:
                        cost, acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
                        print("Batch " + str(step * batch_size) + ", Minibatch Loss= " + \
                              "{:.6f}".format(cost) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
                    step += 1
                    if i % saver_session == 0:
                        saver.save(sess, 'save_session/mnist.model')
            sv.stop()
        print("Final Cost: %.4f" % cost)
        result, test_cost, test_acc = sess.run([pred, loss, accuracy], feed_dict={x: test_data, y: test_label})
        print("--------------------------Testing---------------------------")
        print("Test Cost :", test_cost)
        print("Test Accuracy :", test_acc)
        print("Session ends at ", time.time(), "time")
        print("Total time takes for execution:", time.time()-start)
    print("done")
