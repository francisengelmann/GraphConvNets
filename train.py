import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import utils
import model

# Returns adjacency matrix, degree matrix, labels
adj, degree_matrix, labels_gt = utils.load_data_karate()
labels_gt = np.squeeze([x-1 for x in labels_gt])
print(adj)  # NxN either 0 or 1, Adj. matrix
print(degree_matrix)  # NxN degree matrix: diagonal matrix, entries correspond to number of adj. edges
print(labels_gt)   # 1xN Ground truth labels
# labels_gt[10] = 2
# labels_gt[20] = 3


with tf.Graph().as_default():
    with tf.device('/gpu:' + str(0)):
        num_points = np.shape(adj)[0]  # N, Number of nodes in the graph, points in the point cloud

        inputs = tf.placeholder(tf.float32, shape=(num_points, num_points))  # NxD, D - size of input feature vector
        degM = tf.placeholder(tf.float32, shape=(num_points, num_points))  # NxX, degree matrix
        # labels = tf.placeholder(tf.int32, shape=(num_points))  # 1xN Ground truth labels

        pred = model.gcn(inputs, degM)  # NxF, F dimensionality of output feature vector

        train_mask = np.zeros(num_points)
        train_mask[0] = 1
        train_mask[-1] = 1
        # train_mask[10] = 1
        # train_mask[20] = 1


        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels_gt)
        mask = tf.cast(train_mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        opt_op = optimizer.minimize(loss)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        correct = 0
        num_epochs = 300
        init = tf.global_variables_initializer()
        sess.run(init)

        G = nx.karate_club_graph()


        for i in range(num_epochs):
            _, a, loss_val = sess.run([opt_op, pred, loss], feed_dict={inputs: adj, degM: degree_matrix})
            b = np.argmax(a, 1)  # Nx1
            #print(b)
            correct = np.sum(labels_gt == b)/num_points
            print('Correct:', correct)
            print('Loss:', loss_val)
            #print()
            #correct_total += correct
            print("Correct: ", correct)
            print('a.shape', a.shape)
            pred_labels_dict = {i : b[i] for i in range(len(b)) }
            nodes_pos_dict = {i : a[i][0:2] for i in range(len(b))}
            node_shapes = ['s'] * num_points
            node_shapes[0] = 'o'
            node_shapes[-1] = 'o'

            if False:
                plt.clf()
                plt.axis('on')
                nx.draw(G, cmap=plt.get_cmap('jet'), node_color=labels_gt, labels=pred_labels_dict, font_color='white',
                        node_size=(train_mask*300)+300, pos=nodes_pos_dict)

                plt.axis('on')
                img_name = str(i).zfill(4)+'.png'
                plt.savefig('data/'+img_name)
        plt.axis('on')
        nx.draw(G, cmap=plt.get_cmap('jet'), node_color=labels_gt, labels=pred_labels_dict, font_color='white',
                node_size=(train_mask * 300) + 300, pos=nodes_pos_dict)
        plt.axis('on')
        plt.show()