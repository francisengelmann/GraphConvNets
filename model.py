import tensorflow as tf

def gcn_layer(H, adj_norm, output_dim, scope):

    with tf.variable_scope(scope):
        Dsize = H.get_shape()[1].value
        W = tf.Variable(tf.random_uniform((Dsize, output_dim)))
        # W = tf.Print(W, [W], "W")
        Z = tf.matmul(tf.matmul(adj_norm, H), W)
        H_out = tf.nn.relu(Z)
        #H_out = tf.nn.sigmoid(Z)

    return H_out

def gcn(adj, deg):

    # id = tf.eye(34)
    # W = tf.Variable(tf.random_uniform((34, 2)), trainable=True)
    # y = tf.matmul(id, W)
    # return y

    num_nodes = adj.get_shape()[0].value  # number of nodes in the graph, number of points
    num_classes = 4  # number of ground truth classes

    # inputs X , H(0)
    id = tf.eye(num_nodes)
    H_0 = id  # initial hidden state

    # Normalize adjacency matrix
    A = adj + id  # A' = A + I
    deg = deg + id  # D = D + I = deg_matrix(A+I)
    # D' = D^(-1/2)
    D = tf.sqrt(deg)
    D = tf.reciprocal(D)
    D = tf.where(tf.is_inf(D), tf.zeros_like(D), D)
    adj_norm = tf.matmul(D, tf.matmul(A, D))  # normalization of A

    H_1 = gcn_layer(H_0, adj_norm, num_nodes, "L1")
    H_2 = gcn_layer(H_1, adj_norm, num_nodes//2, "L2")
    H_3 = gcn_layer(H_2, adj_norm, num_classes, "L3")
    y = H_3

    return H_2, y

def gcn_prev(adj, deg):
    num_nodes = adj.get_shape()[0].value  # number of nodes in the graph, number of points
    num_classes = 2  # number of ground truth classes
    print(num_nodes)

    # inputs X , H(0)
    id = tf.eye(num_nodes)
    H = id  # initial hidden state
    Dsize = H.get_shape()[1].value
    W = tf.Variable(tf.random_uniform((Dsize, num_classes)))

    # A' = A + I
    A1 = adj + id
    # D = D + I = deg_matrix(A+I)
    deg = deg + id
    # D' = D^(-1/2)
    D1 = tf.sqrt(deg)
    D2 = tf.reciprocal(D1)
    D2 = tf.where(tf.is_inf(D2), tf.zeros_like(D2), D2)

    # f(H,A) = relu(D'^(-1/2)*A'*D'^(-1/2) * H * W)
    A_norm = tf.matmul(D2, tf.matmul(A1, D2))  # normalization of A
    Z = tf.matmul(tf.matmul(A_norm, H), W)
    y = tf.nn.relu(Z)
    return y