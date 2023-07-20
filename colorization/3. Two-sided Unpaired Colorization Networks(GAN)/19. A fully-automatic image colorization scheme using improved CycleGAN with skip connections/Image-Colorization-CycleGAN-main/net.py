import numpy as np
import tensorflow as tf
import math
 
#构造可训练参数
def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)

#定义卷积层
def conv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output
 
#定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", name = "atrous_conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output
 
#定义反卷积层
def deconv2d(input_, output_dim, kernel_size, stride, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim], [1, 2, 2, 1], padding = "SAME")
        return output
"""
#定义instance_norm
def instance_norm(input_, name="instance_norm"):
    with tf.variable_scope(name):
        epsilon = 1e-9

        mean, var = tf.nn.moments(input_, [1, 2], keep_dims=True)

        return tf.div(tf.subtract(input_, mean), tf.sqrt(tf.add(var, epsilon)))
#定义Groupnorm
def GroupNorm(x, gamma, beta, G=16):
    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5
    x = np.reshape(x, (x.shape[0], G, x.shape[1] / 16, x.shape[2], x.shape[3]))

    x_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(x, axis=(2, 3, 4), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
    """
#定义batchnorm(批次归一化)层
def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_-mean)*inv
        output = scale*normalized + offset
        return output
 
#定义最大池化层
def max_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

#定义lrelu激活层
def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)
 
#定义relu激活层
def relu(input_, name = "relu"):
    return tf.nn.relu(input_, name = name)

#定义残差块
def residule_block_33(input_, output_dim, kernel_size = 3, stride = 1, dilation = 2, atrous = False, name = "res"):
    if atrous:
        conv2dc0 = atrous_conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = atrous_conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))

    else:
        conv2dc0 = conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c0'))
        conv2dc0_norm = batch_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c1'))
        conv2dc1_norm = batch_norm(input_ = conv2dc1, name = (name + '_bn1'))

    add_raw = input_ + conv2dc1_norm
    output = relu(input_ = add_raw)
    return output



#定义生成器
def generator(image, gf_dim=64, reuse=False, name="generator"): 
    #生成器输入尺度: 1*256*256*3

    input_dim = int(image.get_shape()[-1])  # 获取输入通道
    dropout_rate = 0.7  # 定义dropout的比例
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False


        # 第一个卷积层，输出尺度[1, 128, 128, 64]
        e1 = batch_norm(conv2d(input_=image, output_dim=gf_dim, kernel_size=4, stride=2, name='g_e1_conv'),name='g_bn_e1')

        # 第二个卷积层，输出尺度[1, 64, 64, 128]
        e2 = batch_norm(conv2d(input_=lrelu(e1), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_e2_conv'),name='g_bn_e2')

        # 第三个卷积层，输出尺度[1, 32, 32, 256]
        e3 = batch_norm(conv2d(input_=lrelu(e2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_e3_conv'),name='g_bn_e3')

        # 第四个卷积层，输出尺度[1, 16, 16, 512]
        e4 = batch_norm(conv2d(input_=lrelu(e3), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_e4_conv'),name='g_bn_e4')

        # 第五个卷积层，输出尺度[1, 8, 8, 512]
        e5 = batch_norm(conv2d(input_=lrelu(e4), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_e5_conv'),name='g_bn_e5')

        # 第六个卷积层，输出尺度[1, 4, 4, 512]
        e6 = batch_norm(conv2d(input_=lrelu(e5), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_e6_conv'),name='g_bn_e6')

        # 第七个卷积层，输出尺度[1, 2, 2, 512]
        e7 = batch_norm(conv2d(input_=lrelu(e6), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_e7_conv'),name='g_bn_e7')

        # 第八个卷积层，输出尺度[1, 1, 1, 512]
        e8 = batch_norm(conv2d(input_=lrelu(e7), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_e8_conv'),name='g_bn_e8')




        r1 = residule_block_33(input_=e8, output_dim=gf_dim * 8, atrous=True, name='g_r1')
        r2 = residule_block_33(input_=r1, output_dim=gf_dim * 8, atrous=True, name='g_r2')
        r3 = residule_block_33(input_=r2, output_dim=gf_dim * 8, atrous=True, name='g_r3')
        #r4 = residule_block_33(input_=r3, output_dim=gf_dim * 8, atrous=True, name='g_r4')
        #r5 = residule_block_33(input_=r4, output_dim=gf_dim * 8, atrous=True, name='g_r5')
        #r6 = residule_block_33(input_=r5, output_dim=gf_dim * 8, atrous=True, name='g_r6')
        #r7 = residule_block_33(input_ = r6, output_dim = gf_dim*8, atrous = True, name='g_r7')
        #r8 = residule_block_33(input_ = r7, output_dim = gf_dim*8, atrous = True, name='g_r8')
        #r9 = residule_block_33(input_ = r8, output_dim = gf_dim*8, atrous = True, name='g_r9')


        # 第一个反卷积层，输出尺度[1, 2, 2, 512]
        d1 = deconv2d(input_=tf.nn.relu(r3), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)  # 随机扔掉一般的输出
        d1 = tf.concat([batch_norm(d1, name='g_bn_d1'), e7], 3)

        # 第二个反卷积层，输出尺度[1, 4, 4, 512]
        d2 = deconv2d(input_=tf.nn.relu(d1), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)  # 随机扔掉一般的输出
        d2 = tf.concat([batch_norm(d2, name='g_bn_d2'), e6], 3)

        # 第三个反卷积层，输出尺度[1, 8, 8, 512]
        d3 = deconv2d(input_=tf.nn.relu(d2), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)  # 随机扔掉一般的输出
        d3 = tf.concat([batch_norm(d3, name='g_bn_d3'), e5], 3)

        # 第四个反卷积层，输出尺度[1, 16, 16, 512]
        d4 = deconv2d(input_=tf.nn.relu(d3), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_d4')
        d4 = tf.concat([batch_norm(d4, name='g_bn_d4'), e4], 3)

        # 第五个反卷积层，输出尺度[1, 32, 32, 256]
        d5 = deconv2d(input_=tf.nn.relu(d4), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_d5')
        d5 = tf.concat([batch_norm(d5, name='g_bn_d5'), e3], 3)

        # 第六个反卷积层，输出尺度[1, 64, 64, 128]
        d6 = deconv2d(input_=tf.nn.relu(d5), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_d6')
        d6 = tf.concat([batch_norm(d6, name='g_bn_d6'), e2], 3)

        # 第七个反卷积层，输出尺度[1, 128, 128, 64]
        d7 = deconv2d(input_=tf.nn.relu(d6), output_dim=gf_dim, kernel_size=4, stride=2, name='g_d7')
        d7 = tf.concat([batch_norm(d7, name='g_bn_d7'), e1], 3)

        # 第八个反卷积层，输出尺度[1, 256, 256, 3]
        d8 = deconv2d(input_=tf.nn.relu(d7), output_dim=input_dim, kernel_size=4, stride=2, name='g_d8')

        d8 = tf.nn.tanh(d8)

        return d8

#定义判别器
def discriminator(image,targets, df_dim=64, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        dis_input = tf.concat([image, targets], 3)
        h0 = lrelu(conv2d(input_ = dis_input, output_dim = df_dim, kernel_size = 4, stride = 2, name='d_h0_conv'))
        h1 = lrelu(batch_norm(conv2d(input_ = h0, output_dim = df_dim*2, kernel_size = 4, stride = 2, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(batch_norm(conv2d(input_ = h1, output_dim = df_dim*4, kernel_size = 4, stride = 2, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(batch_norm(conv2d(input_ = h2, output_dim = df_dim*8, kernel_size = 4, stride = 1, name='d_h3_conv'), 'd_bn3'))
        output = conv2d(input_ = h3, output_dim = 1, kernel_size = 4, stride = 1, name='d_h4_conv')
        output = tf.sigmoid(output)  # 在输出之前经过sigmoid层，因为需要进行log运算
        return output