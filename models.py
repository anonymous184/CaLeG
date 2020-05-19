import tensorflow as tf

import ops
from datahandler import datashapes


def add_noise(x):
    shape = tf.shape(x)
    return x + tf.truncated_normal(shape, 0.0, 0.01)


def encoder(opts, inputs, reuse=tf.AUTO_REUSE, is_training=False):
    if opts['e_noise'] == 'add_noise':
        inputs = add_noise(inputs)

    with tf.variable_scope("encoder", reuse=reuse):
        res = dcgan_encoder(opts, inputs, is_training, reuse)
        return res


def decoder(opts, noise, reuse=tf.AUTO_REUSE, is_training=True):
    with tf.variable_scope("generator", reuse=reuse):
        proba = ops.linear(opts, noise, opts['n_classes'], 'classifier')
        res = dcgan_decoder(opts, noise, is_training, reuse)
        return res, proba


def discriminator(opts, inputs, reuse=tf.AUTO_REUSE, is_training=False):
    if opts['e_noise'] == 'add_noise':
        inputs = add_noise(inputs)

    with tf.variable_scope("discriminator", reuse=reuse):
        res = dcgan_encoder(opts, inputs, is_training, reuse)
        res_logits = ops.linear(opts, res, 10, scope='softmax')
        res_logits_w = ops.linear(opts, res, 1, scope='w_distance')
        return res_logits, res_logits_w


def dcgan_encoder(opts, inputs, is_training=False, reuse=False):
    num_units = opts['e_num_filters']
    num_layers = opts['e_num_layers']
    layer_x = inputs
    for i in range(num_layers):
        scale = 2 ** (num_layers - i - 1)
        layer_x = ops.conv2d(opts, layer_x, num_units / scale,
                             scope='h%d_conv' % i)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x, is_training,
                                     reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    res = ops.linear(opts, layer_x, opts['zdim'], scope='hfinal_lin')
    return res


def dcgan_decoder(opts, noise, is_training=False, reuse=False):
    assert opts['g_arch'] in ('dcgan', 'dcgan_mod')
    output_shape = datashapes[opts['dataset']]
    num_units = opts['g_num_filters']
    batch_size = tf.shape(noise)[0]
    num_layers = opts['g_num_layers']
    if opts['g_arch'] == 'dcgan':
        height = output_shape[0] // 2 ** num_layers
        width = output_shape[1] / 2 ** num_layers
    else:
        height = output_shape[0] // 2 ** (num_layers - 1)
        width = output_shape[1] // 2 ** (num_layers - 1)

    h0 = ops.linear(
        opts, noise, num_units * height * width, scope='h0_lin')
    h0 = tf.reshape(h0, [-1, height, width, num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2 ** (i + 1)
        _out_shape = [batch_size, height * scale,
                      width * scale, num_units // scale]
        layer_x = ops.deconv2d(opts, layer_x, _out_shape,
                               scope='h%d_deconv' % i)
        if opts['batch_norm']:
            layer_x = ops.batch_norm(opts, layer_x,
                                     is_training, reuse, scope='h%d_bn' % i)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)
    if opts['g_arch'] == 'dcgan':
        last_h = ops.deconv2d(
            opts, layer_x, _out_shape, scope='hfinal_deconv')
    else:
        last_h = ops.deconv2d(
            opts, layer_x, _out_shape, d_h=1, d_w=1, scope='hfinal_deconv')

    return tf.nn.sigmoid(last_h)
