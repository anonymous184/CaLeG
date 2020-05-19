"""
CaLeG without augmentation version
"""
import logging
import math
import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils
from datahandler import datashapes
from models import encoder, decoder, discriminator
from result_logger import ResultLogger


class CaLeG(object):

    def __init__(self, opts, tag):
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.opts = opts
        self.tag = tag
        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        shape = datashapes[opts['dataset']]

        # Placeholders
        self.sample_points = tf.placeholder(tf.float32, [None] + shape, name='real_points_ph')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='label_ph')
        self.sample_noise = tf.placeholder(tf.float32, [None] + [opts['zdim']], name='noise_ph')
        self.fixed_sample_labels = tf.placeholder(tf.int32, shape=[None], name='fixed_sample_label_ph')
        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')

        # Ops
        self.encoded = encoder(opts, inputs=self.sample_points, is_training=self.is_training)
        self.reconstructed, self.probs1 = decoder(opts, noise=self.encoded, is_training=self.is_training)
        self.prob1_softmaxed = tf.nn.softmax(self.probs1, axis=-1)
        self.correct_sum = tf.reduce_sum(
            tf.cast(tf.equal(tf.argmax(self.prob1_softmaxed, axis=1, output_type=tf.int32), self.labels), tf.float32))

        self.De_pro_tilde_logits, self.De_pro_tilde_wdistance = self.discriminate(self.reconstructed)
        self.D_pro_logits, self.D_pro_logits_wdistance = self.discriminate(self.sample_points)

        # Objectives, losses, penalties
        self.loss_cls = self.cls_loss(self.labels, self.probs1)
        self.penalty = self.mmd_penalty(self.encoded)
        self.loss_reconstruct = self.reconstruction_loss(self.opts, self.sample_points, self.reconstructed)
        self.wgan_d_loss = tf.reduce_mean(self.De_pro_tilde_wdistance) - tf.reduce_mean(self.D_pro_logits_wdistance)
        self.wgan_g_loss = -(tf.reduce_mean(self.De_pro_tilde_wdistance))
        self.wgan_d_penalty1 = self.gradient_penalty(self.sample_points, self.reconstructed)
        self.wgan_d_penalty = self.wgan_d_penalty1
        #  G_additional loss
        self.G_tilde_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                           logits=self.De_pro_tilde_logits))
        #  D loss
        self.D_real_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                           logits=self.D_pro_logits))
        self.D_tilde_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.De_pro_tilde_logits))

        self.encoder_objective = self.loss_reconstruct + opts['lambda'] * self.penalty + self.loss_cls
        self.decoder_objective = self.loss_reconstruct + self.G_tilde_loss + self.wgan_g_loss
        self.disc_objective = self.D_real_loss + self.D_tilde_loss + self.wgan_d_loss + self.wgan_d_penalty

        self.total_loss = self.loss_reconstruct + opts['lambda'] * self.penalty + self.loss_cls
        self.loss_pretrain = self.pretrain_loss() if opts['e_pretrain'] else None

        # Optimizers, savers, etc
        opts = self.opts
        lr = opts['lr']
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        optim = self.optimizer(lr, self.lr_decay)
        self.encoder_opt = optim.minimize(loss=self.encoder_objective, var_list=encoder_vars)
        self.decoder_opt = optim.minimize(loss=self.decoder_objective, var_list=decoder_vars)
        self.disc_opt = optim.minimize(loss=self.disc_objective, var_list=discriminator_vars)
        self.ae_opt = optim.minimize(loss=self.total_loss, var_list=encoder_vars + decoder_vars)
        self.pretrain_opt = self.optimizer(lr).minimize(loss=self.loss_pretrain,
                                                        var_list=encoder_vars) if opts['e_pretrain'] else None

        self.saver = tf.train.Saver(max_to_keep=10)
        tf.add_to_collection('real_points_ph', self.sample_points)
        tf.add_to_collection('noise_ph', self.sample_noise)
        tf.add_to_collection('is_training_ph', self.is_training)
        tf.add_to_collection('encoder', self.encoded)

        self.init = tf.global_variables_initializer()
        self.result_logger = ResultLogger(tag, opts['work_dir'], verbose=True)

    def discriminate(self, image):
        res_logits, res_wdistance = discriminator(self.opts, inputs=image, is_training=self.is_training)
        return res_logits, res_wdistance

    def pretrain_loss(self):
        mean_pz = tf.reduce_mean(self.sample_noise, axis=0, keepdims=True)
        mean_qz = tf.reduce_mean(self.encoded, axis=0, keepdims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_pz - mean_qz))
        cov_pz = tf.matmul(self.sample_noise - mean_pz,
                           self.sample_noise - mean_pz, transpose_a=True)
        cov_pz /= tf.cast(tf.shape(self.sample_noise)[0], tf.float32) - 1.
        cov_qz = tf.matmul(self.encoded - mean_qz,
                           self.encoded - mean_qz, transpose_a=True)
        cov_qz /= tf.cast(tf.shape(self.encoded)[0], tf.float32) - 1.
        cov_loss = tf.reduce_mean(tf.square(cov_pz - cov_qz))
        return mean_loss + cov_loss

    def cls_loss(self, labels, logits):
        return tf.reduce_mean(tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)))

    def mmd_penalty(self, sample_qz, mean=0., std=1.):
        opts = self.opts
        sample_pz = tf.random_normal(tf.shape(sample_qz), mean, std)
        sigma2_p = 1.
        n = utils.get_batch_size(sample_qz)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keepdims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keepdims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods
        Cbase = 2. * opts['zdim'] * sigma2_p
        loss_match = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            loss_match += res1 - res2
        return loss_match

    def gradient_penalty(self, real, generated):
        if self.opts['aug_rate'] > 1.0:
            shape = tf.shape(generated)[0]
            idxs = tf.range(shape)
            ridxs = tf.random_shuffle(idxs)
            real = tf.gather(real, ridxs)
        elif self.opts['aug_rate'] < 1.0:
            min_shape = tf.shape(generated)[0]
            shape = tf.shape(real)[0]
            idxs = tf.range(shape)
            ridxs = tf.random_shuffle(idxs)[:min_shape]
            real = tf.gather(real, ridxs)
        else:
            shape = tf.shape(generated)[0]

        alpha = tf.random_uniform(shape=[shape, 1, 1, 1], minval=0., maxval=1.)
        differences = generated - real
        interpolates = real + (alpha * differences)
        gradients = \
            tf.gradients(discriminator(self.opts, interpolates, is_training=self.is_training)[1], [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        return gradient_penalty

    def reconstruction_loss(self, opts, real, reconstr):
        if opts['cost'] == 'l2':
            # c(x,y) = ||x - y||_2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
            loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
            loss = 0.05 * tf.reduce_mean(loss)
        elif opts['cost'] == 'l1':
            # c(x,y) = ||x - y||_1
            loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[1, 2, 3])
            loss = 0.02 * tf.reduce_mean(loss)
        else:
            assert False, 'Unknown cost function %s' % opts['cost']
        return loss

    def optimizer(self, lr, decay=1.):
        opts = self.opts
        lr *= decay
        if opts["optimizer"] == "sgd":
            return tf.train.GradientDescentOptimizer(lr)
        elif opts["optimizer"] == "adam":
            return tf.train.AdamOptimizer(lr, beta1=opts["adam_beta1"])
            # return AdamWOptimizer(5e-4, lr, beta1=opts["adam_beta1"])
        else:
            assert False, 'Unknown optimizer.'

    def sample_pz(self, num=100, z_dist=None, labels=None, label_nums=None):
        opts = self.opts
        if z_dist is None:
            mean = np.zeros(opts["zdim"])
            cov = np.identity(opts["zdim"])
            noise = np.random.multivariate_normal(
                mean, cov, num).astype(np.float32)
            return noise
        else:
            assert labels is not None or label_nums is not None
            means, covariances = z_dist
            if labels is not None:
                return np.array([np.random.multivariate_normal(means[e], covariances[e]) for e in labels])
            noises = []
            for i, cnt in enumerate(label_nums):
                if cnt > 0:
                    noises.append(np.random.multivariate_normal(means[i], covariances[i], cnt))
            return np.concatenate(noises, axis=0)

    def pretrain_encoder(self, data):
        opts = self.opts
        steps_max = 200
        batch_size = opts['e_pretrain_sample_size']
        for step in range(steps_max):
            train_size = data.num_points
            data_ids = np.random.choice(train_size, min(train_size, batch_size),
                                        replace=False)
            batch_images = data.data[data_ids].astype(np.float)
            batch_noise = self.sample_pz(batch_size)

            [_, loss_pretrain] = self.sess.run([self.pretrain_opt, self.loss_pretrain],
                                               feed_dict={self.sample_points: batch_images,
                                                          self.sample_noise: batch_noise, self.is_training: True})

            if opts['verbose']: logging.error('Step %d/%d, loss=%f' % (step, steps_max, loss_pretrain))
            if loss_pretrain < 0.1:
                break

    def adjust_decay(self, epoch, decay):
        opts = self.opts
        if opts['lr_schedule'] == "none":
            pass
        elif opts['lr_schedule'] == "manual":
            if epoch == 30:
                decay = decay / 2.
            if epoch == 50:
                decay = decay / 5.
            if epoch == 100:
                decay = decay / 10.
        elif opts['lr_schedule'] == "manual_smooth":
            enum = opts['epoch_num']
            decay_t = np.exp(np.log(100.) / enum)
            decay = decay / decay_t
        elif opts['lr_schedule'] != "plateau":
            assert type(opts['lr_schedule']) == float
            decay = 1.0 * 10 ** (-epoch / float(opts['lr_schedule']))
        return decay

    def train(self, data):
        self.set_class_ratios(data)
        opts = self.opts
        if opts['verbose']:
            logging.error(opts)
        rec_losses, match_losses, encoder_losses, decoder_losses, disc_losses = [], [], [], [], []
        z_dist = None
        batches_num = math.ceil(data.num_points / opts['batch_size'])
        self.sess.run(self.init)

        if opts['e_pretrain']:
            logging.error('Pretraining the encoder')
            self.pretrain_encoder(data)
            logging.error('Pretraining the encoder done.')

        start_time = time.time()
        decay = 1.
        for epoch in range(opts["epoch_num"]):
            print('Epoch %d:' % epoch)
            # Update learning rate if necessary
            decay = self.adjust_decay(epoch, decay)
            # Iterate over batches
            encoder_loss_total = 0.
            decoder_loss_total = 0.
            disc_loss_total = 0.
            correct_total = 0
            update_z_dist_flag = ((epoch + 1) % 5 == 0)
            z_list, z_new_list, y_new_list = [], [], []
            for it in tqdm(range(batches_num)):
                start_idx = it * opts['batch_size']
                end_idx = (it + 1) * opts['batch_size']
                batch_images = data.data[start_idx:end_idx].astype(np.float32)
                batch_labels = data.labels[start_idx:end_idx].astype(np.int32)
                if z_dist is None:
                    feed_d = {self.sample_points: batch_images,
                              self.labels: batch_labels, self.lr_decay: decay, self.is_training: True}
                    [_, z] = self.sess.run([self.ae_opt, self.encoded], feed_dict=feed_d)

                else:
                    feed_d = {self.sample_points: batch_images, self.labels: batch_labels, self.lr_decay: decay,
                              self.is_training: True}

                    [_, encoder_loss, rec_loss, match_loss, z, correct] = self.sess.run(
                        [self.encoder_opt,
                         self.encoder_objective,
                         self.loss_reconstruct,
                         self.penalty,
                         self.encoded, self.correct_sum
                         ],
                        feed_dict=feed_d)
                    [_, decoder_loss] = self.sess.run(
                        [self.decoder_opt,
                         self.decoder_objective,
                         ],
                        feed_dict=feed_d)
                    [_, disc_loss] = self.sess.run(
                        [self.disc_opt,
                         self.disc_objective,
                         ],
                        feed_dict=feed_d)

                    rec_losses.append(rec_loss)
                    match_losses.append(match_loss)
                    encoder_losses.append(encoder_loss)
                    decoder_losses.append(decoder_loss)
                    disc_losses.append(disc_loss)
                    correct_total += correct
                    encoder_loss_total += encoder_loss
                    decoder_loss_total += decoder_loss
                    disc_loss_total += disc_loss

                z_list.append(z)

            training_acc = correct_total / data.num_points
            avg_encoder_loss = encoder_loss_total / batches_num
            avg_decoder_loss = decoder_loss_total / batches_num
            avg_disc_loss = disc_loss_total / batches_num
            self.result_logger.add_training_metrics(avg_encoder_loss, avg_decoder_loss, avg_disc_loss,
                                                    training_acc, time.time() - start_time)
            z = np.concatenate(z_list, axis=0)
            self.result_logger.save_latent_code(epoch, z, data.labels)
            print('Evaluating...')
            self.evaluate(data, epoch)
            if update_z_dist_flag:
                if len(z_new_list) > 0:
                    z_new = np.concatenate(z_new_list, axis=0)
                    y_new = np.concatenate(y_new_list, axis=0)
                    print('Length of z_gen: %d' % len(z_new))
                else:
                    z_new = None
                    y_new = None
                z_dist = means, covariances = self.get_z_dist(z, data.labels, z_new, y_new)
                np.save(opts['work_dir'] + os.sep + 'means_epoch%02d.npy' % epoch, means)
                np.save(opts['work_dir'] + os.sep + 'covariances_epoch%02d.npy' % epoch, covariances)

            if (epoch + 1) % opts['save_every_epoch'] == 0:
                print('Saving checkpoint...')
                self.saver.save(self.sess, os.path.join(opts['work_dir'], 'checkpoints', 'trained-caleg'),
                                global_step=epoch)

    def get_z_dist(self, z, y, z_new=None, y_new=None):
        opts = self.opts
        print("Computing means and covariances...")
        if z_new is None:
            assert y_new is None
        covariances = []
        means = []
        for c in tqdm(range(opts['n_classes'])):
            z_c = np.concatenate([z[y == c], z_new[y_new == c]], axis=0) if z_new is not None else z[y == c]
            covariances.append(np.cov(np.transpose(z_c)))
            means.append(np.mean(z_c, axis=0))
        covariances = np.array(covariances)
        means = np.array(means)

        return means, covariances

    def evaluate(self, data, epoch):
        batch_size = self.opts['batch_size']
        batches_num = int(math.ceil(len(data.test_data) / batch_size))

        probs = []

        start_time = time.time()
        for it in range(batches_num):
            start_idx = it * batch_size
            end_idx = start_idx + batch_size
            prob = self.sess.run(
                self.probs1,
                feed_dict={self.sample_points: data.test_data[start_idx:end_idx], self.is_training: False})
            probs.append(prob)
        probs = np.concatenate(probs, axis=0)
        predicts = np.argmax(probs, axis=-1)
        assert probs.shape[1] == self.opts['n_classes']
        self.result_logger.save_prediction(epoch, data.test_labels, predicts, probs, time.time() - start_time)
        self.result_logger.save_metrics()

    def set_class_ratios(self, data):
        self.gratio_mode = self.opts['gratio_mode']
        self.dratio_mode = self.opts['dratio_mode']
        class_count = [np.count_nonzero(data.labels == n) for n in range(self.opts['n_classes'])]
        class_cnt = np.array(class_count)
        max_class_cnt = np.max(class_cnt)
        total_aug_nums = (max_class_cnt - class_cnt)
        self.aug_class_rate = total_aug_nums / np.sum(total_aug_nums)

        self.class_aratio = [per_count / sum(class_count) for per_count in class_count]

        n_classes = self.opts['n_classes']
        self.class_dratio = np.full(n_classes, 0.0)
        # Set uniform
        target = 1 / n_classes
        self.class_uratio = np.full(n_classes, target)
        # Set gratio
        self.class_gratio = np.full(n_classes, 0.0)
        for c in range(n_classes):
            if self.gratio_mode == "uniform":
                self.class_gratio[c] = target
            elif self.gratio_mode == "rebalance":
                self.class_gratio[c] = 2 * target - self.class_aratio[c]
            else:
                print("Error while training bgan, unknown gmode " + self.gratio_mode)
                exit()

        # Set dratio
        self.class_dratio = np.full(n_classes, 0.0)
        for c in range(n_classes):
            if self.dratio_mode == "uniform":
                self.class_dratio[c] = target
            elif self.dratio_mode == "rebalance":
                self.class_dratio[c] = 2 * target - self.class_aratio[c]
            else:
                print("Error while training bgan, unknown dmode " + self.dratio_mode)
                exit()

        # if very unbalanced, the gratio might be negative for some classes.
        # In this case, we adjust..
        if self.gratio_mode == "rebalance":
            self.class_gratio[self.class_gratio < 0] = 0
            self.class_gratio = self.class_gratio / sum(self.class_gratio)

        # if very unbalanced, the dratio might be negative for some classes.
        # In this case, we adjust..
        if self.dratio_mode == "rebalance":
            self.class_dratio[self.class_dratio < 0] = 0
            self.class_dratio = self.class_dratio / sum(self.class_dratio)

    def biased_sample_labels(self, num_samples):
        num_samples = math.ceil(num_samples * self.opts['aug_rate'])
        if num_samples == 0:
            return np.full([1], self.opts['n_classes'] - 1, dtype=np.int32)
        aug_num = np.round(self.aug_class_rate * num_samples).astype(np.int32)
        sampled_labels = np.zeros(np.sum(aug_num), dtype=np.int32)
        start = 0
        for i in range(self.opts['n_classes']):
            end = start + aug_num[i]
            sampled_labels[start:end] = i
            start = end
        return sampled_labels

    def load_ckpt(self, epoch):
        self.saver.restore(self.sess, os.path.join(self.opts['work_dir'], 'checkpoints', 'trained-caleg-%d' % epoch))
