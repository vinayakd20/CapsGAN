#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 19:08:13 2018

@author: vinayak
"""
import tensorflow as tf

from config import cfg
#from utils import softmax
from utils import reduce_sum
from capsLayer import CapsLayer

def discriminator(input, isTrain=True, reuse=False):
    epsilon = 1e-9
    if isTrain:
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                labels = tf.constant(0, shape=[cfg.batch_size, ])
            else:
                labels = tf.constant(1, shape=[cfg.batch_size, ])
            Y = tf.one_hot(labels, depth=2, axis=1, dtype=tf.float32)
            X = input
            
        if reuse:
            scope.reuse_variables()
        with tf.variable_scope('Conv1_layer'):
                # Conv1, [batch_size, 20, 20, 256]
                conv1 = tf.contrib.layers.conv2d(X, num_outputs=256,
                                                 kernel_size=9, stride=1,
                                                 padding='VALID')
                assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]
    
            # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
                primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
                caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
                assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]
    
            # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
                """changing the num_outputs to 2 from 10"""
                digitCaps = CapsLayer(num_outputs=2, vec_len=16, with_routing=True, layer_type='FC')
                caps2 = digitCaps(caps1)
                v_length = tf.sqrt(reduce_sum(tf.square(caps2),
                                                   axis=2, keepdims=True) + epsilon)
        
        """Loss """
        max_l = tf.square(tf.maximum(0., cfg.m_plus - v_length))
            # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., v_length - cfg.m_minus))
        """changing assert value to be [batch, 2, 1, 1] from [batch, 10, 1, 1]"""
        assert max_l.get_shape() == [cfg.batch_size, 2, 1, 1]
    
            # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))
    
            # calc T_c: [batch_size, 10]
            # T_c = Y, is my understanding correct? Try it.
        T_c = Y
            # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r
    
        margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        return margin_loss