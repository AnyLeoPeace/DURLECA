'''The GraphSageConv layers are from https://github.com/danielegrattarola/spektral/ and are used as baselines'''

import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from keras import activations, initializers, regularizers, constraints
import numpy as np


def dense_to_sparse(x):
    """
    Converts a Tensor to a SparseTensor.
    :param x: a Tensor.
    :return: a SparseTensor.
    """
    indices = tf.where(tf.not_equal(x, 0))
    values = tf.gather_nd(x, indices)
    shape = tf.shape(x, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


class SparseSoftmax(Layer):

    
    def call(self, inputs):
        features = inputs[0] # None * N * N
        OD = inputs[1] # None * N * N
        OD = tf.cast(tf.cast(OD, tf.bool), tf.float32)

        features_ = features * OD
        features_ = tf.sparse.from_dense(features_)
        features_ = tf.sparse.softmax(features_)
        features_ = tf.sparse.to_dense(features_)

        return features_
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


class GraphSageConvSoftmax(Layer):

    def __init__(self,
                 channels,
                 aggregate_op = None,
                 BN = None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.channels = channels
        self.BN = BN
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(2 * input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        if not K.is_sparse(fltr):
            fltr = tf.sparse.from_dense(tf.transpose(fltr, [0,2,1]))

        weights_neigh = tf.sparse.softmax(fltr)
        weights_neigh = tf.sparse.to_dense(weights_neigh)
        features_neigh = tf.matmul(weights_neigh, features)

        output = K.concatenate([features, features_neigh])
        output = K.dot(output, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        if self.BN is not None:
            output = self.BN(output)
        return output
    
    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.channels,)
        return output_shape
    
    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class GraphSageConv(Layer):
    def __init__(self,
                 channels,
                 aggregate_op = 'binary_mean',
                 BN = None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.channels = channels
        self.BN = BN
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.aggregate_op = 'aggregate_op'
        self.delta = tf.constant(1e-7, tf.float32)

        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(2 * input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0] # None * 323 * d
        fltr = inputs[1]


        weights_neigh = tf.transpose(fltr, [0,2,1]) # None * 323 * 323
        if 'binary' in self.aggregate_op:
            weights_neigh = tf.cast(tf.cast(weights_neigh, tf.bool), tf.float32)

        features_neigh = tf.matmul(weights_neigh, features) # None * 323 * d
        if 'mean' in self.aggregate_op:
            neighbor_sum = tf.reduce_sum(weights_neigh, axis = -1, keepdims = True)
            features_neigh = tf.math.divide_no_nan(features_neigh, (neighbor_sum + self.delta))

        output = K.concatenate([features, features_neigh])
        output = K.dot(output, self.kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        if self.BN is not None:
            output = self.BN(output)
        return output
    
    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.channels,)
        return output_shape
    
    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class GraphSageConvOD(Layer):

    def __init__(self,
                 channels,
                 aggregate_op = None,
                 BN = None,
                 index = [0],
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        # super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.BN = BN
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.delta = tf.constant(1e-7, dtype = tf.float32)
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1] - 1
        self.kernel = self.add_weight(shape=(input_dim * np.power(2,len(self.index)), self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        
        self.built = True
    
    def call(self, inputs):
        input_feature = inputs[0] # None * N * d
        OD_all = inputs[1]

        loc_feature = input_feature
        populations = loc_feature[:,:,:1]  # The first dimension is for saving regional population. Size: None * N * 1
        SIR = loc_feature[:,:,1:] # None * N * d-1
        nb_regions = K.int_shape(OD_all)[-1]
        for index in self.index:
            OD = OD_all[:,index]
            SIR_n = tf.math.divide_no_nan(SIR, self.delta + populations) # None * N * d-1

            # As the regional population is affected by previous mobility restrictions, the current move-out population may be more than the total population in very few cases.
            # Here we force the move-out population <= the total population
            ratio = tf.math.divide_no_nan(populations, self.delta + tf.reduce_sum(OD, axis=2, keepdims = True))
            ratio = tf.repeat(ratio, nb_regions, axis=-1)
            OD = tf.where(ratio < 1, OD*ratio, OD)
            OD_out = tf.reduce_sum(OD, axis=2, keepdims=True)
            OD_in = tf.reduce_sum(OD, axis=1, keepdims=False)

            SIR_in = tf.matmul(tf.transpose(OD, [0,2,1]), SIR_n)
            SIR_stay = SIR - OD_out * SIR_n
            populations = populations - OD_out + K.expand_dims(OD_in,axis=-1) # new population
            SIR = tf.concat((SIR_stay, SIR_in), axis = -1)
            

        output = K.dot(SIR, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        if self.BN is not None:
            output = self.BN(output)
            
        output = tf.concat((populations, output), axis = -1)

        return output
    
    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.channels,)
        return output_shape
    



class GraphSoftmaxPool(Layer):
    def __init__(self,
                 **kwargs):

        super().__init__(**kwargs)

    def call(self, inputs):
        features = inputs[0] # None * N * d
        fltr = inputs[1] # None * N 

        weights = tf.math.softmax(fltr)
        features = tf.expand_dims(weights, -1) * features

        output = tf.reduce_sum(features, axis = 1)

        return output
    
    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = (features_shape[0], features_shape[-1])
        return output_shape
