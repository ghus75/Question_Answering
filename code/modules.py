# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class HighWayNetwork(object):
    def __init__(self):
        pass
    
    def build_graph(self, inputs, activation_bias=0.1, transform_bias=-2.0):
        """Builds a highway network between the embedding layer and the main part of the graph for the model
        
        Inputs : 
          inputs : word embeddings shape(batch_size, sequence_len, embedding_dimension)
          
        Outputs:
          y : a word embedding tensor, same shape as input
        """
    
        with vs.variable_scope("Highway_net",reuse=tf.AUTO_REUSE):
            size = int(inputs.get_shape().as_list()[2]) # dimension of embedding
            
            # Activation
            H = tf.contrib.layers.fully_connected(inputs, num_outputs=size, activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(value=activation_bias))
            # Transform gate
            T = tf.contrib.layers.fully_connected(inputs, num_outputs=size, activation_fn=tf.sigmoid, biases_initializer=tf.constant_initializer(value=transform_bias))
            # Carry gate
            C = tf.subtract(1.0, T, name="Carry_gate")

            y = tf.add(tf.multiply(H, T), tf.multiply(inputs, C), name="HiWay_output")

            return y            
        
       
class biLSTMEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.
    
    This module is used for the Contextual embedding layer : 
        used to encode context and question word embeddings

    Here, we're using the RNN as an "encoder" we're just returning all the hidden states.
    The terminology "encoder" still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional LSTM.
    """

    def __init__(self, hidden_size, keep_prob):
               #(self.FLAGS.hidden_size, self.keep_prob)
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
                  #(self.context_embs, self.context_mask) # (batch_size, context_len, embedding_size)
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("biLSTMEncoder"):
            # input sequence length is given by word_mask tensor
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)
            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)
            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)
            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out
        

class Modeling_layer_biLSTM_Encoder(object):
    """
    Similar to biLSTM_Encoder, with the following modifications:
        - uses a stack of 2 BiLSTM
        - doesn't need a mask placeholder to specify input length, since at that stage all vectors have same length
    """

    def __init__(self, hidden_size, keep_prob):
               #(self.FLAGS.hidden_size, self.keep_prob)
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

        self.rnn_cell_fw2 = rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_fw2 = DropoutWrapper(self.rnn_cell_fw2, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw2 = rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_bw2 = DropoutWrapper(self.rnn_cell_bw2, input_keep_prob=self.keep_prob)

        
    def build_graph(self, inputs):
                  #(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size)
                  #(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)
        """
        Inputs:
          inputs: list of Tensors of shape (batch_size, seq_len, input_size)

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("biLSTM_Modeling"):

            (outputs, _, _) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([self.rnn_cell_fw, self.rnn_cell_fw2], 
                                                                            [self.rnn_cell_bw, self.rnn_cell_bw2],
                                                                            inputs, 
                                                                            dtype=tf.float32)
            # Apply dropout
            outputs = tf.nn.dropout(outputs, self.keep_prob)

            return outputs

class Output_layer_biLSTM_Encoder(object):
    """
    Similar to biLSTM_Encoder, with the following modification:
        - doesn't need a mask placeholder to specify input length, since at that stage all vectors have same length
    """

    def __init__(self, hidden_size, keep_prob):
               #(self.FLAGS.hidden_size, self.keep_prob)
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.BasicLSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)


        
    def build_graph(self, inputs):
                  #(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size)
                  #(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("biLSTM_Output"):

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            # No sequence length : all batch entries are assumed to be full sequences; 
            # and time reversal is applied from time 0 to max_time for each sequence.
            (fw_out, bw_out), (fw_out_state, bw_out_state) = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, dtype=tf.float32, scope='BLSTM_1')

#            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)
#            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

        
class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist



class BiDAFAttn(object):
    """Module for BiDAFAttn attention.
    Module to take set of hidden states, (e.g. one for each context location),
    and return attention distribution and outputs.

    """

    def __init__(self, keep_prob, context_hiddens_vec_size, question_hiddens_vec_size):
               #(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
               #(self.keep_prob,        2h,         2h)
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          context_hiddens_vec_size: size of the context_hiddens vectors. int
          question_hiddens_vec_size: size of the question_hiddens vectors. int
        """
        self.keep_prob = keep_prob
        self.context_hiddens_vec_size = context_hiddens_vec_size
        self.question_hiddens_vec_size = question_hiddens_vec_size

    def build_graph(self, question_hiddens, question_hiddens_mask, context_hiddens, context_hiddens_mask):   
                  #(question_hiddens, self.qn_mask, context_hiddens, self.context_mask)
                  #     q_j,            q_j mask,       c_i        , c_i mask 
        """
        context_hiddens attend to question_hiddens.
        For each context_hiddens, return an attention distribution and an attention output vector.

        Inputs:
          question_hiddens: Tensor shape (batch_size, question_len, 2h).
          question_hiddens_mask: Tensor shape (batch_size, question_len).
            1s where there's real input, 0s where there's padding
          context_hiddens: Tensor shape (batch_size, context_len, 2h)
          context_hiddens_mask: Tensor shape (batch_size, context_len).

        Outputs:
          attn_dist: Tensor shape (batch_size, context_len, question_len).
            For each context_hiddens, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, context_len, hidden_size).
            This is the attention output; the weighted sum of the question_hiddens
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BiDAFAttn"):

            # *******************************
            # *** Build similarity matrix ***
            # *******************************
            
            with vs.variable_scope("Similarity_matrix"):
    
                W_sim1 = tf.get_variable("W_sim1_cn", shape = [self.context_hiddens_vec_size, 1]) # shape (2h, 1)
                W_sim2 = tf.get_variable("W_sim2_qn", shape = [self.context_hiddens_vec_size, 1])
                W_sim3 = tf.get_variable("W_sim3_cq", shape = [self.context_hiddens_vec_size, 1])
                
                question_len = question_hiddens.get_shape().as_list()[1]
                context_len = context_hiddens.get_shape().as_list()[1]
                
                # sous-matrice (W_sim1 . context_repeat) // (?,N,M) = (?, context_len, question_len)
                W_sim1_context = tf.tensordot(W_sim1, context_hiddens, axes=[[0], [2]], name="W_sim1_dot_cn") # (1, ?, context_len)
                W_sim1_context = tf.reshape(W_sim1_context, [-1, context_len, 1]) # (?, context_len, 1)
                W_sim1_context_repeat = tf.tile(W_sim1_context, tf.constant([1, 1, question_len])) # (?, context_len, question_len)
                
                # sous-matrice (W_sim2 . question_repeat) // (?,N,M) = (?, context_len, question_len)
                W_sim2_question = tf.tensordot(W_sim2, question_hiddens, axes=[[0], [2]], name="W_sim2_dot_qn") # (1, ?, question_len)
                W_sim2_question = tf.reshape(W_sim2_question, [-1, question_len, 1]) # (?, question_len, 1)
                W_sim2_question_repeat = tf.tile(W_sim2_question, tf.constant([1, context_len, 1])) # (?, context_len*question_len, 1)
                W_sim2_question_repeat = tf.reshape(W_sim2_question_repeat, [-1, context_len, question_len]) # (?, context_len, question_len)
                
                # sous-matrice (W_sim3 . context_hiddens o question_hiddens) // (?,N,M) = (?, context_len, question_len)
                W_sim3_times_context = tf.multiply(tf.tile(tf.transpose(W_sim3), tf.constant([context_len, 1])), context_hiddens, name="W_sim3_o_cn")
                W_sim3_context_question = tf.matmul(W_sim3_times_context, tf.transpose(question_hiddens, perm=[0, 2, 1]), name="W_sim3_x_qn") # (?, context_len, question_len)
                
                sim_matrix = tf.add_n([W_sim1_context_repeat, W_sim2_question_repeat, W_sim3_context_question], name = "sim_matrix") # shape (?, context_len, question_len)
                
            # ****************************************            
            # *** Calculate attention distribution ***
            # ****************************************

            # *** C2Q Attention ***
            with vs.variable_scope("C2Q_Attention"):
                c2q_attn_logits = sim_matrix # shape (batch_size, context_len, question_len)
                c2q_attn_logits_mask = tf.expand_dims(question_hiddens_mask, 1) # shape (batch_size, 1, question_len)
                _, c2q_attn_dist = masked_softmax(c2q_attn_logits, c2q_attn_logits_mask, 2) # shape (batch_size, context_len, question_len). take softmax over question_hiddens
    
                # Use attention distribution to take weighted sum of question_hiddens
                c2q_output = tf.matmul(c2q_attn_dist, question_hiddens) # shape (batch_size, context_len, 2h)
    
                # Apply dropout
                c2q_output = tf.nn.dropout(c2q_output, self.keep_prob)

            # *** Q2C Attention ***
            with vs.variable_scope("Q2C_Attention"):
                # m_i
                q2c_attn_logits = tf.reduce_max(sim_matrix, axis=2, keep_dims=True) # shape (batch_size, context_len, 1)
                q2c_attn_logits_mask = tf.expand_dims(context_hiddens_mask, 2) # shape (batch_size, context_len, 1)
                # beta
                _, q2c_attn_dist = masked_softmax(q2c_attn_logits, q2c_attn_logits_mask, 1) # shape (batch_size, context_len, 1). take softmax over question_hiddens
                q2c_output = tf.reduce_sum(tf.multiply(q2c_attn_dist, context_hiddens), axis=1) # shape (batch_size, 2h)
                q2c_output = tf.expand_dims(q2c_output, axis=1)# shape (batch_size, 1, 2h)
                # Apply dropout
                q2c_output = tf.nn.dropout(q2c_output, self.keep_prob)# shape (batch_size, 1, 2h)
            
            return c2q_attn_dist, c2q_output, q2c_attn_dist, q2c_output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
