import tensorflow as tf
import numpy as np
from learnMSA.msa_hmm.TotalProbabilityCell import TotalProbabilityCell
from learnMSA.msa_hmm.Utility import deserialize
from learnMSA.msa_hmm.Bidirectional import Bidirectional


class MsaHmmLayer(tf.keras.layers.Layer):
    """ A layer that computes the log-likelihood and posterior state probabilities for batches of observations
        under a number of HMMs.
    Args:
        cell: HMM cell whose forward recursion is used.
        num_seqs: The number of sequences in the dataset. If not provided, the prior is not normalized.
        use_prior: If true, the prior is added to the log-likelihood.
        sequence_weights: A tensor of shape (num_seqs,) that contains the weight of each sequence.
            parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.
                            The parallel factor has to be a divisor of the sequence length.
    """
    def __init__(self, 
                 cell, 
                 num_seqs=None,
                 use_prior=True,
                 sequence_weights=None,
                 parallel_factor=1,
                 **kwargs
                ):
        super(MsaHmmLayer, self).__init__(**kwargs)
        self.cell = cell
        self.num_seqs = num_seqs
        self.use_prior = use_prior 
        self.sequence_weights = sequence_weights
        if sequence_weights is not None:
            self.weight_sum = np.sum(sequence_weights)
        self.parallel_factor = parallel_factor
        
        
    def build(self, input_shape):
        if self.built:
            return
        # build the cell
        self.cell.build((None, input_shape[-2], input_shape[-1]))
        # make a variant of the forward cell configured for backward
        self.reverse_cell = self.cell.make_reverse_direction_offspring()
        # build the reverse cell
        self.reverse_cell.build((None, input_shape[-2], input_shape[-1]))
        #make a forward rnn layer
        self.rnn = tf.keras.layers.RNN(self.cell, 
                                       return_sequences=True, 
                                       return_state=True)
        # make a backward rnn layer
        self.rnn_backward = tf.keras.layers.RNN(self.reverse_cell, 
                                                return_sequences=True, 
                                                return_state=True,
                                                go_backwards=True)
        # make a bidirectional rnn layer to run forward and backward in parallel
        self.bidirectional_rnn = Bidirectional(self.rnn, 
                                                merge_mode="concat" if self.parallel_factor > 1 else "sum", 
                                                backward_layer=self.rnn_backward)
        # Bidirectional makes a copy rather than taking the original rnn, override the copy
        self.bidirectional_rnn.forward_layer = self.rnn 
        self.bidirectional_rnn.backward_layer = self.rnn_backward 
        # build the RNN layers with a different input shape
        rnn_input_shape = (None, input_shape[-2], self.cell.max_num_states)
        self.rnn.build(rnn_input_shape)
        self.rnn_backward.build(rnn_input_shape)
        self.bidirectional_rnn.build(rnn_input_shape)
        if self.parallel_factor > 1:
            self.total_prob_cell = TotalProbabilityCell(self.cell)
            self.total_prob_cell_rev = TotalProbabilityCell(self.reverse_cell, reverse=True)
            self.total_prob_rnn = tf.keras.layers.RNN(self.total_prob_cell, return_sequences=True, return_state=True)
            self.total_prob_rnn_rev = tf.keras.layers.RNN(self.total_prob_cell_rev, return_sequences=True, return_state=True, go_backwards=True)
        else:
            self.total_prob_rnn = None
            self.total_prob_rnn_rev = None
        built = True
        
        
    def forward_recursion(self, inputs, end_hints=None, 
                            return_prior=False, training=False):
        """ Computes the forward recursion for multiple models where each model
            receives a batch of sequences as input.
        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
            end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
            return_prior: If true, the prior is computed and returned.
            training: If true, the cell is run in training mode.
        Returns:
            forward variables: Shape: (num_model, b, seq_len, q)
            log-likelihoods: Shape: (num_model, b)
        """
        #initialize transition- and emission-matricies
        return _forward_recursion_impl(inputs, self.cell, self.rnn, self.total_prob_rnn, 
                                       end_hints=end_hints, return_prior=return_prior,
                                       training=training, parallel_factor=self.parallel_factor)
    
    
    def backward_recursion(self, inputs, end_hints=None, 
                            return_prior=False, training=False):
        """ Computes the backward recursion for multiple models where each model
            receives a batch of sequences as input.
        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
            end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
            return_prior: If true, the prior is computed and returned.
            training: If true, the cell is run in training mode.
        Returns:
            backward variables: Shape: (num_model, b, seq_len, q)
        """
        return _backward_recursion_impl(inputs, self.cell, self.reverse_cell, 
                                        self.rnn_backward, self.total_prob_rnn_rev, 
                                        end_hints=end_hints, return_prior=return_prior,
                                        training=training, parallel_factor=self.parallel_factor)
    

    def state_posterior_log_probs(self, inputs, end_hints=None, 
                                return_prior=False, training=False, no_loglik=False):
        """ Computes the log-probability of state q at position i given inputs.
        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
            end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
            return_prior: If true, the prior is computed and returned.
            training: If true, the cell is run in training mode.
            no_loglik: If true, the loglik is not used in the return value. This can be beneficial for end-to-end training when the
                        normalizing constant of the posteriors is not important and the activation function is the softmax.
        Returns:
            state posterior probbabilities: Shape: (num_model, b, seq_len, q)
        """
        return _state_posterior_log_probs_impl(inputs, self.cell, self.reverse_cell, 
                                                self.bidirectional_rnn, self.total_prob_rnn, 
                                                self.total_prob_rnn_rev,
                                                end_hints=end_hints, return_prior=return_prior,
                                                training=training, no_loglik=no_loglik, 
                                                parallel_factor=self.parallel_factor)
    
    
    def apply_sequence_weights(self, loglik, indices, aggregate=False):
        if self.sequence_weights is not None and indices is not None:
            weights = tf.gather(self.sequence_weights, indices)
            loglik *= weights
            if aggregate:
                loglik = tf.reduce_sum(loglik, axis=1) / tf.reduce_sum(weights, axis=1) #mean over batch 
                loglik = tf.reduce_mean(loglik) #mean over models
        elif aggregate:
            loglik = tf.reduce_mean(loglik) #mean over both models and batches
        return loglik
    
    
    #compute the prior, scale it depending on seq weights
    def compute_prior(self, scaled=True):
        self.cell.recurrent_init() #initialize all relevant tensors
        prior = self.cell.get_prior_log_density()
        if scaled:
            prior = self._scale_prior(prior)
        return prior


    def _scale_prior(self, prior):
        if self.sequence_weights is not None:
            prior /= self.weight_sum
        elif self.num_seqs is not None:
            prior /= self.num_seqs
        return prior
        
        
    def call(self, inputs, indices=None, training=False):
        """ Computes log-likelihoods per model and sequence.
        Args:
            inputs: Sequences. Shape: (num_model, b, seq_len, s)
            indices: Optional sequence indices required to assign sequence weights. Shape: (num_model, b)
        Returns:
            log-likelihoods: Shape: (num_model, b)
            aggregated loglik: Shape: ()
            prior: Shape: (num_model)
            aux_loss: Shape: ()
        """
        inputs = tf.cast(inputs, self.dtype)
        if self.use_prior:
            _, loglik, prior, aux_loss = self.forward_recursion(inputs, return_prior=True, training=training)
            prior = self._scale_prior(prior)
        else:
            _, loglik = self.forward_recursion(inputs, return_prior=False, training=training)
        loglik_mean = self.apply_sequence_weights(loglik, indices, aggregate=True)
        loglik_mean = tf.squeeze(loglik_mean)
        if self.use_prior:
            return loglik, loglik_mean, prior, aux_loss
        else:
            return loglik, loglik_mean
        
        
    def get_config(self):
        config = super(MsaHmmLayer, self).get_config()
        config.update({ 
             "cell" : self.cell,
             "num_seqs" : self.num_seqs,
             "use_prior" : self.use_prior, 
             "sequence_weights" : self.sequence_weights,
             "parallel_factor" : self.parallel_factor
        })
        return config

    
    @classmethod
    def from_config(cls, config):
        config["cell"] = deserialize(config["cell"])
        config["sequence_weights"] = deserialize(config["sequence_weights"])
        return cls(**config)


    
    
        
@tf.function
def _forward_recursion_impl(inputs, cell, rnn, total_prob_rnn, 
                            end_hints=None, return_prior=False,
                            training=False, parallel_factor=1):
    """ Computes the forward recursion for multiple models where each model
        receives a batch of sequences as input.
    Args:
        inputs: Sequences. Shape: (num_model, b, seq_len, s)
        cell: HMM cell used for forward recursion.
        rnn: A RNN layer that runs the forward recursion.
        total_prob_rnn: A RNN layer that computes the total probability of the forward variables.
        end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
        return_prior: If true, the prior is computed and returned.
        training: If true, the cell is run in training mode.
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.
    Returns:
        forward variables: Shape: (num_model, b, seq_len, q)
        log-likelihoods: Shape: (num_model, b)
    """
    cell.recurrent_init()
    #initialize transition- and emission-matricies
    _, b, seq_len, s = tf.unstack(tf.shape(inputs))
    num_model = cell.num_models
    q = cell.max_num_states
    emission_probs = cell.emission_probs(inputs, end_hints=end_hints, training=training)
    #reshape to 3D inputs for RNN (cell will reshape back in each step)
    #if parallel_factor > 1, reshape to equally sized chunks
    chunk_size = seq_len // parallel_factor
    emission_probs = tf.reshape(emission_probs, (num_model*b*parallel_factor, chunk_size, q))
    #do one initialization step
    #this way, tf will compile two versions of the cell call, one with init=True and one without
    initial_state = cell.get_initial_state(batch_size=b*parallel_factor, parallel_factor=parallel_factor)
    forward_1, step_1_state = cell(emission_probs[:,0], initial_state, training=training, init=True)
    #run forward with the output of the first step as initial state
    forward, _, loglik = rnn(emission_probs[:,1:], initial_state=step_1_state, training=training)
    #prepend the separate first step to the other forward steps
    #shape of forward tensor: (num_model*b*factor, chunk_size, z*q + z) z == 1 iff parallel_factor == 1 else z=q
    forward = tf.concat([forward_1[:,tf.newaxis], forward], axis=1) 
    if parallel_factor == 1:
        forward = tf.reshape(forward, (num_model, b, seq_len, -1))
        forward_scaled =  forward[...,:-1]
        forward_scaling_factors = forward[..., -1:]
        forward_result = forward_scaled + forward_scaling_factors
        loglik = tf.reshape(loglik, (num_model, b))
    else:
        forward_result, loglik = _get_total_forward_from_chunks(forward, cell, total_prob_rnn, b, seq_len, parallel_factor=parallel_factor)
    if return_prior:
        prior = cell.get_prior_log_density()
        aux_loss = cell.get_aux_loss()
        return forward_result, loglik, prior, aux_loss
    else:
        return forward_result, loglik


def _get_total_forward_from_chunks(forward, cell, total_prob_rnn, b, seq_len, parallel_factor=1):
    #utility method that computes the actual forward probabilities from the chunked forward variables
    #returns the forward probabilities and the log-likelihood
    q = cell.max_num_states
    num_model = cell.num_models
    chunk_size = seq_len // parallel_factor
    forward_scaled = forward[...,:-q]
    forward_scaling_factors = forward[..., -q:] 
    forward_scaled = tf.reshape(forward_scaled, (num_model*b, parallel_factor, chunk_size, q, -1))
    forward_scaling_factors = tf.reshape(forward_scaling_factors, (num_model*b, parallel_factor, chunk_size, q, 1))
    forward_chunks = forward_scaled + forward_scaling_factors #shape: (num_model*b, factor, chunk_size, q (conditional states), q (actual states))
    #compute the actual forward variables across the chunks via the total probability
    forward_chunks_last = forward_chunks[:,:,-1]  #(num_model*b, factor, q, q)
    forward_chunks_last = tf.reshape(forward_chunks_last, (num_model*b, parallel_factor, q*q))
    forward_total, _, loglik = total_prob_rnn(forward_chunks_last) #(num_model*b, factor, q)
    init, _ = cell.get_initial_state(batch_size=b, parallel_factor=1)
    init = tf.math.log(init + cell.epsilon)
    T = tf.concat([init[:,tf.newaxis], forward_total[:,:-1]], axis=1)
    T = T[:, :, tf.newaxis, :, tf.newaxis]
    forward_result = forward_chunks + T #shape: (num_model*b, factor, chunk_size, q, q)
    forward_result = tf.reshape(forward_result, (num_model, b, seq_len, q, q))
    forward_result = tf.math.reduce_logsumexp(forward_result, axis=-2)
    loglik = tf.reshape(loglik, (num_model, b))
    return forward_result, loglik
    
    
@tf.function
def _backward_recursion_impl(inputs, cell, reverse_cell, 
                            rnn_backward, total_prob_rnn_rev, 
                            end_hints=None, return_prior=False,
                            training=False, parallel_factor=1):
    """ Computes the backward recursion for multiple models where each model
        receives a batch of sequences as input.
    Args:
        inputs: Sequences. Shape: (num_model, b, seq_len, s)
        cell: HMM cell used for forward recursion.
        reverse_cell: HMM cell used for backward recursion.
        rnn_backward: A RNN layer that runs the backward recursion.
        total_prob_rnn_rev: A RNN layer that computes the total probability of the backward variables.
        end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
        return_prior: If true, the prior is computed and returned.
        training: If true, the cell is run in training mode.
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.
    Returns:
        backward variables: Shape: (num_model, b, seq_len, q)
    """
    cell.recurrent_init()
    reverse_cell.recurrent_init()
    _, b, seq_len, s = tf.unstack(tf.shape(inputs))
    num_model = cell.num_models
    q = cell.max_num_states
    emission_probs = reverse_cell.emission_probs(inputs, end_hints=end_hints, training=training)
    #reshape to 3D inputs for RNN (cell will reshape back in each step)
    #if parallel_factor > 1, reshape to equally sized chunks
    chunk_size = seq_len // parallel_factor
    emission_probs = tf.reshape(emission_probs, (num_model*b*parallel_factor, chunk_size, q))
    #do one initialization step
    #this way, tf will compile two versions of the cell call, one with init=True and one without
    initial_state = reverse_cell.get_initial_state(inputs=emission_probs, batch_size=b*parallel_factor, parallel_factor=parallel_factor)
    backward_1, step_1_state = reverse_cell(emission_probs[:,-1], initial_state, training=training, init=True)
    backward, _, _ = rnn_backward(emission_probs[:,:-1], initial_state=step_1_state, training=training)
    backward = tf.concat([backward_1[:,tf.newaxis], backward], axis=1) 
    if parallel_factor == 1:
        backward = tf.reshape(backward, (num_model, b, seq_len, -1))
        backward_scaled = backward[...,:-1]
        backward_scaling_factors = backward[..., -1:]
        backward_result = backward_scaled + backward_scaling_factors
        backward_result = tf.reverse(backward_result, [-2])
    else:
        backward_result = _get_total_backward_from_chunks(backward, cell, reverse_cell, total_prob_rnn_rev, b, seq_len, parallel_factor=parallel_factor)
    if return_prior:
        prior = cell.get_prior_log_density()
        aux_loss = cell.get_aux_loss()
        return backward_result, prior, aux_loss
    else:
        return backward_result


def _get_total_backward_from_chunks(backward, cell, reverse_cell, total_prob_rnn_rev, b, seq_len, revert_chunks=True, parallel_factor=1):
    #utility method that computes the actual backward probabilities from the chunked backward variables
    q = cell.max_num_states
    num_model = cell.num_models
    chunk_size = seq_len // parallel_factor
    backward_scaled = backward[...,:-q]
    backward_scaling_factors = backward[..., -q:]
    backward_scaled = tf.reshape(backward_scaled, (num_model*b, parallel_factor, chunk_size, q, -1))
    backward_scaling_factors = tf.reshape(backward_scaling_factors, (num_model*b, parallel_factor, chunk_size, q, 1))
    backward_chunks = backward_scaled + backward_scaling_factors #shape: (num_model*b, factor, chunk_size, q (conditional states), q (actual states))
    if revert_chunks:
        backward_chunks = tf.reverse(backward_chunks, [-3])
    #compute the actual backward variables across the chunks via the total probability
    backward_chunks_last = backward_chunks[:,:,0]  #(num_model*b, factor, q, q)
    backward_chunks_last = tf.reshape(backward_chunks_last, (num_model*b, parallel_factor, q*q))
    backward_total, _, _ = total_prob_rnn_rev(backward_chunks_last) #(num_model*b, factor, q)
    backward_total = tf.reverse(backward_total, [1])
    init, _ = reverse_cell.get_initial_state(batch_size=b, parallel_factor=1)
    init = tf.math.log(init + reverse_cell.epsilon)
    T = tf.concat([backward_total[:,1:], init[:,tf.newaxis]], axis=1)
    T = T[:, :, tf.newaxis, :, tf.newaxis]
    backward_result = backward_chunks + T #shape: (num_model*b, factor, chunk_size, q, q)
    backward_result = tf.reshape(backward_result, (num_model, b, seq_len, q, q))
    backward_result = tf.math.reduce_logsumexp(backward_result, axis=-2)
    return backward_result


def proper_shape(tensor):
  if tensor.shape.is_fully_defined():
    return tensor.shape
  return tf.shape(tensor)


@tf.function
def _state_posterior_log_probs_impl(inputs, cell, reverse_cell, 
                                    bidirectional_rnn, total_prob_rnn, 
                                    total_prob_rnn_rev, 
                                    end_hints=None, return_prior=False,
                                    training=False, no_loglik=False, parallel_factor=1):
    """ Computes the log-probability of state q at position i given inputs.
    Args:
        inputs: Sequences. Shape: (num_model, b, seq_len, s)
        cell: HMM cell used for forward recursion.
        reverse_cell: HMM cell used for backward recursion.
        bidirectional_rnn: A bidirectional RNN layer that runs forward and backward in parallel.
        total_prob_rnn: A RNN layer that computes the total probability of the forward variables.
        total_prob_rnn_rev: A RNN layer that computes the total probability of the backward variables.
        end_hints: A tensor of shape (..., 2, num_states) that contains the correct state for the left and right ends of each chunk.
        return_prior: If true, the prior is computed and returned.
        training: If true, the cell is run in training mode.
        no_loglik: If true, the loglik is not used in the return value. This can be beneficial for end-to-end training when the
                    normalizing constant of the posteriors is not important and the activation function is the softmax.
        parallel_factor: Increasing this number allows computing likelihoods and posteriors chunk-wise in parallel at the cost of memory usage.
    Returns:
        state posterior probbabilities: Shape: (num_model, b, seq_len, q)
    """
    cell.recurrent_init()
    reverse_cell.recurrent_init()
    _, b, seq_len, s = tf.unstack(tf.shape(inputs))
    num_model = cell.num_models
    q = cell.max_num_states
    emission_probs = cell.emission_probs(inputs, end_hints=end_hints, training=training)
    #reshape to equally sizes chunks according to parallel factor
    chunk_size = seq_len // parallel_factor
    emission_probs = tf.reshape(emission_probs, (num_model*b*parallel_factor, chunk_size, q))
    #make the initial states for both passes
    initial_state = cell.get_initial_state(batch_size=b*parallel_factor, parallel_factor=parallel_factor)
    rev_initial_state = reverse_cell.get_initial_state(inputs=emission_probs, batch_size=b*parallel_factor, parallel_factor=parallel_factor)
    #handle the first observation separately to let tf compile a version of the cell call with init=True
    forward_1, forward_step_1_state = cell(emission_probs[:,0], initial_state, training=training, init=True)
    backward_1, backward_step_1_state = reverse_cell(emission_probs[:,-1], rev_initial_state, training=training, init=True)
    #run forward and backward in parallel
    if proper_shape(emission_probs)[1] > 2:
        posterior, *states = bidirectional_rnn(emission_probs[:,1:-1], initial_state=(*forward_step_1_state, *backward_step_1_state), training=training)
    else:
        #posterior as defined here is never used but required to make the tf autograph work
        posterior, states = tf.zeros(()), forward_step_1_state + backward_step_1_state
        if cell.use_step_counter:
            cell.step_counter.assign_add(1)
    #because of the bidirectionality, we also have to manually do the last forward and backward step
    forward_last, final_state = cell(emission_probs[:,-1], states[:2], training=training)
    backward_last, _ = reverse_cell(emission_probs[:,0], states[2:], training=training)
    posterior_1 = tf.stack([forward_1, backward_last], axis=-2) if parallel_factor > 1 else forward_1 + backward_last
    posterior_last = tf.stack([forward_last, backward_1], axis=-2) if parallel_factor > 1 else forward_last + backward_1
    if proper_shape(emission_probs)[1] > 2:
        if parallel_factor > 1:
            posterior = tf.reshape(posterior, (num_model*b*parallel_factor, chunk_size-2, 2, -1))
        posterior = tf.concat([posterior_1[:,tf.newaxis], posterior, posterior_last[:,tf.newaxis]], axis=1)
    else:
        posterior = tf.concat([posterior_1[:,tf.newaxis], posterior_last[:,tf.newaxis]], axis=1)
    if parallel_factor == 1:
        posterior = tf.reshape(posterior, (num_model, b, seq_len, -1))
        loglik = tf.reshape(final_state[1], (num_model, b))
        posterior = posterior[...,:-1] + posterior[..., -1:] 
    else:
        forward_result, loglik = _get_total_forward_from_chunks(posterior[...,0, :], cell, total_prob_rnn, b, seq_len, parallel_factor=parallel_factor)
        backward_result = _get_total_backward_from_chunks(posterior[...,1, :], cell, reverse_cell, total_prob_rnn_rev, 
                                                            b, seq_len, revert_chunks=False, parallel_factor=parallel_factor)
        posterior = forward_result + backward_result
    if not no_loglik:
        posterior -= loglik[:,:,tf.newaxis,tf.newaxis]
    if return_prior:
        prior = cell.get_prior_log_density()
        aux_loss = cell.get_aux_loss()
        return posterior, prior, aux_loss
    else:
        return posterior


tf.keras.utils.get_custom_objects()["MsaHmmLayer"] = MsaHmmLayer