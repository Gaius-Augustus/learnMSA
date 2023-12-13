import tensorflow as tf
import learnMSA.protein_language_models.Common as common


class SymmetricBilinearReduction(tf.keras.layers.Layer):

    def __init__(self, 
                 reduced_dim, 
                 dropout=0.0, 
                 trainable=True,
                 activation=tf.nn.softmax,
                 scaled=True):
        super(SymmetricBilinearReduction, self).__init__()
        self.reduced_dim = reduced_dim
        self.dropout_prob = dropout
        self.trainable = trainable
        self.activation = activation
        self.scaled = scaled

    def build(self, input_shape):
        self.L2 = 0 #0.001 / (self.reduced_dim * input_shape[-1])
        self.R = self.add_weight(
            shape=(input_shape[-1], self.reduced_dim),
            initializer="random_normal",
            regularizer=tf.keras.regularizers.L2(self.L2),
            trainable=self.trainable,
            name="R")
        self.b = self.add_weight(shape=(1), initializer=tf.constant_initializer(-3), trainable=self.trainable, name="b")
        self.dropout = tf.keras.layers.Dropout(self.dropout_prob)
        
    def _reduce(self, embeddings, training):
        embeddings = self.dropout(embeddings, training=training)
        reduced_emb = tf.matmul(embeddings, self.R) #(..., reduced_dim)
        if self.scaled: #scores should have rougly variance 1
            reduced_emb /= tf.math.sqrt(tf.cast(tf.shape(self.R)[0], reduced_emb.dtype))
        return reduced_emb
        
    def call(self, embeddings_a, embeddings_b, a_is_reduced=False, b_is_reduced=False, training=None, activate_output=True, use_bias=True):
        """ Computes the probability that two embeddings are homolog.
        Args:
            embeddings_a: A 2D-tensor of shape (..., k1, embedding_dim).
            embeddings_b: A 2D-tensor of shape (..., k2, embedding_dim).
            a_is_reduced: Indicates whether the first embedding is already of reduced dimension.
            b_is_reduced: Indicates whether the first embedding is already of reduced dimension.
        Returns:
            Activated dot-product scores for all pairs of embeddings of shape (..., k1, k2). Unactivated if activate_output is False.
        """
        reduced_emb_a = embeddings_a if a_is_reduced else self._reduce(embeddings_a, training)
        reduced_emb_b = embeddings_b if b_is_reduced else self._reduce(embeddings_b, training)
        scores = tf.matmul(reduced_emb_a, reduced_emb_b, transpose_b=True) 
        if self.scaled: 
            #scores should have rougly variance 1
            #assuming that a neuron in the reduced embeddings also have roughly variance 1, since we scaled them
            scores /= tf.math.sqrt(tf.cast(self.reduced_dim, scores.dtype))
        if use_bias:
            scores += self.b
        #make non-padding positions not contribute to the attention distribution
        mask = tf.reduce_all(embeddings_b == 0, axis=-1)
        mask = tf.expand_dims(mask, -2)
        scores -= 1e9 * tf.cast(mask, embeddings_b.dtype)
        if activate_output:
            return self.activation(scores)
        else:
            return scores

    def get_config(self):
        return {"reduced_dim": self.reduced_dim, "dropout" : self.dropout}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
class BackgroundEmbedding(tf.keras.layers.Layer):

    def __init__(self, reduction_layer):
        super(BackgroundEmbedding, self).__init__()
        reduction_layer.trainable = False
        self.reduction_layer = reduction_layer
    
    def build(self, input_shape):
        self.background_embedding = self.add_weight(
            shape=(self.reduction_layer.reduced_dim),
            initializer="zeros",
            name="background_embedding")
        
    def call(self, embedding, training=None):
        return self.reduction_layer(embedding, self.background_embedding, b_is_reduced=True, training=False)

    
def make_scoring_model(config : common.ScoringModelConfig, dropout=0.0, trainable=False):
    # string mapping to allowed functions for conveniece
    if config.activation == "softmax":
        act = tf.nn.softmax
    elif config.activation == "sigmoid":
        act = tf.math.sigmoid
    else:
        act = config.activation
    emb1 = tf.keras.layers.Input(shape=(None, common.dims[config.lm_name]))
    emb2 = tf.keras.layers.Input(shape=(None, common.dims[config.lm_name]))
    # outputs are homology probabilities 
    output = SymmetricBilinearReduction(config.dim,
                                        dropout, 
                                        trainable=trainable, 
                                        activation=act,
                                        scaled=config.scaled)(emb1, emb2, activate_output=True)
    # construct a model and compile for a standard binary classification task
    model = tf.keras.models.Model(inputs=[emb1, emb2], outputs=output)
    return model
    