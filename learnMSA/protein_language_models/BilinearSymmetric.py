import tensorflow as tf


class SymmetricBilinearReduction(tf.keras.layers.Layer):

    def __init__(self, 
                 reduced_dim, 
                 dropout=0.0, 
                 use_attention_scores=False,
                 l2=0.0):
        super(SymmetricBilinearReduction, self).__init__()
        self.reduced_dim = reduced_dim
        self.dropout_prob = dropout
        self.l2 = l2
        self.use_attention_scores = use_attention_scores

    def build(self, input_shape):
        self.R = self.add_weight(
            shape=(input_shape[-1], self.reduced_dim),
            initializer="random_normal",
            regularizer=tf.keras.regularizers.L2(self.l2),
            name="R")
        self.b = self.add_weight(shape=(1), initializer="zeros", name="b")
        self.dropout = tf.keras.layers.Dropout(self.dropout_prob)
        
    def _reduce(self, embeddings, training, softmax_on_reduce_dim=False):
        embeddings = self.dropout(embeddings, training=training)
        reduced_emb = tf.matmul(embeddings, self.R) #(..., reduced_dim)
        if softmax_on_reduce_dim:
            return tf.nn.softmax(reduced_emb)
        else:
            return reduced_emb
        
    def call(self, embeddings_a, embeddings_b, a_is_reduced=False, b_is_reduced=False, training=None, activate_output=True, softmax_on_reduce_dim=False, use_bias=True):
        """ Computes the probability that two embeddings are homolog.
        Args:
            embeddings_a: A 2D-tensor of shape (..., k1, embedding_dim).
            embeddings_b: A 2D-tensor of shape (..., k2, embedding_dim).
            a_is_reduced: Indicates whether the first embedding is already of reduced dimension.
            b_is_reduced: Indicates whether the first embedding is already of reduced dimension.
        Returns:
            Probabilities of shape (..., k1, k2).
            In case of use_attention_scores the last dimension will sum to 1 over all non-padding positions in embeddings_b.
            Otherwise the output will contain independent probabilities (sigmoid activation) for all pairs.
        """
        reduced_emb_a = embeddings_a if a_is_reduced else self._reduce(embeddings_a, training, softmax_on_reduce_dim)
        reduced_emb_b = embeddings_b if b_is_reduced else self._reduce(embeddings_b, training, softmax_on_reduce_dim)
        scores = tf.matmul(reduced_emb_a, reduced_emb_b, transpose_b=True) 
        if use_bias:
            scores += self.b
        if not self.use_attention_scores:
            if activate_output:
                return tf.math.sigmoid(scores) 
            else:
                return scores
        else:
            #make non-padding positions not contribute to the attention distribution
            mask = tf.reduce_all(embeddings_b == 0, axis=-1)
            mask = tf.expand_dims(mask, -2)
            scores -= 1e9 * tf.cast(mask, embeddings_b.dtype)
            if activate_output:
                return tf.nn.softmax(scores)
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

    
def make_scoring_model(emb_dim, reduced_dim, dropout):
    emb1 = tf.keras.layers.Input(shape=(None, emb_dim))
    emb2 = tf.keras.layers.Input(shape=(None, emb_dim))
    # outputs are homology probabilities 
    output = SymmetricBilinearReduction(reduced_dim,
                                        dropout, 
                                        use_attention_scores = True)(emb1, emb2, activate_output=True, softmax_on_reduce_dim=False)
    # construct a model and compile for a standard binary classification task
    model = tf.keras.models.Model(inputs=[emb1, emb2], outputs=output)
    return model
    