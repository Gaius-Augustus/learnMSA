import tensorflow as tf
from learnMSA.msa_hmm.Utility import deserialize


class Bidirectional(tf.keras.layers.Layer):
    """Simple bidirectional wrapper for forward and backward RNNs with shared state.

    Args:
        layer: `keras.layers.RNN` instance.
        merge_mode: Mode by which outputs of the forward and backward RNNs
            will be combined. One of `{"sum", "concat"}`.
            If `None`, the outputs will not be combined,
            they will be returned as a list. Defaults to `"concat"`.
        backward_layer: `keras.layers.RNN` instance.
    """

    def __init__(
        self,
        layer,
        merge_mode,
        backward_layer,
        **kwargs,
    ):
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                "Please initialize `Bidirectional` layer with a "
                f"`keras.layers.Layer` instance. Received: {layer}"
            )
        if backward_layer is not None and not isinstance(backward_layer, tf.keras.layers.Layer):
            raise ValueError(
                "`backward_layer` need to be a `keras.layers.Layer` "
                f"instance. Received: {backward_layer}"
            )
        if merge_mode not in ["sum", "concat"]:
            raise ValueError(
                f"Invalid merge mode. Received: {merge_mode}. "
                "Merge mode should be one of "
                '{"sum", "concat"}'
            )
        super().__init__(**kwargs)

        self.forward_layer = layer
        self.backward_layer = backward_layer
        self._verify_layer_config()

        def force_zero_output_for_mask(layer):
            # Force the zero_output_for_mask to be True if returning sequences.
            if getattr(layer, "zero_output_for_mask", None) is not None:
                layer.zero_output_for_mask = layer.return_sequences

        force_zero_output_for_mask(self.forward_layer)
        force_zero_output_for_mask(self.backward_layer)

        self.merge_mode = merge_mode
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.return_state = layer.return_state
        self.supports_masking = True
        self.input_spec = layer.input_spec

    def _verify_layer_config(self):
        """Ensure the forward and backward layers have valid common property."""
        if self.forward_layer.go_backwards == self.backward_layer.go_backwards:
            raise ValueError(
                "Forward layer and backward layer should have different "
                "`go_backwards` value. Received: "
                "forward_layer.go_backwards "
                f"{self.forward_layer.go_backwards}, "
                "backward_layer.go_backwards="
                f"{self.backward_layer.go_backwards}"
            )

        common_attributes = ("stateful", "return_sequences", "return_state")
        for a in common_attributes:
            forward_value = getattr(self.forward_layer, a)
            backward_value = getattr(self.backward_layer, a)
            if forward_value != backward_value:
                raise ValueError(
                    "Forward layer and backward layer are expected to have "
                    f'the same value for attribute "{a}", got '
                    f'"{forward_value}" for forward layer and '
                    f'"{backward_value}" for backward layer'
                )

    def compute_output_shape(self, sequences_shape, initial_state_shape=None):
        output_shape = self.forward_layer.compute_output_shape(sequences_shape)

        if self.return_state:
            output_shape, state_shape = output_shape[0], output_shape[1:]

        if self.merge_mode == "concat":
            output_shape = list(output_shape)
            output_shape[-1] *= 2
            output_shape = tuple(output_shape)
        elif self.merge_mode is None:
            output_shape = [output_shape, output_shape]

        if self.return_state:
            if self.merge_mode is None:
                return tuple(output_shape) + state_shape + state_shape
            return tuple([output_shape]) + (state_shape) + (state_shape)
        return tuple(output_shape)

    def call(
        self,
        sequences,
        initial_state=None,
        mask=None,
        training=None,
    ):
        kwargs = {"training" : training, "mask" : mask}

        if initial_state is not None:
            # initial_states are not keras tensors, eg eager tensor from np
            # array.  They are only passed in from kwarg initial_state, and
            # should be passed to forward/backward layer via kwarg
            # initial_state as well.
            forward_inputs, backward_inputs = sequences, sequences
            half = len(initial_state) // 2
            forward_state = initial_state[:half]
            backward_state = initial_state[half:]
        else:
            forward_inputs, backward_inputs = sequences, sequences
            forward_state, backward_state = None, None

        y = self.forward_layer(
            forward_inputs, initial_state=forward_state, **kwargs
        )
        y_rev = self.backward_layer(
            backward_inputs, initial_state=backward_state, **kwargs
        )

        if self.return_state:
            states = tuple(y[1:] + y_rev[1:])
            y = y[0]
            y_rev = y_rev[0]

        y = tf.cast(y, self.compute_dtype)
        y_rev = tf.cast(y_rev, self.compute_dtype)

        if self.return_sequences:
            y_rev = tf.reverse(y_rev, axis=[1])
        if self.merge_mode == "concat":
            output = tf.concat([y, y_rev], axis=-1)
        elif self.merge_mode == "sum":
            output = y + y_rev
        else:
            raise ValueError(
                "Unrecognized value for `merge_mode`. "
                f"Received: {self.merge_mode}"
                'Expected one of {"concat", "sum", "ave", "mul"}.'
            )
        if self.return_state:
            if self.merge_mode is None:
                return output + states
            return (output,) + states
        return output

    def reset_states(self):
        # Compatibility alias.
        self.reset_state()

    def reset_state(self):
        if not self.stateful:
            raise AttributeError("Layer must be stateful.")
        self.forward_layer.reset_state()
        self.backward_layer.reset_state()

    @property
    def states(self):
        if self.forward_layer.states and self.backward_layer.states:
            return tuple(self.forward_layer.states + self.backward_layer.states)
        return None

    def build(self, sequences_shape, initial_state_shape=None):
        if not self.forward_layer.built:
            self.forward_layer.build(sequences_shape)
        if not self.backward_layer.built:
            self.backward_layer.build(sequences_shape)
        self.built = True

    def compute_mask(self, _, mask):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_sequences:
            if not self.merge_mode:
                output_mask = (mask, mask)
            else:
                output_mask = mask
        else:
            output_mask = (None, None) if not self.merge_mode else None

        if self.return_state and self.states is not None:
            state_mask = [None for _ in self.states]
            if isinstance(output_mask, list):
                return output_mask + state_mask * 2
            return (output_mask,) + tuple(state_mask * 2)
        return output_mask

    def get_config(self):
        config = {"merge_mode": self.merge_mode,
                  "layer": self.forward_layer,
                  "backward_layer": self.backward_layer }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config["layer"] = deserialize(config["layer"])
        config["backward_layer"] = deserialize(config["backward_layer"])
        layer = cls(**config)
        return layer