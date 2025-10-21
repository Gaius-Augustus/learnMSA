import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import Initializer

import learnMSA.msa_hmm.DirichletMixture as dm
from learnMSA.msa_hmm.MSA2HMM import PHMMTransitionIndexSet, PHMMValueSet
from learnMSA.msa_hmm.SequenceDataset import SequenceDataset
from learnMSA.msa_hmm.Utility import (LG_paml, deserialize, inverse_softplus,
                                      parse_paml)


class EmissionInitializer(Initializer):
    """
    Initializer that broadcasts a given kernel to the desired shape.

    Args:
        dist (np.ndarray): The initial distribution to be broadcasted. Should
            at least match the emitter kernel size in the last dimension.
            Additional dimensions will be broadcasted to match the desired shape.
    """

    def __init__(self, dist : np.ndarray) -> None:
        self.dist = dist

    def __call__(
            self,
            shape : tf.TensorShape | tuple[int, ...] | list[int | None],
            dtype : tf.DType | None = None,
            **kwargs
    ) -> tf.Tensor:
        if dtype is None:
            dtype = tf.float32
        assert shape[-1] == self.dist.size,\
            "Last dimension of shape must match the size of the initial "\
            f"distribution. Shape={shape} dist.size={self.dist.size}"

        dist = tf.cast(self.dist, dtype)
        # Fit to shape by tiling
        return tf.broadcast_to(dist, shape)

    def __repr__(self) -> str:
        return f"EmissionInitializer()"

    def get_config(self) -> dict:
        return {"dist": self.dist.tolist()}

    @classmethod
    def from_config(cls, config : dict) -> "EmissionInitializer":
        return cls(np.array(config["dist"]))


class ConstantInitializer(tf.keras.initializers.Constant):

    def __init__(self, value):
        super(ConstantInitializer, self).__init__(value)

    def __repr__(self):
        if np.isscalar(self.value):
            return f"Const({self.value})"
        elif isinstance(self.value, list):
            return f"Const(size={len(self.value)})"
        else:
            return f"Const(shape={self.value.shape})"

    def get_config(self):  # To support serialization
        return {"value": self.value.tolist() if isinstance(self.value, np.ndarray) else self.value}

    @classmethod
    def from_config(cls, config):
        return cls(np.array(config["value"]))



R, p = parse_paml(LG_paml, SequenceDataset.alphabet[:-1])
exchangeability_init = inverse_softplus(R + 1e-32).numpy()


prior_path = os.path.dirname(__file__)+"/trained_prior/"
model_path = prior_path+"_".join([str(1), "True", "float32", "_dirichlet.h5"])
model = dm.load_mixture_model(model_path, 1, 20, trainable=False, dtype=tf.float32)
dirichlet = model.layers[-1]
background_distribution = dirichlet.expectation()
#the prior was trained on example distributions over the 20 amino acid alphabet
#the additional frequencies for 'B', 'Z',  'X', 'U', 'O' were derived from Pfam
extra = [7.92076933e-04, 5.84256792e-08, 1e-32]
background_distribution = np.concatenate([background_distribution, extra], axis=0)
background_distribution /= np.sum(background_distribution)

def make_default_anc_probs_init(num_models):
    exchangeability_stack = np.stack([exchangeability_init]*num_models, axis=0)
    log_p_stack = np.stack([np.log(p)]*num_models, axis=0)
    exchangeability_stack = np.expand_dims(exchangeability_stack, axis=1) #"k" in AncProbLayer
    log_p_stack = np.expand_dims(log_p_stack, axis=1) #"k" in AncProbLayer
    return [ConstantInitializer(-3),
            ConstantInitializer(exchangeability_stack),
            ConstantInitializer(log_p_stack)]

def make_default_emission_init():
    return EmissionInitializer(np.log(background_distribution))


def make_default_insertion_init():
    return ConstantInitializer(np.log(background_distribution))


class EntryInitializer(Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        #choose such that entry[0] will always be ~0.5 independent of model length
        p0 = tf.zeros([1]+[d for d in shape[1:]], dtype=dtype)
        p = tf.cast(tf.repeat(tf.math.log(1/(shape[0]-1)), shape[0]-1), dtype=dtype)
        return tf.concat([p0, p], axis=0)

    def __repr__(self):
        return f"DefaultEntry()"


class ExitInitializer(Initializer):
    def __call__(self, shape, dtype=None, **kwargs):
        #choose such that all exit probs equal the probs entry[i] for i > 0
        return tf.zeros(shape, dtype=dtype) + tf.cast(tf.math.log(0.5/(shape[0]-1)), dtype=dtype)

    def __repr__(self):
        return f"DefaultExit()"


class MatchTransitionInitializer(Initializer):
    def __init__(self, val, i, scale):
        self.val = val
        self.i = i
        self.scale = scale

    def __call__(self, shape, dtype=None, **kwargs):
        val = tf.constant(self.val, dtype=dtype)[tf.newaxis,:]
        z = tf.random.normal(shape, stddev=self.scale, dtype=dtype)[:,tf.newaxis]
        val_z = val + z
        p_exit_desired = 0.5 / (shape[0]-1)
        prob = (tf.nn.softmax(val_z) * (1-p_exit_desired))[:,self.i]
        return tf.math.log(prob)

    def __repr__(self):
        return f"DefaultMatchTransition({self.val[self.i]})"

    def get_config(self):  # To support serialization
        return {"val": self.val, "i": self.i, "scale": self.scale}


class RandomNormalInitializer(Initializer):
    def __init__(self, mean=0.0, stddev=0.05):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype if dtype != None else tf.float32)

    def __repr__(self):
        return f"Norm({self.mean}, {self.stddev})"

    def get_config(self):  # To support serialization
        return {"mean": self.mean, "stddev": self.stddev}


def make_default_flank_init():
    return ConstantInitializer(0.)


def make_default_transition_init(MM=1,
                                 MI=-1,
                                 MD=-1,
                                 II=-0.5,
                                 IM=0,
                                 DM=0,
                                 DD=-0.5,
                                 FC=0,
                                 FE=-1,
                                 R=-9,
                                 RF=0,
                                 T=0,
                                 scale=0.1):
    transition_init_kernel = {
        "begin_to_match" : EntryInitializer(),
        "match_to_end" : ExitInitializer(),
        "match_to_match" : MatchTransitionInitializer([MM, MI, MD], 0, scale),
        "match_to_insert" : MatchTransitionInitializer([MM, MI, MD], 1, scale),
        "insert_to_match" : RandomNormalInitializer(IM, scale),
        "insert_to_insert" : RandomNormalInitializer(II, scale),
        "match_to_delete" : MatchTransitionInitializer([MM, MI, MD], 2, scale),
        "delete_to_match" : RandomNormalInitializer(DM, scale),
        "delete_to_delete" : RandomNormalInitializer(DD, scale),
        "left_flank_loop" : RandomNormalInitializer(FC, scale),
        "left_flank_exit" : RandomNormalInitializer(FE, scale),
        "right_flank_loop" : RandomNormalInitializer(FC, scale),
        "right_flank_exit" : RandomNormalInitializer(FE, scale),
        "unannotated_segment_loop" : RandomNormalInitializer(FC, scale),
        "unannotated_segment_exit" : RandomNormalInitializer(FE, scale),
        "end_to_unannotated_segment" : RandomNormalInitializer(R, scale),
        "end_to_right_flank" : RandomNormalInitializer(RF, scale),
        "end_to_terminal" : RandomNormalInitializer(T, scale) }
    return transition_init_kernel


@dataclass
class PHMMInitializerSet:
    """ Initializer collection for a pHMM.
    """
    match_emissions : list[Initializer]
    insert_emissions : list[Initializer]
    transitions : list[dict[str, Initializer]]
    start : list[Initializer]


def make_initializers_from(
    values: PHMMValueSet,
    num_models: int = 1,
    random_scale: float = 0.0,
    emission_kernel_extra : np.ndarray | None = None,
) -> PHMMInitializerSet:
    """
    Builds initializers from a given PHMMValueSet.

    Args:
        values (PHMMValueSet): The PHMMValueSet containing the initial values
            (should be log probabilities).
        num_models (int): The number of models to create initializers for.
        random_scale (float): The scale of the random noise to add to the
            initial values.
        emission_kernel_extra (np.ndarray | None): If provided, this array will
            be broadcasted and concatenated to the emission initializers as
            additional dimensions.
    """
    # Gather transition values
    ind = PHMMTransitionIndexSet(values.matches())
    def _get_vec(indices: np.ndarray) -> np.ndarray:
        return values.transitions[indices[:, 0], indices[:, 1]]
    def _get_scalar(index: np.ndarray) -> np.ndarray:
        return np.array([values.transitions[index[0], index[1]]])
    transition_values = {
        "begin_to_match": _get_vec(ind.begin_to_match),
        "match_to_end": _get_vec(ind.match_to_end),
        "match_to_match": _get_vec(ind.match_to_match),
        "match_to_insert": _get_vec(ind.match_to_insert),
        "insert_to_match": _get_vec(ind.insert_to_match),
        "insert_to_insert": _get_vec(ind.insert_to_insert),
        "match_to_delete": np.concatenate([
            _get_scalar(ind.begin_to_delete[0]),
            _get_vec(ind.match_to_delete)
        ]),
        "delete_to_match": np.concatenate([
            _get_vec(ind.delete_to_match), [0.0]
        ]),
        "delete_to_delete": _get_vec(ind.delete_to_delete),
        "left_flank_loop": _get_scalar(ind.left_flank[0]),
        "left_flank_exit": _get_scalar(ind.left_flank[1]),
        "right_flank_loop": _get_scalar(ind.right_flank[0]),
        "right_flank_exit": _get_scalar(ind.right_flank[1]),
        "unannotated_segment_loop": _get_scalar(ind.unannotated[0]),
        "unannotated_segment_exit": _get_scalar(ind.unannotated[1]),
        "end_to_unannotated_segment": _get_scalar(ind.end[0]),
        "end_to_right_flank": _get_scalar(ind.end[1]),
        "end_to_terminal": _get_scalar(ind.end[2])
    }

    # Start distribution value
    start_value = values.start[0] - np.log(1 - np.exp(values.start[0]))

    # Slightly randomize each model's initializers
    def _add_noise(arr: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, random_scale, arr.shape)
        return arr + noise

    # Emission initializers
    match_emissions = []
    insert_emissions = []
    transitions = []
    start = []
    for _ in range(num_models):
        match_emission = _add_noise(values.match_emissions)
        insert_emission = _add_noise(values.insert_emissions)
        if emission_kernel_extra is not None:
            match_extra_broadcasted = np.broadcast_to(
                emission_kernel_extra,
                match_emission.shape[:-1] + emission_kernel_extra.shape
            )
            match_emission = np.concatenate(
                [match_emission, match_extra_broadcasted],
                axis=-1
            )
            insert_extra_broadcasted = np.broadcast_to(
                emission_kernel_extra,
                insert_emission.shape[:-1] + emission_kernel_extra.shape
            )
            insert_emission = np.concatenate(
                [insert_emission, insert_extra_broadcasted],
                axis=-1
            )
        match_emissions.append(ConstantInitializer(match_emission))
        insert_emissions.append(ConstantInitializer(insert_emission))
        transitions.append({
            k: ConstantInitializer(_add_noise(v))
            for k, v in transition_values.items()
        })
        start.append(ConstantInitializer(_add_noise(start_value)))

    return PHMMInitializerSet(
        match_emissions=match_emissions,
        insert_emissions=insert_emissions,
        transitions=transitions,
        start=start
    )


tf.keras.utils.get_custom_objects()["EmissionInitializer"] = EmissionInitializer
tf.keras.utils.get_custom_objects()["ConstantInitializer"] = ConstantInitializer
tf.keras.utils.get_custom_objects()["EntryInitializer"] = EntryInitializer
tf.keras.utils.get_custom_objects()["ExitInitializer"] = ExitInitializer
tf.keras.utils.get_custom_objects()["MatchTransitionInitializer"] = MatchTransitionInitializer
tf.keras.utils.get_custom_objects()["RandomNormalInitializer"] = RandomNormalInitializer