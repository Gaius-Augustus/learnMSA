"""Parse a Dirichlet mixture parameter file.

File format
-----------
Line 1:  <num_chars>  <num_components>
Lines 2…(num_components+1):
    <mixture_weight>  <alpha_1>  …  <alpha_num_chars>

Returns
-------
mixture_weights : np.ndarray, shape (num_components,)
concentrations  : np.ndarray, shape (num_components, num_chars)
"""

import argparse

import numpy as np

from learnMSA.hmm.tf.util import make_dirichlet_model


def parse_dirichlet(path: str):
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]

    num_chars, num_components = map(int, lines[0].split())

    data = []
    for line in lines[1:num_components + 1]:
        vals = list(map(float, line.split()))
        if len(vals) != num_chars + 1:
            raise ValueError(
                f"Expected {num_chars + 1} values per row, got {len(vals)}"
            )
        data.append(vals)

    data = np.array(data)                      # (num_components, num_chars+1)
    mixture_weights = data[:, 0]               # (num_components,)
    concentrations  = data[:, 1:]              # (num_components, num_chars)

    weight_sum = mixture_weights.sum()
    if not np.isclose(weight_sum, 1.0, atol=1e-4):
        print(f"Warning: mixture weights sum to {weight_sum:.6f}, not 1.0")

    return mixture_weights, concentrations


WEIGHTS_PATH = "learnMSA/hmm/weights/"


def build_dirichlet_model(
    name: str,
    mixture_weights: np.ndarray,
    concentrations: np.ndarray,
) -> None:
    """Build a TFDirichletPrior model from parsed Dirichlet mixture
    parameters and save its weights.

    Args:
        name: The name for the weight file (without extension).
        mixture_weights: Mixture weights of shape ``(C,)``.
        concentrations: Concentration parameters of shape ``(C, D)``.
    """
    num_components, num_chars = concentrations.shape

    if num_components == 1:
        # Single component: initializer is just the concentrations
        initializer = concentrations.flatten()
    else:
        # Multi-component: [conc_comp1, ..., conc_compC, mixture_weights]
        initializer = np.concatenate(
            [concentrations.flatten(), mixture_weights]
        )

    model = make_dirichlet_model(
        initializer=initializer,
        dim=num_chars,
        components=num_components,
    )
    prior = model.layers[1]

    # Verify round-trip: matrix concentrations match the input
    matrix = prior.matrix().numpy()  # (1, 1, P)
    recovered_conc = matrix[0, 0, :num_components * num_chars]
    np.testing.assert_allclose(
        recovered_conc.reshape(num_components, num_chars),
        concentrations,
        atol=1e-5,
    )

    model_path = WEIGHTS_PATH + name + ".weights.h5"
    model.save_weights(model_path)
    print(f"Saved weights to {model_path}")

    # Verify the saved weights can be loaded back
    model2 = make_dirichlet_model(
        dim=num_chars, components=num_components
    )
    model2.load_weights(model_path)
    np.testing.assert_allclose(
        model2.layers[1].matrix().numpy(), matrix, atol=1e-7
    )
    print("Round-trip verification passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Parse a Dirichlet mixture parameter file."
    )
    parser.add_argument("input", help="Path to the parameter file")
    parser.add_argument(
        "-n", "--name",
        help="Name for the weight file (without extension). "
             "If provided, builds and saves a TFDirichletPrior model to "
             f"{WEIGHTS_PATH}<name>.weights.h5",
    )
    args = parser.parse_args()

    weights, concentrations = parse_dirichlet(args.input)

    print(f"num_components : {weights.shape[0]}")
    print(f"num_chars      : {concentrations.shape[1]}")
    print(f"\nmixture_weights  shape={weights.shape}  sum={weights.sum():.6f}")
    print(weights)
    print(f"\nconcentrations  shape={concentrations.shape}")
    print(concentrations)

    if args.name:
        build_dirichlet_model(args.name, weights, concentrations)


if __name__ == "__main__":
    main()
