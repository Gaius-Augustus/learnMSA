from setuptools import setup, find_packages

setup(
    name="learnMSA",
    version="1.0",
    url="https://github.com/Ung0d/MSA-HMM",
    author="Felix Becker",
    author_email="beckerfelix94@gmail.com",
    description="learnMSA: Learning and Aligning large Protein Families",
    packages=find_packages(
        where=".",
        include=["msa_hmm"]
    ),
    install_requires=["tensorflow>=2.5.0",
                      "networkx",
                      "logomaker" ],
    include_package_data=True,
    package_data={'': ["trained_prior/*/*", "trained_prior/transition_priors/*/*"]},
    entry_points={
        "console_scripts": [
            "learnMSA = msa_hmm:learnMSA", ] }
)