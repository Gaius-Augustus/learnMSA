from setuptools import setup, find_packages

setup(
    name="learnMSA",
    version="1.0.1",
    url="https://github.com/Gaius-Augustus/learnMSA",
    author="Felix Becker",
    author_email="beckerfelix94@gmail.com",
    description="learnMSA: Learning and Aligning large Protein Families",
    packages=find_packages(
        where=".",
        include=["learnMSA", "learnMSA.run", "learnMSA.msa_hmm"]
    ),
    install_requires=["tensorflow>=2.5.0",
                      "networkx",
                      "logomaker" ],
    include_package_data=True,
    package_data={'': ["msa_hmm/trained_prior/*/*", "msa_hmm/trained_prior/transition_priors/*/*"]},
    license="MIT",
    license_files = ("LICENSE.md"),
    entry_points={
        "console_scripts": [
            "learnMSA = learnMSA.run:run_main", ] }
)