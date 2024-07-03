from setuptools import setup, find_packages

version_file_path = "learnMSA/_version.py"
with open(version_file_path, "rt") as version_file:
    version = version_file.readlines()[0].split("=")[1].strip(' "')
    
setup(
    name="learnMSA",
    version=version,
    url="https://github.com/Gaius-Augustus/learnMSA",
    author="Felix Becker",
    author_email="beckerfelix94@gmail.com",
    description="learnMSA: Learning and Aligning large Protein Families",
    packages=find_packages(
        where=".",
        include=["learnMSA", "learnMSA.run", "learnMSA.msa_hmm", "learnMSA.protein_language_models"]
    ),
    install_requires=["tensorflow>=2.5.0,<2.11",
                      "tensorflow_probability<=0.18.0",
                      "networkx",
                      "logomaker", 
                      "seaborn",
                      "biopython>=1.69",
                      "pyfamsa", 
                      "transformers",
                      "imageio",
                      "SentencePiece",
                      "numpy<2.0"],
    include_package_data=True,
    package_data={'': ["msa_hmm/trained_prior/*/*", 
                        "msa_hmm/trained_prior/transition_priors/*/*",
                        "protein_language_models/new_scoring_models_frozen/*/*",
                        "protein_language_models/priors_V3/*/*"]},
    license="MIT",
    license_files = ("LICENSE.md"),
    entry_points={
        "console_scripts": [
            "learnMSA = learnMSA.run:run_main", ] }
)