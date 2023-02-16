# Overview
This repository stores programs related to EMNIST[^EMNIST] that were created by the author during his studies.

# Treatment
After pulling this repository, you can perform feature extraction and machine learning from EMNIST by running each python file in python3.

# Detail
## Roles
<dl>
    <dt>README.md</dt>
    <dd>This file with a description of the repository.</dd>
    <dt>Summary_EMNIST.py</dt>
    <dd>Program files used to actually know the contents of EMNIST.</dd>
    <dt>FeatureExtraction_EMNIST.py</dt>
    <dd>Program files to perform feature extraction for EMNIST.Unprocessed pixel values and average pixel values per mesh can be extracted.We use an external library, emnist[^emnist], to download EMNIST.</dd>
    <dt>kNN_EMNIST.py</dt>
    <dd>Program file to apply k Nearest Neighbor[^sklearn.neighbors.KNeighborsClassifier] to the features extracted by FeatureExtraction_EMNIST.py.</dd>
    <dt>feature_extraction</dt>
    <dd>A directory of program files to implement each feature extraction method.</dd>
    <dt>EMNIST</dt>
    <dd>Directory for storing the results of feature extraction and machine learning.</dd>
</dl> 

## Common Specifications
* When executing each program file, passing "help" as the second command line argument will result in the standard output of a description of each program file and details of the command line arguments.

# Execution Environment
<dl>
    <dt>OS</dt>
    <dd>Ubuntu 20.04 LTS</dd>
    <dt>Python Version</dt>
    <dd>Python 3.8.10</dd>
</dl> 

# References
[^EMNIST]: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373

[^sklearn.neighbors.KNeighborsClassifier]: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

[^emnist]: (https://pypi.org/project/emnist/)