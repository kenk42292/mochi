
#MOCHI

##v1.0

# Summary

Mochi is a highly configurable, flexible neural network. The neural net can be configured through a user-defined XML file.
This configuration file can be used to configure settings that include:

* Neural net input and output dimensions.

* Each layer and their properties. Layer properties include appropriate attributes for each layer type, including:

    * Layer type
    * Layer input and output dimensions
    * Layer optimizer

* Input data and type.

The project is highly dependent on two external libraries:

*   Armadillo: The Armadillo library is used extensively in processing mathematical computations efficiently and quickly. The library depends on BLAS and LAPACK, or their high-performance replacements: MKL, OpenBLAS.

   * Note: Be sure to use Armadillo v7.600.2 or above. Versions below this have a memory-leak issue with arma::field, which is used throughout the project.

> Conrad Sanderson and Ryan Curtin.

> Armadillo: a template-based C++ library for linear algebra.
 
> Journal of Open Source Software, Vol. 1, pp. 26, 2016.


* pugixml: The pugixml library is used to parse and process the user-provided configuration xml file.

> The pugixml webpage can be found at http://pugixml.org. pugixml is Copyright (C) 2006-2015 Arseny Kapoulkine.


# Design

To achieve the utmost user control over the neural net configuration, each fine-grained mathematical operator was assigned a layer, resulting in layers for each of convolutional, matrix multiplication, sigmoid, softmax, and other such operations.

The following design was created to meet the goals of speed and configurability (presented here as a UML class-diagram).

The design is expected to change (perhaps significantly) over time.


![If UML class diagram image is not visible, the same diagram as a png file is availabe under the project root folder as mochi-uml-classdiagram.png](mochi-uml-classdiagram.png) 

# Testing
The project is tested via the eclipse plug-in for CUTE (C++ Unit Testing Easier). To run the unit-tests in Eclipse, create a build configuration excluding the main mochi.cpp file. Note that building the project for running requires creating a build configuration excluding the main Test.cpp file.
