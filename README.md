# PocketNN
This is an official, proof-of-concept C++ implementation of the paper [PocketNN: Integer-only Training and Inference of Neural Networks via Direct Feedback Alignment and Pocket Activations in Pure C++](https://arxiv.org/abs/2201.02863). The paper was accepted in [TinyML 2022](https://www.tinyml.org/event/summit-2022/) as a full paper.

## The very first run
Just run the `main.cpp` file to see training and testing a PocketNN network with the MNIST dataset! Other sample usages are written in `pktnn_examples.cpp` file.

<img width="640" alt="demo screen capture animation" src="./demo_capture.gif">

## Notes
I used Visual Studio 2019 to write this code. Visual Studio solution file is included in the repository to help importing the project.

## Citing PocketNN
Citation information will be updated soon.

## Presentation video
Please click the image below to watch a youtube video which was recorded at tinyML 2022.

[![A presentation video recorded at tinyML 2022.](http://img.youtube.com/vi/Gcx-b92iXlI/0.jpg)](https://www.youtube.com/watch?v=Gcx-b92iXlI)

## License
PocketNN uses the MIT License. For details, please see the `LICENSE` file.

## Sample datasets
Two sample datasets are copied from their original website.
- MNIST dataset: MNIST dataset is from [the MNIST website](http://yann.lecun.com/exdb/mnist/). The site says "Please refrain from accessing these files from automated scripts with high frequency. Make copies!" So I made the copies and put them in this repository.
- Fashion-MNIST dataset: Fashion-MNIST dataset is from [its github repository](https://github.com/zalandoresearch/fashion-mnist). It follows the MIT License which allows copy and distribution.
