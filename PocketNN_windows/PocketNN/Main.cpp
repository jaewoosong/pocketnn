#include "pktnn_examples.h"

int main() {
    // There are 3 examples
    // 1) very simple integer BP: example_fc_int_bp_very_simple();
    // 2) fc mnist integer DFA: example_fc_int_dfa_mnist();
    // 3) fc fashion mnist integer DFA: example_fc_int_dfa_fashion_mnist();

    // CAUTION:
    // MNIST and Fashion-MNIST datasets should be downloaded in advance
    // and put into ./dataset/ directory with following filenames.
    
    // (1) MNIST (http://yann.lecun.com/exdb/mnist/)
    //  Already downloaded in advance because MNIST website allows making copies.
    //  ("Please refrain from accessing these files from automated scripts with high frequency. Make copies!")
    //     ./dataset/mnist/train-labels.idx1-ubyte
    //     ./dataset/mnist/train-images.idx3-ubyte
    //     ./dataset/mnist/t10k-labels.idx1-ubyte
    //     ./dataset/mnist/t10k-images.idx3-ubyte
    
    // (2) Fashion-MNIST (https://github.com/zalandoresearch/fashion-mnist)
    //  Already downloaded in advance because Fashion-MNIST uses MIT License which allows distribution.
    //     ./dataset/fashion_mnist/train-labels-idx1-ubyte
    //     ./dataset/fashion_mnist/train-images-idx3-ubyte
    //     ./dataset/fashion_mnist/t10k-labels-idx1-ubyte
    //     ./dataset/fashion_mnist/t10k-images-idx3-ubyte
    // (Please download the files from the link.)

    example_fc_int_dfa_mnist();
    return 0;
}
