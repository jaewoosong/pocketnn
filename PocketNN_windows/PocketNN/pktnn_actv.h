#ifndef PKTNN_ACTV_H
#define PKTNN_ACTV_H

#include "pktnn_mat.h"
#include "pktnn_mat3d.h"
#include "pktnn_consts.h"

namespace pktnn {
    class pktactv {
    private:
        static int clamp(int input, const int minVal, const int maxVal);
    public:
        enum class Actv {
            pocket_sigmoid,
            pocket_tanh,
            rescale,
            pocket_softmax,
            pocket_relu8bit,
            pocket_leakyrelu,
            plu,
            as_is
        };

        // wrapper
        static void activate(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, Actv actv = Actv::pocket_tanh, int k = K_BIT, int numItems = 1);
        static void activate3d(pktmat3d& mat3dOut, pktmat3d& mat3dIn, pktmat3d& matActvGradInv, Actv actv = Actv::pocket_tanh, int k = K_BIT, int numItems = 1);
        
        // 2d
        static void pocketSigmoid(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k);
        static void pocketTanh(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k, int numItems = 1);
        static void rescale(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k);
        static void pocketSoftmax(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k);
        static void pocketReLU8Bit(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k);
        static void pocketLeakyReLU(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k);
        static void plu(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k);
        static void asIs(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k);

        // 3d
        static void pocketSigmoid3d(pktmat3d& mat3dOut, pktmat3d& mat3dIn, pktmat3d& matActvGradInv, int k);
        static void pocketTanh3d(pktmat3d& mat3dOut, pktmat3d& mat3dIn, pktmat3d& matActvGradInv, int k, int numItems = 1);
        static void rescale3d(pktmat3d& mat3dOut, pktmat3d& mat3dIn, pktmat3d& matActvGradInv, int k);
    };
}

#endif