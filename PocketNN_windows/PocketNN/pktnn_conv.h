#ifndef PKTNN_CONV_H
#define PKTNN_CONV_H

#include "pktnn_layer.h"
#include "pktnn_fc.h"
#include "pktnn_mat.h"
#include "pktnn_mat3d.h"
#include "pktnn_consts.h"
#include "pktnn_actv.h"

namespace pktnn {

    class pktconv: public pktlayer {
    public:
        enum class Pooling { max_pool, avg_pool, no_pool };

    private:
        // filters
        int mFilterRows;
        int mFilterCols;
        int mFilterDepth;
        int mNumFilters;

        // forward convolution information
        int mStride;
        int mPadding;

        // pooling
        Pooling mPoolType;
        int mPoolRows;
        int mPoolCols;

        // deltas & backpropagation
        pktmat mDelta2dForFc;
        pktmat mDelta2dTransposeForFc;
        pktmat3d mDelta3d;
        pktmat3d mDelta3dRotate180;
        pktmat3d mConvNextFiltersNextDeltas;
        pktmat3d mUpsampled;
        pktmat mBiasUpdate;
        pktmat3d** mFiltersUpdate;
        // mFilters is 4D. Then mFiltersUpdate also should be 4D!!!

        // gradient
        pktmat3d mMaxPoolGrad;
        pktmat3d mActvGradInv3d;

        // core
        pktmat3d* mInput;
        pktmat3d mConvActv;
        pktmat3d** mFilters; // TODO: How to init??
        pktmat mBias;
        pktactv::Actv mActv = pktactv::Actv::pocket_tanh;

        pktmat mFlattened;

    public:
        pktmat3d mOutput;
        pktconv(int d, int r, int c, int n, int stride = 1, int padding = 0, Pooling poolType = Pooling::max_pool, int poolR = 2, int poolC = 2);
        ~pktconv();
        pktconv& init(int d, int r, int c, int n, int stride, int padding, Pooling poolType, int poolR, int poolC);

        // statics
        static void fullConv(pktmat3d& resultMat3d, pktmat3d& image, pktmat3d** filters, int numFilters, pktmat& bias);
        static void fullConvMat(pktmat& resultMat, pktmat& image, pktmat& filter, int bias);
        static void conv(pktmat3d& resultMat3d, pktmat3d& image, pktmat3d** filters, int numFilters, pktmat& bias, int stride = 1, int padding = 0);
        static void conv(pktmat& resultMat, pktmat3d& image, pktmat3d& filter, int bias, int stride = 1, int padding = 0);
        static void convMat(pktmat& resultMat, pktmat& image, pktmat& filter, int bias, int stride = 1, int padding = 0);
        static void calcResultRowCol(int* pResR, int* pResC, int imgR, int imgC, int pad, int ftrR, int ftrC, int stride, int poolR, int poolC);

        // forward
        pktlayer& forward(pktlayer& x);
        pktconv& forward(pktmat3d& x);
        pktconv& forwardConv(pktmat3d& x);
        pktconv& forwardActv();
        pktconv& forwardPool();

        // backward
        pktlayer& backward(pktmat& lastDeltasMat, int lrInv);
        pktlayer& backward(pktmat3d& lastDeltasMat3d, int lrInv); // TODO!!
        pktconv& computeDeltas(pktmat& lastDeltasMat);

        // getters
        pktmat& getFlattenedOutput();
        pktmat& getOutputForFc();
        pktmat3d& getOutputForConv();

        // setters
        pktconv& setRandomWeight();
        pktconv& setRandomBias();
        pktconv& setActv(pktactv::Actv actv);
        pktconv& setPadding(int padding);

        // tools
        pktconv& initEmptyOutput(pktmat3d& x);

        // print
        pktconv& printBias(std::ostream& outTo = std::cout);
        pktconv& printFilters(std::ostream& outTo = std::cout);
        pktconv& printInter(std::ostream& outTo = std::cout);
        pktconv& printOutput(std::ostream& outTo = std::cout);
        
    };
}

#endif