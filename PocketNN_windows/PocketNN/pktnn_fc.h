#ifndef PKTNN_FC_H
#define PKTNN_FC_H

#include <iostream>
#include <string>
#include "pktnn_layer.h"
#include "pktnn_mat3d.h"
#include "pktnn_mat.h"
#include "pktnn_actv.h"
#include "pktnn_tools.h"
#include "pktnn_consts.h"
#include "pktnn_actv.h"

namespace pktnn {
    class pktfc: public pktlayer {

    private:
        int mInDim;
        int mOutDim;
        pktmat* mInput;
        pktmat mWeight;
        pktmat mBias;
        pktmat mInter;
        pktmat mDeltas; // dJ / dmInter = (dJ / dmOutput) elemMul ReLU'(mInter)
        pktmat mDeltasTranspose;
        pktmat mDActvTranspose;
        pktmat mActvGradInv;
        pktmat mWeightUpdate;
        pktmat mBiasUpdate;

        // batch normalization
        bool mUseBn = false;
        pktmat mMean;
        pktmat mVariance;
        pktmat mStdevWithEps;
        pktmat mStandardized;
        pktmat mGamma;
        pktmat mBeta;
        pktmat mBatchNormalized;
        pktmat mDGamma;
        pktmat mDBeta;
        pktmat mDBn;
        pktmat mGammaUpdate;
        pktmat mBetaUpdate;
        pktfc& batchNormalization();

        // DFA
        bool mUseDfa = true;
        pktmat mDfaWeight;
        
        std::string mName = "fc_noname";
        pktactv::Actv mActv = pktactv::Actv::pocket_tanh;
        pktfc& computeDeltas(pktmat& lastDeltasMat, int lrInv);

    public:
        pktmat mOutput;
        const int& rowss = mInDim;
        const int& colss = mOutDim;

        pktfc(int inDim, int outDim);
        ~pktfc();

        // getters
        pktmat& getOutputForFc();
        pktmat3d& getOutputForConv();
        pktmat& getWeight();
        pktmat& getDeltasTranspose();

        // setters
        pktfc& setName(std::string n);
        pktfc& setRandomWeight();
        pktfc& setRandomBias();
        pktfc& setRandomDfaWeight(int r, int c);
        pktfc& setActv(pktactv::Actv actv);
        pktfc& initHeWeightBias();
        pktfc& useBatchNormalization(bool useBn = true);
        pktfc& useDfa(bool useDfa = true);

        // forward and backward
        pktfc& forward(pktmat& x);
        pktlayer& forward(pktlayer& x);
        pktlayer& backward(pktmat& lastDeltasMat, int lrInv);
        
        // print functions
        pktfc& printWeight(std::ostream& outTo = std::cout);
        pktfc& printBias(std::ostream& outTo = std::cout);
        pktfc& printInter(std::ostream& outTo = std::cout);
        pktfc& printOutput(std::ostream& outTo = std::cout);
    };
}
#endif
