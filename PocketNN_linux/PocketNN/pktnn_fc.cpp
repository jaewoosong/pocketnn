#include "pktnn_fc.h"

using namespace pktnn;

pktfc::pktfc(int inDim, int outDim):
    // these sizes are certain
    mWeight(inDim, outDim), mBias(1, outDim), mWeightUpdate(inDim, outDim), mBiasUpdate(1, outDim) {
    // these sizes are dynamic: depend on the input batch size N
    // mInter, mOutput, mDeltas, mActvGradInv: (N, outDim)
    pPrevLayer = nullptr;
    pNextLayer = nullptr;
    mInDim = inDim;
    mOutDim = outDim;
    mLayerType = LayerType::pocket_fc;
    mInput = nullptr;
}

pktfc::~pktfc() {
    
}

pktmat& pktnn::pktfc::getOutputForFc() {
    return mOutput;
}

pktmat3d& pktnn::pktfc::getOutputForConv() {
    // TODO: no need at this stage. Just a placeholder.
    return mDummy3d;
}

pktmat& pktnn::pktfc::getWeight() {
    return mWeight;
}

pktmat& pktnn::pktfc::getDeltasTranspose() {
    return mDeltasTranspose;
}

// setters

pktfc& pktfc::setName(std::string n) {
    mName = n;
    mWeight.setName(n + "_weight");
    mBias.setName(n + "_bias");
    mOutput.setName(n + "_output");

    return *this;
}

pktfc& pktfc::setRandomWeight() {
    mWeight.setRandom();
    return *this;
}

pktfc& pktfc::setRandomBias() {
    mBias.setRandom(true);
    return *this;
}

pktfc& pktnn::pktfc::setRandomDfaWeight(int r, int c) {
    // Using He initialization at this time.
    // Maybe other randomization can work better.
    std::cout << "Initialized DFA!\n";
    mDfaWeight.resetZero(r, c);
    int range = floorSqrt((12 * SHRT_MAX) / (mInDim + mOutDim));
    mDfaWeight.setRandom(false, -range, range);
    return *this;
}

pktfc& pktnn::pktfc::setActv(pktactv::Actv actv) {
    mActv = actv;
    return *this;
}

pktfc& pktnn::pktfc::initHeWeightBias() {
    int range = 0;
    switch (mActv) {
    case pktactv::Actv::pocket_relu8bit:
    case pktactv::Actv::pocket_leakyrelu:
        range = floorSqrt((12 * SHRT_MAX) / (mInDim + mOutDim));
        std::cout << "He: " << range << "\n";
        mWeight.setRandom(false, -range, range);
        mBias.setRandom(false, -range, range);
        break;
    case pktactv::Actv::pocket_tanh:
        range = floorSqrt((12 * SHRT_MAX) / (mInDim + mOutDim));
        std::cout << "He: " << range << "\n";
        mWeight.setRandom(false, -range, range);
        mBias.setRandom(false, -range, range);
        // TODO
        break;
    case pktactv::Actv::pocket_sigmoid:
        range = 0;
        // TODO
        break;
    default:
        range = 0;
        break;
    }
    return *this;
}

pktfc& pktnn::pktfc::useBatchNormalization(bool useBN) {
    mUseBn = useBN;
    return *this;
}

pktfc& pktnn::pktfc::useDfa(bool useDfa) {
    mUseDfa = useDfa;
    if (useDfa) {
        mWeight.setAllConstant(0);
        mBias.setAllConstant(0);
    }
    return *this;
}

pktfc& pktfc::forward(pktmat& xMat) {
    mInput = &xMat;
    mInter.matMulMat(*mInput, mWeight); // (N, Dk) = (N, Dk-1) * (Dk-1, Dk)
    if (mUseBn) {
        batchNormalization();
        pktactv::activate(mOutput, mBatchNormalized, mActvGradInv, mActv, K_BIT, mInDim); // TODO: mInDim needed?
    }
    else {
        mInter.selfAddMat(mBias); // (N, Dk)
        pktactv::activate(mOutput, mInter, mActvGradInv, mActv, K_BIT, mInDim); // TODO: mInDim needed?
    }
    
    if (pNextLayer != nullptr) {
        if (pNextLayer->getLayerType() == LayerType::pocket_fc) {
            (static_cast<pktfc*>(pNextLayer))->forward(*this);
        }
        else {
            // TODO: in what situation can this line be reached?
        }
        
    }

    return *this;
}

pktlayer& pktnn::pktfc::forward(pktlayer& x) {
    assert(static_cast<pktlayer*>(this) != &x);
    return forward(x.getOutputForFc());
}

// backward
pktlayer& pktnn::pktfc::backward(pktmat& lastLayerDeltasMat, int lrInv) {
    computeDeltas(lastLayerDeltasMat, lrInv); // (N, Dlast)
    // this line should be after computeDeltas()
    int batchSize = mDeltas.rows(); // 1 for one item
    //mDeltasDivGradInv.matDivConst(mDeltas, -lrInv); // this might make everything zero
    
    pktmat prevOutputTranspose;
    if (pPrevLayer == nullptr) {
        // first hidden layer
        prevOutputTranspose.transposeOf(*mInput); // (D1, N)
    }
    else {
        prevOutputTranspose.transposeOf(pPrevLayer->getOutputForFc()); // (Dk-1, N)
    }

    // update, TRY: fix mDeltas before here
    // mWeightUpdate and mBiasUpdate:
    // the SUM OF individual updates by each datum
    // ** TODO: I think I don't have to divide them by batchSize because
    //          each individual vector's direction will be quite different
    //          and cancel each other. So if I divide them by batchSize,
    //          then they will easily become zero??

    // what if overflow occurs when there are many parallel data?
    mWeightUpdate.matMulMat(prevOutputTranspose, mDeltas); // (Dk-1, Dk) = (Dk-1, N) * (N, Dk)
    mWeightUpdate.selfDivConst((-lrInv));
    mWeight.selfAddMat(mWeightUpdate);

    if (mUseBn) {
        mGammaUpdate.matDivConst(mDGamma, -lrInv);
        mBetaUpdate.matDivConst(mDBeta, -lrInv);
        mGamma.selfAddMat(mGammaUpdate);
        mBeta.selfAddMat(mBetaUpdate);
    }
    else {
        pktmat allOneMat;
        allOneMat.resetAllOnes(1, batchSize); // (1, N)
        mBiasUpdate.matMulMat(allOneMat, mDeltas);//.printMat(); // (1, Dk) = (1, N) * (N, Dk)
        mBiasUpdate.selfDivConst((-lrInv));
        mBias.selfAddMat(mBiasUpdate);
    }

    // weight and bias upper bound
    mWeight.clampMat(SHRT_MIN + 1, SHRT_MAX);
    mBias.clampMat(SHRT_MIN + 1, SHRT_MAX);

    if (pPrevLayer != nullptr) {
        pPrevLayer->backward(lastLayerDeltasMat, lrInv);
    }

    return *this;
}

pktfc& pktfc::computeDeltas(pktmat& lastLayerDeltasMat, int lrInv) {
    // with batch normalization
    if (mUseBn) {
        // need d_Y (mDBn), d_gamma, d_beta to calculate mDeltas
        
        // step 1: d_Y (mDBn)
        if (pNextLayer == nullptr) {
            // the last layer: lastDelta is assumed to be lossDelta
            mDBn.matElemDivMat(lastLayerDeltasMat, mActvGradInv); // (N, Dlast) == (N, Dk)
            //lastLayerDeltasMat.printMat();
        }
        else {
            mDActvTranspose.matMulMat((static_cast<pktfc*>(getNextLayer()))->mWeight,
                (static_cast<pktfc*>(getNextLayer()))->mDeltasTranspose); // (Dk, N) = (Dk, Dk+1) * (Dk+1, N)
            mDBn.transposeOf(mDActvTranspose); // (N, Dk)
            mDBn.selfElemDivMat(mActvGradInv);
        }

        // step 2 & 3: d_gamma, d_beta
        const int numItems = mDBn.rows(); // N
        const int featureDims = mDBn.cols(); // Dk
        mDGamma.resetZero(1, featureDims);
        mDBeta.resetZero(1, featureDims);

        for (int c = 0; c < featureDims; ++c) {
            for (int r = 0; r < numItems; ++r) {
                mDGamma.selfElemAddConst(0, c, mDBn.getElem(r, c) * mStandardized.getElem(r, c));
                mDBeta.selfElemAddConst(0, c, mDBn.getElem(r, c));
            }
        }

        // step 4: mDeltas: THE COMPLICATED formula is here!
        // TODO: maybe delaying DIV to the end can make better result
        pktmat gammaStdev;
        gammaStdev.matElemDivMat(mGamma, mStdevWithEps); // (1, Dk)

        pktmat dGammaXhat;
        dGammaXhat.matElemMulMat(mDGamma, mStandardized); // (1, Dk) elemMul (N, Dk)
        dGammaXhat.selfMulConst(-1);

        pktmat dYtimesN;
        dYtimesN.matMulConst(mDBn, numItems); // (N, Dk)

        pktmat oneColumnVec;
        oneColumnVec.resetAllOnes(numItems, 1);

        pktmat dBetaMatrix;
        dBetaMatrix.matMulMat(oneColumnVec, mDBeta); // (N, 1) * (1, Dk) = (N, Dk)
        dBetaMatrix.selfMulConst(-1);

        mDeltas.resetZero(numItems, featureDims); // (N, Dk)
        mDeltas.matAddMat(dGammaXhat, dYtimesN); // (N, Dk) + (N, Dk)
        mDeltas.selfAddMat(dBetaMatrix); // (N, Dk)
        mDeltas.matElemMulSelf(gammaStdev); // (1, Dk) Hadamard* (N, Dk)
        mDeltas.selfDivConst(numItems);
    }
    // without batch normalization
    else {
        if (pNextLayer == nullptr) {
            // the last layer: lastDelta is assumed to be lossDelta
            mDeltas.matElemDivMat(lastLayerDeltasMat, mActvGradInv);
        }
        else {
            // hidden or the first layer
            if (mUseDfa) {
                // Direct Feedback Alignment
                if (!mDfaWeight.dimsEqual(lastLayerDeltasMat.cols(), mWeight.cols())) {
                    setRandomDfaWeight(lastLayerDeltasMat.cols(), mWeight.cols()); // (Dlast, Dk)
                }
                mDeltas.matMulMat(lastLayerDeltasMat, mDfaWeight); // (N, Dk) = (N, Dlast) * (Dlast, Dk)
                mDeltas.selfElemDivMat(mActvGradInv);
            }
            else {
                // Vanilla Gradient Descent
                mDActvTranspose.matMulMat((static_cast<pktfc*>(getNextLayer()))->mWeight,
                    (static_cast<pktfc*>(getNextLayer()))->mDeltasTranspose); // (Dk, N) = (Dk, Dk+1) * (Dk+1, N)
                mDeltas.transposeOf(mDActvTranspose); // (N, Dk)
                mDeltas.selfElemDivMat(mActvGradInv);
            }
        }
    }

    // --- NEW: average the deltas! ---
    //if (mDeltas.rows() > 1) {
    //    mDeltas.averageColwise();
    //}
    // --- End averaging ---
    
    mDeltasTranspose.transposeOf(mDeltas);
    return *this;
}

pktfc& pktnn::pktfc::batchNormalization() {
    // calculates means and variances of each dimension
    // and then standardize the input matrix X into X_hat
    const int numItems = mInter.rows();
    const int featureDims = mInter.cols();

    mMean.resetZero(1, featureDims);
    mVariance.resetZero(1, featureDims);

    // calculate means
    for (int c = 0; c < featureDims; ++c) {
        for (int r = 0; r < numItems; ++r) {
            mMean.setElem(0, c, mMean.getElem(0, c) + mInter.getElem(r, c));
        }
    }
    mMean.selfDivConst(numItems);

    // calculate variances & stdevs
    for (int c = 0; c < featureDims; ++c) {
        int thisMean = mMean.getElem(0, c);
        for (int r = 0; r < numItems; ++r) {
            int devi = mInter.getElem(r, c) - thisMean;
            mVariance.setElem(0, c, mVariance.getElem(0, c) + (devi * devi));
        }
    }
    mVariance.selfDivConst(numItems);
    mStdevWithEps.squareRootOf(mVariance);

    // add epsilon to stdev
    for (int c = 0; c < featureDims; ++c) {
        if (mStdevWithEps.getElem(0, c) == 0) {
            // avoid divide-by-zero (eps in the original paper)
            mStdevWithEps.setElem(0, c, 1);
        }
    }

    // batchNormalization
    mStandardized.resetZero(numItems, featureDims);
    for (int c = 0; c < featureDims; ++c) {
        int thisMean = mMean.getElem(0, c);
        int thisStdev = mStdevWithEps.getElem(0, c);
        for (int r = 0; r < numItems; ++r) {
            // 0 or 1 is too discrete for PocketNN
            // multiply by PKT_MAX (127) to make it useful
            int stddzd8bit = (PKT_MAX * (mInter.getElem(r, c) - thisMean)) / thisStdev;
            mStandardized.setElem(r, c, stddzd8bit);
        }
    }

    // gamma, beta initialization when necessary (first run)
    if (!mGamma.dimsEqual(1, featureDims)) {
        mGamma.resetAllOnes(1, featureDims);
    }

    if (!mBeta.dimsEqual(1, featureDims)) {
        mBeta.resetZero(1, featureDims);
    }

    mBatchNormalized.resetZero(numItems, featureDims);
    for (int c = 0; c < featureDims; ++c) {
        const int gamma = mGamma.getElem(0, c);
        const int beta = mBeta.getElem(0, c);
        // for each item
        for (int r = 0; r < numItems; ++r) {
            const int xHat = mStandardized.getElem(r, c);
            mBatchNormalized.setElem(r, c, gamma * xHat + beta);
        }
    }
    
    return *this;
}

// print

pktfc& pktfc::printWeight(std::ostream& outTo) {
    outTo << "Weight's average: " << mWeight.average() << "\n";
    mWeight.printMat(outTo);
    return *this;
}

pktfc& pktfc::printBias(std::ostream& outTo) {
    outTo << "Bias\n";
    mBias.printMat(outTo);
    return *this;
}

pktfc& pktfc::printInter(std::ostream& outTo) {
    outTo << "Inter\n";
    mInter.printMat(outTo);
    return *this;
}

pktfc& pktnn::pktfc::printOutput(std::ostream& outTo) {
    outTo << "Output: \n";
    mOutput.printMat(outTo);
    return *this;
}
