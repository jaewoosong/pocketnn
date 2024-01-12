#include "pktnn_conv.h"

using namespace pktnn;

void pktnn::pktconv::fullConv(pktmat3d& resultMat3d, pktmat3d& image, pktmat3d** filters, int numFilters, pktmat& bias) {
    const int rowPadding = filters[0]->rows() - 1;
    const int colPadding = filters[0]->cols() - 1;
    const int padding = rowPadding < colPadding ? rowPadding : colPadding;
    conv(resultMat3d, image, filters, numFilters, bias, 1, padding);
}

void pktnn::pktconv::fullConvMat(pktmat& resultMat, pktmat& image, pktmat& filter, int bias) {
    const int rowPadding = filter.rows() - 1;
    const int colPadding = filter.cols() - 1;
    const int padding = rowPadding < colPadding ? rowPadding : colPadding;
    convMat(resultMat, image, filter, bias, 1, padding);
}

void pktnn::pktconv::conv(pktmat3d& resultMat3d, pktmat3d& image, pktmat3d** filters, int numFilters, pktmat& bias, int stride, int padding) {
    assert(0 < numFilters);
    const int filterDepth = filters[0]->depth();
    const int filterRows = filters[0]->rows();
    const int filterCols = filters[0]->cols();
    const int imageRows = image.rows();
    const int imageCols = image.cols();
    int resultRows = 0;
    int resultCols = 0;
    calcResultRowCol(&resultRows, &resultCols, imageRows, imageCols, padding, filterRows, filterCols, stride, 1, 1);
    resultMat3d.resetZero3d(numFilters, resultRows, resultCols);

    // convolution (correlation)
    for (int numF = 0; numF < numFilters; ++numF) {
        conv(resultMat3d.getMatAtDepth(numF), image, *(filters[numF]), bias.getElem(0, numF), stride, padding);        
    }
}

void pktnn::pktconv::conv(pktmat& resultMat, pktmat3d& image, pktmat3d& filter, int bias, int stride, int padding) {
    // resultMat is assumed to be well-initialized in advance
    // TODO: refactor to use convMat() inside conv()
    for (int rR = 0; rR < resultMat.rows(); ++rR) {
        for (int rC = 0; rC < resultMat.cols(); ++rC) {
            int result = 0;
            for (int fD = 0; fD < filter.depth(); ++fD) {
                for (int fR = 0; fR < filter.rows(); ++fR) {
                    for (int fC = 0; fC < filter.cols(); ++fC) {
                        int iR = rR + fR - padding;
                        int iC = rC + fC - padding;
                        if ((iR < 0) || (iR >= image.rows()) || (iC < 0) || (iC >= image.cols())) {
                            continue; // zero padding
                        }
                        result += (image.getElem(fD, iR, iC) * (filter.getElem(fD, fR, fC)));
                    }
                }
            }
            result += bias;
            resultMat.setElem(rR, rC, result);
        }
    }
}

void pktnn::pktconv::convMat(pktmat& resultMat, pktmat& image, pktmat& filter, int bias, int stride, int padding) {
    int resultRows = 0;
    int resultCols = 0;
    calcResultRowCol(&resultRows, &resultCols, image.rows(), image.cols(), padding, filter.rows(), filter.cols(), stride, 1, 1);
    resultMat.resetZero(resultRows, resultCols);

    for (int rR = 0; rR < resultMat.rows(); ++rR) {
        for (int rC = 0; rC < resultMat.cols(); ++rC) {
            int result = 0;
            for (int fR = 0; fR < filter.rows(); ++fR) {
                for (int fC = 0; fC < filter.cols(); ++fC) {
                    int iR = rR + fR - padding;
                    int iC = rC + fC - padding;
                    if ((iR < 0) || (iR >= image.rows()) || (iC < 0) || (iC >= image.cols())) {
                        continue; // zero padding
                    }
                    result += (image.getElem(iR, iC) * (filter.getElem(fR, fC)));
                }
            }            
            result += bias;
            resultMat.setElem(rR, rC, result);
        }
    }
}

void pktnn::pktconv::calcResultRowCol(int* pResR, int* pResC, int imgR, int imgC, int pad, int ftrR, int ftrC, int stride, int poolR, int poolC) {
    *pResR = (((imgR + (2 * pad) - ftrR) / stride) + 1) / poolR;
    *pResC = (((imgC + (2 * pad) - ftrC) / stride) + 1) / poolC;
}

pktnn::pktconv::pktconv(int d, int r, int c, int n, int stride, int padding, Pooling poolType, int poolR, int poolC) {
    mLayerType = LayerType::pocket_conv;
    init(d, r, c, n, stride, padding, poolType, poolR, poolC);
}

pktnn::pktconv::~pktconv() {
    for (int i = 0; i < mNumFilters; ++i) {
        delete mFilters[i];
        delete mFiltersUpdate[i];
        mFilters[i] = nullptr;
        mFiltersUpdate[i] = nullptr;
    }
    delete[] mFilters;
    delete[] mFiltersUpdate;
    mFilters = nullptr;
    mFiltersUpdate = nullptr;
}

pktconv& pktconv::init(const int d, const int r, const int c, const int n, const int stride, const int padding,
    Pooling poolType, const int poolR, const int poolC) {
    mFilterDepth = d;
    mFilterRows = r;
    mFilterCols = c;    
    mNumFilters = n;
    mStride = stride;
    mPadding = padding;
    mPoolType = poolType;
    mPoolRows = poolR;
    mPoolCols = poolC;
    mInput = nullptr;
    
    // TODO: possible memory leak???
    mFilters = new pktmat3d* [mNumFilters];
    mFiltersUpdate = new pktmat3d * [mNumFilters];
    for (int i = 0; i < mNumFilters; ++i) {
        mFilters[i] = new pktmat3d(mFilterDepth, mFilterRows, mFilterCols);
        mFiltersUpdate[i] = new pktmat3d(mFilterDepth, mFilterRows, mFilterCols);
    }

    mBias.resetZero(1, mNumFilters); // one bias per one filter
    
    return *this;
}

pktlayer& pktnn::pktconv::forward(pktlayer& x) {
    assert(static_cast<pktlayer*>(this) != &x);
    return forward(x.getOutputForConv());
}

pktconv& pktnn::pktconv::forward(pktmat3d& x) {
    assert(x.depth() == mFilterDepth);
    mInput = &x;
    initEmptyOutput(x);
    forwardConv(x);
    forwardActv();
    forwardPool();

    if (pNextLayer != nullptr) {
        pNextLayer->forward(*this);
    }

    return *this;
}

pktconv& pktnn::pktconv::forwardConv(pktmat3d& x) {
    conv(mConvActv, x, mFilters, mNumFilters, mBias, mStride, mPadding);
    //fullConv(mConvActv, x, mFilters, mNumFilters, mBias);
    return *this;
}

pktconv& pktnn::pktconv::forwardActv() {
    pktactv::activate3d(mConvActv, mConvActv, mActvGradInv3d, mActv, K_BIT);// , mFilterDepth* mFilterRows* mFilterCols);
    return *this;
}

pktconv& pktnn::pktconv::forwardPool() {
    const int outputRows = mConvActv.rows() / mPoolRows;
    const int outputCols = mConvActv.cols() / mPoolCols;
    mOutput.resetZero3d(mNumFilters, outputRows, outputCols);
    mMaxPoolGrad.resetZero3d(mConvActv.depth(), mConvActv.rows(), mConvActv.cols());

    // max pooling only at this moment
    // "VALID" pooling (size shrinks), not "SAME" (size same)
    for (int outD = 0; outD < mNumFilters; ++outD) {
        for (int outR = 0; outR < outputRows; ++outR) {
            for (int outC = 0; outC < outputCols; ++outC) {
                // "neighbours"
                // pktmat type variables are automatically zero-initialized
                int rx = outR * mPoolRows;
                int cy = outC * mPoolCols;
                int result = mConvActv.getElem(outD, rx, cy);
                int maxR = rx;
                int maxC = cy;                
                for (int offsetR = 0; offsetR < mPoolRows; ++offsetR) {
                    for (int offsetC = 0; offsetC < mPoolCols; ++offsetC) {
                        int thisElem = mConvActv.getElem(outD, rx + offsetR, cy + offsetC);
                        if (result < thisElem) {
                            result = thisElem;
                            maxR = rx + offsetR;
                            maxC = cy + offsetC;
                        }
                    }
                }
                mOutput.setElem(outD, outR, outC, result);
                mMaxPoolGrad.setElem(outD, maxR, maxC, 1);
            }
        }
    }

    return *this;
}

pktlayer& pktnn::pktconv::backward(pktmat& lastDeltasMat, int lrInv) {
    // this function is used only when CONV <- FC.
    // I will convert pktmat lastDeltasMat to pktmat3d lastDeltasMat3d.
    computeDeltas(lastDeltasMat);
    // batchSize was used in FC. Not now at Conv.
    int batchSize = mDelta3d.rows(); // this line should be after computeDeltas()
    
    mBiasUpdate.resetZero(1, mNumFilters);
    // mDelta3d.mDepth == mFiltersUpdate.mNumFilters
    // mDelta3d.mRows != mFiltersUpdate.mFilterRows
    // mInput.mDepth == mFiltersUpdate.mDepth
    // ...
    for (int numF = 0; numF < mNumFilters; ++numF) {
        // bias update
        mBiasUpdate.setElem(0, numF, mDelta3d.getMatAtDepth(numF).sum());
        // filters update
        mFiltersUpdate[numF]->resetZero3d(mFilterDepth, mFilterRows, mFilterCols);
        for (int d = 0; d < mFilterDepth; ++d) {
            if (pPrevLayer == nullptr) {
                convMat(mFiltersUpdate[numF]->getMatAtDepth(d),
                    (*mInput).getMatAtDepth(d),
                    mDelta3d.getMatAtDepth(numF), 0);
            }
            else {
                convMat(mFiltersUpdate[numF]->getMatAtDepth(d),
                    (pPrevLayer->getOutputForConv()).getMatAtDepth(d),
                    mDelta3d.getMatAtDepth(numF), 0);
            }
        }
    }

    mBiasUpdate.selfDivConst(-lrInv);
    mBias.matAddMat(mBias, mBiasUpdate);

    for (int numF = 0; numF < mNumFilters; ++numF) {
        mFiltersUpdate[numF]->selfDivConst3d(-lrInv);
        mFilters[numF]->mat3dAddMat3d(*(mFilters[numF]), *(mFiltersUpdate[numF]));
    }

    if (pPrevLayer != nullptr) {
        // backpropagate
        // TODO: need to convert lastDeltasMat into pktmat3d type!!
        (static_cast<pktconv*>(pPrevLayer))->backward(lastDeltasMat, lrInv);
    }
    return *this;
}

pktlayer& pktnn::pktconv::backward(pktmat3d& lastDeltasMat3d, int lrInv) {
    // TODO
    return *this;
}

pktconv& pktnn::pktconv::computeDeltas(pktmat& lastDeltasMat) {
    if (!mDelta3d.dimsEqual(mConvActv)) {
        mDelta3d.resetZero3d(mConvActv.depth(), mConvActv.rows(), mConvActv.cols());
    }

    // delta_l = Upsampling[(w_l+1)^T delta_l+1] elemMul actv'(z_l)

    if (pNextLayer == nullptr) {
        // TODO: this will usually not happen.
        // the last layer: lastDelta is assumed to be lossDelta
        //mDelta3d.deepCopyOf(lastDeltasMat);
    }
    else if (pNextLayer->getLayerType() == LayerType::pocket_fc) {
        // Conv-Pool-FC by flattening
        // need to "Unflatten" w_l+1 * delta_l+1
        mDelta2dTransposeForFc.matMulMat(
            (static_cast<pktfc*>(pNextLayer))->getWeight(),
            (static_cast<pktfc*>(pNextLayer))->getDeltasTranspose()); // (Dk, 1) = (Dk, Dk+1) * (Dk+1, 1)
        mDelta2dForFc.transposeOf(mDelta2dTransposeForFc); // (N, Dk)
        mConvNextFiltersNextDeltas.makeMat3dFromMat(mOutput.depth(), mOutput.rows(), mOutput.cols(), mDelta2dForFc);
        // need to "Unflatten" mDelta2dForFc to mDelta3d! (seems working...)
        // TODO: no need to calculate mDelta3d here?!
        //mDelta3d.makeMat3dFromMat(mConvActv.depth(), mConvActv.rows(), mConvActv.cols(), mDelta2dForFc);
        //mDelta3d.printMat3d();
    }
    else if (pNextLayer->getLayerType() == LayerType::pocket_conv) {
        // Conv-Pool-Conv
        // size initialization
        pktmat oneConv;
        /*
        fullConvMat(oneConv,
            (static_cast<pktconv*>(pNextLayer))->mFilters[0]->getMatAtDepth(0),
            (static_cast<pktconv*>(pNextLayer))->mDelta3dRotate180.getMatAtDepth(0), 0);
        mConvNextFiltersNextDeltas.resetZero3d(mConvActv.depth(), oneConv.rows(), oneConv.cols());
        */
        mConvNextFiltersNextDeltas.resetZero3d(mOutput.depth(), mOutput.rows(), mOutput.cols());

        for (int currDeltaD = 0; currDeltaD < mDelta3d.depth(); ++currDeltaD) { // max < 6
            // for each depth in the current layer
            const int nextNumFilters = (static_cast<pktconv*>(pNextLayer))->mNumFilters;
            for (int numF = 0; numF < nextNumFilters; ++numF) { // max < 4
                // for each filter in the next layer
                // backward delta convolution
                // (curr_filters, next_deltas) are (images, filters) of backward conv
                fullConvMat(oneConv,
                    (static_cast<pktconv*>(pNextLayer))->mFilters[numF]->getMatAtDepth(currDeltaD),
                    //(static_cast<pktconv*>(pNextLayer))->mFilters[currDeltaD]->getMatAtDepth(numF),
                    (static_cast<pktconv*>(pNextLayer))->mDelta3dRotate180.getMatAtDepth(numF), 0);
                // is "ADD" a right choice here?
                // assume that for different next filters....sdlfjsdlfdsf
                assert(mConvNextFiltersNextDeltas.getMatAtDepth(currDeltaD).dimsEqual(oneConv));
                mConvNextFiltersNextDeltas.getMatAtDepth(currDeltaD).selfAddMat(oneConv);
            }
        }
    }
    else {
        // error
    }

    // upsamling
    // TODO:
    //  mConvNextFiltersNextDeltas is (1, 7, 7) (d, r, c)
    //  while mMaxPoolGrad and mUpsampled are (1, 12, 12) (d, r, c),
    //  and mPoolRows, mPoolCols are (2, 2).
    //  So mMaxPoolGrad.getElem(0, 12, sth) causes error.
    //  Moreover, mMaxPoolGrad.getElem(0, sth, 12) is problematic but runs smoothly... alert!
    mUpsampled.resetZero3d(mMaxPoolGrad.depth(), mMaxPoolGrad.rows(), mMaxPoolGrad.cols());
    for (int d = 0; d < mConvNextFiltersNextDeltas.depth(); ++d) {
        for (int r = 0; r < mConvNextFiltersNextDeltas.rows(); ++r) {
            for (int c = 0; c < mConvNextFiltersNextDeltas.cols(); ++c) {
                int rx = r * mPoolRows;
                int cy = c * mPoolCols;
                for (int offsetR = 0; offsetR < mPoolRows; ++offsetR) {
                    for (int offsetC = 0; offsetC < mPoolCols; ++offsetC) {
                        // TODO: error happened at mMaxPoolGrad.getElem(0, 12, 0);
                        int val = mConvNextFiltersNextDeltas.getElem(d, r, c) * mMaxPoolGrad.getElem(d, rx + offsetR, cy + offsetC);
                        mUpsampled.setElem(d, rx + offsetR, cy + offsetC, val);
                    }
                }
            }
        }
    }

    // actv'(z)
    mDelta3d.mat3dElemDivMat3d(mUpsampled, mActvGradInv3d);
    mDelta3dRotate180.rotate180Of(mDelta3d);
    
    return *this;
}

pktconv& pktconv::setRandomWeight() {
    for (int i = 0; i < mNumFilters; ++i) {
        mFilters[i]->setRandom();
    }
    return *this;
}

pktconv& pktconv::setRandomBias() {
    mBias.setRandom(true); // TODO: can bias be zero?
    return *this;
}

pktconv& pktnn::pktconv::setActv(pktactv::Actv actv) {
    mActv = actv;
    return *this;
}

pktconv& pktnn::pktconv::setPadding(int padding) {
    mPadding = padding;
    return *this;
}

pktconv& pktnn::pktconv::initEmptyOutput(pktmat3d& x) {
    //int outputRows = (((x.rows() + (2 * mPadding) - mFilterRows) / mStride) + 1) / mPoolRows;
    //int outputCols = (((x.cols() + (2 * mPadding) - mFilterCols) / mStride) + 1) / mPoolCols;
    int outputRows = 0;
    int outputCols = 0;
    calcResultRowCol(&outputRows, &outputCols, x.rows(), x.cols(), mPadding, mFilterRows, mFilterCols, mStride, mPoolRows, mPoolCols);
    mOutput.resetZero3d(mNumFilters, outputRows, outputCols);
    return *this;
}

pktmat& pktnn::pktconv::getFlattenedOutput() {
    // (d, r, c) to (d, rC + c) because PocketNN uses row-wise vectors.
    int D = mOutput.depth();
    int R = mOutput.rows();
    int C = mOutput.cols();
    mFlattened.resetZero(1, D * R * C);
    for (int d = 0; d < D; ++d) {
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                mFlattened.setElem(0, (d*R*C) + (r*C) + c, mOutput.getElem(d, r, c));
            }
        }
    }
    return mFlattened;
}

pktmat& pktnn::pktconv::getOutputForFc() {
    return getFlattenedOutput();
}

pktmat3d& pktnn::pktconv::getOutputForConv() {
    return mOutput;
}

pktconv& pktnn::pktconv::printBias(std::ostream& outTo) {
    mBias.printMat(outTo);
    return *this;
}

pktconv& pktnn::pktconv::printFilters(std::ostream& outTo) {
    for (int i = 0; i < mNumFilters; ++i) {
        mFilters[i]->printMat3d(outTo);
    }
    return *this;
}

pktconv& pktnn::pktconv::printInter(std::ostream& outTo) {
    mConvActv.printMat3d(outTo);
    return *this;
}

pktconv& pktnn::pktconv::printOutput(std::ostream& outTo) {
    mOutput.printMat3d(outTo);
    return *this;
}
