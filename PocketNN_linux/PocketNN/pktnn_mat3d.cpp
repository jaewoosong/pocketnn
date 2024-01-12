#include "pktnn_mat3d.h"

using namespace pktnn;

pktnn::pktmat3d::pktmat3d() {
    initZero3d(0, 0, 0);
}

pktnn::pktmat3d::pktmat3d(int d, int r, int c) {
    initZero3d(d, r, c);
}

pktmat3d::~pktmat3d() {
    if (mDeleteOnDestruct) {
        deleteMat3d();
    }
}

pktmat3d& pktmat3d::initZero3d(int d, int r, int c) {
    mDepth = maxVal(0, d);
    mRows = maxVal(0, r);
    mCols = maxVal(0, c);    
    mMat3d = new pktmat* [mDepth];
    for (int i = 0; i < mDepth; ++i) {
        mMat3d[i] = new pktmat(mRows, mCols);
    }
    mDeleteOnDestruct = true;
    return *this;
}

pktmat3d& pktmat3d::deleteMat3d() {
    for (int i = 0; i < mDepth; ++i) {
        //mMat3d[i]->deleteMat();
        delete mMat3d[i]; // should use this?
        mMat3d[i] = nullptr;
    }
    delete[] mMat3d;
    mMat3d = nullptr;
    return *this;
}

pktmat3d& pktnn::pktmat3d::resetZero3d(int d, int r, int c) {
    deleteMat3d();
    initZero3d(d, r, c);
    return *this;
}

int pktnn::pktmat3d::rows() {
    return mRows;
}

int pktnn::pktmat3d::cols() {
    return mCols;
}

int pktnn::pktmat3d::depth() {
    return mDepth;
}

int pktnn::pktmat3d::getElem(int d, int r, int c) {
    return mMat3d[d]->getElem(r, c);
}

pktmat& pktnn::pktmat3d::getMatAtDepth(int d) {
    assert((0 <= d) && (d < mDepth));
    return *(mMat3d[d]);
}

bool pktnn::pktmat3d::dimsEqual(pktmat3d& mat1) {
    return ((mDepth == mat1.mDepth) && (mRows == mat1.mRows) && (mCols == mat1.mCols));
}

bool pktnn::pktmat3d::dimsEqual(int d, int r, int c) {
    return ((mDepth == d) && (mRows == r) && (mCols == c));
}

pktmat3d& pktnn::pktmat3d::normalizeMinMax3d(int newMin, int newMax) {
    for (int d = 0; d < mDepth; ++d) {
        mMat3d[d]->normalizeMinMax(newMin, newMax);
    }
    return *this;
}

pktmat3d& pktnn::pktmat3d::mat3dAddMat3d(pktmat3d& mat3d1, pktmat3d& mat3d2) {
    assert(mat3d1.dimsEqual(mat3d2));
    // BUG: EVERYTHING BECOMES ZERO HERE!!
    if (!dimsEqual(mat3d1)) {
        // if mat3d1 is equal to *this, this if statment will not be executed.
        resetZero3d(mat3d1.mDepth, mat3d1.mRows, mat3d1.mCols);
    }
    
    for (int d = 0; d < mDepth; ++d) {
        mMat3d[d]->matAddMat(mat3d1.getMatAtDepth(d), mat3d2.getMatAtDepth(d));
    }
    return *this;
}

pktmat3d& pktnn::pktmat3d::mat3dElemDivMat3d(pktmat3d& mat3d1, pktmat3d& mat3d2) {
    assert(mat3d1.dimsEqual(mat3d2));
    resetZero3d(mat3d1.mDepth, mat3d1.mRows, mat3d1.mCols);
    for (int d = 0; d < mDepth; ++d) {
        mMat3d[d]->matElemDivMat(mat3d1.getMatAtDepth(d), mat3d2.getMatAtDepth(d));
    }
    return *this;
}

pktmat3d& pktnn::pktmat3d::setElem(int d, int r, int c, int val) {
    mMat3d[d]->setElem(r, c, val);
    return *this;
}

pktmat3d& pktnn::pktmat3d::setRandom(bool allowZero, int minVal, int maxVal) {
    for (int d = 0; d < mDepth; ++d) {
        for (int r = 0; r < mRows; ++r) {
            for (int c = 0; c < mCols; ++c) {
                mMat3d[d]->setElem(r, c, randomRange(minVal, maxVal));
                if (!allowZero) {
                    while (mMat3d[d]->getElem(r, c) == 0) {
                        mMat3d[d]->setElem(r, c, randomRange(minVal, maxVal));
                    }
                }
            }
        }
    }
    return *this;
}

pktmat3d& pktnn::pktmat3d::selfAddMat3d(pktmat3d& mat3d1) {
    mat3dAddMat3d(*this, mat3d1);
    return *this;
}

pktmat3d& pktnn::pktmat3d::selfDivConst3d(int const1) {
    for (int d = 0; d < mDepth; ++d) {
        mMat3d[d]->selfDivConst(const1);
    }
    return *this;
}

pktmat3d& pktnn::pktmat3d::selfElemMulMat3d(pktmat3d& mat3d1) {
    assert(dimsEqual(mat3d1));
    for (int d = 0; d < mDepth; ++d) {
        mMat3d[d]->selfElemMulMat(*(mat3d1.mMat3d[d]));
    }
    return *this;
}

pktmat3d& pktnn::pktmat3d::selfElemDivMat3d(pktmat3d& mat3d1) {
    assert(dimsEqual(mat3d1));
    for (int d = 0; d < mDepth; ++d) {
        mMat3d[d]->selfElemDivMat(*(mat3d1.mMat3d[d]));
    }
    return *this;
}

pktmat3d& pktnn::pktmat3d::rotate180Of(pktmat3d& mat3d1) {
    resetZero3d(mat3d1.mDepth, mat3d1.mRows, mat3d1.mCols);
    for (int d = 0; d < mDepth; ++d) {
        mMat3d[d]->rotate180Of(*(mat3d1.mMat3d[d]));
    }
    return *this;
}

pktmat3d& pktnn::pktmat3d::deepCopyOf(const pktmat3d& mat3d1) {
    resetZero3d(mat3d1.mDepth, mat3d1.mRows, mat3d1.mCols);
    for (int d = 0; d < mDepth; ++d) {
        mMat3d[d]->deepCopyOf(*(mat3d1.mMat3d[d]));
    }
    return *this;
}

pktmat3d& pktnn::pktmat3d::makeMat3dFromMat(const int D, const int R, const int C, pktmat& mat1) {
    assert(D * R * C == mat1.numElems());
    resetZero3d(D, R, C);

    for (int d = 0; d < D; ++d) {
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                int idx = d * (R + C) + r * C + c;
                mMat3d[d]->setElem(r, c, mat1.getElem(idx / mat1.cols(), idx % mat1.cols()));
            }
        }
    }

    return *this;
}

pktmat3d& pktnn::pktmat3d::printMat3d(std::ostream& outTo) {
    outTo << "Matrix3d\n";
    for (int i = 0; i < mDepth; ++i) {
        mMat3d[i]->printMat(outTo);
        outTo << "\n";
    }
    return *this;
}

