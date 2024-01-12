#ifndef PKTNN_MAT_H
#define PKTNN_MAT_H

#include <iostream>
#include <random>
#include <limits>
#include <assert.h>
#include "pktnn_tools.h"
#include "pktnn_consts.h"

namespace pktnn {

    class pktmat {
        // TODO: is 'rule of 5' needed?
    private:
        int mRows;
        int mCols;
        int** mMat;
        bool mDeleteOnDestruct;
        std::string mName;
        void fastCopy(const pktmat& t);
        pktmat& initZero(int r, int c);
        
    public:
        pktmat();
        pktmat(int r, int c);
        pktmat(const pktmat& t); // (rule of 3) II. copy constructor
        ~pktmat(); // (rule of 3) I. destructor
        pktmat& operator=(const pktmat& t); // (rule of 3) III. copy assignment

        // important
        pktmat& deleteMat();
        pktmat& resetZero(int r, int c);
        int average();
        int variance();
        int variance(int avg);
        int stdev();
        int stdev(int avg);
        pktmat& averageColwise();
        pktmat& standardize(int numSigma = 2, int low = PKT_MIN, int high = PKT_MAX);
        pktmat& normalizeRowwise(int newMin = PKT_MIN, int newMax = PKT_MAX);
        pktmat& normalizeColwise(int newMin = PKT_MIN, int newMax = PKT_MAX);
        pktmat& normalizeMinMax(int newMin = PKT_MIN, int newMax = PKT_MAX);
        pktmat& clampMat(int low, int high);
        bool dimsEqual(pktmat& mat1);
        bool dimsEqual(int r, int c);

        // getters
        int rows() const;
        int cols() const;
        int sum() const;
        int numElems() const;
        int getRowMin(int row);
        int getRowMax(int row);
        int getColMin(int col);
        int getColMax(int col);
        int** getMat(); // TODO: possible memory leak
        int getElem(int r, int c) const;
        bool getDeleteOnDestruct();
        int getMaxIndexInRow(int r) const;

        // setters
        pktmat& setMat(int r, int c, int* m);
        pktmat& setElem(int r, int c, int val);
        pktmat& setAllConstant(int constant);
        pktmat& setAllConstant(int r, int c, int constant);
        pktmat& resetAllOnes(int r, int c);
        pktmat& setRandom(bool allowZero = false, int minVal = SHRT_MIN, int maxVal = SHRT_MAX);
        pktmat& setName(std::string n);
        pktmat& setDeleteOnDestruct(bool b);
        pktmat& fastReturn();

        pktmat& printMat(std::ostream& outTo = std::cout);

        // mat calculations
        pktmat& matMulMat(pktmat& mat1, pktmat& mat2);
        pktmat& matAddMat(pktmat& mat1, pktmat& mat2);
        pktmat& matMulConst(pktmat& mat1, int con1);
        pktmat& matDivConst(pktmat& mat1, int con1);
        pktmat& matAddConst(pktmat& mat1, int con1);
        pktmat& matElemAddMat(pktmat& mat1, pktmat& mat2);
        pktmat& matElemMulMat(pktmat& mat1, pktmat& mat2);
        pktmat& matElemDivMat(pktmat& mat1, pktmat& mat2);
        pktmat& matMulSelf(pktmat& mat1);
        pktmat& matElemMulSelf(pktmat& mat1);
        pktmat& selfMulConst(int con1);
        pktmat& selfDivConst(int con1);
        pktmat& selfAddConst(int con1);
        pktmat& selfElemAddConst(int r, int c, int con1);
        pktmat& selfMulMat(pktmat& mat2);
        pktmat& selfAddMat(pktmat& mat1);
        pktmat& selfElemMulMat(pktmat& mat1);
        pktmat& selfElemDivMat(pktmat& mat1);
        pktmat& mulGradOf(pktmat& mat1);
        pktmat& transposeOf(pktmat& mat1);
        pktmat& rotate180Of(pktmat& mat1);
        pktmat& squareRootOf(pktmat& mat1);
        pktmat& randomKSamplesOf(pktmat& mat1, int k);
        pktmat& indexedSlicedSamplesOf(pktmat& mat1, int* indices, int start, int end);
        pktmat& sliceOf(pktmat& mat1, int rowStart, int rowEnd, int colStart, int colEnd);
        pktmat& deepCopyOf(const pktmat& mat1);
        pktmat& matUpdateLr(pktmat& update, int lr_inverse);
    };

}
#endif