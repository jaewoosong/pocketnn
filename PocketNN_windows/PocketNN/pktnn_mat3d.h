#ifndef PKTNN_MAT3D_H
#define PKTNN_MAT3D_H
#include <assert.h>
#include "pktnn_tools.h"
#include "pktnn_mat.h"

namespace pktnn {

    class pktmat3d {

    private:
        int mRows;
        int mCols;
        int mDepth;
        pktmat** mMat3d;
        bool mDeleteOnDestruct;
        pktmat3d& initZero3d(int d, int r, int c);
        pktmat3d& deleteMat3d();

    public:
        pktmat3d();
        pktmat3d(int d, int r, int c);
        ~pktmat3d();
        pktmat3d& resetZero3d(int d, int r, int c);
        int rows();
        int cols();
        int depth();
        int getElem(int d, int r, int c);
        pktmat& getMatAtDepth(int d);
        bool dimsEqual(pktmat3d& mat1);
        bool dimsEqual(int d, int r, int c);
        pktmat3d& normalizeMinMax3d(int newMin = PKT_MIN, int newMax = PKT_MAX);
        pktmat3d& mat3dAddMat3d(pktmat3d& mat3d1, pktmat3d& mat3d2);
        pktmat3d& mat3dElemDivMat3d(pktmat3d& mat3d1, pktmat3d& mat3d2);
        pktmat3d& setElem(int d, int r, int c, int val);
        pktmat3d& setRandom(bool allowZero = false, int minVal = SHRT_MIN, int maxVal = SHRT_MAX);
        pktmat3d& selfAddMat3d(pktmat3d& mat3d1);
        pktmat3d& selfDivConst3d(int const1);
        pktmat3d& selfElemMulMat3d(pktmat3d& mat3d1);
        pktmat3d& selfElemDivMat3d(pktmat3d& mat3d1);
        pktmat3d& rotate180Of(pktmat3d& mat3d1);
        pktmat3d& makeMat3dFromMat(const int d, const int r, const int c, pktmat& mat1);
        pktmat3d& deepCopyOf(const pktmat3d& mat3d1);
        
        pktmat3d& printMat3d(std::ostream &outTo = std::cout);
    };

}

#endif
