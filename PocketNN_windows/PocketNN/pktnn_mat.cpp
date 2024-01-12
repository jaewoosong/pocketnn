#include "pktnn_mat.h"

// calling a constructor in another constructor caused unexpected behaviour:
// https://isocpp.org/wiki/faq/ctors#init-methods

using namespace pktnn;

pktmat& pktmat::initZero(int r, int c) {
    mRows = 0 < r ? r : 0;
    mCols = 0 < c ? c : 0;
    mMat = new int* [mRows];
    for (int i = 0; i < mRows; ++i) {
        // cppreference: value_initialization to zero by using ()
        mMat[i] = new int[mCols]();
    }
    mDeleteOnDestruct = true;
    if (mName == "") {
        mName = "mat_noname";
    }

    return *this;
}

pktmat& pktmat::deleteMat() {
    for (int i = 0; i < mRows; ++i) {
        delete[] mMat[i];
        mMat[i] = nullptr;
    }
    delete[] mMat;
    mMat = nullptr;

    mRows = 0;
    mCols = 0;

    return *this;
}

pktmat& pktmat::resetZero(int r, int c) {
    deleteMat();
    initZero(r, c);
    return *this;
}

int pktmat::average() {
    if ((mRows == 0) || (mCols == 0)) {
        return 0;
    }

    int sum_all = 0;
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            sum_all += mMat[r][c];
        }
    }

    return sum_all / (mRows * mCols);
}

int pktmat::variance() {
    return variance(average());
}

int pktmat::variance(int avg) {
    if ((mRows == 0) || (mCols == 0)) {
        return 0;
    }

    // TODO: How to avoid overflow? check before square?
    // Bypass: use long or unsigned int?
    int sum_var = 0;
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            sum_var += ((mMat[r][c] - avg) * (mMat[r][c] - avg));
        }
    }

    return sum_var / (mRows * mCols);
}

int pktmat::stdev() {
    return stdev(average());
}

int pktmat::stdev(int avg) {
    return floorSqrt(variance(avg));
}

pktmat& pktnn::pktmat::averageColwise() {
    pktmat tempMat(1, mCols);
    for (int c = 0; c < mCols; ++c) {
        for (int r = 0; r < mRows; ++r) {
            tempMat.selfElemAddConst(0, c, mMat[r][c]);
        }
    }
    tempMat.selfDivConst(mRows);

    deepCopyOf(tempMat);
    return *this;
}

/*
Consider Y = XW + B, where X, W and B are all sampleSize bits.
Then approximately (informal proof),
(1) Xi, Wi, Bi: sampleSize bits
(2) XiWi: sampleSize bits * sampleSize bits = 2sampleSize bits
(3) sum_i (XiWi): log (n * 2^2sampleSize) = (log n) + 2sampleSize bits
(4) sum_i (XiWi) + Bi: log (n * 2^2sampleSize + 2^sampleSize)
                       = log (2^sampleSize ( n*2^sampleSize + 1))
                       = sampleSize + log (n*2^sampleSize + 1)
                       < sampleSize + log ((n+1)(2^sampleSize)) (for sampleSize >= 1)
                       = 2sampleSize + log (n+1) bits
    (e.g., if n == 1023, sum_i (XiWi) + Bi is 2sampleSize + 10 bits)
(5) sampleSize = 8 is fine because 2sampleSize + log (n+1) is less than 32 bits
    sampleSize = 16 is not good because 2sampleSize + log (n+1) is greater than 32 bits
(Therefore) set sampleSize = 8.
*/

pktmat& pktmat::standardize(int numSigma, int low, int high) {
    // 1, 2, 3 sigma: 68, 95, 99.7%
    if (((mRows == 0) || (mCols == 0)) || (mRows * mCols == 1)) {
        return *this;
    }

    // linear mapping in range (avg - sampleSize*std, avg + sampleSize*std)
    // y = ax (no +b!) passes
    // (avg - sampleSize*std, MIN) and (avg + sampleSize*std, MAX)
    // a: (MAX - MIN) / (2sampleSize*std)

    int avg = average();
    int std = stdev(avg);
    int nume = high - low;
    int deno = 2 * numSigma * std;
    if (deno <= 0) {
        deno = 1; // TODO: add 1 is workaround
    }

    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            // parentheses are IMPORTANT!
            // (nume / deno) might be zero.
            mMat[r][c] = (mMat[r][c] * nume) / deno;
        }
    }

    return *this;// clampValue(low, high);
}

pktmat& pktnn::pktmat::normalizeRowwise(int newMin, int newMax) {
    for (int r = 0; r < mRows; ++r) {
        int rowMin = getRowMin(r);
        int rowMax = getRowMax(r);
        int a1000 = (1000 * (newMax - newMin)) / (rowMax - rowMin);
        int b1000 = (1000 * newMax) - (a1000 * rowMax);
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] = (a1000 * mMat[r][c] + b1000) / 1000;
        }
    }

    return *this;
}

pktmat& pktnn::pktmat::normalizeColwise(int newMin, int newMax) {
    for (int c = 0; c < mCols; ++c) {
        int colMin = getColMin(c);
        int colMax = getColMax(c);
        int a1000 = (1000 * (newMax - newMin)) / (colMax - colMin);
        int b1000 = (1000 * newMax) - (a1000 * colMax);
        for (int r = 0; r < mRows; ++r) {
            mMat[r][c] = (a1000 * mMat[r][c] + b1000) / 1000;
        }
    }

    return *this;
}

pktmat& pktnn::pktmat::normalizeMinMax(int newMin, int newMax) {
    int min = PKT_MAX;
    int max = PKT_MIN;
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            int thisVal = mMat[r][c];
            if (thisVal < min) {
                min = thisVal;
            }
            if (thisVal > max) {
                max = thisVal;
            }
        }
    }

    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] = (mMat[r][c] - min) / (max - min);
        }
    }

    return *this;
}

pktmat& pktmat::clampMat(int low, int high) {
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] = mMat[r][c] < low ? low : mMat[r][c];
            mMat[r][c] = mMat[r][c] > high ? high : mMat[r][c];
        }
    }

    return *this;
}

bool pktmat::dimsEqual(pktmat& mat1) {
    return ((mRows == mat1.mRows) && (mCols == mat1.mCols));
}

bool pktnn::pktmat::dimsEqual(int r, int c) {
    return ((mRows == r) && (mCols == c));
}

void pktmat::fastCopy(const pktmat& t) {
    // TODO: change this function for moving functionality (delete t after moving)
    mRows = t.mRows;
    mCols = t.mCols;
    mMat = t.mMat;
    mDeleteOnDestruct = true; // important!!

    if ((mName == "") || (mName == "mat_noname")) {
        // "": directly created via fastCopy()
        mName = t.mName + "_copy";
    }
    else {
        mName = "abcdefg"; // This line should be unreachable??
    }
}

pktmat::pktmat() {
    initZero(0, 0);
}

pktmat::pktmat(int r, int c) {
    initZero(r, c);
}

pktmat::pktmat(const pktmat& t) {
    deepCopyOf(t);
}

pktmat::~pktmat() {
    // TODO: Can copy elision be problematic in my code??
    deleteMat();
    if (mDeleteOnDestruct) {
        //deleteMat();
    }
    else {
        //std::cout << "__" + mName + ": DO NOT Delete " << mMat << "\n";
    }
}

pktmat& pktmat::operator=(const pktmat& t) {
    // TODO: didn't check. need to confirm the soundness of the code.
    if (this != &t) {
        deepCopyOf(t);
    }
    return *this;
}

int pktmat::rows() const {
    return mRows;
}

int pktmat::cols() const {
    return mCols;
}

int pktnn::pktmat::sum() const {
    int result = 0;
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            result += mMat[r][c];
        }
    }
    return result;
}

int pktnn::pktmat::numElems() const {
    return mRows * mCols;
}

int pktnn::pktmat::getRowMin(int row) {
    int returnVal = PKT_MAX;
    for (int c = 0; c < mCols; ++c) {
        if (mMat[row][c] < returnVal) {
            returnVal = mMat[row][c];
        }
    }
    return returnVal;
}

int pktnn::pktmat::getRowMax(int row) {
    int returnVal = PKT_MIN;
    for (int c = 0; c < mCols; ++c) {
        if (mMat[row][c] > returnVal) {
            returnVal = mMat[row][c];
        }
    }
    return returnVal;
}

int pktnn::pktmat::getColMin(int col) {
    int returnVal = PKT_MAX;
    for (int r = 0; r < mRows; ++r) {
        if (mMat[r][col] < returnVal) {
            returnVal = mMat[r][col];
        }
    }
    return returnVal;
}

int pktnn::pktmat::getColMax(int col) {
    int returnVal = PKT_MIN;
    for (int r = 0; r < mRows; ++r) {
        if (mMat[r][col] > returnVal) {
            returnVal = mMat[r][col];
        }
    }
    return returnVal;
}

int** pktmat::getMat() {
    int** returnMat = new int* [mRows];

    for (int i = 0; i < mRows; ++i) {
        returnMat[i] = new int[mCols]();
        for (int j = 0; j < mCols; ++j) {
            returnMat[i][j] = mMat[i][j];
        }
    }

    return returnMat;
}

int pktmat::getElem(int r, int c) const {
    return mMat[r][c];
}

bool pktmat::getDeleteOnDestruct() {
    std::cout << "__" << mDeleteOnDestruct << "__\n";
    return mDeleteOnDestruct;
}

int pktnn::pktmat::getMaxIndexInRow(int r) const {
    int result = 0;
    for (int c = 0; c < mCols; ++c) {
        if (mMat[r][c] > mMat[r][result]) {
            result = c;
        }
    }
    return result;
}

pktmat& pktmat::setMat(int r, int c, int* m) {
    resetZero(r, c);
    for (int i = 0; i < mRows; ++i) {
        for (int j = 0; j < mCols; ++j) {
            mMat[i][j] = *(m + (i * mCols) + j);
        }
    }

    return *this;
}

pktmat& pktmat::setElem(int r, int c, int value) {
    mMat[r][c] = value;
    return *this;
}

pktmat& pktnn::pktmat::setAllConstant(int constant) {
    for (int i = 0; i < mRows; ++i) {
        for (int j = 0; j < mCols; ++j) {
            mMat[i][j] = constant;
        }
    }
    return *this;
}

pktmat& pktnn::pktmat::setAllConstant(int r, int c, int constant) {
    resetZero(r, c);
    setAllConstant(constant);
    return *this;
}

pktmat& pktnn::pktmat::resetAllOnes(int r, int c) {
    return setAllConstant(r, c, 1);
}

pktmat& pktmat::setRandom(bool allowZero, int minVal, int maxVal) {
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            // initialized to 0 by calloc!
            mMat[r][c] = randomRange(minVal, maxVal);
            if (!allowZero) {
                while (mMat[r][c] == 0) {
                    mMat[r][c] = randomRange(minVal, maxVal);
                }
            }
        }
    }
    return *this;
}

pktmat& pktmat::setName(std::string n) {
    mName = n;
    return *this;
}

pktmat& pktmat::setDeleteOnDestruct(bool b)
{
    mDeleteOnDestruct = b;
    return *this;
}

pktmat& pktmat::fastReturn() {
    return setDeleteOnDestruct(false);
}

pktmat& pktmat::printMat(std::ostream& outTo) {
    for (int i = 0; i < mRows; ++i) {
        for (int j = 0; j < mCols; ++j) {
            outTo << mMat[i][j] << " ";
        }
        outTo << "\n";
    }

    return *this;
}

// * CANNOT use chaining because i.e., mWeight is a member of pktfc.
pktmat& pktmat::matMulMat(pktmat& mat1, pktmat& mat2) {
    // ASSUMPTION: this is NOT in-place
    assert(mat1.mCols == mat2.mRows); // (1, Dk) * (N, Dk)??
    assert((this != &mat1) && (this != &mat2));
    
    resetZero(mat1.mRows, mat2.mCols);
    for (int r1 = 0; r1 < mat1.mRows; ++r1) {
        for (int c2 = 0; c2 < mat2.mCols; ++c2) {
            int total = 0;
            int prevTotal = 0;
            for (int c1_r2 = 0; c1_r2 < mat1.mCols; ++c1_r2) {
                prevTotal = total;
                total += (mat1.mMat[r1][c1_r2] * mat2.mMat[c1_r2][c2]);
                // debug
                if (mat1.mMat[r1][c1_r2] * mat2.mMat[c1_r2][c2] > 0) {
                    if (total < prevTotal) {
                        std::cout << "Overflow: " << mat1.mMat[r1][c1_r2] << ", " << mat2.mMat[c1_r2][c2] << " | " << prevTotal << " -> " << total << "\n";
                    }
                }
            }
            mMat[r1][c2] = total;
        }
    }

    return *this;
}

// * CANNOT use chaining because i.e., mBias is a member of pktfc.
pktmat& pktmat::matAddMat(pktmat& mat1, pktmat& mat2) {
    assert(mat1.mCols == mat2.mCols);
    if (mat1.mRows != mat2.mRows) {
        assert((mat2.mRows == 1) && (mat1.mRows >= mat2.mRows));
    }

    if ((mRows != mat1.mRows) || (mCols != mat1.mCols)) {
        // if mat2 is equal to *this, this if statment will not be executed.
        resetZero(mat1.mRows, mat1.mCols);
    }

    if ((mat1.mRows != mat2.mRows) && (mat2.mRows == 1)) {
        // broadcasting
        for (int r = 0; r < mat1.mRows; ++r) {
            for (int c = 0; c < mat1.mCols; ++c) {
                mMat[r][c] = mat1.mMat[r][c] + mat2.mMat[0][c];
            }
        }
    }
    else if (mat1.mRows == mat2.mRows) {
        for (int r = 0; r < mat1.mRows; ++r) {
            for (int c = 0; c < mat1.mCols; ++c) {
                mMat[r][c] = mat1.mMat[r][c] + mat2.mMat[r][c];
            }
        }
    }
    else {
        // TODO: error
    }

    return *this;
}

pktmat& pktmat::matMulConst(pktmat& mat1, int con1) {
    resetZero(mat1.mRows, mat1.mCols);
    // TODO: can cause overflow
    for (int r = 0; r < mat1.mRows; ++r) {
        for (int c = 0; c < mat1.mCols; ++c) {
            mMat[r][c] = mat1.mMat[r][c] * con1;
        }
    }
    return *this;
}

pktmat& pktmat::matDivConst(pktmat& mat1, int con1) {
    assert(con1 != 0);
    resetZero(mat1.mRows, mat1.mCols);
    for (int r = 0; r < mat1.mRows; ++r) {
        for (int c = 0; c < mat1.mCols; ++c) {
            mMat[r][c] = mat1.mMat[r][c] / con1;
        }
    }
    return *this;
}

pktmat& pktmat::matAddConst(pktmat& mat1, int con1) {
    assert(dimsEqual(mat1));
    for (int r = 0; r < mat1.mRows; ++r) {
        for (int c = 0; c < mat1.mCols; ++c) {
            mMat[r][c] = mat1.mMat[r][c] + con1;
        }
    }
    return *this;
}



pktmat& pktmat::matElemAddMat(pktmat& mat1, pktmat& mat2) {
    assert(dimsEqual(mat1) && mat1.dimsEqual(mat2));
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] = mat1.mMat[r][c] + mat2.mMat[r][c];
        }
    }
    return *this;
}

pktmat& pktmat::matElemMulMat(pktmat& mat1, pktmat& mat2) {
    bool equalDim = mat1.dimsEqual(mat2);
    bool canBroadcast = (mat1.rows() == 1) && (mat1.cols() == mat2.cols());
    assert(equalDim || canBroadcast);

    if (!dimsEqual(mat2)) {
        resetZero(mat2.mRows, mat2.mCols);
    }
    
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            if (equalDim) {
                mMat[r][c] = mat1.mMat[r][c] * mat2.mMat[r][c];
            }
            else if (canBroadcast) {
                mMat[r][c] = mat1.mMat[0][c] * mat2.mMat[r][c];
            }
        }
    }
    
    return *this;
}

pktmat& pktmat::matElemDivMat(pktmat& mat1, pktmat& mat2) {
    assert(mat1.dimsEqual(mat2));
    if (!dimsEqual(mat1)) {
        resetZero(mat1.mRows, mat1.mCols);
    }

    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            if (mat2.mMat[r][c] == 0) {
                mMat[r][c] = INT_MAX;
            }
            /*
            else if (mat2.mMat[r][c] == PKT_MAX) {
                mMat[r][c] = 0;
            }
            */
            else {
                mMat[r][c] = mat1.mMat[r][c] / mat2.mMat[r][c];
            }
        }
    }
    return *this;
}

pktmat& pktmat::mulGradOf(pktmat& mat1) {
    assert((mRows == mat1.mRows) && (mCols == mat1.mCols));
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            if ((mat1.mMat[r][c] <= PKT_MIN) || (mat1.mMat[r][c] >= PKT_MAX)) {
                mMat[r][c] = 0;
            }
        }
    }
    return *this;
}

pktmat& pktmat::selfMulConst(int con1) {
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] *= con1;
        }
    }
    return *this;
}

pktmat& pktmat::selfDivConst(int con1) {
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] /= con1;
        }
    }
    return *this;
}

pktmat& pktmat::selfAddConst(int con1) {
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] += con1;
        }
    }
    return *this;
}

pktmat& pktnn::pktmat::selfElemAddConst(int r, int c, int con1) {
    mMat[r][c] += con1;
    return *this;
}

pktmat& pktnn::pktmat::selfMulMat(pktmat& mat2) {
    pktmat tempMat;
    tempMat.deepCopyOf(*this);
    matMulMat(tempMat, mat2);
    return *this;
}

pktmat& pktnn::pktmat::matMulSelf(pktmat& mat1) {
    pktmat tempMat;
    tempMat.deepCopyOf(*this);
    matMulMat(mat1, tempMat);
    return *this;
}

pktmat& pktnn::pktmat::matElemMulSelf(pktmat& mat1) {
    pktmat tempMat;
    tempMat.deepCopyOf(*this);
    matElemMulMat(mat1, tempMat);
    return *this;
}

pktmat& pktnn::pktmat::selfAddMat(pktmat& mat1) {
    matAddMat(*this, mat1);
    return *this;
}

pktmat& pktnn::pktmat::selfElemMulMat(pktmat& mat1) {
    matElemMulMat(*this, mat1);
    return *this;
}

pktmat& pktnn::pktmat::selfElemDivMat(pktmat& mat1) {
    matElemDivMat(*this, mat1);
    return *this;
}

pktmat& pktmat::transposeOf(pktmat& mat1) {
    resetZero(mat1.mCols, mat1.mRows);
    for (int r = 0; r < mat1.mRows; ++r) {
        for (int c = 0; c < mat1.mCols; ++c) {
            mMat[c][r] = mat1.mMat[r][c];
        }
    }
    return *this;
}

pktmat& pktnn::pktmat::rotate180Of(pktmat& mat1) {
    if (!dimsEqual(mat1)) {
        resetZero(mat1.mRows, mat1.mCols);
    }

    int midRow = mRows / 2;
    if (mRows % 2 != 0) {
        for (int c = 0; c <= mCols / 2; ++c) {
            mMat[midRow][c] = mat1.mMat[midRow][mCols - c - 1];
            mMat[midRow][mCols - c - 1] = mat1.mMat[midRow][c];
        }
    }
    
    for (int r = 0; r < midRow; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] = mat1.mMat[mRows - r - 1][mCols - c - 1];
            mMat[mRows - r - 1][mCols - c - 1] = mat1.mMat[r][c];
        }
    }
    
    return *this;    
}

pktmat& pktnn::pktmat::squareRootOf(pktmat& mat1) {
    if (!dimsEqual(mat1)) {
        resetZero(mat1.mRows, mat1.mCols);
    }

    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] = floorSqrt(mat1.mMat[r][c]);
        }
    }
    return *this;
}

pktmat& pktnn::pktmat::randomKSamplesOf(pktmat& mat1, int k) {
    assert(k <= mat1.mRows);

    int totalNum = mat1.mRows;
    int* indices = new int[totalNum];

    for (int i = 0; i < totalNum; ++i) {
        indices[i] = i;
    }

    for (int i = totalNum - 1; i > 0; --i) {
        // Pick a random index from 0 to i
        int j = rand() % (i + 1);
        int temp = indices[j];
        indices[j] = indices[i];
        indices[i] = temp;
    }

    resetZero(k, mat1.mCols);
    for (int r = 0; r < k; ++r) {
        for (int c = 0; c < mat1.mCols; ++c) {
            mMat[r][c] = mat1.mMat[indices[r]][c];
        }
    }
    
    return *this;
}

pktmat& pktnn::pktmat::indexedSlicedSamplesOf(pktmat& mat1, int* indices, int start, int end) {
    // (python style) start: inclusive, end: exclusive
    assert((0 <= start) && (start < end));
    int sampleSize = end - start;
    resetZero(sampleSize, mat1.mCols);
    for (int r = 0; r < sampleSize; ++r) {
        for (int c = 0; c < mat1.mCols; ++c) {
            mMat[r][c] = mat1.mMat[indices[start + r]][c];
        }
    }

    return *this;
}

pktmat& pktmat::sliceOf(pktmat& mat1, int rowStart, int rowEnd, int colStart, int colEnd) {
    assert((rowStart >= 0) && (colStart >= 0));
    assert((rowStart <= rowEnd) && (colStart <= colEnd));
    assert((rowEnd < mat1.mRows) && (colEnd < mat1.mCols));

    resetZero(rowEnd - rowStart + 1, colEnd - colStart + 1);
    for (int r = rowStart; r <= rowEnd; ++r) {
        for (int c = colStart; c <= colEnd; ++c) {
            mMat[r - rowStart][c - colStart] = mat1.mMat[r][c];
        }
    }

    return *this;
}

pktmat& pktmat::deepCopyOf(const pktmat& mat1) {
    resetZero(mat1.mRows, mat1.mCols);
    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] = mat1.mMat[r][c];
        }
    }
    return *this;
}

pktmat& pktmat::matUpdateLr(pktmat& update, int lr_inverse) {
    assert(dimsEqual(update));
    if (lr_inverse <= 0) {
        lr_inverse = 1;
    }

    for (int r = 0; r < mRows; ++r) {
        for (int c = 0; c < mCols; ++c) {
            mMat[r][c] += (update.mMat[r][c] / lr_inverse);
        }
    }

    return *this;
}