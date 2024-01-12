#include "pktnn_actv.h"

using namespace pktnn;

int pktactv::clamp(int input, const int minVal, const int maxVal) {
    if (input < minVal) { return minVal; }
    else if (input > maxVal) { return maxVal; }
    else { return input; }
}

void pktactv::activate(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, Actv actv, int k, int numItems) {
    if (!matOut.dimsEqual(matIn)) {
        matOut.resetZero(matIn.rows(), matIn.cols());
    }

    switch (actv) {
    case Actv::pocket_sigmoid:
        pocketSigmoid(matOut, matIn, matActvGradInv, k);
        break;
    case Actv::pocket_tanh:
        pocketTanh(matOut, matIn, matActvGradInv, k, numItems);
        break;
    case Actv::rescale:
        rescale(matOut, matIn, matActvGradInv, k);
        break;
    case Actv::pocket_softmax:
        pocketSoftmax(matOut, matIn, matActvGradInv, k);
        break;
        // -- below are not yet supported in CONV
    case Actv::pocket_relu8bit:
        pocketReLU8Bit(matOut, matIn, matActvGradInv, k);
        break;
    case Actv::pocket_leakyrelu:
        pocketLeakyReLU(matOut, matIn, matActvGradInv, k);
        break;
    case Actv::plu:
        plu(matOut, matIn, matActvGradInv, k);
        break;
    case Actv::as_is:
        asIs(matOut, matIn, matActvGradInv, k);
        break;
    default:
        std::cout << "No activation function!\n";
        break;
    }
}

void pktactv::activate3d(pktmat3d& mat3dOut, pktmat3d& mat3dIn, pktmat3d& matActvGradInv3d, Actv actv, int k, int numItems) {
    if (!mat3dOut.dimsEqual(mat3dIn)) {
        mat3dOut.resetZero3d(mat3dIn.depth(), mat3dIn.rows(), mat3dIn.cols());
    }
    matActvGradInv3d.resetZero3d(mat3dOut.depth(), mat3dOut.rows(), mat3dOut.cols());

    switch (actv) {
    case Actv::pocket_sigmoid:
        pocketSigmoid3d(mat3dOut, mat3dIn, matActvGradInv3d, k);
        break;
    case Actv::pocket_tanh:
        pocketTanh3d(mat3dOut, mat3dIn, matActvGradInv3d, k, numItems);
        break;
    case Actv::rescale:
        rescale3d(mat3dOut, mat3dIn, matActvGradInv3d, k);
        break;
    default:
        std::cout << "No activation function!\n";
        break;
    }
}

void pktnn::pktactv::pocketSigmoid3d(pktmat3d& mat3dOut, pktmat3d& mat3dIn, pktmat3d& matActvGradInv3d, int k) {
    for (int i = 0; i < mat3dIn.depth(); ++i) {
        pocketSigmoid(mat3dOut.getMatAtDepth(i), mat3dIn.getMatAtDepth(i), matActvGradInv3d.getMatAtDepth(i), k);
    }
}

void pktnn::pktactv::pocketTanh3d(pktmat3d& mat3dOut, pktmat3d& mat3dIn, pktmat3d& matActvGradInv3d, int k, int numItems) {
    for (int i = 0; i < mat3dIn.depth(); ++i) {
        // numItems == mat3dIn.depth() ??
        // also, need to consider not only numItems but also filter size??
        pocketTanh(mat3dOut.getMatAtDepth(i), mat3dIn.getMatAtDepth(i), matActvGradInv3d.getMatAtDepth(i), k, numItems);
    }
}

void pktnn::pktactv::rescale3d(pktmat3d& mat3dOut, pktmat3d& mat3dIn, pktmat3d& matActvGradInv3d, int k) {
    for (int i = 0; i < mat3dIn.depth(); ++i) {
        rescale(mat3dOut.getMatAtDepth(i), mat3dIn.getMatAtDepth(i), matActvGradInv3d.getMatAtDepth(i), k);
    }
}

void pktactv::pocketSigmoid(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k) {
    // Ver. 3
    // Ref: Piecewise linear approximation applied to nonlinear function of a neural network
    // I modified it to match my purpose
    // (0) x / (2^k) so that 2k bit will become k bit (ignoring leading 0s)
    //     cannot use >> operator because >> for negative number might differ on systems.
    // (1) [-127, -75]: x/8 + 20
    // (2) [-74 , -32]: x/2 + 48
    // (3) [-31 ,  31]: x   + 64
    // (4) [ 32 ,  74]: x/2 + 80
    // (5) [ 75 , 127]: x/8 + 108

    assert(matOut.dimsEqual(matIn));
    if (!matActvGradInv.dimsEqual(matOut)) {
        matActvGradInv.resetZero(matOut.rows(), matOut.cols());
    }
    const int yMax = PKT_MAX;
    const int yMin = 1; // NEVER zero!! Why not? to avoid div-by-zero?
    const int joints[6] = { -127, -74, -31, 32, 75, 128 };
    const int divisor = 1 << k;
    const int slopesInv[7] = { PKT_MAX, 8, 2, 1, 2, 8, PKT_MAX };
    //const int slopesInv[7] = {PKT_MAX * divisor, 8 * divisor, 2 * divisor, 1 * divisor, 2 * divisor, 8 * divisor, PKT_MAX * divisor };
    
    
    for (int r = 0; r < matOut.rows(); ++r) {
        for (int c = 0; c < matOut.cols(); ++c) {
            int x = matIn.getElem(r, c) / divisor;
            if (x < joints[0]) {
                matOut.setElem(r, c, yMin);
                matActvGradInv.setElem(r, c, slopesInv[0]);
            }
            else if (x < joints[1]) {
                int y = x / 8 + 20;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[1]);
            }
            else if (x < joints[2]) {
                int y = x / 2 + 48;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[2]);
            }
            else if (x < joints[3]) {
                int y = x + 64;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[3]);
            }
            else if (x < joints[4]) {
                int y = x / 2 + 80;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[4]);
            }
            else if (x < joints[5]) {
                int y = x / 8 + 108;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[5]);
            }
            else {
                matOut.setElem(r, c, yMax);
                matActvGradInv.setElem(r, c, slopesInv[6]);
            }
        }
    }
}

void pktactv::pocketTanh(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k, int numItems) {
    // Ver. 2
    // (0) x / (2^k) so that 2k bit will become k bit (ignoring leading 0s)
    //     cannot use >> operator because >> for negative number might differ on systems.
    // (1) [-127, -75]: x/4 - 88 
    // (2) [-74 , -32]: x   - 32
    // (3) [-31 ,  31]: 2x
    // (4) [ 32 ,  74]: x   + 32
    // (5) [ 75 , 127]: x/4 + 88

    assert(matOut.dimsEqual(matIn));
    if (!matActvGradInv.dimsEqual(matOut)) {
        matActvGradInv.resetZero(matOut.rows(), matOut.cols());
    }
    const int yMax = PKT_MAX;
    const int yMin = PKT_MIN;
    const int joints[6] = { -127, -74, -31, 32, 75, 128 };
    const int divisor = (1 << k) * numItems; //TODO! * numItems works better.
    const int slopesInv[7] = { PKT_MAX, 8, 2, 1, 2, 8, PKT_MAX }; // Strictly, they should be PKT_MAX, 4, 1, 0.5, 1, 4, PKT_MAX
    //const int slopesInv[7] = {PKT_MAX * numItems, 8 * numItems, 2 * numItems, 1 * numItems, 2 * numItems, 8 * numItems, PKT_MAX * numItems };

    for (int r = 0; r < matOut.rows(); ++r) {
        for (int c = 0; c < matOut.cols(); ++c) {
            int x = matIn.getElem(r, c) / divisor;
            if (x < joints[0]) {
                matOut.setElem(r, c, yMin);
                matActvGradInv.setElem(r, c, slopesInv[0]);
            }
            else if (x < joints[1]) {
                int y = x / 4 - 88;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[1]);
            }
            else if (x < joints[2]) {
                int y = x - 32;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[2]);
            }
            else if (x < joints[3]) {
                int y = 2 * x;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[3]);
            }
            else if (x < joints[4]) {
                int y = x + 32;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[4]);
            }
            else if (x < joints[5]) {
                int y = x / 4 + 88;
                matOut.setElem(r, c, y);
                matActvGradInv.setElem(r, c, slopesInv[5]);
            }
            else {
                matOut.setElem(r, c, yMax);
                matActvGradInv.setElem(r, c, slopesInv[6]);
            }
        }
    }
}

void pktnn::pktactv::rescale(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k) {
    assert(matOut.dimsEqual(matIn));
    if (!matActvGradInv.dimsEqual(matOut)) {
        matActvGradInv.resetZero(matOut.rows(), matOut.cols());
    }
    const int divisor = 1 << k;
    // TODO: I put 1 here and divisor in the for loop. Why?
    // TODO: if I put 1<<k here, then will all backprop signals become zero??
    // It's for: x / (2^k) so that 2k bit will become k bit (ignoring leading 0s)
    matActvGradInv.setAllConstant(1);
    for (int r = 0; r < matOut.rows(); ++r) {
        for (int c = 0; c < matOut.cols(); ++c) {
            matOut.setElem(r, c, matIn.getElem(r, c) / divisor);
        }
    }
}

void pktnn::pktactv::pocketSoftmax(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k) {
    // ROWWISE: one row is considered to be one item
    // Original Softmax: (e^x < 1) when (x < 0)
    // Pocket Softmax: 0 for (x < 0) to avoid sum==0, and rescale them to be summed up
    assert(matOut.dimsEqual(matIn));
    if (!matActvGradInv.dimsEqual(matOut)) {
        matActvGradInv.resetZero(matOut.rows(), matOut.cols());
    }

    const int newTotal = INT_MAX;
    for (int r = 0; r < matOut.rows(); ++r) {
        int rowSum = 0;
        int rowMax = 0;
        for (int c = 0; c < matOut.cols(); ++c) {
            int thisVal = matIn.getElem(r, c);
            // set all nonpositive numbers to 0
            if (thisVal <= 0) {
                matOut.setElem(r, c, 0);
            }
            else {
                matOut.setElem(r, c, thisVal);
                rowSum += thisVal;
            }
        }

        if (rowSum == 0) {
            rowSum = 1; // avoid overflow
        }

        for (int c = 0; c < matOut.cols(); ++c) {
            int newVal = matOut.getElem(r, c) * (newTotal / rowSum); // rescale
            if (newVal == 0) {
                matActvGradInv.setElem(r, c, INT_MAX);
            }
            else {
                matOut.setElem(r, c, newVal);
                matActvGradInv.setElem(r, c, 1); // actually int(rowSum / newTotal) == 0
            }
        }
    }
}

void pktactv::pocketReLU8Bit(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k) {
    assert(matOut.dimsEqual(matIn));
    matActvGradInv.resetZero(matOut.rows(), matOut.cols());

    const int minVal = 0;
    const int maxVal = 127; // CHAR_MAX - 1
    // so that the multiplication result will be smaller than INT_MAX ?
    // (1 << (k - 1)) - 1; // -(2^(k-1)-1)
    
    for (int r = 0; r < matOut.rows(); ++r) {
        for (int c = 0; c < matOut.cols(); ++c) {
            int currElem = matIn.getElem(r, c);
            if (currElem < minVal) {
                matOut.setElem(r, c, minVal);
                matActvGradInv.setElem(r, c, INT_MAX); // slope 0: div by PKT_MAX
            }
            else if (currElem > maxVal) {
                matOut.setElem(r, c, maxVal);
                matActvGradInv.setElem(r, c, INT_MAX); // slope 0: div by PKT_MAX
            }
            else {
                matOut.setElem(r, c, currElem);
                matActvGradInv.setElem(r, c, 1); // slope 1: linear
            }
        }
    }

}

void pktnn::pktactv::pocketLeakyReLU(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k) {
    assert(matOut.dimsEqual(matIn));
    matActvGradInv.resetZero(matOut.rows(), matOut.cols());

    const int maxVal = SHRT_MAX; //(1 << (k - 1)) - 1; // -(2^(k-1)-1)
    const int midVal = 0;
    const int minVal = -maxVal;
    const int leakySlopeDiv = 5;
    
    for (int r = 0; r < matOut.rows(); ++r) {
        for (int c = 0; c < matOut.cols(); ++c) {
            int currElem = matIn.getElem(r, c);
            if (currElem < minVal) {
                matOut.setElem(r, c, minVal);
                matActvGradInv.setElem(r, c, INT_MAX);
            }
            else if (currElem < midVal) {
                matOut.setElem(r, c, currElem / leakySlopeDiv);
                matActvGradInv.setElem(r, c, leakySlopeDiv);
            }
            else if (currElem < maxVal) {
                matOut.setElem(r, c, currElem);
                matActvGradInv.setElem(r, c, 1);
            }
            else {
                matOut.setElem(r, c, maxVal);
                matActvGradInv.setElem(r, c, INT_MAX); // slope 1: linear
            }
        }
    }
}

void pktnn::pktactv::plu(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k) {
    // Ref. PLU: The Piecewise Linear Unit Activation Function
    // PLU(x) = max[a(x+c)-c, min{a(x-c)+c, x}]
    const int maxVal = PKT_MAX;
    const int minVal = PKT_MIN;
    const int slopeInv = 10; // 1/a
    const int c = 1;
    assert(matOut.dimsEqual(matIn));
    if (!matActvGradInv.dimsEqual(matOut)) {
        matActvGradInv.resetZero(matOut.rows(), matOut.cols());
    }

    for (int r = 0; r < matOut.rows(); ++r) {
        for (int c = 0; c < matOut.cols(); ++c) {
            const int x = matIn.getElem(r, c);
            const int thresMax = (x + c) / slopeInv - c;
            const int thresMin = (x - c) / slopeInv + c;
            
            int newVal = x;
            if (thresMin < newVal) { newVal = thresMin; }
            if (thresMax > newVal) { newVal = thresMax; }

            if (newVal < minVal) {
                matOut.setElem(r, c, minVal);
                matActvGradInv.setElem(r, c, PKT_MAX); // slope 0: div by PKT_MAX
            }
            else if (newVal > maxVal) {
                matOut.setElem(r, c, maxVal);
                matActvGradInv.setElem(r, c, PKT_MAX); // slope 0: div by PKT_MAX
            }
            else {
                matOut.setElem(r, c, newVal);
                matActvGradInv.setElem(r, c, x / newVal); // grad = newVal / x. Now inverse.
            }
        }
    }

}

void pktactv::asIs(pktmat& matOut, pktmat& matIn, pktmat& matActvGradInv, int k) {
    assert(matOut.dimsEqual(matIn));
    if (!matActvGradInv.dimsEqual(matOut)) {
        matActvGradInv.resetZero(matOut.rows(), matOut.cols());
    }
    matActvGradInv.setAllConstant(1);
    for (int r = 0; r < matOut.rows(); ++r) {
        for (int c = 0; c < matOut.cols(); ++c) {
            matOut.setElem(r, c, matIn.getElem(r, c));
        }
    }
}
