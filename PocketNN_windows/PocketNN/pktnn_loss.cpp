#include "pktnn_loss.h"

using namespace pktnn;


int pktloss::scalarL2Loss(int y, int yHat) {
    return (yHat - y) * (yHat - y) / 2;
}

int pktloss::scalarL2LossDelta(int y, int yHat) {
    return (yHat - y);
}

int pktloss::batchL2Loss(pktmat& lossMat, pktmat& yMat, pktmat& yHatMat) {
    assert(yMat.dimsEqual(yHatMat));
    lossMat.resetZero(yHatMat.rows(), 1);
    // IMPORTANT: One sumLoss value per one sample
    
    int accumLoss = 0;
    // Each row corresponds to one input
    for (int r = 0; r < yMat.rows(); ++r) {
        int columnLoss = 0;
        for (int c = 0; c < yMat.cols(); ++c) {
            int y = yMat.getElem(r, c);
            int yHat = yHatMat.getElem(r, c);
            columnLoss += scalarL2Loss(y, yHat);
        }
        // TODO: need to avoid overflow
        lossMat.setElem(r, 0, columnLoss);
        accumLoss += lossMat.getElem(r, 0);
    }

    return accumLoss;
}

int pktloss::batchL2LossDelta(pktmat& lossDeltaMat, pktmat& yMat, pktmat& yHatMat) {
    assert(yMat.dimsEqual(yHatMat));
    // Assumption: 1 input -> 1 scalar sumLoss value
    // for 1 output of dimention T, lossDeltaMat = (1, T)
    lossDeltaMat.resetZero(yHatMat.rows(), yHatMat.cols()); // (N, Dlast)

    int accumLossDelta = 0;
    // Per each input item
    for (int r = 0; r < yMat.rows(); ++r) {
        int columnLossDelta = 0;
        // each item has output of size (1, yMat.cols())
        for (int c = 0; c < yMat.cols(); ++c) {
            int y = yMat.getElem(r, c);
            int yHat = yHatMat.getElem(r, c);
            lossDeltaMat.setElem(r, c, scalarL2LossDelta(y, yHat));
            columnLossDelta += lossDeltaMat.getElem(r, c);
        }
        accumLossDelta += columnLossDelta;
    }

    return accumLossDelta; // return sum! (average is meaningless)
}

int pktnn::pktloss::vectorPocketCrossLoss(pktmat& lossVec, pktmat& yVec, pktmat& yHatVec) {
    assert(yVec.dimsEqual(yHatVec) && (yVec.rows() == 1));
    lossVec.resetZero(yVec.rows(), 1);

    int sumLoss = 0;
    for (int c = 0; c < yVec.cols(); ++c) {
        if (yVec.getElem(0, c) == INT_MAX) {
            lossVec.setElem(0, 0, INT_MAX - yHatVec.getElem(0, c));
            sumLoss += INT_MAX - yHatVec.getElem(0, c);
        }
    }
    return sumLoss;
}

int pktnn::pktloss::batchPocketCrossLoss(pktmat& lossMat, pktmat& yMat, pktmat& yHatMat) {
    assert(yMat.dimsEqual(yHatMat));
    lossMat.resetZero(yMat.rows(), 1); // 1 item per row, one-hot encoding

    int sumLoss = 0;
    for (int r = 0; r < yMat.rows(); ++r) {
        for (int c = 0; c < yMat.cols(); ++c) {
            if (yMat.getElem(r, c) == INT_MAX) {
                lossMat.setElem(r, 0, INT_MAX - yHatMat.getElem(r, c));
                sumLoss += lossMat.getElem(r, 0);
            }
        }
    }
    return sumLoss;
}

int pktnn::pktloss::batchPocketCrossLossDelta(pktmat& lossDeltaMat, pktmat& yMat, pktmat& yHatMat) {
    assert(yMat.dimsEqual(yHatMat));
    lossDeltaMat.resetZero(yHatMat.rows(), yHatMat.cols());

    int sumLossDelta = 0;
    for (int r = 0; r < yMat.rows(); ++r) {
        for (int c = 0; c < yMat.cols(); ++c) {
            if (yMat.getElem(r, c) == INT_MAX) {
                lossDeltaMat.setElem(r, c, -1);
                sumLossDelta += lossDeltaMat.getElem(r, c);
            }
        }
    }
    return sumLossDelta;
}

int pktloss::batchCrossEntropyLoss(pktmat& lossMat, pktmat& yMat, pktmat& yHatMat) {
    assert(yMat.dimsEqual(yHatMat));
    lossMat.resetZero(yMat.rows(), 1); // one sumLoss value per one sample
    
    int accumLoss = 0;
    int yShift = -7; // -intRoundLog(2, PKT_MAX);
    for (int r = 0; r < yMat.rows(); ++r) {
        // one row is one output vector
        int columnLoss = 0;
        for (int c = 0; c < yMat.cols(); ++c) {
            int yHat = yHatMat.getElem(r, c);
            if (yMat.getElem(r, c) == 1) {
                // pocket sigmoid range = [1, 127 (== PKT_MAX)]
                // as_is range = +- 16 bit range??
                // there can be multiple correct answers. SO ACCUMULATE.
                // yShift achieves what we want
                // multiply PKT_MAX for lossDelta

                //columnLoss -= (/*PKT_MAX **/ intRoundLog(2, minVal(yHat, PKT_MAX), 0, yShift));
                columnLoss += ((yHat - PKT_MAX) * (yHat - PKT_MAX)) / 2;
            }
            else {
                //columnLoss += (yHat * yHat) / 2;
            }
        }
        lossMat.setElem(r, 0, columnLoss);
        accumLoss += lossMat.getElem(r, 0);
    }

    // It is useful to return sum of cross entropy in integer-only mode
    // because individual sumLoss has too few steps
    return accumLoss;
}

int pktloss::batchCrossEntropyLossDelta(pktmat& lossDeltaMat, pktmat& yMat, pktmat& yHatMat) {
    assert(yMat.dimsEqual(yHatMat));
    if (!lossDeltaMat.dimsEqual(yHatMat)) {
        lossDeltaMat.resetZero(yHatMat.rows(), yHatMat.cols());
    }
    else {
        lossDeltaMat.setAllConstant(0);
    }

    int accumLossDelta = 0;
    for (int r = 0; r < yMat.rows(); ++r) {
        int columnLossDelta = 0;
        for (int c = 0; c < yMat.cols(); ++c) {
            // int y = yMat.getElem(r, c);
            int yHat = yHatMat.getElem(r, c);
            if (yMat.getElem(r, c) == 1) {                
                lossDeltaMat.setElem(r, c, (yHat - PKT_MAX));
            }
            else {
                //lossDeltaMat.setElem(r, c, yHat);
            }
            columnLossDelta += lossDeltaMat.getElem(r, c);
        }
        accumLossDelta += columnLossDelta;
    }
    return accumLossDelta;
}

