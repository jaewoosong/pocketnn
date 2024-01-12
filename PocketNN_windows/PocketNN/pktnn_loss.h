#ifndef PKTNN_LOSS_H
#define PKTNN_LOSS_H

#include "pktnn_mat.h"
#include "pktnn_tools.h"

namespace pktnn {

    class pktloss {
    public:
        // regression
        static int scalarL2Loss(int y, int yHat);
        static int scalarL2LossDelta(int y, int yHat);
        static int batchL2Loss(pktmat& lossMat, pktmat& yMat, pktmat& yHatMat);
        static int batchL2LossDelta(pktmat& lossDeltaMat, pktmat& yMat, pktmat& yHatMat);
        
        // classification
        static int vectorPocketCrossLoss(pktmat& lossVec, pktmat& yVec, pktmat& yHatVec);
        static int batchPocketCrossLoss(pktmat& lossMat, pktmat& yMat, pktmat& yHatMat);
        static int batchPocketCrossLossDelta(pktmat& lossDeltaMat, pktmat& yMat, pktmat& yHatMat);
        static int batchCrossEntropyLoss(pktmat& lossMat, pktmat& yMat, pktmat& yHatMat);
        static int batchCrossEntropyLossDelta(pktmat& lossDeltaMat, pktmat& yMat, pktmat& yHatMat);
    };

}

#endif