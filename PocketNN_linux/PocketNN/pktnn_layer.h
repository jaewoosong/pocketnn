#ifndef PKTNN_LAYER_H
#define PKTNN_LAYER_H

#include <iostream>
#include "pktnn_mat.h"
#include "pktnn_mat3d.h"

namespace pktnn {

    class pktlayer {
    public:
        enum class LayerType {
            pocket_fc,
            pocket_conv
        };
    protected:
        LayerType mLayerType;
        pktlayer* pPrevLayer;
        pktlayer* pNextLayer;
        pktmat3d mDummy3d;
    public:
        pktlayer();
        ~pktlayer();
        LayerType getLayerType();
        pktlayer* getPrevLayer();
        pktlayer* getNextLayer();
        pktlayer& setPrevLayer(pktlayer& layer1);
        pktlayer& setNextLayer(pktlayer& layer1);
        virtual pktlayer& forward(pktlayer& x) = 0;
        virtual pktlayer& backward(pktmat& lastDeltasMat, int lrInv) = 0;
        virtual pktmat& getOutputForFc() = 0;
        virtual pktmat3d& getOutputForConv() = 0;
    };

}
#endif
