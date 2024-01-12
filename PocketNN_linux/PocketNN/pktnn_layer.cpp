#include "pktnn_layer.h"

using namespace pktnn;

pktnn::pktlayer::pktlayer() {
    mLayerType = LayerType::pocket_fc;
    pPrevLayer = nullptr;
    pNextLayer = nullptr;
}

pktnn::pktlayer::~pktlayer() {
    //mDummy3d.deleteMat3d(); // problematic!!
    //mDummy3d.~pktmat3d(); // no difference
}

pktnn::pktlayer::LayerType pktnn::pktlayer::getLayerType() {
    return mLayerType;
}

pktnn::pktlayer* pktnn::pktlayer::getPrevLayer() {
    return pPrevLayer;
}

pktnn::pktlayer* pktnn::pktlayer::getNextLayer() {
    return pNextLayer;
}

pktlayer& pktnn::pktlayer::setPrevLayer(pktlayer& layer1) {
    pPrevLayer = &layer1;
    layer1.pNextLayer = this; // CAUTION (inf. loop): layer1.setNextLayer(*this);
    return *this;
}

pktlayer& pktnn::pktlayer::setNextLayer(pktlayer& layer1) {
    pNextLayer = &layer1;
    layer1.pPrevLayer = this; // CAUTION (inf. loop): layer1.setPrevLayer(*this);
    return *this;
}
