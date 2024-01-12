#ifndef PKTNN_CONSTS_H
#define PKTNN_CONSTS_H

namespace pktnn {
    const int K_BIT = 8;
    const int PKT_MIN = SCHAR_MIN + 1; // -127
    const int PKT_MAX = SCHAR_MAX; // 127
    const int UNSIGNED_4BIT_MAX = 15; // 0 to 15
    const std::string TYPE_PKTFC = "pktfc";
    const std::string TYPE_PKTCONV = "pktconv";
}

#endif