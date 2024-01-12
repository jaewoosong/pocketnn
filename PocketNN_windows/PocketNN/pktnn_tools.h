#ifndef PKTNN_TOOLS_H
#define PKTNN_TOOLS_H

#include <iostream>
#include <limits.h>
#include <assert.h>
#include "pktnn_consts.h"

namespace pktnn {
    int maxVal(int a, int b);
    int minVal(int a, int b);
    int clampValue(int value, int lower, int upper);
    int randomRange(int lower, int upper);
    int floorSqrt(int x);
    int intRoundLog(int base, int x, bool getClosest = true);
    int intRoundLog(int base, int x, int xShift, int yShift, bool getClosest = true);
    int round(int n, int unit);
}
#endif
