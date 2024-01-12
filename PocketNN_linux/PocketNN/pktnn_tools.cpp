#include "pktnn_tools.h"

int pktnn::maxVal(int a, int b) {
    return a > b ? a : b;
}

int pktnn::minVal(int a, int b) {
    return a < b ? a : b;
}

int pktnn::clampValue(int value, int lower, int upper) {
    value = (value < lower) ? lower : value;
    value = (value > upper) ? upper : value;
    return value;
}

int pktnn::randomRange(int lower, int upper) {
    // both inclusive
    return (int)((std::rand() % (upper - lower + 1)) + lower);
}

int pktnn::floorSqrt(int x) {
    if (x <= 0) {
        return 0;
    }

    int ans = 1;
    while (ans * ans <= x) {
        ++ans;
    }

    return ans - 1;
}

int pktnn::intRoundLog(int base, int x, bool getClosest) {
    // 2 = log_2(5) because of integer approximation
    return intRoundLog(base, x, 0, 0, getClosest);
}

int pktnn::intRoundLog(int base, int x, int xShift, int yShift, bool getClosest) {
    // y = log (x - xShift) + yShift
    // xShift == xMin: so that asymptote is x == xMin
    // yShift == -log (xMax - xMin): so that log (x - xShift) + yShift == 0
    assert(base > 1);
    int xMinusP = x - xShift;

    //std::cout << "log base: " << base << ", log input: " << x << "\n";

    if (xMinusP <= 0) {
        // log 0 is assumed to be minus infinity
        // TODO: workaround for (-) values
        return PKT_MIN;
    }
    else if (xMinusP < 1) {
        // TODO: impossible in integer-only mode
        // logarithm is negative in this range
    }
    else if (xMinusP == 1) {
        return 0;
    }
    else { // (xMinusP > 1)
        // logarithm is positive in this range
        int exponent = 0;
        int curr = 1; // base^0 == 1
        int prev = 0; // dummy

        while (curr < xMinusP) {
            prev = curr;
            curr *= base;
            ++exponent;
        }

        // take a closer x
        int y = (exponent - 1);
        if (getClosest) {
            if ((curr - xMinusP) < (xMinusP - prev)) {
                y = exponent;
            }
        }

        return y + yShift;
    }
    return 1; // TODO: Dummy
}

int pktnn::round(int n, int unit) {
    // Smaller multiple
    int a = (n / unit) * unit;

    // Larger multiple
    int b = a + unit;

    // Return of closest of two
    return (n - a > b - n) ? b : a;
}
