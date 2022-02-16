#ifndef PKTNN_EXAMPLES_H
#define PKTNN_EXAMPLES_H

#include <iostream>
#include <fstream>
#include <limits.h>
#include <assert.h>
#include "pktnn_fc.h"
#include "pktnn_mat.h"
#include "pktnn_tools.h"
#include "pktnn_loss.h"
#include "pktnn_loader.h"

int example_fc_int_bp_very_simple();
int example_fc_int_dfa_mnist();
int example_fc_int_dfa_fashion_mnist();

#endif