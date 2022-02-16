#ifndef PKTNN_LOADER_H
#define PKTNN_LOADER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include "urlmon.h"
#pragma comment(lib, "urlmon.lib")

#include "pktnn_mat.h"
#include "pktnn_mat3d.h"

namespace pktnn {
    class pktloader {
    public:
        enum class Dataset {
            diabetes,
            mnist
        };
        static void csvLoader(pktmat& saveToMat, std::string fileName);
        static bool file_exists(const std::string& name);
        static void downloadDataset(Dataset dataset);
        static void parseDatasetDiabetes(pktmat& saveToMat, std::string fileName);
        static int reverseInt(int i);
        static pktmat3d** loadMnistImages(int numImagesToLoad);
        static void loadMnistImages(pktmat& images, int numImagesToLoad, bool isTrain);
        static void loadMnistLabels(pktmat& labels, int numLabelsToLoad, bool isTrain);
        static void loadFashionMnistImages(pktmat& images, int numImagesToLoad, bool isTrain);
        static void loadFashionMnistLabels(pktmat& labels, int numLabelsToLoad, bool isTrain);
    };

    extern "C" {
        void _downloadDataset(pktloader::Dataset dataset);
    }
}

#endif
