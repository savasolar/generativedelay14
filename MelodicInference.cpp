#include "MelodicInference.h"
#include <random>
#include "model.h.c"


MelodicInference::MelodicInference() {
    File logFile = File::getSpecialLocation(File::userDesktopDirectory).getChildFile("plugin_debug.txt");
    String message = "Plugin constructor called\n";
    logFile.appendText(message);
}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {


    return {};
}






std::vector<std::string> MelodicInference::generate(
    const std::vector<std::string>& prompt,
    float temperature,
    int topK)
{
    
    return {};
}





bool MelodicInference::simple_test() {

    // implement generation test

    return {};
}

