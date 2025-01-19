#pragma once
#include <JuceHeader.h>
#include <vector>
#include <string>

//#include <RTNeural/RTNeural.h>
//#include <memory>

class MelodicInference {
public:
    MelodicInference();
    ~MelodicInference();

    bool loadModel();
    std::vector<std::string> generate(const std::vector<std::string>& prompt,
        float temperature = 0.8f,
        int topK = 200);

private:

//    std::unique_ptr<RTNeural::Model<float>> model;





    bool simple_test();


};