#include "MelodicInference.h"
#include <fstream>
#include <random>
#include <filesystem>


MelodicInference::MelodicInference() {}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {


    try {
        auto modelPath = "model_traced.pt";
        
//        if (!std::filesystem::exists(modelPath)) {
//            DBG("Model file not found: " + juce::String(modelPath));
//            return false;
//        }
//        model = torch::jit::load(modelPath);
//        model.eval();
//
//
//        // Load and verify model
////        DBG("Model device: " + juce::String(model.parameters().begin()->device().str()));
//        DBG("Model parameters: " + juce::String(model.parameters().size()));
//
//
//
//        DBG("Model loaded successfully");
//
//        return true;


        // Debug filesystem state
        DBG("Current working directory: " + juce::String(std::filesystem::current_path().string()));
        DBG("Attempting to load from: " + juce::String(modelPath));
        DBG("File exists check: " + juce::String(std::filesystem::exists(modelPath) ? "YES" : "NO"));
        DBG("Is regular file: " + juce::String(std::filesystem::is_regular_file(modelPath) ? "YES" : "NO"));

        // Try to open file directly to verify permissions
        std::ifstream f(modelPath, std::ios::binary);
        DBG("Can open file: " + juce::String(f.good() ? "YES" : "NO"));
        if (f.good()) {
            f.close();
        }

        if (!std::filesystem::exists(modelPath)) {
            DBG("Model file not found: " + juce::String(modelPath));
            return false;
        }

        model = torch::jit::load(modelPath);
        model.eval();

        DBG("Model parameters: " + juce::String(model.parameters().size()));

        DBG("Model loaded successfully");

        return true;

    }
    catch (const c10::Error& e) {
        DBG("LibTorch error: " + juce::String(e.what()));
        return false;
    }

}






std::vector<std::string> MelodicInference::generate(
    const std::vector<std::string>& prompt,
    float temperature,
    int topK)
{
    try {
        torch::NoGradGuard no_grad;

        // Input preparation and validation
        DBG("Input size: " + juce::String(prompt.size()));

        std::vector<int64_t> input_tokens;
        for (const auto& token : prompt) {
            input_tokens.push_back(0); // Will replace with actual token lookup
        }

        // create and validate tensor
        
        
        
        //auto input = torch::tensor(input_tokens).unsqueeze(0);
        ////DBG("Input tensor shape: " + juce::String(input.sizes()[0]) + "x" + juce::String(input.sizes()[1]));
        //DBG("Input tensor shape: " + juce::String(input.sizes()[0]) + "x" + juce::String(input.sizes()[1]));
        //DBG("Input tensor device: " + juce::String(input.device().str()));



        auto input = torch::tensor(input_tokens, torch::dtype(torch::kInt64)).unsqueeze(0);
//        DBG("Input shape: " + juce::String(input.sizes()[0]) + "x" + juce::String(input.sizes()[1]));
//        DBG("Input dtype: " + juce::String(input.dtype().name()));
//        DBG("Input device: " + juce::String(input.device().str()));



        // Forward pass
        std::vector<torch::jit::IValue> inputs = { input };
        auto output = model.forward(inputs).toTensor();

        
        // Validate output
        DBG("Output shape: " + juce::String(output.sizes()[0]) + "x" +
            juce::String(output.sizes()[1]) + "x" + juce::String(output.sizes()[2]));



        return { "70" }; // Return dummy value to test if crash happens before forward()
    }
    catch (const c10::Error& e) {
        DBG("Generate error: " + juce::String(e.what()));
        return { "0" };
    }
}





bool MelodicInference::simple_test() {

    // implement generation test

    return {};
}

