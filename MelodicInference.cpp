#include "MelodicInference.h"
//#include <fstream>
#include <random>
//#include <filesystem>


MelodicInference::MelodicInference() {}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {


    try {
        model = torch::jit::load("C:/Users/savas/Documents/JUCE Projects/generativedelay14/Model/model_traced.pt");
        model.eval();

        // load token mappings
        tokenMappings = torch::jit::load("C:/Users/savas/Documents/JUCE Projects/generativedelay14/Model/token_mappings.pt");

        DBG("Model and mappings loaded successfully");

        return true;

    }
    catch (const c10::Error& e) {
        DBG("Model loading error: " + juce::String(e.what()));
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

        //std::vector<int64_t> input_tokens;
        //for (const auto& token : prompt) {
        //    input_tokens.push_back(0); // Will replace with actual token lookup
        //}

        // convert prompt to tokens
        std::vector<int64_t> inputTokens;
        for (const auto& token : prompt) {
            inputTokens.push_back(std::stoi(token == "_" ? "0" : token));
        }

        // Create input tensor
        auto input = torch::tensor(inputTokens, torch::dtype(torch::kInt64)).unsqueeze(0);

        // Forward pass
        std::vector<torch::jit::IValue> inputs = { input };
        auto output = model.forward(inputs).toTensor();

        // Get last logits
        auto logits = output.select(1, -1);

        // Apply temperature
        logits = logits / temperature;

        // Apply top-k
        auto topk_values = std::get<0>(logits.topk(std::min(topK, (int)logits.size(-1))));
        auto probs = torch::softmax(topk_values, -1);

        // Sample from distribution
        auto nextToken = torch::multinomial(probs, 1);

        // Convert back to string
        std::vector<std::string> result;
        result.push_back(std::to_string(nextToken.item<int64_t>()));

        return result;





//        return { "70" }; // Return dummy value to test if crash happens before forward()
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

