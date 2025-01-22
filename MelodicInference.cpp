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
        DBG("Prompt size: " + juce::String(prompt.size()));
        for (int i = 0; i < 5; i++) {
            DBG("Prompt[" + juce::String(i) + "]: " + juce::String(prompt[i]));
        }

        // In generate(), before processing:
        /*DBG("Input melody:");
        for (const auto& token : prompt) {
            DBG(token + " ");
        }*/
        juce::String melodyStr;
        for (const auto& token : prompt) {
            melodyStr += juce::String(token) + " ";
        }
        DBG("Input melody: " + melodyStr.trimEnd());

        torch::NoGradGuard no_grad;

        DBG("Starting token conversion");


        //std::vector<int64_t> input_tokens;
        //for (const auto& token : prompt) {
        //    input_tokens.push_back(0); // Will replace with actual token lookup
        //}

        //std::vector<int64_t> inputTokens;
        //for (const auto& token : prompt) {
        //    inputTokens.push_back(token == "_" ? 0 : std::stoi(token));
        //}

        std::vector<int64_t> inputTokens;
        for (const auto& token : prompt) {
            if (token == "_" || token == "-") {
                inputTokens.push_back(0);
            }
            else {
                try {
                    int val = std::stoi(token);
                    DBG("Converting: " + juce::String(token) + " to " + juce::String(val));
                    inputTokens.push_back(val);
                }
                catch (...) {
                    DBG("Failed to convert: " + juce::String(token));
                    inputTokens.push_back(0);
                }
            }
        }


        // Debug print
        DBG("Input tokens size: " + juce::String(inputTokens.size()));

        auto input = torch::tensor(inputTokens, torch::dtype(torch::kInt64)).unsqueeze(0);



        DBG("Input tensor shape: " + juce::String(input.sizes()[0]) + "x" + juce::String(input.sizes()[1]));


        // Forward pass
        std::vector<torch::jit::IValue> inputs = { input };


        std::vector<std::string> result;
        for (int i = 0; i < 32; i++) {
            auto logits = model.forward(inputs).toTensor();
            //logits = logits.select(1, -1) / temperature;
            logits = logits.index({ torch::indexing::Slice(), -1 }).div(temperature);


            auto topk_values = std::get<0>(logits.topk(std::min(topK, (int)logits.size(-1))));
            auto probs = torch::softmax(topk_values, -1);
            auto next_token = torch::multinomial(probs, 1);


            // Debug shapes before cat
            auto next_sizes = next_token.sizes();
            auto input_sizes = input.sizes();
            DBG("next_token shape: [" + juce::String(next_sizes[0]) + "]; input shape: [" + juce::String(input_sizes[0]) + "," + juce::String(input_sizes[1]) + "]");


            //input = torch::cat({ input, next_token.unsqueeze(1) }, 1);

            input = torch::cat({ input, next_token.view({1, 1}) }, 1);

            inputs = { input };
            result.push_back(std::to_string(next_token.item<int64_t>()));
        }




        // After processing but before return:
        /*DBG("Generated melody:");
        for (const auto& token : result) {
            DBG(token + " ");
        }*/
        juce::String resultStr;
        for (const auto& token : result) {
            resultStr += juce::String(token) + " ";
        }
        DBG("Generated melody: " + resultStr.trimEnd());



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

