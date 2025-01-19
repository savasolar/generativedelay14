#include "MelodicInference.h"
#include <fstream>
#include <random>


MelodicInference::MelodicInference() {}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {
//    //juce::String modelPath = "C:/Users/savas/Documents/JUCE Projects/generativedelay14/Model/model_rtneural.json";
//    
//
//    try {
//        model = torch::jit::load("C:/Users/savas/Documents/JUCE Projects/generativedelay14/Model/model_traced.pt");
//        if (!loadTokenMappings("C:/Users/savas/Documents/JUCE Projects/generativedelay14/Model/token_mappings.pt")) {
//            return false;
//        }
//        return true;
//    }
//    catch (const std::exception& e) {
//        DBG("Error loading model: " + String(e.what()));
//        return false;
//    }
//
//
//    return {};
//
//

    try {
        std::string modelPath = "C:/Users/savas/Documents/JUCE Projects/generativedelay14/Model/model_traced.pt";

        // Check file exists
        std::ifstream file(modelPath);
        if (!file.good()) {
            DBG("Model file does not exist: " + String(modelPath));
            return false;
        }

        model = torch::jit::load(modelPath);

        std::string tokenPath = "C:/Users/savas/Documents/JUCE Projects/generativedelay14/Model/token_mappings.pt";
        if (!loadTokenMappings(tokenPath)) {
            return false;
        }

        return true;
    }
    catch (const c10::Error& e) {
        DBG("LibTorch Error loading model: " + String(e.what()));
        return false;
    }
    catch (const std::exception& e) {
        DBG("Standard Error loading model: " + String(e.what()));
        return false;
    }

}






std::vector<std::string> MelodicInference::generate(
    const std::vector<std::string>& prompt,
    float temperature,
    int topK)
{
    try {
        auto input = preprocess(prompt);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        auto output = model.forward(inputs).toTensor();
        output = output.slice(1, -1, output.size(1));  // Get last token
        output = output / temperature;

        return postprocess(output, temperature, topK);
    }
    catch (const std::exception& e) {
        DBG("Error during generation: " + String(e.what()));
        return {};
    }
}




bool MelodicInference::loadTokenMappings(const std::string& path) {
    try {
        auto checkpoint = torch::jit::load(path);

        auto loaded = checkpoint.forward({});

        auto stoi_tensor = loaded.toGenericDict().at("stoi").toTensor();
        auto itos_tensor = loaded.toGenericDict().at("itos").toTensor();





        DBG(juce::String("stoi_tensor shape: ") + juce::String(stoi_tensor.sizes().size()));
        DBG(juce::String("itos_tensor shape: ") + juce::String(itos_tensor.sizes().size()));

        for (int64_t i = 0; i < std::min(int64_t(5), stoi_tensor.size(0)); i++) {
            DBG("stoi_tensor[" + String(i) + "]: " +
                String(stoi_tensor[i][0].toString()) + ", " +
                String(stoi_tensor[i][1].item<int64_t>()));
        }




        for (int64_t i = 0; i < stoi_tensor.size(0); i++) {
            std::string key = stoi_tensor[i][0].toString();
            int64_t val = stoi_tensor[i][1].item<int64_t>();
            stoi[key] = val;
        }

        for (int64_t i = 0; i < itos_tensor.size(0); i++) {
            int64_t key = std::stoi(itos_tensor[i][0].toString());
            std::string val = itos_tensor[i][1].toString();
            itos[key] = val;
        }

        return true;
    }
    catch (const std::exception& e) {
        DBG("Error loading mappings: " + String(e.what()));
        return false;
    }
}


torch::Tensor MelodicInference::preprocess(const std::vector<std::string>& tokens) {
    std::vector<int64_t> indices;
    for (const auto& token : tokens) {
        indices.push_back(stoi[token]);
    }
    return torch::tensor(indices).unsqueeze(0); // Add batch dimension
}

std::vector<std::string> MelodicInference::postprocess(
    const torch::Tensor& logits,
    float temperature,
    int topK)
{
    auto probs = logits.softmax(-1);
    if (topK > 0) {
        auto topk_out = torch::topk(probs, topK);
        probs = std::get<0>(topk_out);
    }

    auto next_token = torch::multinomial(probs, 1);
    int64_t token_id = next_token.item<int64_t>();

    return { itos[token_id] };
}





bool MelodicInference::simple_test() {

    // implement generation test

    return {};
}

