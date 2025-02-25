#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <random>
#include <map>

class MelodicInference {
public:
    MelodicInference();
    ~MelodicInference();

    // Load model from file
    bool loadModel(const juce::String& modelPath = "C:/Users/savas/Desktop/2025-02-09-melodic-nanogpt-2-onnx/melodygpt_v02_quantized.ort",
        const juce::String& tokensPath = "C:/Users/savas/Desktop/2025-02-09-melodic-nanogpt-2-onnx/token_mappings.json");

    // Generate melody from input tokens
    std::vector<std::string> generate(const std::vector<std::string>& inputTokens,
        float temperature = 0.8f,
        int topK = 200);

private:
    // ONNX Runtime session and environment
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;

    // Token mappings (string to int and back)
    std::map<std::string, int> stoi_;
    std::map<int, std::string> itos_;

    // Random generator for sampling
    std::mt19937 rng_;

    // Helper methods
    std::vector<float> softmax(const std::vector<float>& x);
    int sampleFromLogits(const std::vector<float>& logits, float temperature, int topK);
    void loadTokenMappings(const juce::String& tokensPath);
};