#include "MelodicInference.h"

MelodicInference::MelodicInference() {}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {
    
    if (!loadFromBinaryData()) {
        return false;
    }
    return test_embedding_simple();

}

bool MelodicInference::loadFromBinaryData() {
    auto* data = BinaryData::model_weights_bin;
    size_t pos = 0;

    // Read config
    config.vocab_size = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;
    config.embedding_dim = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;
    config.hidden_size = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;

    // Add debug prints here
    DBG("First token mapping indices:");
    int count = 0;
    for (const auto& [token, idx] : tokenToIdx) {
        if (count++ < 5) DBG(token << ": " << idx);
    }

    // Read token mappings
    while (pos < BinaryData::model_weights_binSize) {
        int32_t tokenLen = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;
        if (pos + tokenLen + 4 > BinaryData::model_weights_binSize) break;

        std::string token(data + pos, tokenLen); pos += tokenLen;
        int32_t idx = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;

        tokenToIdx[token] = idx;
        idxToToken[idx] = token;
    }

    //size_t token_60_start = 60 * config.embedding_dim;
    //DBG("token_60_start: " << token_60_start);
    //DBG("config.vocab_size: " << config.vocab_size);
    //DBG("config.embedding_dim: " << config.embedding_dim);

    // Load weights
    auto loadTensor = [&pos, data](std::vector<float>& vec, size_t size) {
        vec.resize(size);
        memcpy(vec.data(), data + pos, size * sizeof(float));
        pos += size * sizeof(float);
        };

    // Load all tensors in order matching Python export
    loadTensor(weights.token_embedding, config.vocab_size * config.embedding_dim);
    loadTensor(weights.position_embedding, 32 * config.embedding_dim);
    loadTensor(weights.attention_qkv, 3 * config.embedding_dim * config.embedding_dim);
    loadTensor(weights.attention_bias, 3 * config.embedding_dim);
    loadTensor(weights.lstm_ih, 4 * config.hidden_size * config.embedding_dim);
    loadTensor(weights.lstm_hh, 4 * config.hidden_size * config.hidden_size);
    loadTensor(weights.lstm_bias, 4 * config.hidden_size);
    loadTensor(weights.output, config.vocab_size * (config.hidden_size * 2));
    loadTensor(weights.output_bias, config.vocab_size);

    DBG("Loaded token_embedding size: " << weights.token_embedding.size());
    DBG("Expected size: " << config.vocab_size * config.embedding_dim);
    DBG("Config vocab_size: " << config.vocab_size);
    DBG("Config embedding_dim: " << config.embedding_dim);

    return validateWeights();
}

bool MelodicInference::test_embedding_simple() {
    std::vector<int> test_tokens = { 60, 45 };
    auto result = embedding_forward(test_tokens);

    // First 5 values for token "60" should match Python output
    float expected[5] = { 0.28037292f, -1.8811982f, 0.6628347f, -0.52908814f, 0.3716367f };

    for (int i = 0; i < 5; i++) {
        float diff = std::abs(result[i] - expected[i]);
        if (diff > 0.1f) {
            DBG("Embedding test failed at " << i);
            DBG("Expected: " << expected[i] << " Got: " << result[i]);
            return false;
        }
    }
    return true;
}

std::vector<std::string> MelodicInference::generate(const std::vector<std::string>& prompt, float temperature, int topK) {
    // Debug melody for testing
    return { "60", "-", "-", "-", "64", "-", "-", "-" };
}

std::vector<float> MelodicInference::embedding_forward(const std::vector<int>& input_tokens) {
    size_t seq_len = input_tokens.size();
    size_t max_positions = std::min(seq_len, size_t(32));
    std::vector<float> output(seq_len * config.embedding_dim, 0.0f);

    // Print dimensions for debugging
    DBG("config.embedding_dim: " << config.embedding_dim);
    DBG("weights.token_embedding.size(): " << weights.token_embedding.size());

    for (size_t i = 0; i < seq_len; i++) {
        // Debug the index calculation
        size_t token_offset = input_tokens[i] * config.embedding_dim;
        DBG("Token " << input_tokens[i] << " offset: " << token_offset);

        for (size_t j = 0; j < config.embedding_dim; j++) {
            if (token_offset + j >= weights.token_embedding.size()) {
                DBG("Index out of bounds: " << token_offset + j);
                continue;
            }
            output[i * config.embedding_dim + j] = weights.token_embedding[token_offset + j];
        }
    }
    return output;
}

bool MelodicInference::loadNPZ(const char* path) {
    return true;
}

bool MelodicInference::validateWeights() {
    return true;
}