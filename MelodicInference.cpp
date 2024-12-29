#include "MelodicInference.h"

MelodicInference::MelodicInference() {}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {
    // For now, just return true
    return true;
}

bool MelodicInference::loadFromBinaryData() {
    auto* data = BinaryData::model_weights_bin;
    size_t pos = 0;

    // Read config
    config.vocab_size = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;
    config.embedding_dim = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;
    config.hidden_size = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;

    // Read token mappings
    while (pos < BinaryData::model_weights_binSize) {
        int32_t tokenLen = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;
        if (pos + tokenLen + 4 > BinaryData::model_weights_binSize) break;

        std::string token(data + pos, tokenLen); pos += tokenLen;
        int32_t idx = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;

        tokenToIdx[token] = idx;
        idxToToken[idx] = token;
    }

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

    return validateWeights();
}

std::vector<std::string> MelodicInference::generate(const std::vector<std::string>& prompt, float temperature, int topK) {
    // Debug melody for testing
    return { "60", "-", "-", "-", "64", "-", "-", "-" };
}

std::vector<float> MelodicInference::embedding_forward(const std::vector<int>& input_tokens) {
    size_t seq_len = input_tokens.size();
    size_t max_positions = std::min(seq_len, size_t(32));
    std::vector<float> output(seq_len * config.embedding_dim, 0.0f);

    // Token embeddings
    for (size_t i = 0; i < seq_len; i++) {
        size_t token_idx = input_tokens[i] * config.embedding_dim;
        for (size_t j = 0; j < config.embedding_dim; j++) {
            output[i * config.embedding_dim + j] = weights.token_embedding[token_idx + j];
        }
    }

    // Position embeddings
    for (size_t i = 0; i < seq_len; i++) {
        size_t pos = i % 32;
        for (size_t j = 0; j < config.embedding_dim; j++) {
            output[i * config.embedding_dim + j] += weights.position_embedding[pos * config.embedding_dim + j];
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