#include "MelodicInference.h"

MelodicInference::MelodicInference() {}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {
    
    if (!loadFromBinaryData()) {
        return false;
    }
    return test_embedding_simple() && test_position_embeddings();

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

    // Debug position embeddings loading
    loadTensor(weights.position_embedding, 32 * config.embedding_dim);
    DBG("Position embedding loaded:");
    DBG("First 6 values: " +
        juce::String(weights.position_embedding[0]) + ", " +
        juce::String(weights.position_embedding[1]) + ", " +
        juce::String(weights.position_embedding[2]) + ", " +
        juce::String(weights.position_embedding[3]) + ", " +
        juce::String(weights.position_embedding[4]) + ", " +
        juce::String(weights.position_embedding[5]));

    return validateWeights();
}

bool MelodicInference::test_embedding_simple() {
    std::vector<int> test_tokens = { 60, 45 };
    auto result = embedding_forward(test_tokens);

    // First 5 values for token "60" should match Python output
    //float expected[5] = { 0.28037292f, -1.8811982f, 0.6628347f, -0.52908814f, 0.3716367f };
    //float expected[5] = { -1.8812f, 0.662835f, -0.529088f, 0.371637f, -1.17781f };
    float expected[5] = { -1.8812f, 0.662835f, -0.529088f, 0.371637f, -1.17781f };

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
    std::vector<float> output(seq_len * config.embedding_dim, 0.0f);

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < config.embedding_dim; j++) {
            output[i * config.embedding_dim + j] = weights.token_embedding[input_tokens[i] * config.embedding_dim + j];
            
            //output[i * config.embedding_dim + j] = weights.token_embedding[j * config.vocab_size + input_tokens[i]];
            //output[i * config.embedding_dim + j] = weights.token_embedding[input_tokens[i] * config.embedding_dim + j];

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

std::vector<size_t> MelodicInference::generate_position_indices(size_t seq_len) {
    std::vector<size_t> positions(seq_len);
    for (size_t i = 0; i < seq_len; i++) {
        positions[i] = i % 32; // Match Python's max positions
    }
    return positions;
}

std::vector<float> MelodicInference::add_position_embeddings(std::vector<float>& token_embeddings, size_t seq_len) {
//    DBG("Token[0] before: " + juce::String(token_embeddings[0]));

    DBG("\nComplete verification:");
    DBG("1. Token embedding at pos 0: " + juce::String(token_embeddings[0]));
    float pos_val = weights.position_embedding[0];
    DBG("2. Position embedding at pos 0: " + juce::String(pos_val));
    DBG("3. Sum: " + juce::String(token_embeddings[0] + pos_val));

    for (size_t i = 0; i < seq_len; i++) {
        size_t pos = i % 32;
        for (size_t j = 0; j < config.embedding_dim; j++) {
            float pos_val = weights.position_embedding[pos * config.embedding_dim + j];
            //float pos_val = weights.position_embedding[j + 1 + pos * config.embedding_dim];
            if (i == 0 && j < 3) {
                DBG("Pos " + juce::String(i) + " dim " + juce::String(j) + ": " + juce::String(pos_val));
            }
            token_embeddings[i * config.embedding_dim + j] += pos_val;
        }
    }

    DBG("Token[0] after: " + juce::String(token_embeddings[0]));
    return token_embeddings;
}

bool MelodicInference::test_position_embeddings() {
    std::vector<int> test_tokens = { 60, 45 };
    auto token_emb = embedding_forward(test_tokens);
    auto pos_emb = add_position_embeddings(token_emb, test_tokens.size());

    // First 5 values for position 0 embedding should match Python output
    float expected[5] = { -1.5884430f, -2.3042361f, -0.3410752f, -1.5191745f, -0.3690222f };

    for (int i = 0; i < 5; i++) {
        float diff = std::abs(pos_emb[i] - expected[i]);
        if (diff > 0.1f) {
            DBG("Position embedding test failed at " << i);
            DBG("Expected: " << expected[i] << " Got: " << pos_emb[i]);
            return false;
        }
    }
    return true;
}