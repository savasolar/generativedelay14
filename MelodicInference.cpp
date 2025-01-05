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

    // Read token mappings
    while (pos < BinaryData::model_weights_binSize) {
        int32_t tokenLen = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;
        if (pos + tokenLen + 4 > BinaryData::model_weights_binSize) break;

        std::string token(data + pos, tokenLen); pos += tokenLen;
        int32_t idx = *reinterpret_cast<const int32_t*>(data + pos); pos += 4;

        tokenToIdx[token] = idx;
        idxToToken[idx] = token;
    }

    // Load weights helper
    auto loadTensor = [&pos, data](std::vector<float>& vec, size_t size) {
        vec.resize(size);
        memcpy(vec.data(), data + pos, size * sizeof(float));
        pos += size * sizeof(float);
        };


    auto loadTensor2 = [&pos, data](std::vector<float>& vec, size_t size) {
        vec.resize(size);
        // Step back one float before copying
        pos -= sizeof(float);
        memcpy(vec.data(), data + pos, size * sizeof(float));
        pos += size * sizeof(float);
        };

    // Load all tensors in order matching Python export
    loadTensor(weights.token_embedding, config.vocab_size * config.embedding_dim);
    loadTensor2(weights.position_embedding, 32 * config.embedding_dim);
    //loadTensor(weights.attention_qkv, /* size */);  // TODO: Add correct sizes
    //loadTensor(weights.attention_bias, /* size */);
    //loadTensor(weights.lstm_ih, /* size */);
    //loadTensor(weights.lstm_hh, /* size */);
    //loadTensor(weights.lstm_bias, /* size */);
    //loadTensor(weights.output, /* size */);
    //loadTensor(weights.output_bias, /* size */);

    //DBG("\nEntire position embedding array (" << weights.position_embedding.size() << " values):");
    //for (size_t i = 0; i < weights.position_embedding.size(); i++) {
    //    DBG("Index " << i << ": " << weights.position_embedding[i]);
    //    if (i % config.embedding_dim == 0) {
    //        DBG("----- Position " << (i / config.embedding_dim) << " -----");
    //    }
    //}

    return true;
}

std::vector<std::string> MelodicInference::generate(const std::vector<std::string>& prompt, float temperature, int topK) {
    std::vector<int> tokens;
    for (const auto& token : prompt) {
        tokens.push_back(tokenToIdx[token]);
    }

    // TODO: Implement generation logic using forward()
    return {};
}

Eigen::MatrixXf MelodicInference::getTokenEmbeddings(const std::vector<int>& input_tokens) {
    Eigen::MatrixXf embeddings(input_tokens.size(), config.embedding_dim);

    for (int i = 0; i < input_tokens.size(); i++) {
        int token_idx = input_tokens[i];

        // Debug inside the loop
        if (token_idx == 60) {
            DBG("Token " << token_idx << " embedding starts at offset: " << (token_idx * config.embedding_dim));
            DBG("First 5 values in weights.token_embedding:");
            for (int j = 0; j < 5; j++) {
                DBG(weights.token_embedding[token_idx * config.embedding_dim + j]);
            }
        }

        const float* token_emb = weights.token_embedding.data() + token_idx * config.embedding_dim;
        embeddings.row(i) = Eigen::Map<const Eigen::VectorXf>(token_emb, config.embedding_dim);
    }

    return embeddings;
}

//Eigen::MatrixXf MelodicInference::addPositionEmbeddings(const Eigen::MatrixXf& token_embeddings) {
//    
//    // get sequence length
//    int seq_len = token_embeddings.rows();
//    // limit to max 32 positions
//    seq_len = std::min(seq_len, 32);
//
//    // create matrix same size as input
//    Eigen::MatrixXf output = token_embeddings;
//
//    // for each position in sequence
//    for (int pos = 0; pos < seq_len; pos++) {
//        // get position embedding vector
//        //const float* pos_emb = weights.position_embedding.data() + pos * config.embedding_dim;
//        const float* pos_emb = weights.position_embedding.data() + pos;
//        output.row(pos) += Eigen::Map<const Eigen::VectorXf>(pos_emb, config.embedding_dim, config.embedding_dim);
//
//
//        // add position embedding to token embedding at this position
//        output.row(pos) += Eigen::Map<const Eigen::VectorXf>(
//            pos_emb, config.embedding_dim);
//    }
//
//    // Debug position 0 values
//    DBG("Position 0 embedding values:");
//    for (int i = 0; i < 5; i++) {
//        DBG(weights.position_embedding[i]);
//    }
//
//    return output;
//}

//Eigen::MatrixXf MelodicInference::addPositionEmbeddings(const Eigen::MatrixXf& token_embeddings) {
//    int seq_len = std::min((int)token_embeddings.rows(), 32);
//    Eigen::MatrixXf output = token_embeddings;
//
//    for (int pos = 0; pos < seq_len; pos++) {
//        // Start reading one position earlier in the position embeddings
//        const float* pos_emb = weights.position_embedding.data() + (pos * config.embedding_dim) - config.embedding_dim;
//        output.row(pos) += Eigen::Map<const Eigen::VectorXf>(pos_emb, config.embedding_dim);
//    }
//
//    return output;
//}

Eigen::MatrixXf MelodicInference::addPositionEmbeddings(const Eigen::MatrixXf& token_embeddings) {
    int seq_len = std::min((int)token_embeddings.rows(), 32);
    Eigen::MatrixXf output = token_embeddings;

    // Add offset correction
    const float* base_pos_emb = weights.position_embedding.data();

    for (int pos = 0; pos < seq_len; pos++) {
        const float* pos_emb = base_pos_emb + pos * config.embedding_dim;
        output.row(pos) += Eigen::Map<const Eigen::VectorXf>(pos_emb, config.embedding_dim);

        // Debug first few values
        if (pos < 2) {
            DBG("Position " << pos << " first 3 values: "
                << pos_emb[0] << ", " << pos_emb[1] << ", " << pos_emb[2]);
        }
    }
    return output;
}

Eigen::MatrixXf MelodicInference::computeAttention(const Eigen::MatrixXf& embeddings) {
    // TODO: Implement self-attention using weights.attention_*
    return Eigen::MatrixXf();
}

Eigen::MatrixXf MelodicInference::processLSTM(const Eigen::MatrixXf& attention_output) {
    // TODO: Implement LSTM using weights.lstm_*
    return Eigen::MatrixXf();
}

Eigen::VectorXf MelodicInference::computeLogits(const Eigen::MatrixXf& lstm_output) {
    // TODO: Implement final linear layer using weights.output*
    return Eigen::VectorXf();
}

Eigen::VectorXf MelodicInference::forward(const std::vector<int>& tokens) {
    // TODO: Implement full forward pass
    return Eigen::VectorXf();
}

bool MelodicInference::test_embedding_simple() {
    DBG("Running token embedding test...");
    std::vector<int> test_tokens = { 60, 45 };
    auto result = getTokenEmbeddings(test_tokens);

    DBG("Token 60 first 5 embeddings: " << result(0, 0) << ", " << result(0, 1) << ", "
        << result(0, 2) << ", " << result(0, 3) << ", " << result(0, 4));

    float expected[5] = { -1.8812f, 0.662835f, -0.529088f, 0.371637f, -1.17781f };

    for (int i = 0; i < 5; i++) {
        float diff = std::abs(result(0, i) - expected[i]);
        if (diff > 0.1f) {
            DBG("Embedding test failed at " << i);
            DBG("Expected: " << expected[i] << " Got: " << result(0, i));
            return false;
        }
    }

    DBG("Token embedding test passed!");
    return true;
}

bool MelodicInference::test_position_embeddings() {
    DBG("Testing raw position embeddings first:");
    float expected_raw[5] = { -1.8688159f, -0.42303798f, -1.00391f, -0.99008644f, -0.74065894f };

    // Check raw values match Python's output
    for (int i = 0; i < 5; i++) {
        float diff = std::abs(weights.position_embedding[i] - expected_raw[i]);
        if (diff > 0.1f) {
            DBG("Raw position value mismatch at " << i);
            DBG("Expected: " << expected_raw[i] << " Got: " << weights.position_embedding[i]);
            return false;
        }
    }
    DBG("Raw position embeddings match Python values");

    // Now test the combined embeddings
    std::vector<int> test_tokens = { 60, 45 };
    auto token_emb = getTokenEmbeddings(test_tokens);
    auto result = addPositionEmbeddings(token_emb);

    // Combined values (token + position) for first position
    float expected_combined[5] = { -1.5884430f, -2.3042361f, -0.3410752f, -1.5191745f, -0.3690222f };

    for (int i = 0; i < 5; i++) {
        float diff = std::abs(result(0, i) - expected_combined[i]);
        if (diff > 0.1f) {
            DBG("Combined embedding test failed at " << i);
            DBG("Expected: " << expected_combined[i] << " Got: " << result(0, i));
            return false;
        }
    }
    return true;
}
