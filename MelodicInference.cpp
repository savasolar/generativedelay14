#include "MelodicInference.h"

MelodicInference::MelodicInference() {}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {
    // first load config and token mappings
    if (!loadConfig("model_weights/config.bin")) {
        return false;
    }

    // load each weight matrix separately
    if (!loadTokenEmbeddings("model_weights/token_embedding.bin") ||
        !loadPositionEmbeddings("model_weights/position_embedding.bin")) {
        return false;
    }

    return test_embedding_simple()/* && test_position_embeddings()*/;
}

bool MelodicInference::loadConfig(const std::string& filename) {
    // Use absolute path where we know the files are
    juce::File configFile(R"(C:\Users\savas\Documents\JUCE Projects\generativedelay14\Model\)" + filename);

    if (!configFile.existsAsFile()) {
        DBG("Config file not found: " << configFile.getFullPathName());
        return false;
    }

    juce::FileInputStream stream(configFile);
    if (stream.failedToOpen()) {
        return false;
    }

    // Read config values
    config.vocab_size = stream.readInt();
    config.embedding_dim = stream.readInt();
    config.hidden_size = stream.readInt();

    DBG("\nToken mapping debug - Reading:");
    // Read token mappings
    /*while (!stream.isExhausted()) {
        int32_t tokenLen = stream.readInt();
        if (stream.isExhausted()) break;

        juce::String token = stream.readString();
        int32_t idx = stream.readInt();

        DBG("Token '" << token.toStdString() << "' -> index " << idx);

        tokenToIdx[token.toStdString()] = idx;
        idxToToken[idx] = token.toStdString();
    }*/

    while (!stream.isExhausted()) {
        // Read token length
        int32_t tokenLen = stream.readInt();
        if (stream.isExhausted()) break;

        // Read exactly tokenLen bytes into a string
        std::vector<char> tokenBuffer(tokenLen);
        if (stream.read(tokenBuffer.data(), tokenLen) != tokenLen) {
            DBG("Failed to read token bytes");
            return false;
        }
        std::string token(tokenBuffer.begin(), tokenBuffer.end());

        // Read token index
        int32_t idx = stream.readInt();

        DBG("Token '" << token << "' -> index " << idx);

        tokenToIdx[token] = idx;
        idxToToken[idx] = token;
    }

    return true;
}

bool MelodicInference::loadTokenEmbeddings(const std::string& filename) {
    juce::File embedFile(R"(C:\Users\savas\Documents\JUCE Projects\generativedelay14\Model\)" + filename);
    ;

    if (!embedFile.existsAsFile()) {
        DBG("Token embeddings file not found: " << embedFile.getFullPathName());
        return false;
    }

    juce::FileInputStream stream(embedFile);
    if (stream.failedToOpen()) {
        return false;
    }

    // Read dimensions
    int32_t numDims = stream.readInt();
    std::vector<int32_t> dims;
    for (int i = 0; i < numDims; i++) {
        dims.push_back(stream.readInt());
    }

    //DBG("Token embeddings dimensions:");
    //for (auto dim : dims) {
    //    DBG(" - " << dim);
    //}

    // Allocate and read data
    size_t totalSize = config.vocab_size * config.embedding_dim;
    weights.token_embedding.resize(totalSize);
    bool success = stream.read(weights.token_embedding.data(), totalSize * sizeof(float)) == totalSize * sizeof(float);

    //// Debug first 5 values
    //DBG("First 5 token embedding values:");
    //for (int i = 0; i < 5; i++) {
    //    DBG(weights.token_embedding[i]);
    //}

    return success;
}

bool MelodicInference::loadPositionEmbeddings(const std::string& filename) {
    juce::File embedFile(R"(C:\Users\savas\Documents\JUCE Projects\generativedelay14\Model\)" + filename);

    if (!embedFile.existsAsFile()) {
        DBG("Position embeddings file not found: " << embedFile.getFullPathName());
        return false;
    }

    juce::FileInputStream stream(embedFile);
    if (stream.failedToOpen()) {
        return false;
    }

    // Read dimensions
    int32_t numDims = stream.readInt();
    std::vector<int32_t> dims;
    for (int i = 0; i < numDims; i++) {
        dims.push_back(stream.readInt());
    }

    //DBG("Position embeddings dimensions:");
    //for (auto dim : dims) {
    //    DBG(" - " << dim);
    //}

    // Allocate and read data
    size_t totalSize = 32 * config.embedding_dim;
    weights.position_embedding.resize(totalSize);
    bool success = stream.read(weights.position_embedding.data(), totalSize * sizeof(float)) == totalSize * sizeof(float);

    //// Debug first 5 values
    //DBG("First 5 position embedding values:");
    //for (int i = 0; i < 5; i++) {
    //    DBG(weights.position_embedding[i]);
    //}

    return success;
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

        //// Debug inside the loop
        //if (token_idx == 60) {
        //    DBG("Token " << token_idx << " embedding starts at offset: " << (token_idx * config.embedding_dim));
        //    DBG("First 5 values in weights.token_embedding:");
        //    for (int j = 0; j < 5; j++) {
        //        DBG(weights.token_embedding[token_idx * config.embedding_dim + j]);
        //    }
        //}

        const float* token_emb = weights.token_embedding.data() + token_idx * config.embedding_dim;
        embeddings.row(i) = Eigen::Map<const Eigen::VectorXf>(token_emb, config.embedding_dim);

    }

    return embeddings;
}

Eigen::MatrixXf MelodicInference::addPositionEmbeddings(const Eigen::MatrixXf& token_embeddings) {
    int seq_len = std::min((int)token_embeddings.rows(), 32);
    Eigen::MatrixXf output = token_embeddings;

    // Add offset correction
    const float* base_pos_emb = weights.position_embedding.data();

    for (int pos = 0; pos < seq_len; pos++) {
        const float* pos_emb = base_pos_emb + pos * config.embedding_dim;
        output.row(pos) += Eigen::Map<const Eigen::VectorXf>(pos_emb, config.embedding_dim);
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
    std::vector<int> test_tokens = { tokenToIdx["60"], tokenToIdx["45"] };
    auto result = getTokenEmbeddings(test_tokens);

    DBG("\nPrompt '60 45' debug:");
    DBG("Token embedding first 5 values:");
    for (int i = 0; i < 5; i++) {
        DBG(result(0, i));
    }

    DBG("\nPosition embedding first 5 values:");
    for (int i = 0; i < 5; i++) {
        DBG(weights.position_embedding[i]);
    }

    auto combined = addPositionEmbeddings(result);
    DBG("\nCombined first 5 values:");
    for (int i = 0; i < 5; i++) {
        DBG(combined(0, i));
    }

    return true;
}

