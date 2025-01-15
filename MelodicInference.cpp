#include "MelodicInference.h"


inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

MelodicInference::MelodicInference() {}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {
    // first load config and token mappings
    if (!loadConfig("model_weights/config.bin")) {
        return false;
    }

    // load each weight matrix separately
    if (!loadTokenEmbeddings("model_weights/token_embedding.bin") ||
        !loadPositionEmbeddings("model_weights/position_embedding.bin") ||
        !loadAttentionWeights() ||
        !loadAttentionBias() ||
        !loadLSTMWeights()) {
        return false;
    }

    DBG("loadModel check"); // Add this

    return test_attention();

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

    //DBG("\nToken mapping debug - Reading:");
    // Read token mappings
    
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

        //DBG("Token '" << token << "' -> index " << idx);

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

    // Allocate and read data
    size_t totalSize = config.vocab_size * config.embedding_dim;
    weights.token_embedding.resize(totalSize);
    bool success = stream.read(weights.token_embedding.data(), totalSize * sizeof(float)) == totalSize * sizeof(float);

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

    // Allocate and read data
    size_t totalSize = 32 * config.embedding_dim;
    weights.position_embedding.resize(totalSize);
    bool success = stream.read(weights.position_embedding.data(), totalSize * sizeof(float)) == totalSize * sizeof(float);

    return success;
}

bool MelodicInference::loadAttentionWeights() {
    // Load QKV weights
    juce::File qkvFile(R"(C:\Users\savas\Documents\JUCE Projects\generativedelay14\Model\model_weights\attention_qkv.bin)");

    if (!qkvFile.existsAsFile()) {
        DBG("QKV weights file not found");
        return false;
    }

    juce::FileInputStream stream(qkvFile);
    if (stream.failedToOpen()) return false;

    // Read dimensions
    int32_t numDims = stream.readInt();
    std::vector<int32_t> dims;
    for (int i = 0; i < numDims; i++) {
        dims.push_back(stream.readInt());
    }

    // Allocate and read data
    size_t totalSize = 3 * config.embedding_dim * config.embedding_dim; // 3x for Q,K,V
    weights.attention_qkv.resize(totalSize);
    
    bool success = stream.read(weights.attention_qkv.data(), totalSize * sizeof(float)) == totalSize * sizeof(float);

    //if (success) {
    //    DBG("\nLoaded attention weights verification:");

    //    // Print first row of Q weight
    //    DBG("Q weight first row:");
    //    for (int i = 0; i < config.embedding_dim; i++) {
    //        DBG(weights.attention_qkv[i]);
    //    }

    //    // Print first row of K weight
    //    DBG("\nK weight first row:");
    //    for (int i = 0; i < config.embedding_dim; i++) {
    //        DBG(weights.attention_qkv[config.embedding_dim * config.embedding_dim + i]);
    //    }

    //    // Print first row of V weight
    //    DBG("\nV weight first row:");
    //    for (int i = 0; i < config.embedding_dim; i++) {
    //        DBG(weights.attention_qkv[2 * config.embedding_dim * config.embedding_dim + i]);
    //    }
    //}

    return success;

}

bool MelodicInference::loadAttentionBias() {
    juce::File biasFile(R"(C:\Users\savas\Documents\JUCE Projects\generativedelay14\Model\model_weights\attention_bias.bin)");

    if (!biasFile.existsAsFile()) {
        DBG("Attention bias file not found");
        return false;
    }

    juce::FileInputStream stream(biasFile);
    if (stream.failedToOpen()) return false;

    // Read dimensions
    int32_t numDims = stream.readInt();
    std::vector<int32_t> dims;
    for (int i = 0; i < numDims; i++) {
        dims.push_back(stream.readInt());
    }

    // Allocate and read data  
    size_t totalSize = 3 * config.embedding_dim; // 3x for Q,K,V biases
    weights.attention_bias.resize(totalSize);
    
    bool success = stream.read(weights.attention_bias.data(), totalSize * sizeof(float)) == totalSize * sizeof(float);

    //if (success) {
    //    DBG("\nLoaded attention bias verification:");

    //    // Print Q bias
    //    DBG("Q bias:");
    //    for (int i = 0; i < config.embedding_dim; i++) {
    //        DBG(weights.attention_bias[i]);
    //    }

    //    // Print K bias
    //    DBG("\nK bias:");
    //    for (int i = 0; i < config.embedding_dim; i++) {
    //        DBG(weights.attention_bias[config.embedding_dim + i]);
    //    }

    //    // Print V bias
    //    DBG("\nV bias:");
    //    for (int i = 0; i < config.embedding_dim; i++) {
    //        DBG(weights.attention_bias[2 * config.embedding_dim + i]);
    //    }
    //}

    return success;

}



bool MelodicInference::loadLSTMWeights() {
    juce::File ihFile(R"(C:\Users\savas\Documents\JUCE Projects\generativedelay14\Model\model_weights\lstm_ih.bin)");
    juce::File hhFile(R"(C:\Users\savas\Documents\JUCE Projects\generativedelay14\Model\model_weights\lstm_hh.bin)");
    juce::File biasFile(R"(C:\Users\savas\Documents\JUCE Projects\generativedelay14\Model\model_weights\lstm_bias.bin)");

    if (!ihFile.existsAsFile() || !hhFile.existsAsFile() || !biasFile.existsAsFile()) {
        DBG("LSTM files not found");
        return false;
    }

    juce::FileInputStream ihStream(ihFile), hhStream(hhFile), biasStream(biasFile);
    if (!ihStream.openedOk() || !hhStream.openedOk() || !biasStream.openedOk()) return false;

    // Load dimensions
    ihStream.readInt();
    hhStream.readInt();
    biasStream.readInt();



    ihStream.readInt();
    hhStream.readInt();
    biasStream.readInt();

    ihStream.setPosition(12);
    hhStream.setPosition(12);
    biasStream.setPosition(12);

    // Debug reads for both files
    std::vector<uint8_t> raw_bytes_ih(9);
    std::vector<uint8_t> raw_bytes_hh(9);
    std::vector<uint8_t> raw_bytes_bias(9);
    ihStream.read(raw_bytes_ih.data(), 9);
    hhStream.read(raw_bytes_hh.data(), 9);
    biasStream.read(raw_bytes_bias.data(), 9);

    DBG("\nFirst 9 bytes read from ih file:");
    for (int i = 0; i < 9; i++) {
        DBG((int)raw_bytes_ih[i]);
    }

    DBG("\nFirst 9 bytes read from hh file:");
    for (int i = 0; i < 9; i++) {
        DBG((int)raw_bytes_hh[i]);
    }

    DBG("\nFirst 9 bytes read from bias file:");
    for (int i = 0; i < 9; i++) {
        DBG((int)raw_bytes_bias[i]);
    }

    ihStream.setPosition(12);  // Reset after debug read
    hhStream.setPosition(12);
    biasStream.setPosition(12);

    // allocate
    //weights.lstm_ih.resize(4 * config.hidden_size * config.embedding_dim);
    //weights.lstm_hh.resize(4 * config.hidden_size * config.hidden_size);
    //weights.lstm_bias.resize(4 * config.hidden_size);

    // Allocate without padding
    size_t ih_size = 4 * config.hidden_size * config.embedding_dim;
    size_t hh_size = 4 * config.hidden_size * config.hidden_size;
    size_t bias_size = 4 * config.hidden_size;
    weights.lstm_ih.resize(ih_size);
    weights.lstm_hh.resize(hh_size);
    weights.lstm_bias.resize(bias_size);

    //// Read data
    //return ihStream.read(weights.lstm_ih.data(), weights.lstm_ih.size() * sizeof(float)) == weights.lstm_ih.size() * sizeof(float) &&
    //    hhStream.read(weights.lstm_hh.data(), weights.lstm_hh.size() * sizeof(float)) == weights.lstm_hh.size() * sizeof(float) &&
    //    biasStream.read(weights.lstm_bias.data(), weights.lstm_bias.size() * sizeof(float)) == weights.lstm_bias.size() * sizeof(float);


    // Direct read into vectors
    return ihStream.read(weights.lstm_ih.data(), ih_size * sizeof(float)) == ih_size * sizeof(float) &&
        hhStream.read(weights.lstm_hh.data(), hh_size * sizeof(float)) == hh_size * sizeof(float) &&
        biasStream.read(weights.lstm_bias.data(), bias_size * sizeof(float)) == bias_size * sizeof(float);

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

    // Map weights to Eigen matrices with correct layout
    Eigen::Map<const Eigen::MatrixXf> q_weight_mat(weights.attention_qkv.data(),
        config.embedding_dim, config.embedding_dim);
    Eigen::Map<const Eigen::MatrixXf> k_weight_mat(weights.attention_qkv.data() + config.embedding_dim * config.embedding_dim,
        config.embedding_dim, config.embedding_dim);
    Eigen::Map<const Eigen::MatrixXf> v_weight_mat(weights.attention_qkv.data() + 2 * config.embedding_dim * config.embedding_dim,
        config.embedding_dim, config.embedding_dim);

    // Linear projections matching PyTorch F.linear
    Eigen::MatrixXf Q = embeddings * q_weight_mat + Eigen::Map<const Eigen::VectorXf>(weights.attention_bias.data(),
        config.embedding_dim).replicate(1, embeddings.rows()).transpose();
    Eigen::MatrixXf K = embeddings * k_weight_mat + Eigen::Map<const Eigen::VectorXf>(weights.attention_bias.data() + config.embedding_dim,
        config.embedding_dim).replicate(1, embeddings.rows()).transpose();
    Eigen::MatrixXf V = embeddings * v_weight_mat + Eigen::Map<const Eigen::VectorXf>(weights.attention_bias.data() + 2 * config.embedding_dim,
        config.embedding_dim).replicate(1, embeddings.rows()).transpose();


    // add debug prints
//    DBG("\nQ first 5 values:");
//    for (int i = 0; i < 5; i++) DBG(Q(0, i));



    // Scale Q
    float scale = 1.0f / std::sqrt(config.embedding_dim);
    Q *= scale;

    // Compute attention scores
    Eigen::MatrixXf scores = Q * K.transpose();

    

    // Create attention mask
    int seq_len = embeddings.rows();
    int window_size = 8;

    Eigen::MatrixXf mask = Eigen::MatrixXf::Ones(seq_len, seq_len);
    for (int i = 0; i < seq_len; i++) {
        int start = std::max(0, i - window_size);
        int end = std::min(seq_len, i + 1);
        mask.block(i, start, 1, end - start).setZero();
    }
    scores = (mask.array() == 1).select(-std::numeric_limits<float>::infinity(), scores);


//    DBG("\nScores after masking (first row):");
//    for (int i = 0; i < 5; i++) DBG(scores(0, i));






    // Apply softmax row-wise
    Eigen::MatrixXf attention_weights = Eigen::MatrixXf::Zero(seq_len, seq_len);
    for (int i = 0; i < seq_len; i++) {
        // Get max for numerical stability
        float max_val = scores.row(i).maxCoeff();
        Eigen::VectorXf exp_scores = (scores.row(i).array() - max_val).exp();
        attention_weights.row(i) = exp_scores / exp_scores.sum();
    }

    // Final multiplication with V
    Eigen::MatrixXf output = attention_weights * V;

//    DBG("\nAttention output (first row):");
//    for (int i = 0; i < 5; i++) DBG(output(0, i));

    return output;




}



Eigen::MatrixXf MelodicInference::processLSTM(const Eigen::MatrixXf& attention_output) {
    
    DBG("Entering processLSTM");

    int seq_len = attention_output.rows();

    // Map weights for forward direction
    Eigen::Map<const Eigen::MatrixXf> weight_ih_forward(
        weights.lstm_ih.data(),
        4 * config.hidden_size,  // 4 gates
        config.embedding_dim
    );

    Eigen::Map<const Eigen::MatrixXf> weight_hh_forward(
        weights.lstm_hh.data(),
        4 * config.hidden_size,
        config.hidden_size
    );

    // First half of bias is for forward direction
    Eigen::Map<const Eigen::VectorXf> bias_forward(
        weights.lstm_bias.data(),
        4 * config.hidden_size
    );

    // Map weights for reverse direction - starting from halfway point in memory
    Eigen::Map<const Eigen::MatrixXf> weight_ih_reverse(
        weights.lstm_ih.data() + (4 * config.hidden_size * config.embedding_dim),
        4 * config.hidden_size,
        config.embedding_dim
    );

    Eigen::Map<const Eigen::MatrixXf> weight_hh_reverse(
        weights.lstm_hh.data() + (4 * config.hidden_size * config.hidden_size),
        4 * config.hidden_size,
        config.hidden_size
    );

    // Second half of bias is for reverse direction
    Eigen::Map<const Eigen::VectorXf> bias_reverse(
        weights.lstm_bias.data() + (4 * config.hidden_size),
        4 * config.hidden_size
    );




    DBG("\nFirst 5 values after mapping:");
    DBG("weight_ih_l0:");
    for (int i = 0; i < 5; i++) {
        DBG(weight_ih_forward(0, i));
    }
    DBG("\nbias_ih_l0:");
    for (int i = 0; i < 5; i++) {
        DBG(bias_forward(i));
    }

    // debug prints:
    DBG("\nC++ LSTM debug:");
    DBG("Input shape: " << attention_output.rows() << "x" << attention_output.cols());

    DBG("\nLSTM weights:");
    DBG("weight_ih_l0 shape: " << weight_ih_forward.rows() << "x" << weight_ih_forward.cols());
    DBG("weight_hh_l0 shape: " << weight_hh_forward.rows() << "x" << weight_hh_forward.cols());
    DBG("bias_ih_l0 shape: " << bias_forward.size());

    DBG("weight_ih_l0 first row:");
    for (int i = 0; i < weight_ih_forward.cols(); i++) {
        DBG(weight_ih_forward(0, i));
    }

    DBG("weight_hh_l0 first row:");
    for (int i = 0; i < weight_hh_forward.cols(); i++) {
        DBG(weight_hh_forward(0, i));
    }

    DBG("bias_ih_l0:");
    for (int i = 0; i < bias_forward.size(); i++) {
        DBG(bias_forward(i));
    }





    // Process both directions
    Eigen::MatrixXf forward_output = processLSTMDirection(
        attention_output,
        weight_ih_forward,
        weight_hh_forward,
        bias_forward,
        false  // forward direction
    );

    Eigen::MatrixXf reverse_output = processLSTMDirection(
        attention_output,
        weight_ih_reverse,
        weight_hh_reverse,
        bias_reverse,
        true  // reverse direction
    );

    // Concatenate outputs along hidden dimension
    Eigen::MatrixXf combined_output(seq_len, 2 * config.hidden_size);
    combined_output << forward_output, reverse_output;

    return combined_output;


}

Eigen::MatrixXf MelodicInference::computeGates(
    const Eigen::VectorXf& x_t,
    const Eigen::VectorXf& h_prev,
    const Eigen::MatrixXf& w_ih,
    const Eigen::MatrixXf& w_hh,
    const Eigen::VectorXf& bias) {

    // Compute gates using matrix operations
    Eigen::VectorXf gates = (w_ih * x_t) + (w_hh * h_prev) + bias;

    // Split into individual gates
    int gate_size = config.hidden_size;
    Eigen::VectorXf i_g = gates.segment(0 * gate_size, gate_size);
    Eigen::VectorXf f_g = gates.segment(1 * gate_size, gate_size);
    Eigen::VectorXf g_g = gates.segment(2 * gate_size, gate_size);
    Eigen::VectorXf o_g = gates.segment(3 * gate_size, gate_size);



    for (int i = 0; i < gate_size; i++) {
        i_g[i] = sigmoid(i_g[i]);
        f_g[i] = sigmoid(f_g[i]);
        //g_g[i] = tanh(g_g[i]);
        g_g[i] = std::tanh(g_g[i]);
        o_g[i] = sigmoid(o_g[i]);
    }

    // Combine gates
    Eigen::VectorXf combined(4 * gate_size);
    combined << i_g, f_g, g_g, o_g;

    return combined;
}

Eigen::MatrixXf MelodicInference::processLSTMDirection(
    const Eigen::MatrixXf& input,
    const Eigen::MatrixXf& weight_ih,
    const Eigen::MatrixXf& weight_hh,
    const Eigen::VectorXf& bias,
    bool reverse) {

    int seq_len = input.rows();
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(seq_len, config.hidden_size);

    // Initialize states
    Eigen::VectorXf h_t = Eigen::VectorXf::Zero(config.hidden_size);
    Eigen::VectorXf c_t = Eigen::VectorXf::Zero(config.hidden_size);

    // Process sequence
    for (int t = 0; t < seq_len; t++) {
        int idx = reverse ? (seq_len - 1 - t) : t;

        // Get input at current timestep
        Eigen::VectorXf x_t = input.row(idx);

        // Compute gates
        Eigen::VectorXf gates = computeGates(x_t, h_t, weight_ih, weight_hh, bias);

        // Extract individual gates
        Eigen::VectorXf i_g = gates.segment(0, config.hidden_size);
        Eigen::VectorXf f_g = gates.segment(config.hidden_size, config.hidden_size);
        Eigen::VectorXf g_g = gates.segment(2 * config.hidden_size, config.hidden_size);
        Eigen::VectorXf o_g = gates.segment(3 * config.hidden_size, config.hidden_size);

        // Update cell state
        c_t = f_g.array() * c_t.array() + i_g.array() * g_g.array();

        // Update hidden state
        h_t = o_g.array() * c_t.array().tanh();

        // Store output
        output.row(idx) = h_t;
    }

    return output;
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




bool MelodicInference::test_attention() {
    
    // Create input of size [129, 32] to match Python
    Eigen::MatrixXf test_input = Eigen::MatrixXf::Zero(129, config.embedding_dim);
    // Fill first rows with actual token embeddings
    std::vector<int> test_tokens = { tokenToIdx["60"], tokenToIdx["45"] };
    auto token_emb = getTokenEmbeddings(test_tokens);
    auto combined = addPositionEmbeddings(token_emb);
    test_input.topRows(combined.rows()) = combined;

    auto attn_output = computeAttention(test_input);
    auto lstm_output = processLSTM(attn_output);

    return true;

}