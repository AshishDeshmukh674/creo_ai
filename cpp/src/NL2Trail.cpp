#include "NL2Trail.h"
#include <stdexcept>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cstring>   
#include <array>    

NL2Trail::NL2Trail(const std::string& onnxPath, const std::string& spmPath,
                   size_t maxNewTokens)
    : env_(ORT_LOGGING_LEVEL_WARNING, "nl2trail"),
      opts_(),
      session_(nullptr),
      max_new_tokens_(maxNewTokens) {

    // Initialize session with proper API
    session_ = std::make_unique<Ort::Session>(env_, std::wstring(onnxPath.begin(), onnxPath.end()).c_str(), opts_);

    // Load tokenizer
    if (!tok_.load(spmPath)) {
        throw std::runtime_error("Failed to load SentencePiece model: " + spmPath);
    }
}

int64_t NL2Trail::argmax(const float* logits, size_t vocab) {
    size_t best = 0;
    float maxv = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab; ++i) {
        if (logits[i] > maxv) {
            maxv = logits[i];
            best = i;
        }
    }
    return static_cast<int64_t>(best);
}

std::vector<int32_t> NL2Trail::greedyDecode(const std::vector<int32_t>& srcIds) {
    // Real implementation using ONNX Runtime
    // Shapes:
    // input_ids: [1, src_seq]
    // attention_mask: [1, src_seq]
    // decoder_input_ids: [1, tgt_seq] (grows each step)
    // logits: [1, tgt_seq, vocab]

    // Prepare static inputs (encoder side)
    const int64_t batch = 1;
    const int64_t src_len = static_cast<int64_t>(srcIds.size());

    std::vector<int64_t> input_ids(srcIds.begin(), srcIds.end());
    std::vector<int64_t> attention_mask(src_len, 1);

    // Start decoder with a single <pad> token for T5
    std::vector<int64_t> dec_ids_vec = { tok_.pad_id() };

    // IO Names
    const char* in_names[] = {"input_ids", "attention_mask", "decoder_input_ids"};
    const char* out_names[] = {"logits"};

    // Create memory info
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // We don't know vocab size; we'll get it from output shape on first run
    size_t vocab = 0;

    for (size_t step = 0; step < max_new_tokens_; ++step) {
        int64_t tgt_len = static_cast<int64_t>(dec_ids_vec.size());

        // Create Ort tensors
        std::array<int64_t, 2> src_shape{batch, src_len};
        std::array<int64_t, 2> dec_shape{batch, tgt_len};

        Ort::Value input_ids_ort = Ort::Value::CreateTensor<int64_t>(mem, input_ids.data(), input_ids.size(), src_shape.data(), 2);
        Ort::Value attn_mask_ort = Ort::Value::CreateTensor<int64_t>(mem, attention_mask.data(), attention_mask.size(), src_shape.data(), 2);
        Ort::Value dec_ids_ort   = Ort::Value::CreateTensor<int64_t>(mem, dec_ids_vec.data(), dec_ids_vec.size(), dec_shape.data(), 2);

        // Run session
        std::vector<Ort::Value> output;
        {
            std::array<Ort::Value,3> inputs = {std::move(input_ids_ort), std::move(attn_mask_ort), std::move(dec_ids_ort)};
            output = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), inputs.size(), out_names, 1);
        }

        // Read logits
        auto info = output[0].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape(); // [1, tgt_len, vocab]
        if (shape.size() != 3) throw std::runtime_error("Unexpected logits rank");
        vocab = static_cast<size_t>(shape[2]);

        const float* logits = output[0].GetTensorData<float>();

        // Get the last time step logits (index tgt_len-1)
        const size_t last_step_offset = (static_cast<size_t>(tgt_len) - 1) * vocab;
        const float* last_logits = logits + last_step_offset;

        int64_t next_id = argmax(last_logits, vocab);

        // Stop on EOS
        if (next_id == tok_.eos_id()) {
            break;
        }
        // Append token and continue
        dec_ids_vec.push_back(next_id);
    }

    // Remove the initial <pad> and return the generated IDs
    if (!dec_ids_vec.empty() && dec_ids_vec.front() == tok_.pad_id())
        dec_ids_vec.erase(dec_ids_vec.begin());

    // Cast to int32_t for decoding
    std::vector<int32_t> out_ids(dec_ids_vec.begin(), dec_ids_vec.end());
    return out_ids;
}

std::string NL2Trail::generate(const std::string& nl) {
    // Encode NL
    auto src_ids32 = tok_.encode(nl);
    if (src_ids32.empty()) return "";

    std::vector<int32_t> gen_ids = greedyDecode(std::vector<int32_t>(src_ids32.begin(), src_ids32.end()));
    // Decode to text
    return tok_.decode(gen_ids);
}


