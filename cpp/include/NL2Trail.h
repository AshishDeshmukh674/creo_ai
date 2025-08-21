#pragma once
#include <string>
#include <vector>
#include <memory>

// Include ONNX Runtime for production inference
#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable: 4800 4996)
#endif

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
    #pragma warning(pop)
#endif

#include "SpmTokenizer.h"

/**
 * NL2Trail: Main class for Natural Language to Creo Trail conversion
 * 
 * This class loads an ONNX model exported from a fine-tuned T5 transformer
 * and performs text generation using greedy decoding to convert natural
 * language descriptions into Creo Parametric trail file commands.
 * 
 * Architecture:
 * - Uses ONNX Runtime for efficient CPU/GPU inference
 * - Integrates SentencePiece tokenizer for text preprocessing
 * - Implements greedy decoding for text generation
 */
class NL2Trail {
public:
    /**
     * Constructor: Initialize the model and tokenizer
     * 
     * @param onnxPath Path to the exported ONNX model file (.onnx)
     * @param spmPath Path to the SentencePiece model file (spiece.model)
     * @param maxNewTokens Maximum number of tokens to generate (default: 256)
     * 
     * @throws std::runtime_error if model or tokenizer fails to load
     */
    NL2Trail(const std::string& onnxPath, const std::string& spmPath,
             size_t maxNewTokens = 256);

    /**
     * Generate Creo trail commands from natural language input
     * 
     * @param nl Natural language description (e.g., "Create a 50mm cube")
     * @return Generated Creo trail file commands as a string
     * 
     * Example:
     *   Input: "Create a 50mm cube"
     *   Output: "~ Command `ProCmdDashboardActivate`\n~ Activate..."
     */
    std::string generate(const std::string& nl);

private:
    /**
     * Perform greedy decoding to generate token sequence
     * 
     * @param srcIds Tokenized input sequence (encoder input)
     * @return Vector of generated token IDs (decoder output)
     * 
     * This implements the core text generation loop:
     * 1. Initialize decoder with start token
     * 2. For each step, run the model to get next token probabilities
     * 3. Select highest probability token (greedy)
     * 4. Stop on EOS token or max length reached
     */
    std::vector<int32_t> greedyDecode(const std::vector<int32_t>& srcIds);

    /**
     * Find the index of maximum value in logits array
     * 
     * @param logits Array of probability scores for each vocabulary token
     * @param vocab Size of vocabulary (length of logits array)
     * @return Index of token with highest probability
     */
    int64_t argmax(const float* logits, size_t vocab);

private:
    // ONNX Runtime components for model inference
    Ort::Env env_;                    // ONNX Runtime environment
    Ort::SessionOptions opts_;        // Session configuration options
    std::unique_ptr<Ort::Session> session_;  // Model session for inference
    Ort::AllocatorWithDefaultOptions alloc_;  // Memory allocator

    // Tokenizer for text preprocessing and postprocessing
    SpmTokenizer tok_;

    // Generation configuration
    size_t max_new_tokens_;          // Maximum tokens to generate
};
