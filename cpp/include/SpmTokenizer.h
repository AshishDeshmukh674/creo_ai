#pragma once
#include <string>
#include <vector>

#ifndef USE_SIMPLE_TOKENIZER
// Include SentencePiece for production tokenization
#include <sentencepiece_processor.h>
#else
// Simple tokenizer fallback when SentencePiece not available
#include <sstream>
#include <algorithm>
#endif

/**
 * SpmTokenizer: Wrapper class for SentencePiece tokenization
 * 
 * SentencePiece is a subword tokenization method used by T5 models.
 * It handles text preprocessing by breaking words into subword units,
 * enabling the model to handle out-of-vocabulary words and maintain
 * a fixed vocabulary size.
 * 
 * T5 Special Token IDs (standard configuration):
 * - PAD (padding): 0 - Used to pad sequences to same length
 * - EOS (end of sequence): 1 - Marks the end of generated text
 * - UNK (unknown): 2 - Represents unknown/rare tokens
 * 
 * Note: These IDs may vary depending on the specific T5 model variant.
 * Verify with your model's tokenizer configuration if needed.
 */
class SpmTokenizer {
public:
    /**
     * Load SentencePiece model from file
     * 
     * @param spmModelPath Path to the .model file (e.g., "spiece.model")
     * @return true if loading successful, false otherwise
     * 
     * The .model file contains:
     * - Vocabulary mappings (token ↔ ID)
     * - Subword segmentation rules
     * - Special token definitions
     */
    bool load(const std::string& spmModelPath);

    /**
     * Encode text string to token IDs
     * 
     * @param text Input text to tokenize (e.g., "Create a cube")
     * @return Vector of token IDs representing the text
     * 
     * Example:
     *   Input: "Create a cube"
     *   Output: [1234, 89, 5678] (actual IDs depend on vocabulary)
     */
    std::vector<int32_t> encode(const std::string& text) const;

    /**
     * Decode token IDs back to text string
     * 
     * @param ids Vector of token IDs to decode
     * @return Reconstructed text string
     * 
     * Example:
     *   Input: [1234, 89, 5678]
     *   Output: "Create a cube"
     */
    std::string decode(const std::vector<int32_t>& ids) const;

    /**
     * Get the token ID for padding
     * Used to pad variable-length sequences to the same length
     */
    int pad_id() const { return 0; }  // T5 standard padding token ID

    /**
     * Get the token ID for end-of-sequence
     * Used to mark the end of generated text during decoding
     */
    int eos_id() const { return 1; }  // T5 standard EOS token ID

private:
#ifndef USE_SIMPLE_TOKENIZER
    // SentencePiece processor instance that handles the actual tokenization
    sentencepiece::SentencePieceProcessor sp_;
#else
    // Simple tokenizer fallback - basic word splitting
    // For production, you would replace this with proper subword tokenization
    std::vector<std::string> simple_vocab_;
    void initSimpleVocab();
    int32_t getTokenId(const std::string& token) const;
#endif
};
