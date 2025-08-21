/**
 * SpmTokenizer implementation for SentencePiece tokenization
 * 
 * This file implements the wrapper class for Google's SentencePiece library,
 * which provides subword tokenization used by T5 and other transformer models.
 * 
 * When USE_SIMPLE_TOKENIZER is defined, it falls back to a basic tokenizer.
 */

#include "SpmTokenizer.h"
#include <stdexcept>

#ifdef USE_SIMPLE_TOKENIZER
#include <iostream>
#include <unordered_map>
#include <cctype>
#endif

#ifdef USE_SIMPLE_TOKENIZER
/**
 * Initialize simple vocabulary for fallback tokenizer
 * This is a basic implementation - in production you'd load a proper vocabulary
 */
void SpmTokenizer::initSimpleVocab() {
    // Basic vocabulary for Creo commands (simplified)
    simple_vocab_ = {
        "<pad>", "<eos>", "<unk>",  // Special tokens
        "create", "sketch", "extrude", "revolve", "sweep", "blend",
        "circle", "rectangle", "line", "arc", "spline", "point",
        "dimension", "constraint", "pattern", "mirror", "copy",
        "cube", "cylinder", "sphere", "cone", "torus",
        "mm", "inch", "degree", "radius", "diameter", "length",
        "width", "height", "depth", "angle", "distance",
        "feature", "surface", "solid", "assembly", "part",
        "modify", "edit", "delete", "hide", "show", "zoom",
        "view", "rotate", "translate", "scale", "measure"
    };
}

/**
 * Get token ID for a given token (simple mapping with reverse lookup)
 */
int32_t SpmTokenizer::getTokenId(const std::string& token) const {
    // Find token in vocabulary
    auto it = std::find(simple_vocab_.begin(), simple_vocab_.end(), token);
    if (it != simple_vocab_.end()) {
        return static_cast<int32_t>(std::distance(simple_vocab_.begin(), it));
    }
    return 2; // Return <unk> token ID for unknown words
}
#endif

/**
 * Load SentencePiece model from file
 * 
 * @param path Path to the .model file (typically "spiece.model")
 * @return true if successful, false if loading failed
 */
bool SpmTokenizer::load(const std::string& path) {
#ifndef USE_SIMPLE_TOKENIZER
    // Load the SentencePiece model using the processor
    // The model file contains vocabulary and segmentation rules
    auto status = sp_.Load(path);
    
    // Check if loading was successful using SentencePiece status
    return status.ok();
#else
    // Simple tokenizer - initialize vocabulary
    initSimpleVocab();
    std::cout << "Using simple tokenizer (SentencePiece not available)" << std::endl;
    return true;  // Always succeeds for simple tokenizer
#endif
}

/**
 * Encode text string into sequence of token IDs
 * 
 * @param text Input text to tokenize
 * @return Vector of token IDs representing the input text
 */
std::vector<int32_t> SpmTokenizer::encode(const std::string& text) const {
#ifndef USE_SIMPLE_TOKENIZER
    // Temporary vector to hold integer token IDs from SentencePiece
    std::vector<int> ids;
    
    // Use SentencePiece to tokenize the text into integer IDs
    // This breaks the text into subword units based on the trained model
    sp_.Encode(text, &ids);
    
    // Convert from int to int32_t for consistency with ONNX Runtime
    // (ONNX Runtime expects int32_t for input token IDs)
    return std::vector<int32_t>(ids.begin(), ids.end());
#else
    // Simple tokenizer implementation
    std::vector<int32_t> result;
    std::istringstream iss(text);
    std::string token;
    
    // Simple whitespace tokenization
    while (iss >> token) {
        // Convert to lowercase for consistent matching
        std::transform(token.begin(), token.end(), token.begin(), ::tolower);
        result.push_back(getTokenId(token));
    }
    
    return result;
#endif
}

/**
 * Decode sequence of token IDs back into text string
 * 
 * @param ids Vector of token IDs to decode
 * @return Reconstructed text string
 */
std::string SpmTokenizer::decode(const std::vector<int32_t>& ids) const {
#ifndef USE_SIMPLE_TOKENIZER
    // Convert int32_t back to int for SentencePiece API compatibility
    std::vector<int> v(ids.begin(), ids.end());
    
    // Output string to hold the decoded text
    std::string out;
    
    // Use SentencePiece to convert token IDs back to readable text
    // This combines subword units back into natural language
    sp_.Decode(v, &out);
    
    return out;
#else
    // Simple tokenizer decode - generate meaningful Creo commands
    std::string result;
    
    // For simple tokenizer, generate actual Creo trail commands based on input patterns
    // This is a demo implementation - in production you'd use the trained model output
    bool has_cube = false, has_circle = false, has_rectangle = false;
    
    // Check what type of command was requested (basic pattern matching)
    for (int32_t id : ids) {
        if (id >= 0 && id < static_cast<int32_t>(simple_vocab_.size())) {
            std::string token = simple_vocab_[id];
            if (token == "cube") has_cube = true;
            else if (token == "circle") has_circle = true;
            else if (token == "rectangle") has_rectangle = true;
        }
    }
    
    // Generate appropriate Creo trail commands
    if (has_cube) {
        result = "~ Command `ProCmdDashboardActivate`\n"
                "~ Activate sketch\n"
                "~ Command `ProCmdSquare`\n"
                "~ Create square sketch\n"
                "~ Command `ProCmdDimLinear`\n"
                "~ Set dimension 50mm\n"
                "~ Command `ProCmdSketchDone`\n"
                "~ Exit sketch\n"
                "~ Command `ProCmdExtrude`\n"
                "~ Extrude 50mm\n"
                "~ Command `ProCmdFeatureDone`\n"
                "! Created 50mm cube";
    } else if (has_circle) {
        result = "~ Command `ProCmdDashboardActivate`\n"
                "~ Activate sketch\n"
                "~ Command `ProCmdCircle`\n"
                "~ Create circle\n"
                "~ Command `ProCmdDimDiameter`\n"
                "~ Set diameter\n"
                "~ Command `ProCmdSketchDone`\n"
                "! Created circle";
    } else if (has_rectangle) {
        result = "~ Command `ProCmdDashboardActivate`\n"
                "~ Activate sketch\n"
                "~ Command `ProCmdRectangle`\n"
                "~ Create rectangle\n"
                "~ Command `ProCmdDimLinear`\n"
                "~ Set dimensions\n"
                "~ Command `ProCmdSketchDone`\n"
                "! Created rectangle";
    } else {
        // Generic response for unknown commands
        result = "~ Command `ProCmdDashboardActivate`\n"
                "~ Activate modeling environment\n"
                "! Ready for feature creation";
    }
    
    return result;
#endif
}
