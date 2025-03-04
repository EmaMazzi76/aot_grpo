#!/bin/bash
# Script to run AOT-GRPO test with a local Phi-mini model

# Set default values
MODEL_PATH="${1:-$HOME/Downloads/phi-mini}"
DATASET="${2:-gsm8k}"
SAMPLES="${3:-1}"
OPT_STEPS="${4:-5}"
USE_CPU="${5:-true}"  # Default to using CPU due to memory constraints

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== AOT-GRPO Test Runner ===${NC}"
echo -e "${YELLOW}Model Path:${NC} $MODEL_PATH"
echo -e "${YELLOW}Dataset:${NC} $DATASET"
echo -e "${YELLOW}Number of Samples:${NC} $SAMPLES"
echo -e "${YELLOW}Optimization Steps:${NC} $OPT_STEPS"
echo -e "${YELLOW}Force CPU:${NC} $USE_CPU"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model directory not found at $MODEL_PATH${NC}"
    echo "Please download the model or specify the correct path as the first argument"
    exit 1
fi

# Set environment variables to control PyTorch behavior
if [ "$USE_CPU" = "true" ]; then
    # Force CPU usage by disabling CUDA and MPS
    export CUDA_VISIBLE_DEVICES=""
    export PYTORCH_MPS_ENABLE=0
    echo -e "${YELLOW}Forcing CPU execution for memory stability${NC}"
else
    # For MPS (Metal GPU on macOS), adjust memory limits
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9
    echo -e "${YELLOW}Using MPS with adjusted memory settings${NC}"
fi

# Run the test with reduced memory settings
echo -e "${GREEN}Starting test with low memory settings...${NC}"
python aot_grpo.py --model_path "$MODEL_PATH" --dataset "$DATASET" --samples "$SAMPLES" --opt_steps "$OPT_STEPS" --low_memory

# Check if test was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Test completed successfully!${NC}"
else
    echo -e "${RED}Test failed. See above for errors.${NC}"
    echo -e "${YELLOW}Try running with CPU only: ./run_test.sh \"$MODEL_PATH\" \"$DATASET\" \"$SAMPLES\" \"$OPT_STEPS\" true${NC}"
fi