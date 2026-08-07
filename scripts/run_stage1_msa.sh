#!/bin/bash
# ==============================================================================
# stage1_msa_buffer.sh
# Stage 1: MSA generation for buffer-based aptamer-ligand study
# ==============================================================================
#
# Designed for the buffer approach (e.g., 1TRS_2NA_1K_4MG_12CL) instead of
# Mg intervals. Supports running multiple targets simultaneously.
#
# Usage:
#   ./stage1_msa_buffer.sh <input_dir> [output_suffix]
#
# Examples:
#   # Run all inputs from af3_inputs directory
#   ./stage1_msa_buffer.sh ~/af3_inputs
#
#   # Run cocaine-only inputs
#   ./stage1_msa_buffer.sh ~/af3_inputs/cocaine cocaine
#
#   # Run both cocaine and morphine simultaneously
#   ./stage1_msa_buffer.sh ~/af3_inputs/combined combined
#
# Input files should be named like:
#   NC001_cocaine_1TRS_2NA_1K_4MG_12CL.json
#   NC001_morphine_1TRS_2NA_1K_4MG_12CL.json
#
# ==============================================================================

set -uo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_dir> [output_suffix]"
    echo ""
    echo "Examples:"
    echo "  $0 ~/af3_inputs"
    echo "  $0 ~/af3_inputs/cocaine cocaine"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_SUFFIX="${2:-buffer}"

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AF3_DIR="$HOME/alphafold3"
MODELS_DIR="/media/Data/models"
DB_DIR="/media/Data/alphafold3_dbs"
INFERENCE_OUTPUT_DIR="$HOME/afinput_inference_${OUTPUT_SUFFIX}"

# Temp output for MSA (will copy _data.json to inference dir)
MSA_TEMP_DIR="$HOME/afoutput_msa_temp_${OUTPUT_SUFFIX}"

# Parallel jobs (MSA is CPU-bound, can run multiple)
MAX_PARALLEL=4

# ==============================================================================
# SETUP
# ==============================================================================
mkdir -p "$INFERENCE_OUTPUT_DIR"
mkdir -p "$MSA_TEMP_DIR"

echo "=========================================="
echo "STAGE 1: MSA Generation (Buffer Mode)"
echo "=========================================="
echo "Started: $(date)"
echo ""
echo "Input directory:      $INPUT_DIR"
echo "Inference output:     $INFERENCE_OUTPUT_DIR"
echo "MSA temp directory:   $MSA_TEMP_DIR"
echo "Max parallel jobs:    $MAX_PARALLEL"
echo ""

# ==============================================================================
# FIND INPUT FILES
# ==============================================================================
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}ERROR: Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

# Find all JSON files (excluding _data.json which are outputs)
mapfile -t INPUT_FILES < <(find "$INPUT_DIR" -maxdepth 1 -name "*.json" ! -name "*_data.json" -type f | sort)
TOTAL_JOBS=${#INPUT_FILES[@]}

if [ $TOTAL_JOBS -eq 0 ]; then
    echo -e "${RED}ERROR: No JSON input files found in $INPUT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Found $TOTAL_JOBS input files${NC}"
echo ""

# Show sample files
echo "Sample files:"
for i in "${!INPUT_FILES[@]}"; do
    if [ $i -lt 5 ]; then
        echo "  $(basename "${INPUT_FILES[$i]}")"
    fi
done
if [ $TOTAL_JOBS -gt 5 ]; then
    echo "  ... and $((TOTAL_JOBS - 5)) more"
fi
echo ""

# ==============================================================================
# MSA GENERATION FUNCTION
# ==============================================================================
run_msa_job() {
    local INPUT_JSON="$1"
    local JOB_NUM="$2"
    local TOTAL="$3"
    
    local BASENAME=$(basename "$INPUT_JSON" .json)
    local JOB_OUTPUT_DIR="$MSA_TEMP_DIR/${BASENAME}_msa"
    local LOG_FILE="$MSA_TEMP_DIR/${BASENAME}_msa.log"
    
    # Check if already processed
    if [ -f "$INFERENCE_OUTPUT_DIR/${BASENAME}_data.json" ]; then
        echo -e "  ${YELLOW}[$JOB_NUM/$TOTAL] SKIP${NC} $BASENAME (already exists)"
        return 0
    fi
    
    echo -e "  ${BLUE}[$JOB_NUM/$TOTAL] START${NC} $BASENAME"
    
    mkdir -p "$JOB_OUTPUT_DIR"
    
    # Run AlphaFold3 MSA only (no inference)
    docker run --rm \
        --volume "$AF3_DIR:/app/alphafold" \
        --volume "$MODELS_DIR:/root/models" \
        --volume "$DB_DIR:/root/public_databases" \
        --volume "$INPUT_DIR:/root/af_input" \
        --volume "$JOB_OUTPUT_DIR:/root/af_output" \
        alphafold3 \
        python /app/alphafold/run_alphafold.py \
        --json_path="/root/af_input/$(basename "$INPUT_JSON")" \
        --model_dir=/root/models \
        --db_dir=/root/public_databases \
        --output_dir=/root/af_output \
        --run_inference=false \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "  ${RED}[$JOB_NUM/$TOTAL] FAILED${NC} $BASENAME (exit code: $EXIT_CODE)"
        echo "    Log: $LOG_FILE"
        return 1
    fi
    
    # Find and copy _data.json to inference directory
    # AlphaFold creates: JOB_OUTPUT_DIR/<job_name>_<timestamp>/<job_name>_data.json
    local DATA_JSON=$(find "$JOB_OUTPUT_DIR" -name "*_data.json" -type f 2>/dev/null | head -1)
    
    if [ -z "$DATA_JSON" ] || [ ! -f "$DATA_JSON" ]; then
        echo -e "  ${RED}[$JOB_NUM/$TOTAL] ERROR${NC} $BASENAME: No _data.json found"
        return 1
    fi
    
    # Copy with consistent naming (basename_data.json)
    cp "$DATA_JSON" "$INFERENCE_OUTPUT_DIR/${BASENAME}_data.json"
    
    echo -e "  ${GREEN}[$JOB_NUM/$TOTAL] DONE${NC} $BASENAME"
    return 0
}

export -f run_msa_job
export MSA_TEMP_DIR INFERENCE_OUTPUT_DIR AF3_DIR MODELS_DIR DB_DIR INPUT_DIR
export RED GREEN YELLOW BLUE NC

# ==============================================================================
# PARALLEL EXECUTION
# ==============================================================================
echo "Starting MSA generation..."
echo ""

COMPLETED=0
FAILED=0
SKIPPED=0

# Process in batches for cleaner output
for ((i=0; i<TOTAL_JOBS; i+=MAX_PARALLEL)); do
    BATCH_END=$((i + MAX_PARALLEL))
    if [ $BATCH_END -gt $TOTAL_JOBS ]; then
        BATCH_END=$TOTAL_JOBS
    fi
    
    echo -e "${BLUE}Batch $((i/MAX_PARALLEL + 1)): Jobs $((i+1))-$BATCH_END of $TOTAL_JOBS${NC}"
    
    PIDS=()
    for ((j=i; j<BATCH_END; j++)); do
        INPUT_JSON="${INPUT_FILES[$j]}"
        BASENAME=$(basename "$INPUT_JSON" .json)
        
        # Check if already done
        if [ -f "$INFERENCE_OUTPUT_DIR/${BASENAME}_data.json" ]; then
            echo -e "  ${YELLOW}[$((j+1))/$TOTAL_JOBS] SKIP${NC} $BASENAME (already exists)"
            ((SKIPPED++))
            continue
        fi
        
        # Run in background
        (
            run_msa_job "$INPUT_JSON" "$((j+1))" "$TOTAL_JOBS"
        ) &
        PIDS+=($!)
    done
    
    # Wait for batch to complete
    for PID in "${PIDS[@]}"; do
        wait $PID
        if [ $? -eq 0 ]; then
            ((COMPLETED++))
        else
            ((FAILED++))
        fi
    done
    
    echo ""
done

# ==============================================================================
# SUMMARY
# ==============================================================================
echo "=========================================="
echo "STAGE 1 COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Results:"
echo "  Total jobs:     $TOTAL_JOBS"
echo "  Completed:      $COMPLETED"
echo "  Skipped:        $SKIPPED"
echo "  Failed:         $FAILED"
echo ""
echo "Output files in: $INFERENCE_OUTPUT_DIR"
echo ""

# List generated files
DATA_COUNT=$(find "$INFERENCE_OUTPUT_DIR" -name "*_data.json" -type f 2>/dev/null | wc -l)
echo "Generated _data.json files: $DATA_COUNT"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}WARNING: $FAILED jobs failed. Check logs in $MSA_TEMP_DIR${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Ready for Stage 2 inference!${NC}"
echo "Run: ./stage2_inference_buffer.sh $INFERENCE_OUTPUT_DIR"
