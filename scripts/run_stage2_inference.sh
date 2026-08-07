#!/bin/bash
# ==============================================================================
# stage2_inference_buffer.sh
# Stage 2: Parallel Inference on 4 GPUs for buffer-based study
# ==============================================================================
#
# Uses pre-computed MSA data from Stage 1.
# Distributes jobs across 4 GPUs using round-robin assignment.
#
# Key flags:
#   --run_data_pipeline=false  → Skip MSA, use existing _data.json
#   --flash_attention_implementation=xla  → RTX 5090 compatibility
#   --gpus "device=N"  → GPU isolation
#
# Usage:
#   ./stage2_inference_buffer.sh <input_dir> [output_dir]
#
# Examples:
#   ./stage2_inference_buffer.sh ~/afinput_inference_buffer
#   ./stage2_inference_buffer.sh ~/afinput_inference_cocaine ~/afoutput_cocaine
#
# ==============================================================================

set -uo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_dir> [output_dir]"
    echo ""
    echo "Examples:"
    echo "  $0 ~/afinput_inference_buffer"
    echo "  $0 ~/afinput_inference_cocaine ~/afoutput_cocaine"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="${2:-$HOME/afoutput_buffer}"

# ==============================================================================
# CONFIGURATION
# ==============================================================================
AF3_DIR="/home/patrick/alphafold3"
MODELS_DIR="/media/Data/models"
DB_DIR="/media/Data/alphafold3_dbs"
LOG_DIR="/home/patrick/af3_inputs_crossreactivity/STAGE2/isolation/isolation_all/inference_logs"

# 4 GPUs available
NUM_GPUS=3

# Max concurrent jobs (1 per GPU for memory safety)
MAX_CONCURRENT=$NUM_GPUS

# Throttle: seconds between launching jobs
THROTTLE_DELAY=5

# ==============================================================================
# SETUP
# ==============================================================================
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "STAGE 2: Inference (4 GPUs, Buffer Mode)"
echo "=========================================="
echo "Started: $(date)"
echo ""
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Log directory:    $LOG_DIR"
echo "GPUs:             $NUM_GPUS"
echo ""

# ==============================================================================
# FIND INPUT FILES
# ==============================================================================
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}ERROR: Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

mapfile -t DATA_FILES < <(find "$INPUT_DIR" -name "*_data.json" -type f | sort)
TOTAL_JOBS=${#DATA_FILES[@]}

if [ $TOTAL_JOBS -eq 0 ]; then
    echo -e "${RED}ERROR: No _data.json files found in $INPUT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Found $TOTAL_JOBS _data.json files${NC}"
echo ""

# Show distribution across targets
echo "File distribution:"
for target in cocaine morphine; do
    count=$(printf '%s\n' "${DATA_FILES[@]}" | grep -c "_${target}_" 2>/dev/null || echo "0")
    if [ "$count" -gt 0 ]; then
        echo "  $target: $count files"
    fi
done
echo ""

# ==============================================================================
# INFERENCE FUNCTION
# ==============================================================================
run_inference_job() {
    local DATA_JSON="$1"
    local GPU_ID="$2"
    local JOB_NUM="$3"
    local TOTAL="$4"
    
    # Extract job name from filename: NC001_cocaine_1TRS_2NA_1K_4MG_12CL_data.json -> NC001_cocaine_1TRS_2NA_1K_4MG_12CL
    local BASENAME=$(basename "$DATA_JSON" _data.json)
    local JOB_TEMP_DIR="$OUTPUT_DIR/temp_${BASENAME}_$$"
    local LOG_FILE="$LOG_DIR/${BASENAME}_gpu${GPU_ID}.log"
    
    # Check if already completed
    local OUTPUT_SUBDIR="$OUTPUT_DIR/$BASENAME"
    if [ -d "$OUTPUT_SUBDIR" ] && [ -n "$(find "$OUTPUT_SUBDIR" -name '*.cif' 2>/dev/null)" ]; then
        echo -e "  ${YELLOW}[$JOB_NUM/$TOTAL] GPU$GPU_ID SKIP${NC} $BASENAME (already done)"
        return 0
    fi
    
    echo -e "  ${CYAN}[$JOB_NUM/$TOTAL] GPU$GPU_ID START${NC} $BASENAME"
    
    # Create isolated temp input directory with just this file
    mkdir -p "$JOB_TEMP_DIR/input"
    cp "$DATA_JSON" "$JOB_TEMP_DIR/input/"
    mkdir -p "$JOB_TEMP_DIR/output"
    
    # Run inference
    docker run --rm \
        --runtime=nvidia \
        -e NVIDIA_VISIBLE_DEVICES="$GPU_ID" \
        --volume "$AF3_DIR:/app/alphafold" \
        --volume "$MODELS_DIR:/root/models" \
        --volume "$DB_DIR:/root/public_databases" \
        --volume "$JOB_TEMP_DIR/input:/root/af_input" \
        --volume "$JOB_TEMP_DIR/output:/root/af_output" \
        alphafold3:rtx5090 \
        python /app/alphafold/run_alphafold.py \
        --json_path="/root/af_input/$(basename "$DATA_JSON")" \
        --model_dir=/root/models \
        --db_dir=/root/public_databases \
        --output_dir=/root/af_output \
        --run_data_pipeline=false \
        --flash_attention_implementation=xla \
        --gpu_device=0 \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "  ${RED}[$JOB_NUM/$TOTAL] GPU$GPU_ID FAILED${NC} $BASENAME"
        echo "    Log: $LOG_FILE"
        rm -rf "$JOB_TEMP_DIR"
        return 1
    fi
    
    # Move results to final output directory
    # AlphaFold creates: temp/output/<job_name>_<timestamp>/
    local RESULT_DIR=$(find "$JOB_TEMP_DIR/output" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
    
    if [ -z "$RESULT_DIR" ] || [ ! -d "$RESULT_DIR" ]; then
        echo -e "  ${RED}[$JOB_NUM/$TOTAL] GPU$GPU_ID ERROR${NC} $BASENAME: No output directory"
        rm -rf "$JOB_TEMP_DIR"
        return 1
    fi
    
    # Check for structure files
    if [ -z "$(find "$RESULT_DIR" -name '*.cif' 2>/dev/null)" ]; then
        echo -e "  ${RED}[$JOB_NUM/$TOTAL] GPU$GPU_ID ERROR${NC} $BASENAME: No .cif files"
        rm -rf "$JOB_TEMP_DIR"
        return 1
    fi
    
    # Move to final location with clean naming
    mkdir -p "$OUTPUT_SUBDIR"
    mv "$RESULT_DIR"/* "$OUTPUT_SUBDIR/"
    
    # Cleanup temp
    rm -rf "$JOB_TEMP_DIR"
    
    echo -e "  ${GREEN}[$JOB_NUM/$TOTAL] GPU$GPU_ID DONE${NC} $BASENAME"
    return 0
}

export -f run_inference_job
export OUTPUT_DIR LOG_DIR AF3_DIR MODELS_DIR DB_DIR
export RED GREEN YELLOW BLUE CYAN NC

# ==============================================================================
# PARALLEL EXECUTION WITH GPU ROUND-ROBIN
# ==============================================================================
echo "Starting inference..."
echo ""

# Track running jobs per GPU
declare -A GPU_PIDS

COMPLETED=0
FAILED=0
SKIPPED=0
JOB_IDX=0

for DATA_JSON in "${DATA_FILES[@]}"; do
    ((JOB_IDX++))
    
    # Assign GPU using round-robin
    GPU_ID=$(( (JOB_IDX - 1) % NUM_GPUS ))
    
    # Check if already done
    BASENAME=$(basename "$DATA_JSON" _data.json)
    OUTPUT_SUBDIR="$OUTPUT_DIR/$BASENAME"
    if [ -d "$OUTPUT_SUBDIR" ] && [ -n "$(find "$OUTPUT_SUBDIR" -name '*.cif' 2>/dev/null)" ]; then
        echo -e "  ${YELLOW}[$JOB_IDX/$TOTAL_JOBS] GPU$GPU_ID SKIP${NC} $BASENAME (exists)"
        ((SKIPPED++))
        continue
    fi
    
    # Wait if this GPU has a running job
    if [ -n "${GPU_PIDS[$GPU_ID]:-}" ]; then
        OLD_PID="${GPU_PIDS[$GPU_ID]}"
        if kill -0 "$OLD_PID" 2>/dev/null; then
            wait "$OLD_PID"
            if [ $? -eq 0 ]; then
                ((COMPLETED++))
            else
                ((FAILED++))
            fi
        fi
    fi
    
    # Launch job on this GPU
    (
        run_inference_job "$DATA_JSON" "$GPU_ID" "$JOB_IDX" "$TOTAL_JOBS"
    ) &
    GPU_PIDS[$GPU_ID]=$!
    
    # Throttle between launches
    sleep $THROTTLE_DELAY
done

# Wait for all remaining jobs
echo ""
echo "Waiting for final jobs to complete..."
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    if [ -n "${GPU_PIDS[$GPU_ID]:-}" ]; then
        wait "${GPU_PIDS[$GPU_ID]}" 2>/dev/null
        if [ $? -eq 0 ]; then
            ((COMPLETED++))
        else
            ((FAILED++))
        fi
    fi
done

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "=========================================="
echo "STAGE 2 COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Results:"
echo "  Total jobs:  $TOTAL_JOBS"
echo "  Completed:   $COMPLETED"
echo "  Skipped:     $SKIPPED"
echo "  Failed:      $FAILED"
echo ""

# Count output structures
CIF_COUNT=$(find "$OUTPUT_DIR" -name "*.cif" -type f 2>/dev/null | wc -l)
echo "Generated structure files (.cif): $CIF_COUNT"
echo ""

# Summary by target
echo "Structures by target:"
for target in cocaine morphine; do
    count=$(find "$OUTPUT_DIR" -path "*_${target}_*" -name "*.cif" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  $target: $count structures"
    fi
done
echo ""

echo "Output directory: $OUTPUT_DIR"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}WARNING: $FAILED jobs failed. Check logs in $LOG_DIR${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All inference complete!${NC}"
