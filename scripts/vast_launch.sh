#!/bin/bash
#
# Vast.ai Launch Script for TFG Peptide Design
#
# Usage:
#   ./scripts/vast_launch.sh [num_designs] [output_name]
#
# Examples:
#   ./scripts/vast_launch.sh 1000 test_run
#   ./scripts/vast_launch.sh 60000 production_run
#

set -e  # Exit on error

# Configuration
NUM_DESIGNS=${1:-1000}
OUTPUT_NAME=${2:-baseline_$(date +%Y%m%d_%H%M%S)}
DOCKER_IMAGE="YOUR_DOCKERHUB_USERNAME/boltzgen-tfg:latest"  # Update this!
GPU_TYPE="RTX_4090"
MIN_GPU_RAM=24
MAX_PRICE=0.60
DISK_SIZE=50

# Vast.ai configuration
RELIABILITY_THRESHOLD=0.95

echo "================================"
echo "TFG Vast.ai Launch Script"
echo "================================"
echo "Designs:     $NUM_DESIGNS"
echo "Output:      $OUTPUT_NAME"
echo "Docker:      $DOCKER_IMAGE"
echo "GPU:         $GPU_TYPE"
echo "Max Price:   \$${MAX_PRICE}/hr"
echo "================================"

# Check if vastai CLI is installed
if ! command -v vastai &> /dev/null; then
    echo "Error: vastai CLI not found"
    echo "Install with: pip install vastai"
    exit 1
fi

# Check if API key is set
if ! vastai show user &> /dev/null; then
    echo "Error: Vast.ai API key not configured"
    echo "Set with: vastai set api-key YOUR_API_KEY"
    exit 1
fi

# Search for available instances
echo ""
echo "Searching for available GPU instances..."
SEARCH_QUERY="reliability > $RELIABILITY_THRESHOLD gpu_ram >= $MIN_GPU_RAM dph < $MAX_PRICE gpu_name=$GPU_TYPE"

vastai search offers "$SEARCH_QUERY" | head -20

read -p "Select offer ID (or press Ctrl+C to cancel): " OFFER_ID

if [ -z "$OFFER_ID" ]; then
    echo "Error: No offer ID provided"
    exit 1
fi

# Launch instance
echo ""
echo "Launching instance $OFFER_ID..."

INSTANCE_ID=$(vastai create instance \
    $OFFER_ID \
    --image $DOCKER_IMAGE \
    --disk $DISK_SIZE \
    --env "NUM_DESIGNS=$NUM_DESIGNS" \
    --env "OUTPUT_NAME=$OUTPUT_NAME" \
    --onstart-cmd "mkdir -p /workspace/output/$OUTPUT_NAME" \
    | grep -oP 'Created instance \K\d+')

if [ -z "$INSTANCE_ID" ]; then
    echo "Error: Failed to launch instance"
    exit 1
fi

echo "Instance $INSTANCE_ID launched successfully"

# Wait for instance to start
echo ""
echo "Waiting for instance to start..."
sleep 30

# Check instance status
vastai show instance $INSTANCE_ID

# Get SSH connection info
echo ""
echo "Getting SSH connection..."
SSH_INFO=$(vastai ssh-url $INSTANCE_ID)
SSH_HOST=$(echo $SSH_INFO | cut -d'@' -f2 | cut -d':' -f1)
SSH_PORT=$(echo $SSH_INFO | cut -d':' -f2)

echo "SSH: ssh -p $SSH_PORT root@$SSH_HOST"

# Create run script
RUN_SCRIPT="/tmp/vast_run_${INSTANCE_ID}.sh"

cat > $RUN_SCRIPT << 'EOF'
#!/bin/bash
set -e

cd /workspace

echo "================================"
echo "Starting BoltzGen Generation"
echo "================================"
echo "Designs: $NUM_DESIGNS"
echo "Output:  /workspace/output/$OUTPUT_NAME"
echo "================================"

# Run BoltzGen
boltzgen run /workspace/design_spec.yaml \
    --output /workspace/output/$OUTPUT_NAME \
    --protocol peptide-anything \
    --num_designs $NUM_DESIGNS \
    --budget 2 \
    --cache /workspace/cache

echo ""
echo "================================"
echo "BoltzGen Complete - Starting Reranking"
echo "================================"

# Run reranking
python /workspace/scripts/rerank.py \
    --input /workspace/output/$OUTPUT_NAME \
    --output /workspace/output/${OUTPUT_NAME}_reranked \
    --top_k 100

echo ""
echo "================================"
echo "Pipeline Complete!"
echo "================================"
echo "Results in: /workspace/output/${OUTPUT_NAME}_reranked"
echo ""
echo "To download results:"
echo "  vastai copy $INSTANCE_ID:/workspace/output/${OUTPUT_NAME}_reranked ./results"
echo ""
EOF

# Upload and execute run script
echo ""
echo "Uploading run script..."
vastai copy-put $INSTANCE_ID $RUN_SCRIPT /workspace/run.sh

echo "Starting BoltzGen pipeline..."
vastai execute $INSTANCE_ID "chmod +x /workspace/run.sh && /workspace/run.sh"

# Save instance info
INFO_FILE="vast_instance_${INSTANCE_ID}.txt"
cat > $INFO_FILE << EOF
Instance ID: $INSTANCE_ID
Offer ID: $OFFER_ID
Output Name: $OUTPUT_NAME
Num Designs: $NUM_DESIGNS
SSH: ssh -p $SSH_PORT root@$SSH_HOST

To monitor:
  vastai show instance $INSTANCE_ID
  vastai ssh $INSTANCE_ID "tail -f /workspace/output/\${OUTPUT_NAME}/boltzgen.log"

To download results:
  vastai copy $INSTANCE_ID:/workspace/output/${OUTPUT_NAME}_reranked ./results

To destroy instance:
  vastai destroy instance $INSTANCE_ID
EOF

echo ""
echo "Instance info saved to: $INFO_FILE"
echo ""
echo "================================"
echo "Launch Complete"
echo "================================"
echo "Instance ID: $INSTANCE_ID"
echo ""
echo "Monitor progress:"
echo "  vastai show instance $INSTANCE_ID"
echo ""
echo "Download results when done:"
echo "  vastai copy $INSTANCE_ID:/workspace/output/${OUTPUT_NAME}_reranked ./results"
echo ""
echo "Destroy instance:"
echo "  vastai destroy instance $INSTANCE_ID"
echo "================================"
