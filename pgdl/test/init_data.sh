#!/bin/bash
# ==============================================================================
# PGDL Dataset Initialization Script
# Initializes all datasets and registers ML models into MorphingDB.
# Usage: bash init_data.sh
# ==============================================================================

set -e

# --- Database Configuration ---
DB_NAME="postgres"
DB_HOST="localhost"
DB_PORT="5432"
DB_USER="why"
DB_PASS="123456"

export PGPASSWORD="$DB_PASS"
PSQL="/home/why/dbagent/pg_base/bin/psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"

# --- Path Configuration ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/morphingdb_test/data"
MODEL_DIR="$SCRIPT_DIR/morphingdb_test/models"
TMP_DIR="$SCRIPT_DIR/.tmp_init"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

mkdir -p "$TMP_DIR"

echo "=========================================="
echo " PGDL Dataset Initialization"
echo "=========================================="

# ==============================================================================
# Step 0: Ensure database is running
# ==============================================================================
echo ""
echo "[Step 0] Checking database connectivity..."
if ! $PSQL -t -c "SELECT 1;" > /dev/null 2>&1; then
    echo "  Database not reachable. Starting PostgreSQL..."
    /home/why/dbagent/pg_base/bin/pg_ctl -D /home/why/dbagent/pg_base/data start
    sleep 3
    if ! $PSQL -t -c "SELECT 1;" > /dev/null 2>&1; then
        echo "  ERROR: Cannot connect to database after startup."
        exit 1
    fi
fi
echo "  Database is running."

START_TIME=$(date +%s)

# ==============================================================================
# Step 1: CREATE EXTENSION pgdl
# ==============================================================================
echo ""
echo "[Step 1] CREATE EXTENSION pgdl"
$PSQL -c "CREATE EXTENSION IF NOT EXISTS pgdl;" 2>&1 | grep -v "^NOTICE" || true
echo "  pgdl extension loaded."

# ==============================================================================
# Step 2: Import Series Datasets (via Python helpers)
# ==============================================================================
echo ""
echo "[Step 2] Importing series datasets..."

# --- 2a: Slice Test ---
echo -n "  [2a] slice_test ... "
SLICE_CSV="$DATA_DIR/series/slice/slice_localization_data.csv"
if [ ! -f "$SLICE_CSV" ]; then
    echo "SKIPPED (CSV not found)"
else
    python3 -m morphingdb_test.series_test.slice_test.import_dataset > /dev/null 2>&1 && \
        echo "OK" || echo "FAILED"
fi

# --- 2b: Swarm Test ---
echo -n "  [2b] swarm_test ... "
SWARM_CSV="$DATA_DIR/series/swarm/Swarm_Behaviour.csv"
if [ -f "$SWARM_CSV" ]; then
    python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from morphingdb_test.series_test.swarm_test.import_dataset import import_swarmm_mvec_table, import_swarm_table
import_swarmm_mvec_table()
import_swarm_table()
" > /dev/null 2>&1 && echo "OK" || echo "FAILED"
else
    echo "SKIPPED (CSV not found)"
fi

# --- 2c: Year Predict Test ---
echo -n "  [2c] year_predict_test ... "
YEAR_CSV="$DATA_DIR/series/yead_predict/YearPredictionMSD.csv"
if [ -f "$YEAR_CSV" ]; then
    python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from morphingdb_test.series_test.year_predict_test.import_dataset import import_year_predict_mvec_table, import_year_predict_table
import_year_predict_mvec_table()
import_year_predict_table()
" > /dev/null 2>&1 && echo "OK" || echo "FAILED"
else
    echo "SKIPPED (CSV not found)"
fi

# ==============================================================================
# Step 3: Import Image Datasets (batched SQL inserts)
# ==============================================================================
echo ""
echo "[Step 3] Importing image datasets..."

# --- 3a: CIFAR-10 ---
echo -n "  [3a] cifar10 ... "
CIFAR_DIR="$DATA_DIR/image/cifar10/test/"
if [ -d "$CIFAR_DIR" ]; then
    $PSQL -c "DROP TABLE IF EXISTS cifar_image_vector_table;" > /dev/null 2>&1
    $PSQL -c "DROP TABLE IF EXISTS cifar_image_table;" > /dev/null 2>&1
    $PSQL -c "CREATE TABLE cifar_image_vector_table (id int, image_vector mvec);" > /dev/null 2>&1
    $PSQL -c "CREATE TABLE cifar_image_table (id int, image_path text);" > /dev/null 2>&1

    # Collect image paths
    mapfile -t IMG_PATHS < <(find "$CIFAR_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" \) | sort)
    TOTAL=${#IMG_PATHS[@]}
    BATCH=500
    echo -n "("
    for ((i=0; i<TOTAL; i+=BATCH)); do
        SQL_FILE="$TMP_DIR/cifar10_batch_$((i/BATCH)).sql"
        echo "BEGIN;" > "$SQL_FILE"
        end=$((i+BATCH))
        if [ $end -gt $TOTAL ]; then end=$TOTAL; fi
        for ((j=i; j<end; j++)); do
            IMG_ID=$((j+1))
            img="${IMG_PATHS[$j]}"
            echo "INSERT INTO cifar_image_vector_table (id, image_vector) VALUES ($IMG_ID, image_to_vector(224,224,0.4914,0.4822,0.4465,0.2023,0.1994,0.2010, '$img'));" >> "$SQL_FILE"
            echo "INSERT INTO cifar_image_table (id, image_path) VALUES ($IMG_ID, '$img');" >> "$SQL_FILE"
        done
        echo "COMMIT;" >> "$SQL_FILE"
        $PSQL -f "$SQL_FILE" > /dev/null 2>&1 || $PSQL -f "$SQL_FILE" > /dev/null 2>&1
        echo -n "."
    done
    echo -n ") "
    COUNT=$($PSQL -t -c "SELECT COUNT(*) FROM cifar_image_table;" 2>/dev/null | tr -d ' ')
    echo "OK ($COUNT images)"
else
    echo "SKIPPED (dir not found)"
fi

# --- 3b: ImageNet ---
echo -n "  [3b] imagenet ... "
IMAGENET_DIR="$DATA_DIR/image/image-net/data/"
if [ -d "$IMAGENET_DIR" ]; then
    $PSQL -c "DROP TABLE IF EXISTS imagenet_image_vector_table;" > /dev/null 2>&1
    $PSQL -c "DROP TABLE IF EXISTS imagenet_image_table;" > /dev/null 2>&1
    $PSQL -c "CREATE TABLE imagenet_image_vector_table (id int, image_vector mvec);" > /dev/null 2>&1
    $PSQL -c "CREATE TABLE imagenet_image_table (id int, image_path text);" > /dev/null 2>&1

    mapfile -t IMG_NAMES < <(ls "$IMAGENET_DIR" | head -10000)
    TOTAL=${#IMG_NAMES[@]}
    BATCH=1000
    echo -n "("
    for ((i=0; i<TOTAL; i+=BATCH)); do
        SQL_FILE="$TMP_DIR/imagenet_batch_$((i/BATCH)).sql"
        echo "BEGIN;" > "$SQL_FILE"
        end=$((i+BATCH))
        if [ $end -gt $TOTAL ]; then end=$TOTAL; fi
        for ((j=i; j<end; j++)); do
            IMG_ID=$((j+1))
            img="${IMG_NAMES[$j]}"
            FULL_PATH="${IMAGENET_DIR}${img}"
            echo "INSERT INTO imagenet_image_vector_table (id, image_vector) VALUES ($IMG_ID, image_to_vector(224,224,0.4914,0.4822,0.4465,0.2023,0.1994,0.2010, '$FULL_PATH'));" >> "$SQL_FILE"
            echo "INSERT INTO imagenet_image_table (id, image_path) VALUES ($IMG_ID, '$FULL_PATH');" >> "$SQL_FILE"
        done
        echo "COMMIT;" >> "$SQL_FILE"
        $PSQL -f "$SQL_FILE" > /dev/null 2>&1 || $PSQL -f "$SQL_FILE" > /dev/null 2>&1
        echo -n "."
    done
    echo -n ") "
    COUNT=$($PSQL -t -c "SELECT COUNT(*) FROM imagenet_image_table;" 2>/dev/null | tr -d ' ')
    echo "OK ($COUNT images)"
else
    echo "SKIPPED (dir not found)"
fi

# --- 3c: Stanford Dogs ---
echo -n "  [3c] stanford_dogs ... "
DOGS_DIR="$DATA_DIR/image/Stanford_Dogs/images/Images/"
if [ -d "$DOGS_DIR" ]; then
    $PSQL -c "DROP TABLE IF EXISTS stanford_dogs_image_vector_table;" > /dev/null 2>&1
    $PSQL -c "DROP TABLE IF EXISTS stanford_dogs_image_table;" > /dev/null 2>&1
    $PSQL -c "CREATE TABLE stanford_dogs_image_vector_table (id int, image_vector mvec);" > /dev/null 2>&1
    $PSQL -c "CREATE TABLE stanford_dogs_image_table (id int, image_path text);" > /dev/null 2>&1

    mapfile -t IMG_PATHS < <(find "$DOGS_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" \) | sort | head -10000)
    TOTAL=${#IMG_PATHS[@]}
    BATCH=1000
    echo -n "("
    for ((i=0; i<TOTAL; i+=BATCH)); do
        SQL_FILE="$TMP_DIR/stanford_dogs_batch_$((i/BATCH)).sql"
        echo "BEGIN;" > "$SQL_FILE"
        end=$((i+BATCH))
        if [ $end -gt $TOTAL ]; then end=$TOTAL; fi
        for ((j=i; j<end; j++)); do
            IMG_ID=$((j+1))
            img="${IMG_PATHS[$j]}"
            echo "INSERT INTO stanford_dogs_image_vector_table (id, image_vector) VALUES ($IMG_ID, image_to_vector(256,224,0.485,0.456,0.406,0.229,0.224,0.225, '$img'));" >> "$SQL_FILE"
            echo "INSERT INTO stanford_dogs_image_table (id, image_path) VALUES ($IMG_ID, '$img');" >> "$SQL_FILE"
        done
        echo "COMMIT;" >> "$SQL_FILE"
        $PSQL -f "$SQL_FILE" > /dev/null 2>&1 || $PSQL -f "$SQL_FILE" > /dev/null 2>&1
        echo -n "."
    done
    echo -n ") "
    COUNT=$($PSQL -t -c "SELECT COUNT(*) FROM stanford_dogs_image_table;" 2>/dev/null | tr -d ' ')
    echo "OK ($COUNT images)"
else
    echo "SKIPPED (dir not found)"
fi

# ==============================================================================
# Step 4: Import Text Datasets (via Python helpers for parquet/tokenizer)
# ==============================================================================
echo ""
echo "[Step 4] Importing text datasets..."

# --- 4a: IMDB (raw text) ---
echo -n "  [4a] imdb (raw text) ... "
IMDB_PARQUET="$DATA_DIR/text/imdb/data/test-00000-of-00001.parquet"
if [ -f "$IMDB_PARQUET" ]; then
    python3 "$SCRIPT_DIR/import_text_data.py" 2>&1 | tail -3 && echo "OK" || echo "FAILED"
else
    echo "SKIPPED (parquet not found)"
fi

# --- 4a2: IMDB vector (mvec) ---
echo -n "  [4a2] imdb (vector) ... "
IMDB_VEC=$($PSQL -t -c "SELECT COUNT(*) FROM imdb_vector_test;" 2>/dev/null | tr -d ' ')
if [ "$IMDB_VEC" != "" ] && [ "$IMDB_VEC" -gt 0 ] 2>/dev/null; then
    echo "EXISTS ($IMDB_VEC rows, skipping)"
elif [ -f "$IMDB_PARQUET" ]; then
    python3 -c "
import sys, os; sys.path.insert(0, '$SCRIPT_DIR')
from morphingdb_test.text_test.imdb.import_dataset import import_imdb_mvec_dataset
import_imdb_mvec_dataset()
" > /dev/null 2>&1 && echo "OK" || echo "FAILED"
else
    echo "SKIPPED (parquet not found)"
fi

# --- 4b: SST2 ---
echo -n "  [4b] sst2 ... "
SST2_TSV="$DATA_DIR/text/sst2/data/train.tsv"
if [ -f "$SST2_TSV" ]; then
    python3 -c "
import sys, os; sys.path.insert(0, '$SCRIPT_DIR')
from morphingdb_test.text_test.sst2.import_dataset import import_sst2_dataset, import_sst2_mvec_dataset
import_sst2_dataset()
import_sst2_mvec_dataset()
" > /dev/null 2>&1 && echo "OK" || echo "FAILED"
else
    echo "SKIPPED (tsv not found)"
fi

# --- 4c: Financial Phrasebank ---
echo -n "  [4c] financial_phrasebank ... "
FP_DIR="$DATA_DIR/text/financial_phrasebank"
if [ -d "$FP_DIR" ] && [ -f "$FP_DIR/data/Sentences_50Agree.txt" ]; then
    python3 -c "
import sys, os; sys.path.insert(0, '$SCRIPT_DIR')
from morphingdb_test.text_test.financial_phrasebank.import_dataset import import_financial_phrasebank_dataset, import_financial_phrasebank_mvec_dataset
import_financial_phrasebank_dataset()
import_financial_phrasebank_mvec_dataset()
" > /dev/null 2>&1 && echo "OK" || echo "FAILED"
else
    echo "SKIPPED (dir not found)"
fi

# ==============================================================================
# Step 4d: Import Reasoning Datasets
# ==============================================================================
echo ""
echo "[Step 4d] Importing reasoning datasets..."

echo -n "  [4d] reasoning_test ... "
python3 -c "
import sys, os; sys.path.insert(0, '$SCRIPT_DIR')
from morphingdb_test.reasoning_test.import_dataset import import_reasoning_dataset, import_reasoning_mvec_dataset
import_reasoning_dataset()
import_reasoning_mvec_dataset()
" > /dev/null 2>&1 && echo "OK" || echo "FAILED"

# ==============================================================================
# Step 5: Register ML Models
# ==============================================================================
echo ""
echo "[Step 5] Registering ML models..."

register_model() {
    local model_name="$1"
    local model_path="$2"
    local exists
    exists=$($PSQL -t -c "SELECT COUNT(*) FROM model_info WHERE model_name = '$model_name';" 2>/dev/null | tr -d ' ')
    if [ "$exists" = "0" ] || [ -z "$exists" ]; then
        $PSQL -c "SELECT create_model('$model_name', '$model_path', '', '');" > /dev/null 2>&1
        echo "  [OK] $model_name"
    else
        echo "  [exists] $model_name"
    fi
}

register_model "slice"                    "$MODEL_DIR/slice.pt"
register_model "swarm"                    "$MODEL_DIR/swarm.pt"
register_model "year_predict"             "$MODEL_DIR/year_predict.pt"
register_model "googlenet_cifar10"        "$MODEL_DIR/googlenet_cifar10.pt"
register_model "defect_vec"               "$MODEL_DIR/resnet18_imagenet.pt"
register_model "alexnet_stanford_dogs"    "$MODEL_DIR/alexnet_stanford_dogs.pt"
register_model "finance"                 "$MODEL_DIR/sentiment_analysis_model.pt"
register_model "sst2_vec"                "$MODEL_DIR/traced_albert_vec.pt"

# reasoning models
register_model "cross_encoder"           "$MODEL_DIR/cross_encoder.pt"
register_model "deberta_reader"          "$MODEL_DIR/deberta_reader.pt"
register_model "flan_t5_reader"          "$MODEL_DIR/flan_t5_reader_gpu.pt"

# ==============================================================================
# Cleanup
# ==============================================================================
rm -rf "$TMP_DIR"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=========================================="
echo " Initialization Complete"
echo "=========================================="
echo ""
echo "Registered models:"
$PSQL -c "SELECT model_name, model_path FROM model_info ORDER BY model_name;"
echo ""
echo "Dataset tables:"
$PSQL -c "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;"
echo ""
echo "All available datasets are written into the database."

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))
SECS=$((ELAPSED % 60))
echo ""
echo "Total time: ${HOURS}h ${MINS}m ${SECS}s (${ELAPSED}s)"
