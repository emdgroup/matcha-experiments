#!/usr/bin/env bash
# Generate the MATCHA evaluation configs for every OpenADMET task group and architecture,
# plus the finetuning configs of the pretrained models.
#
# Usage: bash configs/generate_eval.sh [OUT_DIR]
#   OUT_DIR defaults to configs/eval

set -euo pipefail

OUT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)/eval}"

GROUPS_LIST="caco2 clint physchem unbound"

# model short name : MATCHA architecture
MODELS="
attentivefp:AttentiveFPRegressor
chemprop:ChempropRegressor
cnn:CNNRegressor
e3gnn:E3GNNRegressor
gatedgcn:GatedGCNRegressor
gin:GINRegressor
gps:GPSRegressor
gps3d:GPS3DRegressor
gt:GTRegressor
gt3d:GT3DRegressor
mlp:MLPRegressor
rnn:RNNRegressor
roformer:RoFormerRegressor
snn:SNNRegressor
"

endpoints() {
  case "$1" in
    caco2)    echo "CACO2_Papp_AB_endpoint CACO2_ER_endpoint" ;;
    clint)    echo "HLM_CLint_endpoint MLM_CLint_endpoint" ;;
    physchem) echo "LogD_endpoint KSOL_endpoint" ;;
    unbound)  echo "MPPB_endpoint MBPB_endpoint MGMB_endpoint" ;;
    *) echo "Unknown task group: $1" >&2; exit 1 ;;
  esac
}

label_encoder() {
  local i=0
  for ep in $(endpoints "$1"); do
    printf '      %d:\n        task_label: "%s"\n' "$i" "$ep"
    i=$((i + 1))
  done
}

# Shared dataset and split sections
header() {
  cat <<EOF
dataset:
  path: "datasets/openadmet/train_$1.csv"
  smiles_key: "SMILES"
  label_key: "endpoint"
  operator_key: "OPERATOR"

split:
  method: "file"
  n_subset: 1
  n_bootstrap: 1000
  frac_bootstrap: 1.0
  method_params:
    path:
      - "datasets/openadmet/test_$1.csv"
EOF
}

for group in $GROUPS_LIST; do
  mkdir -p "$OUT_DIR/$group"
  n_endpoints=$(endpoints "$group" | wc -w | tr -d ' ')

  # Architectures with hyperparameters from autotune
  for entry in $MODELS; do
    name="${entry%%:*}"
    arch="${entry##*:}"

    cat > "$OUT_DIR/$group/$name.yaml" <<EOF
$(header "$group")

model:
  architecture: "$arch"
  config_path: "optima/$group/$name.yaml"
  params:
    loss_fn: "bounded-mse"
    num_endpoints: $n_endpoints
    num_epochs: 50
    batch_size: 64
    label_encoder_params:
$(label_encoder "$group")

output:
  serialization:
    path: "eval/$group/$name"
  mlflow:
    experiment_name: "eval-$group-$name"
    log_dir: "mlflow"
    tags:
      architecture: "$arch"
      dataset: "openadmet-$group"
EOF
  done

  # Finetuning of the GatedGCN pretrained on public ADMET data
  cat > "$OUT_DIR/$group/finetuning.yaml" <<EOF
$(header "$group")

model:
  architecture: "FinetuningRegressor"
  config_path: null
  params:
    loss_fn: "bounded-mse"
    num_endpoints: $n_endpoints
    num_epochs: 100
    batch_size: 64
    patience: 10
    path_to_pretrained: "pretrain/admet/gatedgcn"
    finetuning_strategy: full
    pred_hidden_dims: null
    pretrain_lr: 1.0e-4
    pretrain_decay: 0.9
    optimizer_args:
      lr: 1.0e-4
    scheduler: warmup_cosine_annealing
    scheduler_args:
      peak_lr_factor: 10.0
      min_lr: 1.0e-4
    label_encoder_params:
$(label_encoder "$group")

output:
  serialization:
    path: "eval/$group/finetuning-gatedgcn"
  mlflow:
    experiment_name: "eval-$group-finetuning"
    log_dir: "mlflow"
    tags:
      architecture: "FinetuningRegressor"
      dataset: "openadmet-$group"
EOF

  # Finetuning of the GatedGCN pretrained on B3LYP properties
  cat > "$OUT_DIR/$group/finetuning-b3lyp-gatedgcn.yaml" <<EOF
$(header "$group")

model:
  architecture: "FinetuningRegressor"
  config_path: null
  params:
    loss_fn: "bounded-mse"
    num_endpoints: $n_endpoints
    num_epochs: 50
    batch_size: 64
    patience: 10
    path_to_pretrained: "pretrain/b3lyp/gatedgcn"
    finetuning_strategy: full
    pred_hidden_dims:
      - 256
      - 256
    activation: swish
    dropout: 0.1
    pretrain_lr: 1.0e-4
    pretrain_decay: 0.9
    optimizer: adamw
    optimizer_args:
      lr: 1.0e-3
    scheduler: cosine_annealing
    scheduler_args:
      min_lr: 1.0e-4
    label_encoder_params:
$(label_encoder "$group")

output:
  serialization:
    path: "eval/$group/finetuning-b3lyp-gatedgcn"
  mlflow:
    experiment_name: "eval-$group-finetuning-b3lyp-gatedgcn"
    log_dir: "mlflow"
    tags:
      architecture: "FinetuningRegressor"
      dataset: "openadmet-$group"
EOF

  # LoRA finetuning of the RoFormer pretrained with masked language modeling
  cat > "$OUT_DIR/$group/finetuning-roformer.yaml" <<EOF
$(header "$group")

model:
  architecture: "FinetuningRegressor"
  config_path: null
  params:
    loss_fn: "bounded-mse"
    num_endpoints: $n_endpoints
    num_epochs: 30
    batch_size: 64
    patience: 10
    path_to_pretrained: "pretrain/mlm/roformer"
    finetuning_strategy: lora
    pred_hidden_dims:
      - 256
      - 256
    activation: gelu
    dropout: 0.1
    pretrain_lr: 1.0e-4
    pretrain_decay: 0.9
    optimizer: adamw
    optimizer_args:
      lr: 1.0e-3
    scheduler: cosine_annealing
    scheduler_args:
      min_lr: 1.0e-4
    label_encoder_params:
$(label_encoder "$group")

output:
  serialization:
    path: "eval/$group/finetuning-roformer"
  mlflow:
    experiment_name: "eval-$group-finetuning-roformer"
    log_dir: "mlflow"
    tags:
      architecture: "FinetuningRegressor"
      dataset: "openadmet-$group"
EOF
done

echo "Configs written to $OUT_DIR"
