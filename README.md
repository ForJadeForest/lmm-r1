
## Quick Start


> [!NOTE]
>We recommend using vLLM 0.7.2 or higher.

### Training

The training process follows a two-stage approach:

#### Stage 1: First Stage Training

```bash
# Train with text-only data
bash examples/scripts/train_stage1_text.sh

# Train with multimodal data for comparison
bash examples/scripts/train_stage1_multi.sh
```

#### Stage 2: Second Stage Training

```bash
# Train on specific domain A
bash examples/scripts/train_stage2_a.sh

# Train on specific domain B
bash examples/scripts/train_stage2_b.sh
```

#### Direct Training

```bash
# Direct training on specific domain
bash examples/scripts/train_direct.sh
```

