# Multimodal Speculative Decoding

Исследование методов ускорения мультимодального инференса LLM на примере Qwen2-VL.
Реализованы 5 архитектур драфтера, дистилляция знаний и классический спекулятивный декодинг.

**Железо:** 8 × V100 32 GB  
**Целевая модель:** Qwen2-VL-2B-Instruct (~3B параметров)  
**Драфтер:** ~88M параметров (малый Qwen2 + визуальные компоненты)

---

## Содержание

1. [Архитектуры драфтера](#архитектуры-драфтера)
2. [Структура проекта](#структура-проекта)
3. [Установка](#установка)
4. [Подготовка данных](#подготовка-данных)
5. [Обучение](#обучение)
6. [Инференс](#инференс)
7. [Сравнение архитектур](#сравнение-архитектур)
8. [Формат результатов](#формат-результатов)
9. [Логирование в W&B](#логирование-в-wb)

---

## Архитектуры драфтера

| ID | Название | Визуальный энкодер | Проектор | LLM | Обучаемые параметры |
|----|----------|--------------------|----------|-----|---------------------|
| `arch1` | ViT(T) + Proj(T) + Adapter(D) + LLM(D) | Заморожен (target) | Заморожен (target) | Свой | Adapter + LLM |
| `arch2` | ViT(T) + Proj(D) + LLM(D) | Заморожен (target, raw) | Свой MLP | Свой | Projector + LLM |
| `arch3` | ViT(D) + Proj(D) + LLM(D) | Свой SmallViT | Свой MLP | Свой | Всё |
| `arch4` | LLM(D) only | — | — | Свой | LLM |
| `eagle3` | EAGLE-3: ViT(T) + Adapter(D) + FeatureFusion + LLM(D) | Заморожен (target) | — | Свой | Projections + Fusion + Adapter + LLM |

`(T)` — используется компонент целевой модели; `(D)` — обучается с нуля.

**EAGLE-3** отличается от остальных архитектур: на каждой позиции _t_ вход в LLM — это
`FeatureFusion(projected_h_{t-1}, embed(x_t))`, где `h_{t-1}` во время тренировки берётся из
целевой модели, а во время генерации — из собственных скрытых состояний драфтера.
Это даёт более высокий acceptance rate без дополнительных вызовов target.

---

## Структура проекта

```
diploma/
├── configs/
│   ├── data_sources.jsonl        # пути к датасетам
│   ├── distill_arch1.yaml
│   ├── distill_arch2.yaml
│   ├── distill_arch3.yaml
│   ├── distill_arch4.yaml
│   └── distill_eagle3.yaml
├── data/
│   ├── dataset.py                # MultimodalDataset, build_datasets
│   └── collator.py               # DataCollator
├── distillation/
│   ├── losses.py                 # DistillationLoss (KL, JS, topk_kl, ...)
│   ├── trainer.py                # DistillationTrainer (DDP, AMP, scheduler)
│   ├── eagle3_losses.py          # Eagle3Loss (KL + feature MSE + CE)
│   └── eagle3_trainer.py        # Eagle3Trainer
├── inference/
│   ├── speculative_decoding.py   # SpeculativeDecoder
│   ├── eagle3_speculative_decoding.py  # Eagle3SpeculativeDecoder
│   └── run_inference.py          # CLI для запуска инференса
├── models/
│   ├── target.py                 # TargetModel (Qwen2-VL wrapper)
│   ├── base_drafter.py           # BaseDrafter (ABC)
│   ├── components/               # VisualAdapter, MLPProjector, SmallViT
│   └── drafters/
│       ├── arch1.py … arch4.py
│       ├── arch5_eagle3.py       # EAGLE-3
│       └── small_llm_config.py
└── train.py                      # точка входа для обучения
```

---

## Установка

```bash
git clone <repo>
cd diploma

pip install -r requirements.txt
# Для W&B логирования (опционально):
pip install wandb
```

`requirements.txt`:
```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.45.0
accelerate>=0.26.0
Pillow>=10.0.0
pyyaml>=6.0
einops>=0.7.0
```

Загрузка весов целевой модели:
```bash
# Hugging Face Hub (автоматически при первом запуске)
# или вручную:
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct
```

---

## Подготовка данных

### Формат семпла (JSONL)

Каждая строка в файле аннотаций — отдельный диалог:

```jsonl
{"image": "coco/000000123.jpg", "conversations": [{"from": "human", "value": "What is in the image?"}, {"from": "gpt", "value": "A cat sitting on a sofa."}]}
```

Поддерживаются многоходовые диалоги:
```jsonl
{"image": "path/to/img.jpg", "conversations": [
  {"from": "human", "value": "Describe the scene."},
  {"from": "gpt",   "value": "The image shows ..."},
  {"from": "human", "value": "What color is the car?"},
  {"from": "gpt",   "value": "The car is red."}
]}
```

### Конфиг источников данных (`configs/data_sources.jsonl`)

Каждая строка — один источник:

```jsonl
{"image_folder": "/data/images/coco",    "jsonl_path": "/data/coco_train.jsonl",  "weight": 1.0, "split": "train"}
{"image_folder": "/data/images/coco",    "jsonl_path": "/data/coco_eval.jsonl",   "weight": 1.0, "split": "eval"}
{"image_folder": "/data/images/llava",   "jsonl_path": "/data/llava_train.jsonl", "weight": 2.0, "split": "train"}
```

- `weight` — множитель повторения датасета в `ConcatDataset` (округляется до целого).
- `split` — `"train"` или `"eval"`.

---

## Обучение

### Одна GPU

```bash
python train.py --config configs/distill_arch1.yaml
```

### 8 × V100 (torchrun / DDP)

```bash
torchrun --nproc_per_node=8 train.py --config configs/distill_arch1.yaml
```

### Обучение всех 5 архитектур последовательно

```bash
for ARCH in arch1 arch2 arch3 arch4 eagle3; do
    echo "=== Training $ARCH ==="
    torchrun --nproc_per_node=8 train.py --config configs/distill_${ARCH}.yaml
done
```

### Ключевые гиперпараметры в YAML

| Параметр | Описание | Рекомендуемые значения |
|----------|----------|------------------------|
| `learning_rate` | начальный LR (cosine decay) | `2e-4` |
| `num_train_epochs` | число эпох | `3` |
| `per_device_train_batch_size` | батч на GPU | `4` (V100 32 GB) |
| `gradient_accumulation_steps` | шаги до optimizer.step | `4` → effective batch = 128 |
| `temperature` | температура distill-лосса | `2.0` |
| `alpha` | вес KL-дистилляции | `0.9` (arch1–4), `0.7` (eagle3) |
| `beta` | вес feature alignment (eagle3) | `0.2` |
| `loss_type` | тип лосса | `forward_kl` / `topk_kl` |
| `bf16` | смешанная точность bf16 | `true` |

### Конфиг EAGLE-3 (`configs/distill_eagle3.yaml`)

```yaml
target_model: Qwen/Qwen2-VL-2B-Instruct
arch: eagle3
arch_kwargs:
  feature_dim: 512
  adapter_dropout: 0.0
training:
  alpha: 0.7
  beta: 0.2
  temperature: 2.0
  loss_type: forward_kl
  output_dir: checkpoints/eagle3
  # ... остальные параметры как у arch1
```

### Чекпоинты

После обучения в `checkpoints/<arch>/` сохраняются:
```
checkpoints/
├── arch1/
│   ├── step-500/drafter.pt
│   ├── step-1000/drafter.pt
│   └── final/drafter.pt
├── arch2/final/drafter.pt
...
└── eagle3/final/drafter.pt
```

Сохраняются только веса драфтера; target заморожен и не сохраняется.

---

## Инференс

### CLI

```bash
# Arch1–Arch4 (классический спекулятивный декодинг)
python -m inference.run_inference \
    --target  Qwen/Qwen2-VL-2B-Instruct \
    --drafter checkpoints/arch1/final/drafter.pt \
    --arch    arch1 \
    --image   /path/to/image.jpg \
    --prompt  "Describe the image in detail." \
    --gamma   5 \
    --max_new_tokens 256 \
    --temperature 1.0 \
    --do_sample

# EAGLE-3
python -m inference.run_inference \
    --target  Qwen/Qwen2-VL-2B-Instruct \
    --drafter checkpoints/eagle3/final/drafter.pt \
    --arch    eagle3 \
    --image   /path/to/image.jpg \
    --prompt  "Describe the image in detail." \
    --gamma   5
```

Вывод:
```
Response:
The image shows a brown dog playing with a yellow tennis ball...

Stats:
Tokens generated: 128 | Acceptance rate: 74.22% | Mean tokens/target call: 4.71 | Tokens/sec: 52.3
```

### Программный API

```python
import torch
from transformers import Qwen2VLProcessor
from models import TargetModel, Arch1Drafter, Arch5Eagle3Drafter
from inference import SpeculativeDecoder, SpeculativeDecodingConfig
from inference.eagle3_speculative_decoding import Eagle3SpeculativeDecoder

device = torch.device("cuda")
dtype  = torch.bfloat16

target    = TargetModel("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=dtype).to(device).eval()
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# --- Arch1 ---
drafter = Arch1Drafter(target)
drafter.load_state_dict(torch.load("checkpoints/arch1/final/drafter.pt"))
drafter = drafter.to(device).eval()

cfg     = SpeculativeDecodingConfig(gamma=5, max_new_tokens=256, do_sample=True)
decoder = SpeculativeDecoder(target, drafter, cfg)

# --- EAGLE-3 ---
drafter_e3 = Arch5Eagle3Drafter(target)
drafter_e3.load_state_dict(torch.load("checkpoints/eagle3/final/drafter.pt"))
drafter_e3 = drafter_e3.to(device).eval()

decoder_e3 = Eagle3SpeculativeDecoder(target, drafter_e3, cfg)

# --- Generate ---
with torch.no_grad():
    output_ids, stats = decoder.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        attention_mask=inputs.get("attention_mask"),
    )
```

### Параметры спекулятивного декодинга

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `gamma` | число токенов, генерируемых драфтером за один шаг | `5` |
| `temperature` | температура сэмплирования | `1.0` |
| `top_p` | nucleus sampling | `1.0` (отключён) |
| `max_new_tokens` | лимит новых токенов | `256` |
| `do_sample` | сэмплирование vs жадный декодинг | `True` |

---

## Сравнение архитектур

### Метрики

| Метрика | Описание |
|---------|----------|
| **Acceptance rate (β)** | Доля принятых токенов драфтера: `accepted / (draft_total)` |
| **Tokens / target call** | Среднее число новых токенов за один вызов target `= β·γ + 1` |
| **Speedup** | `tokens_per_sec_speculative / tokens_per_sec_baseline` |
| **Eval loss** | KL/distill loss на валидационной выборке после обучения |

### Запуск бенчмарка на одном изображении

```bash
for ARCH in arch1 arch2 arch3 arch4 eagle3; do
    echo "=== $ARCH ==="
    python -m inference.run_inference \
        --target  Qwen/Qwen2-VL-2B-Instruct \
        --drafter checkpoints/${ARCH}/final/drafter.pt \
        --arch    ${ARCH} \
        --image   assets/benchmark_image.jpg \
        --prompt  "Describe the image in detail." \
        --gamma   5 \
        --max_new_tokens 256 \
        --do_sample
done
```

### Ожидаемые ориентиры (γ = 5, температура = 1.0)

| Архитектура | Acceptance rate | Tokens / call | Ускорение |
|-------------|-----------------|---------------|-----------|
| arch4 (text-only) | ~40–50% | ~3.0–3.5 | ~1.8–2.0× |
| arch3 (full drafter) | ~50–60% | ~3.5–4.0 | ~2.0–2.4× |
| arch2 (target ViT, own proj) | ~55–65% | ~3.8–4.3 | ~2.2–2.6× |
| arch1 (target ViT + proj) | ~60–70% | ~4.0–4.5 | ~2.4–2.8× |
| eagle3 | ~70–80% | ~4.5–5.0 | ~2.8–3.5× |

Реальные числа зависят от данных, длины ответа и оборудования.

---

## Формат результатов

`DecodingStats` возвращается из `decoder.generate()`:

```python
@dataclass
class DecodingStats:
    total_tokens_generated: int    # всего новых токенов
    total_draft_tokens:     int    # всего токенов от драфтера (= N_calls × γ)
    total_accepted_tokens:  int    # принято (включая бонус)
    total_target_calls:     int    # число вызовов target
    wall_time_seconds:      float

    # Вычисляемые свойства:
    acceptance_rate          # total_accepted / total_draft
    tokens_per_second        # total_tokens_generated / wall_time
    mean_tokens_per_target_call  # total_tokens_generated / total_target_calls
```

Пример лога:
```
Tokens generated: 256 | Acceptance rate: 76.40% | Mean tokens/target call: 4.82 | Tokens/sec: 58.1
```

---

## Логирование в W&B

Включить W&B в YAML:
```yaml
training:
  use_wandb: true
  wandb_project: multimodal-speculative-decoding
```

Или через переменную окружения:
```bash
WANDB_PROJECT=multimodal-speculative-decoding \
torchrun --nproc_per_node=8 train.py --config configs/distill_eagle3.yaml
```

Логируемые метрики:
- `train/loss`, `train/distill_loss`, `train/task_loss`
- `train/feature_loss` (только EAGLE-3)
- `train/lr`, `train/step`
- `eval/loss`, `eval/distill_loss`, `eval/task_loss`

---

## Возможные проблемы

**OOM на V100 32 GB**  
Уменьшить `per_device_train_batch_size` до `2` и увеличить `gradient_accumulation_steps` до `8`.

**Медленная загрузка данных**  
Увеличить `dataloader_num_workers` до `8` или предварительно токенизировать датасет.

**`RuntimeError: Expected all tensors to be on the same device`**  
Убедиться, что target и drafter переведены на одно устройство до передачи в тренер/декодер.

**NCCL timeout при DDP**  
```bash
export NCCL_TIMEOUT=3600
torchrun --nproc_per_node=8 ...
```
