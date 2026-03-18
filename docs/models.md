# AI/ML Models

All models run locally via OpenVINO. No cloud API calls. Models are stored in the `models/` directory (gitignored) and loaded on demand through the Model Control Panel.

## Model Inventory

| Model | Purpose | Framework | Device | Dimensions |
|-------|---------|-----------|--------|------------|
| Qwen3-4B-Instruct | Agent LLM (tool calling) | OpenVINO GenAI | GPU | N/A |
| Qwen2.5-VL-3B-Instruct | Image captioning (VLM) | OpenVINO GenAI | GPU | N/A |
| all-MiniLM-L6-v2 | Text embeddings | OpenVINO Core | GPU (configurable) | 384-dim |
| face-detection-retail-0004 | Face bounding boxes | OpenVINO Core | CPU | N/A |
| landmarks-regression-retail-0009 | Facial landmarks | OpenVINO Core | CPU | 5 points |
| face-reidentification-retail-0095 | Face descriptors | OpenVINO Core | CPU | 512-dim |

## LLM: Qwen3-4B-Instruct

**File:** `tools/llm.py` — `LLMService`

Agent reasoning model. Reads user queries and decides which search tools to invoke using Qwen3's native tool-calling format.

- **Path:** `models/Qwen3-4B-Instruct-ov/`
- **Quantization:** OpenVINO IR (INT4/INT8)
- **Input:** Chat messages + tool definitions (OpenAI function-calling schema)
- **Output:** XML `<tool_call>` blocks or natural language response
- **Generation config:** `max_new_tokens=2048`, `do_sample=False` (greedy)

**Tool call parsing:** The service parses Qwen3's XML tool-call format:
```xml
<tool_call>
{"name": "search_by_caption", "arguments": {"query": "beach sunset"}}
</tool_call>
```

**Thread safety:** Load/unload protected by `threading.Lock`. Unload calls `gc.collect()`.

## VLM: Qwen2.5-VL-3B-Instruct

**File:** `ingestion/captioner.py` — `Captioner`

Vision-language model for generating image captions.

- **Path:** `models/Qwen2.5-VL-3B-Instruct/INT4/`
- **Quantization:** INT4
- **Input:** Image tensor (NHWC, uint8, resized to ~1 megapixel max) + text prompt
- **Output:** One-sentence description

**Two modes:**
1. **Unconditional:** `generate(image_path)` — "Describe this image in one sentence."
2. **With names:** `generate_with_names(image_path, names)` — Uses a short, direct prompt telling the VLM who is in the photo and asking it to describe the scene using their names. Single-person and multi-person prompts are separate templates. This approach works reliably with the small INT4 model for both English and non-ASCII names.

**Image preprocessing:**
- Load via PIL, convert to RGB numpy array
- Resize if pixel count exceeds ~1 megapixel (preserves aspect ratio)
- Convert to `ov.Tensor` (NHWC uint8 format for GenAI VLM pipeline)

## Text Embeddings: all-MiniLM-L6-v2

**File:** `vector/embeddings.py` — `EmbeddingService`

Sentence-transformer model for encoding text into semantic vectors.

- **Path:** `models/all-MiniLM-L6-v2-ov/`
- **HuggingFace ID:** `sentence-transformers/all-MiniLM-L6-v2`
- **Output:** 384-dimensional float vector
- **Pooling:** Mean pooling over non-padding tokens
- **Normalization:** L2-normalized
- **Device:** GPU (configurable via `text_embedding_device` setting, default GPU)

**Auto-download:** If the OpenVINO model directory doesn't exist, exports from HuggingFace using `optimum-cli export openvino`.

**Methods:**
- `embed_text(text) → list[float]` — Encode text to 384-dim vector
- `embed_image(image_path) → list[float]` — Currently embeds the file path as text (placeholder for CLIP-style image embedding)

## Face Detection: Intel OMZ Retail Models

**Files:** `ingestion/faces.py` — `FaceRecognizer`, `face_recognition/` module

Three-model pipeline from Intel Open Model Zoo:

### 1. face-detection-retail-0004

Detects face bounding boxes in images.
- **Input:** Image (BGR, resized to model input)
- **Output:** List of `[x_min, y_min, x_max, y_max, confidence]`
- **Confidence threshold:** Configurable (default 0.6)

### 2. landmarks-regression-retail-0009

Detects 5 facial landmarks (eyes, nose, mouth corners) for face alignment.
- **Input:** Cropped face region
- **Output:** 5 (x, y) landmark coordinates

### 3. face-reidentification-retail-0095

Extracts a 512-dimensional face descriptor for identity matching.
- **Input:** Aligned face crop
- **Output:** 512-dim float vector (L2-normalized)

**Identity matching:** Face descriptors are stored in ChromaDB's `face_identities` collection. During ingestion, each detected face is compared against known identities using cosine distance. Candidates (top-3 within configurable `face_identity_threshold`, default 0.5) are stored for user confirmation — the system never auto-assigns names.

**Auto-download:** Models downloaded via `omz_downloader` (Intel Open Model Zoo) if not present.

## Model Lifecycle

Models are managed via the `/models/*` API and the frontend Model Control Panel.

### States
- **Unloaded** — Not in memory, no resources consumed
- **Loading** — Being loaded (thread-safe, lock-protected)
- **Loaded** — Ready for inference

### Presets

The Model Control Panel provides tab-aware presets:

| Preset | LLM | VLM | Embeddings | Face Detection |
|--------|-----|-----|------------|----------------|
| Search Mode | Load | Unload | Load | Unload |
| Ingest Mode | Unload | Load | Load | Load |

### Pre-flight Validation

Before running search or ingestion, the frontend checks that required models are loaded:
- **Search** requires: LLM + Embeddings
- **Ingestion** requires: VLM + Embeddings + Face Detection

If models are missing, a warning banner is shown with instructions.

## Directory Layout

```
models/
├── Qwen3-4B-Instruct-ov/              # Agent LLM
├── Qwen2.5-VL-3B-Instruct/INT4/       # VLM captioner
├── all-MiniLM-L6-v2-ov/               # Text embeddings
└── intel/                              # Face detection (OMZ)
    ├── face-detection-retail-0004/FP32/
    ├── landmarks-regression-retail-0009/FP32/
    └── face-reidentification-retail-0095/FP32/
```
