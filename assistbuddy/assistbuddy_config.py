# AssistBuddy Configuration

## Model Architecture
VISION_ENCODER = "openai/clip-vit-base-patch16"
TEXT_ENCODER = "bert-base-uncased"
AUDIO_ENCODER = "openai/whisper-small"
DECODER = "gpt2"

HIDDEN_DIM = 768
FUSION_LAYERS = 4
FUSION_HEADS = 8

## Training Hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
MAX_EPOCHS = 5
MAX_GRAD_NORM = 1.0

## Generation Parameters
# ADMIN style (precise, factual)
ADMIN_TEMPERATURE = 0.3
ADMIN_TOP_P = 0.9
ADMIN_MAX_TOKENS = 512

# FRIEND style (casual, fluid)
FRIEND_TEMPERATURE = 0.7
FRIEND_TOP_P = 0.9
FRIEND_MAX_TOKENS = 512

## Dataset Configuration
SYNTHETIC_DATA_DIR = "./synthetic_data"
NUM_TRAIN_SAMPLES = 1000
NUM_VAL_SAMPLES = 200

## Privacy & Compliance
PII_DETECTION_CONFIDENCE_THRESHOLD = 0.7
AUTO_REDACT_PII = True  # Automatically redact unless authorized
LOG_PROVENANCE = True  # Always log sources

## OCR & Processing
OCR_BACKEND = "tesseract"  # or "easyocr"
OCR_LANGUAGES = ["en", "hi"]  # English and Hindi
OCR_MIN_CONFIDENCE = 0.4

## Camera/Video Analysis
VIDEO_KEYFRAME_METHOD = "scene_change"  # uniform, scene_change, histogram
VIDEO_FRAMES_PER_SUMMARY = 10
ACTIVITY_DETECTION_ENABLED = True
ACTIVITY_CLASSES = ["working", "idle", "absent", "transit", "unclear"]

## File Size Limits
MAX_PDF_PAGES = 100
MAX_VIDEO_DURATION_SECONDS = 600  # 10 minutes
MAX_IMAGE_SIZE_MB = 10
MAX_EXCEL_ROWS = 10000

## Checkpointing
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_EVERY_N_STEPS = 500
SAVE_TOTAL_LIMIT = 5  # Keep last 5 checkpoints

## Logging
LOG_EVERY_N_STEPS = 50
USE_WANDB = False  # Set to True if using Weights & Biases
USE_TENSORBOARD = True

## Special Tokens
SPECIAL_TOKENS = {
    'pad_token': '[PAD]',
    'admin_token': '[ADMIN]',
    'friend_token': '[FRIEND]',
    'redacted_token': '[REDACTED]'
}
