"""Smart Stethoscope -- AI-assisted heart sound screening.

Kuwait College of Science & Technology, Group 07 capstone project.

This package is organized as:
    config           -- environment-driven paths & audio pipeline settings
    signal_pipeline   -- pure DSP functions (filtering, windowing, features)
    clinical_info     -- per-class explanation/recommendation text
    inference         -- TFLite model loading + murmur-first decision rule
    audio_capture     -- microphone recording + demo-dataset playback
    report            -- PDF diagnosis report generation
    gui               -- PySide6 touchscreen application

See README.md for setup instructions and docs/MODEL_CARD.md +
docs/DATASET_CARD.md for what this model was trained on, how it was
evaluated, and its known limitations.
"""

__version__ = "1.0.0"
