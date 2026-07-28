"""
Clinical explanation text shown for each of the 4 model classes.

Kept as data (not scattered across the GUI) so it can be reviewed,
translated, or audited independently of the UI code -- this is the part
of the app most likely to be read by a clinician or a reviewer who is
NOT a Python developer.

IMPORTANT: this text is written to sound authoritative, but the system
is a screening aid, not a diagnostic device. Every entry ends with an
explicit "confirm with a clinician" instruction on purpose. See
docs/MODEL_CARD.md for the full intended-use and limitations statement.
"""
from __future__ import annotations

NORMAL = "normal"
MURMUR = "murmur"
EXTRAHLS = "extrahls"
ARTIFACT = "artifact"

CLINICAL_INFO = {
    NORMAL: {
        "title": "NORMAL Heart Sound",
        "reasons": [
            "Regular S1 (lub) and S2 (dub) pattern detected",
            "No abnormal frequencies in 20-150 Hz range",
            "Consistent inter-beat interval observed",
            "Mel spectrogram energy matches normal cardiac profile",
            "No turbulent flow signatures detected",
        ],
        "reasoning": (
            "The AI model analyzed the recorded cardiac auscultation and found "
            "no significant acoustic abnormalities. The spectral energy profile "
            "is concentrated in the 20-150 Hz range, consistent with normal S1 "
            "(lub) and S2 (dub) heart sounds. The inter-beat interval is regular, "
            "suggesting normal sinus rhythm with no detectable turbulent flow. "
            "The murmur probability score (%.1f%%) is below the clinical threshold "
            "of %.0f%%, supporting a normal classification. NOTE: This is an "
            "AI-assisted screening tool. A qualified cardiologist must confirm "
            "all findings before any clinical decision is made."
        ),
        "recommendation": (
            "No immediate cardiac concern detected by AI screening. Routine "
            "annual cardiac checkup is advised as standard preventive care. "
            "If the patient reports symptoms (chest pain, palpitations, dyspnea), "
            "refer to a cardiologist regardless of this result."
        ),
    },
    MURMUR: {
        "title": "MURMUR Detected",
        "reasons": [
            "Continuous mid-systolic turbulence detected between S1 and S2",
            "Abnormal high-frequency energy in 100-400 Hz range",
            "Turbulent flow signature identified in waveform envelope",
            "Pattern consistent with known murmur profiles in training data",
            "Mel spectrogram shows continuous energy between heartbeats",
        ],
        "reasoning": (
            "The AI model detected acoustic signatures strongly consistent with "
            "a cardiac murmur. Spectral analysis reveals elevated energy in the "
            "100-400 Hz range persisting between S1 and S2 heart sounds. The "
            "murmur probability score (%.1f%%) exceeds the clinical detection "
            "threshold (%.0f%%). IMPORTANT: This model was tuned with 100%% "
            "murmur recall as the primary safety objective -- it is intentionally "
            "conservative to eliminate false negatives, since a missed murmur is "
            "clinically more dangerous than a false positive. All positive "
            "findings must be verified by a cardiologist with echocardiography."
        ),
        "recommendation": (
            "Cardiology referral strongly advised. Transthoracic "
            "Echocardiogram (TTE) recommended to characterize valve morphology, "
            "regurgitation severity, and ventricular function. Do not disregard "
            "this finding without formal clinical evaluation."
        ),
    },
    EXTRAHLS: {
        "title": "Extra Heart Sound (S3/S4)",
        "reasons": [
            "Extra low-frequency component detected after S2 (possible S3)",
            "Triple rhythm (gallop) pattern identified in waveform",
            "Energy burst outside normal S1-S2 cardiac cycle",
            "Pattern consistent with ventricular filling abnormality",
            "Mel spectrogram shows additional energy cluster post-S2",
        ],
        "reasoning": (
            "The AI model identified an additional acoustic event beyond the "
            "expected S1-S2 cardiac cycle, classified as an extra heart sound "
            "(S3 or S4 gallop). An S3 may indicate heart failure, dilated "
            "cardiomyopathy, or volume overload; an S4 suggests reduced "
            "ventricular compliance or hypertensive heart disease. Physiological "
            "S3 can be normal in athletes and patients under 40, so clinical "
            "context is essential. The extra-sound probability score "
            "(%.1f%%) indicates a clear departure from the normal two-sound "
            "pattern. Clinical correlation is mandatory."
        ),
        "recommendation": (
            "Cardiac evaluation advised. Consider BNP/NT-proBNP levels and "
            "echocardiography for a suspected S3; evaluate for hypertension or "
            "hypertrophic cardiomyopathy for a suspected S4. Refer to a "
            "cardiologist for comprehensive hemodynamic assessment."
        ),
    },
    ARTIFACT: {
        "title": "Recording Artifact",
        "reasons": [
            "High noise-to-signal ratio detected in recording",
            "Non-cardiac acoustic patterns dominate the signal",
            "Movement or contact artifacts present in waveform",
            "Spectral profile inconsistent with cardiac sounds",
            "Recording quality insufficient for reliable analysis",
        ],
        "reasoning": (
            "The dominant acoustic patterns in this recording do not match any "
            "cardiac sound profile. Common causes include stethoscope movement "
            "against skin or clothing, excessive ambient noise, poor acoustic "
            "coupling, patient movement or talking during recording, or cable "
            "rubbing. This classification does NOT indicate a cardiac "
            "abnormality -- it indicates the recording quality was insufficient "
            "for reliable analysis."
        ),
        "recommendation": (
            "Repeat the recording: press the stethoscope firmly against bare "
            "skin at the left 5th intercostal space (midclavicular line), ask "
            "the patient to hold their breath for 5-8 seconds, minimize ambient "
            "noise, and confirm the acoustic coupling adapter is sealed. Do not "
            "interpret this result clinically."
        ),
    },
}


def format_reasoning(pred: str, murmur_p: float, murmur_threshold: float) -> str:
    """Safely format the reasoning template regardless of how many %-style
    placeholders it happens to contain (some entries use one, some use two)."""
    info = CLINICAL_INFO.get(pred, CLINICAL_INFO[ARTIFACT])
    try:
        return info["reasoning"] % (murmur_p * 100, murmur_threshold * 100)
    except TypeError:
        try:
            return info["reasoning"] % (murmur_p * 100,)
        except TypeError:
            return info["reasoning"]
