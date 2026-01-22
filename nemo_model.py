import os
import json
import logging
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.models import EncDecDiarLabelModel

from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# DIARIZATION PIPELINE
# -----------------------------------------------------------------------------
class DiarizationPipeline:
    def __init__(self, output_dir="diarization_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # MANIFEST CREATION
    # -------------------------------------------------------------------------
    def create_manifest(self, audio_path, rttm_path=None):
        manifest_path = os.path.join(self.output_dir, "input_manifest.json")

        meta = {
            "audio_filepath": os.path.abspath(audio_path),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": os.path.abspath(rttm_path) if rttm_path else None,
            "uem_filepath": None,
        }

        with open(manifest_path, "w") as f:
            f.write(json.dumps(meta) + "\n")

        return manifest_path

    # -------------------------------------------------------------------------
    # MSDD DIARIZATION
    # -------------------------------------------------------------------------
    def run_msdd(self, audio_file):
        print(f"\n[INFO] Running MSDD diarization on {audio_file}")

        manifest_path = self.create_manifest(audio_file)

        # Download NeMo diarization config if needed
        config_path = "diar_infer_telephonic.yaml"
        if not os.path.exists(config_path):
            os.system(
                "wget https://raw.githubusercontent.com/NVIDIA/NeMo/main/"
                "examples/speaker_tasks/diarization/conf/inference/"
                "diar_infer_telephonic.yaml"
            )

        # Load base config
        cfg = OmegaConf.load(config_path)

        # Override required fields
        override_cfg = OmegaConf.from_dotlist([
            f"diarizer.manifest_filepath={manifest_path}",
            f"diarizer.out_dir={self.output_dir}",
        ])
        cfg = OmegaConf.merge(cfg, override_cfg)

        # Set pretrained models
        cfg.diarizer.vad.model_path = "vad_multilingual_marblenet"
        cfg.diarizer.speaker_embeddings.model_path = "titanet_large"
        cfg.diarizer.msdd_model.model_path = "diar_msdd_telephonic"

        # Optional tuning
        cfg.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7]

        # Run diarization
        diarizer = NeuralDiarizer(cfg=cfg)
        diarizer.diarize()

        base = os.path.splitext(os.path.basename(audio_file))[0]
        return os.path.join(self.output_dir, "pred_rttms", f"{base}.rttm")

    # -------------------------------------------------------------------------
    # SORTFORMER DIARIZATION (END-TO-END)
    # -------------------------------------------------------------------------
    def run_sortformer(self, audio_file):
        print(f"\n[INFO] Running Sortformer diarization on {audio_file}")

        model = EncDecDiarLabelModel.from_pretrained(
            model_name="diar_sortformer_telephonic"
        )

        model.diarize(
            paths2audio_files=[audio_file],
            batch_size=1,
            out_dir=self.output_dir,
        )

        base = os.path.splitext(os.path.basename(audio_file))[0]
        return os.path.join(self.output_dir, "pred_rttms", f"{base}.rttm")


# -----------------------------------------------------------------------------
# DER CALCULATION
# -----------------------------------------------------------------------------
def calculate_der(reference_rttm, hypothesis_rttm):
    def load_rttm(path):
        ann = Annotation()
        with open(path) as f:
            for line in f:
                p = line.strip().split()
                start = float(p[3])
                dur = float(p[4])
                speaker = p[7]
                ann[Segment(start, start + dur)] = speaker
        return ann

    ref = load_rttm(reference_rttm)
    hyp = load_rttm(hypothesis_rttm)

    metric = DiarizationErrorRate()
    return metric(ref, hyp)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    WAV_FILE = "/content/Track_1_SD_DevData_1/Track_1_SD_DevData_1/Hindi/data/wav/2006763.wav"
    GROUND_TRUTH_RTTM = (
        "/content/Track_1_SD_DevData_1/Track_1_SD_DevData_1/"
        "Hindi/data/rttm/2006763_SPEAKER.rttm"
    )

    MODEL_TYPE = "msdd"  # "msdd" or "sortformer"

    pipeline = DiarizationPipeline(output_dir="my_diarization_results")

    # Run diarization
    if MODEL_TYPE == "msdd":
        pred_rttm = pipeline.run_msdd(WAV_FILE)
    else:
        pred_rttm = pipeline.run_sortformer(WAV_FILE)

    print(f"\n[SUCCESS] RTTM saved at: {pred_rttm}")

    # Compute DER if GT exists
    if os.path.exists(GROUND_TRUTH_RTTM):
        der = calculate_der(GROUND_TRUTH_RTTM, pred_rttm)
        print("-----------------------------------")
        print(f"Diarization Error Rate (DER): {der * 100:.2f}%")
        print("-----------------------------------")
    else:
        print("Ground-truth RTTM not found. DER skipped.")
