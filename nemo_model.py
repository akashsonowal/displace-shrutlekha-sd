import os
import json
import logging
import torch
import numpy as np
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer, EncDecDiarLabelModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment

# Initialize NeMo Logging
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

class DiarizationPipeline:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_manifest(self, audio_path, rttm_path=None):
        """Creates a NeMo-compatible manifest JSON."""
        manifest_path = os.path.join(self.output_dir, "input_manifest.json")
        meta = {
            "audio_filepath": os.path.abspath(audio_path),
            "offset": 0,
            "duration": None, # NeMo will calculate this
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": os.path.abspath(rttm_path) if rttm_path else None,
            "uem_filepath": None
        }
        
        with open(manifest_path, "w") as fout:
            fout.write(json.dumps(meta) + "\n")
            
        return manifest_path

    def run_msdd(self, audio_file):
        """
        Runs the Multi-scale Diarization Decoder (MSDD) pipeline.
        This uses TitaNet for embeddings + Clustering + MSDD for refinement.
        """
        print(f"\n[INFO] Running MSDD Diarization on {audio_file}...")
        
        # 1. Create Config for MSDD
        # We load the standard telephonic MSDD config structure
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
        config = OmegaConf.load(OmegaConf.from_cli_loader([f"diarizer.manifest_filepath={self.create_manifest(audio_file)}",
                                                          f"diarizer.out_dir={self.output_dir}"]))
        
        # Configure standard Pretrained Models
        config.diarizer.vad.model_path = "vad_multilingual_marblenet"
        config.diarizer.speaker_embeddings.model_path = "titanet_large"
        config.diarizer.msdd_model.model_path = "diar_msdd_telephonic" 
        
        # Enable MSDD
        config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7] 
        
        # Run Pipeline
        model = NeuralDiarizer(cfg=config)
        model.diarize()
        
        # The output RTTM is saved by NeMo in {output_dir}/pred_rttms/
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        generated_rttm = os.path.join(self.output_dir, "pred_rttms", f"{base_name}.rttm")
        return generated_rttm

    def run_sortformer(self, audio_file):
        """
        Runs the Sortformer End-to-End Diarization model.
        """
        print(f"\n[INFO] Running Sortformer on {audio_file}...")
        
        manifest_path = self.create_manifest(audio_file)
        
        # Load Sortformer from NGC
        # Note: Check for exact model name on NGC, usually 'diar_sortformer_telephonic' or similar
        # If specific checkpoint is needed, replace model_name with .nemo file path
        model = EncDecDiarLabelModel.from_pretrained(model_name="diar_sortformer_telephonic")
        
        # Run Inference
        model.diarize(
            paths2audio_files=[audio_file], 
            batch_size=1,
            out_dir=self.output_dir
        )
        
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        generated_rttm = os.path.join(self.output_dir, "pred_rttms", f"{base_name}.rttm")
        return generated_rttm

def calculate_der(reference_rttm, hypothesis_rttm):
    """
    Calculates DER using pyannote.metrics.
    """
    def load_rttm(path):
        annotations = Annotation()
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # RTTM format: SPEAKER file 1 start duration <NA> <NA> label <NA> <NA>
                start = float(parts[3])
                duration = float(parts[4])
                label = parts[7]
                annotations[Segment(start, start + duration)] = label
        return annotations

    ref = load_rttm(reference_rttm)
    hyp = load_rttm(hypothesis_rttm)

    metric = DiarizationErrorRate()
    der = metric(ref, hyp)
    
    return der

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # INPUTS
    WAV_FILE = "sample_audio.wav"
    GROUND_TRUTH_RTTM = "sample_audio.rttm" # Optional, for DER calculation
    
    # CHOOSE MODEL: 'msdd' or 'sortformer'
    MODEL_TYPE = 'msdd' 

    pipeline = DiarizationPipeline(output_dir="my_diarization_results")

    # 1. Run Diarization
    if MODEL_TYPE == 'msdd':
        pred_rttm = pipeline.run_msdd(WAV_FILE)
    else:
        pred_rttm = pipeline.run_sortformer(WAV_FILE)

    print(f"\n[SUCCESS] RTTM generated at: {pred_rttm}")

    # 2. Calculate DER (if ground truth exists)
    if os.path.exists(GROUND_TRUTH_RTTM):
        score = calculate_der(GROUND_TRUTH_RTTM, pred_rttm)
        print(f"--------------------------------------")
        print(f"Diarization Error Rate (DER): {score * 100:.2f}%")
        print(f"--------------------------------------")
    else:
        print("No ground truth RTTM provided. Skipping DER calculation.")