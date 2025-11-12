import os
import uuid
import json
import traceback
import torch
import torchaudio
import whisper
import comfy.model_patcher
import comfy.model_management as mm
import folder_paths

WHISPER_MODEL_SUBDIR = os.path.join("stt", "whisper")
WHISPER_PATCHER_CACHE = {}

class WhisperModelWrapper(torch.nn.Module):
    """Wrapper around a Whisper model for dynamic loading and memory tracking."""

    def __init__(self, model_name: str, download_root: str):
        super().__init__()
        self.model_name = model_name
        self.download_root = download_root
        self.whisper_model = None
        self.model_memory_bytes = 0

    def load_model(self, device: torch.device):
        """Load the Whisper model onto a device and track memory usage."""
        print(f"[whisper] Loading model '{self.model_name}' to device {device}")
        self.whisper_model = whisper.load_model(
            self.model_name,
            download_root=self.download_root,
            device=device
        )
        # Estimate memory usage of model weights
        try:
            self.model_memory_bytes = sum(p.numel() * p.element_size() for p in self.whisper_model.parameters())
        except Exception as e:
            print(f"[whisper] Failed to estimate model memory for '{self.model_name}': {e}")
            self.model_memory_bytes = 0
        print(f"[whisper] Loaded model '{self.model_name}' size={self.model_memory_bytes} bytes")


class WhisperPatcher(comfy.model_patcher.ModelPatcher):
    """Handles model patching, loading, and offloading for Whisper."""
    
    def patch_model(self, device_to=None, *args, **kwargs):
        print(f"[whisper] patch_model called for '{self.model.model_name}', load_device={self.load_device}")
        if self.model.whisper_model is None:
            print(f"[whisper] Model '{self.model.model_name}' not loaded, loading now")
            self.model.load_model(self.load_device)
            self.size = self.model.model_memory_bytes
            print(f"[whisper] Model '{self.model.model_name}' loaded with estimated size {self.size}")
        return super().patch_model(device_to=self.load_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        print(f"[whisper] unpatch_model called for '{self.model.model_name}', unpatch_weights={unpatch_weights}")
        if unpatch_weights:
            self.model.whisper_model = None
            self.model.model_memory_bytes = 0
            mm.soft_empty_cache()
            print(f"[whisper] Model '{self.model.model_name}' weights cleared and cache emptied")
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights, *args, **kwargs)


class WhisperTranscribeNode:
    @classmethod
    def INPUT_TYPES(cls):
        download_root = os.path.join(folder_paths.models_dir, WHISPER_MODEL_SUBDIR)
        local_models = [f.split(".")[0] for f in os.listdir(download_root) if f.endswith(".pt")] if os.path.exists(download_root) else []
        default_models = [
            'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 
            'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 
            'large-v3-turbo', 'turbo'
        ]
        available_models = sorted(set(local_models + default_models))
        languages = ["auto"] + [lang.capitalize() for lang in sorted(whisper.tokenizer.LANGUAGES.values())]

        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The input audio file to transcribe."}),
                "model": (available_models, {"default": "base", "tooltip": "Choose the Whisper model variant. Larger models are more accurate but slower."}),
            },
            "optional": {
                "language": (languages, {"default": "auto", "tooltip": "Language spoken in the audio. Set to 'auto' for automatic detection."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional initial prompt that can help Whisper maintain context (e.g., names or phrases)."}),
            }
        }

    RETURN_TYPES = ("STRING", "whisper_alignment", "STRING", "whisper_alignment", "STRING")
    RETURN_NAMES = ("text", "segments", "segments_str", "words", "words_str")
    FUNCTION = "transcribe"
    CATEGORY = "Audio"

    def transcribe(self, audio, model: str, language: str, prompt: str):
        """Transcribe audio into text, segments, and word-level timestamps."""
        print(f"[whisper] transcribe called: model={model}, language={language}, prompt_len={len(prompt) if prompt is not None else 0}")
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, f"{uuid.uuid1()}.wav")
        try:
            print(f"[whisper] Saving audio to {audio_path} (sr={audio['sample_rate']}, waveform_shape={getattr(audio['waveform'], 'shape', None)})")
            torchaudio.save(audio_path, audio['waveform'].squeeze(0), audio["sample_rate"])
        except Exception as e:
            print(f"[whisper] Failed to save audio to {audio_path}: {e}")
            traceback.print_exc()
            raise

        # Load or retrieve cached model patcher
        if model not in WHISPER_PATCHER_CACHE:
            print(f"[whisper] Creating new patcher for model '{model}'")
            model_wrapper = WhisperModelWrapper(model, os.path.join(folder_paths.models_dir, WHISPER_MODEL_SUBDIR))
            patcher = WhisperPatcher(
                model=model_wrapper,
                load_device=mm.get_torch_device(),
                offload_device=mm.unet_offload_device(),
                size=0
            )
            WHISPER_PATCHER_CACHE[model] = patcher
        else:
            print(f"[whisper] Using cached patcher for model '{model}'")
        patcher = WHISPER_PATCHER_CACHE[model]

        print(f"[whisper] Requesting mm.load_model_gpu for '{model}'")
        mm.load_model_gpu(patcher)
        whisper_model = patcher.model.whisper_model
        if whisper_model is None:
            print(f"[whisper] ERROR: whisper_model is None after mm.load_model_gpu for '{model}'")

        try:
            print(f"[whisper] Calling whisper_model.transcribe on {audio_path}")
            result = whisper_model.transcribe(audio_path, language=language, word_timestamps=True, initial_prompt=prompt)
            print(f"[whisper] Transcription finished for '{model}'. text_len={len(result.get('text',''))}, segments={len(result.get('segments', []))}")
        except Exception as e:
            print(f"[whisper] Exception during transcribe for model '{model}': {e}")
            traceback.print_exc()
            raise

        # Prepare segments
        segments = [{"value": seg["text"].strip(), "start": seg["start"], "end": seg["end"]} for seg in result['segments']]

        # Prepare word-level timestamps
        words = [
            {"value": word["word"].strip(), "start": word["start"], "end": word["end"]}
            for seg in result['segments'] for word in seg["words"]
        ]

        try:
            os.remove(audio_path)
            print(f"[whisper] Removed temporary audio file {audio_path}")
        except Exception as e:
            print(f"[whisper] Failed to remove temporary audio file {audio_path}: {e}")

        return result['text'], segments, json.dumps(segments), words, json.dumps(words)
