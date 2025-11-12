import os
import uuid
import json
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
        self.whisper_model = whisper.load_model(
            self.model_name,
            download_root=self.download_root,
            device=device
        )
        # Estimate memory usage of model weights
        self.model_memory_bytes = sum(p.numel() * p.element_size() for p in self.whisper_model.parameters())


class WhisperPatcher(comfy.model_patcher.ModelPatcher):
    """Handles model patching, loading, and offloading for Whisper."""
    
    def patch_model(self, device_to=None, *args, **kwargs):
        if self.model.whisper_model is None:
            self.model.load_model(self.load_device)
            self.size = self.model.model_memory_bytes
        return super().patch_model(device_to=self.load_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        if unpatch_weights:
            self.model.whisper_model = None
            self.model.model_memory_bytes = 0
            mm.soft_empty_cache()
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights, *args, **kwargs)


class WhisperTranscribeNode:
    FUNCTION = "transcribe"
    CATEGORY = "Audio"
    RETURN_TYPES = ("STRING", "whisper_alignment", "STRING", "whisper_alignment", "STRING")
    RETURN_NAMES = ("text", "segments", "segments_str", "words", "words_str")

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
                "audio": ("AUDIO",),
                "model": (available_models,),
            },
            "optional": {
                "language": (languages,),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def transcribe(self, audio, model: str, language: str, prompt: str):
        """Transcribe audio into text, segments, and word-level timestamps."""
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        audio_path = os.path.join(temp_dir, f"{uuid.uuid1()}.wav")
        torchaudio.save(audio_path, audio['waveform'].squeeze(0), audio["sample_rate"])

        # Load or retrieve cached model patcher
        if model not in WHISPER_PATCHER_CACHE:
            model_wrapper = WhisperModelWrapper(model, os.path.join(folder_paths.models_dir, WHISPER_MODEL_SUBDIR))
            patcher = WhisperPatcher(
                model=model_wrapper,
                load_device=mm.get_torch_device(),
                offload_device=mm.unet_offload_device(),
                size=0
            )
            WHISPER_PATCHER_CACHE[model] = patcher
        patcher = WHISPER_PATCHER_CACHE[model]

        mm.load_model_gpu(patcher)
        whisper_model = patcher.model.whisper_model

        result = whisper_model.transcribe(audio_path, language=language, word_timestamps=True, initial_prompt=prompt)

        # Prepare segments
        segments = [{"value": seg["text"].strip(), "start": seg["start"], "end": seg["end"]} for seg in result['segments']]

        # Prepare word-level timestamps
        words = [
            {"value": word["word"].strip(), "start": word["start"], "end": word["end"]}
            for seg in result['segments'] for word in seg["words"]
        ]

        os.remove(audio_path)

        return result['text'], segments, json.dumps(segments), words, json.dumps(words)
