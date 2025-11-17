import os
import time
import shutil
from moviepy import VideoFileClip, concatenate_videoclips 
from pathlib import Path
import folder_paths 
import torch
import torchaudio

class CustomVideoConcatenator:
    """
    ComfyUI Custom Node to concatenate multiple video files listed in a multiline string 
    using the moviepy library, supporting external audio file paths or direct audio 
    waveform input from other ComfyUI nodes.
    """

    CATEGORY = "Custom Video Generation"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multiline_video_paths": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "Enter a multiline string of video paths here"
                }),
                "output_filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "output_directory": ("STRING", {"default": "output/moviepy"}), # Set a descriptive default subdirectory
            },
            "optional": { 
                # String path input (e.g., from a Load Text node or text entry)
                "audio_path": ("STRING", { 
                    "default": "",
                    "placeholder": "Optional: Path to an external audio file (e.g., .mp3, .wav)"
                }),
                # Audio waveform object from an audio node
                "audio": ("AUDIO", {
                    "placeholder": "Optional: Audio waveform and sample rate",
                    "forceInput": True
                })
            }
        }

    # Define the output types for the node
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)

    # Define the name of the function to execute when the node runs
    FUNCTION = "concat_videos"

    def concat_videos(self, multiline_video_paths, output_filename_prefix, output_directory, audio_path=None, audio=None):
        
        print(f"Starting video concatenation for files with prefix: {output_filename_prefix}")

        final_audio_path = None
        temp_audio_dir = None
        
        # --- 0. Determine Audio Source ---
        if audio_path and os.path.exists(audio_path):
            final_audio_path = os.path.abspath(audio_path.strip())
            print(f"Using external audio file: {final_audio_path}")
            
        elif audio is not None and 'waveform' in audio and 'sample_rate' in audio:
            # Create a dedicated temporary folder for the audio file
            try:
                base_temp_dir = folder_paths.get_temp_directory()
                temp_audio_dir = os.path.join(base_temp_dir, f"video_concat_audio_temp_{int(time.time())}")
                os.makedirs(temp_audio_dir, exist_ok=True)
                
                # Define the final path inside the temp folder
                final_audio_path = os.path.join(temp_audio_dir, "temp_audio.wav")
                waveform = audio['waveform']
                sample_rate = audio['sample_rate']
                
                # Normalize and format tensor for torchaudio.save (standard practice)
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)
                elif waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                
                if waveform.dtype != torch.float32:
                    waveform = waveform.float()
                waveform = waveform.clamp(-1, 1)
                
                torchaudio.save(final_audio_path, waveform, sample_rate)
                print(f"Using waveform audio, saved temporarily to: {final_audio_path}")
                
            except Exception as e:
                # Log error and proceed without audio if waveform saving fails
                print(f"WARNING: Failed to process or save waveform audio. Error: {e}. Proceeding without audio.")
                final_audio_path = None 
                temp_audio_dir = None
        
        if not final_audio_path:
            print("No valid audio track will be attached. Concatenating video clips silently.")
        
        # --- 1. Parse Video Paths and Pre-Checks ---
        video_paths = [path.strip() for path in multiline_video_paths.split('\n') if path.strip()]
        video_paths = [os.path.abspath(path) for path in video_paths]
        
        if len(video_paths) < 1:
            raise ValueError("ERROR: Must provide at least one video path to process.")
        
        for path in video_paths:
            if not Path(path).exists():
                raise ValueError(f"Video path does not exist: {path}")

        if len(video_paths) == 1:
            print(f"WARNING: Only one video provided. Returning the path to the original file: {video_paths[0]}")
            return (video_paths[0],)

        # --- 2. Load and Concatenate Video Clips ---
        clips = []
        final_clip = None
        try:
            for path in video_paths:
                clip = VideoFileClip(path)
                clips.append(clip)
            
            # concatenate_videoclips joins the clips end-to-end.
            final_clip = concatenate_videoclips(clips, method='compose')
            print(f"Successfully concatenated {len(clips)} clips. Total duration: {final_clip.duration:.2f}s")   
        except Exception as e:
            # Handle error during load or concatenation
            error_msg = f"ERROR during video processing: {e}"
            print(error_msg)
            for c in clips:
                c.close()
            if final_clip:
                final_clip.close()
            raise RuntimeError(error_msg)
            
        
        # --- 4. Define Output Path ---
        # Use ComfyUI standard output directory with the user-defined subdirectory
        output_dir_path = Path(os.path.abspath(output_directory))
        output_dir_path.mkdir(parents=True, exist_ok=True) 
            
        timestamp = int(time.time())
        clean_prefix = "".join(c for c in output_filename_prefix if c.isalnum() or c in ('_', '-')).rstrip()
        filename = f"{clean_prefix}_{timestamp}.mp4"
        output_path = str(output_dir_path / filename)

        # --- 5. Write the final video file (with quality optimization) ---
        try:
            final_clip.write_videofile(
                output_path, 
                codec='libx264',  
                audio=final_audio_path,         
                audio_codec='aac',        
            )
            print(f"Video saved successfully to: {output_path}")
            
            return (output_path,)
            
        except Exception as e:
            error_msg = f"ERROR writing video file to {output_path}. Error: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
            
        finally:
            # --- 6. Cleanup ---
            for c in clips:
                c.close()
            if final_clip:
                final_clip.close()
            
            # Clean up the temporary directory created for the waveform audio
            if temp_audio_dir and os.path.exists(temp_audio_dir):
                try:
                    shutil.rmtree(temp_audio_dir)
                    print(f"Cleaned up temp audio directory: {temp_audio_dir}")
                except Exception as e:
                    print(f"WARNING: Failed to clean up temp audio directory {temp_audio_dir}. Error: {e}")