import os
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path

class CustomVideoConcatenator:
    """
    ComfyUI Custom Node to concatenate multiple video files listed in a JSON string 
    using the moviepy library.
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
                "output_directory": ("STRING", {"default": "output"}),
            },
        }

    # Define the output types for the node
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)

    # Define the name of the function to execute when the node runs
    FUNCTION = "concat_videos"

    def concat_videos(self, multiline_video_paths, output_filename_prefix, output_directory):
        """
        Loads video clips, concatenates them, and saves the final video file.
        The output video is saved to the location specified by output_directory.
        """
        
        print(f"Starting video concatenation for files with prefix: {output_filename_prefix}")
        
        # 1. Parse the multiline string input into a Python list of paths
        video_paths = [path.strip() for path in multiline_video_paths.split('\n') if path.strip()]
        video_paths = [os.path.abspath(path) for path in video_paths]
        for path in video_paths:
            if not Path(path).exists():
                raise ValueError(f"Video path does not exist: {path}")

        if len(video_paths) < 2:
            warning_msg = "WARNING: Need at least two videos to concatenate. Returning original path or empty string."
            print(warning_msg)
            return (video_paths[0] if video_paths else "",)

        # 2. Load the video clips
        clips = []
        path = "" # Initialize path for cleanup/error reporting
        try:
            for path in video_paths:
                # Use VideoFileClip to load the video object from the path
                clip = VideoFileClip(path)
                clips.append(clip)
                print(f"Loaded clip: {path} (Duration: {clip.duration:.2f}s)")
        except Exception as e:
            # Cleanup any already loaded clips
            for c in clips:
                c.close()
            error_msg = f"ERROR loading video clip at path: {path}. Error: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)

        # 3. Concatenate the clips
        final_clip = None
        try:
            # concatenate_videoclips joins the clips end-to-end.
            # 'compose' method handles clips of different sizes by centering them.
            final_clip = concatenate_videoclips(clips, method='compose')
            print(f"Successfully concatenated {len(clips)} clips. Total duration: {final_clip.duration:.2f}s")
        except Exception as e:
            # Cleanup and report error
            for c in clips:
                c.close()
            error_msg = f"ERROR during moviepy concatenation: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)

        # 4. Define the output path using the provided output_directory
        output_dir_path = Path(os.path.abspath(output_directory)) 
        # Ensure the directory structure exists
        output_dir_path.mkdir(parents=True, exist_ok=True) 
            
        timestamp = int(time.time())
        # Clean the prefix and generate a unique filename
        clean_prefix = "".join(c for c in output_filename_prefix if c.isalnum() or c in ('_', '-')).rstrip()
        filename = f"{clean_prefix}_{timestamp}.mp4"
        output_path = str(output_dir_path / filename)

        # 5. Write the final video file
        try:
            final_clip.write_videofile(
                output_path, 
                codec='libx264',           
                audio_codec='aac',         
                temp_audiofile='temp-audio.m4a', # Temporary file for audio processing
                remove_temp=True,
                verbose=False, 
                logger=None # Suppress moviepy console spam
            )
            print(f"Video saved successfully to: {output_path}")
            
            return (output_path,)
            
        except Exception as e:
            error_msg = f"ERROR writing video file to {output_path}. Error: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
            
        finally:
            for c in clips:
                c.close()
            if final_clip:
                final_clip.close()