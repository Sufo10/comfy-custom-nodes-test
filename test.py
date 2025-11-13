import json
import time
import logging
import requests
from pathlib import Path
import concurrent.futures

class SceneVideoWanIteratorNode:
    """
    Node that:
      1. Sends each scene to a ComfyUI video generation workflow via HTTP.
      2. Polls for completion.
      3. Downloads the result.
      4. Logs all events with timing and error details.
    
    This node expects the 'scenes_json' input, which is a required string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return  {
            "required": {
                # Changed to JSON string input, which is typical for passing structured data between nodes
                "scenes_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": "JSON array of scenes with 'scene', 'start', 'end', 'dialogue', and 'scenario' keys"
                }),
                "comfy_api_url": ("STRING", {"default": "http://localhost:8188", "tooltip": "Base URL of the ComfyUI API (e.g., http://localhost:8188)"}),
                "workflow_path": ("STRING", {"default": "./workflow.json", "tooltip": "Path to the ComfyUI workflow template JSON file."}),
                "video_output_dir": ("STRING", {"default": "output/comfy_videos", "tooltip": "Directory to save the final MP4 videos."}),
            },
            "optional": {
                "max_workers": ("INT", {"default": 3, "min": 1, "max": 20, "tooltip": "Maximum number of concurrent scenes to process (API calls/downloads)."}),
                "trigger": ("INT", { 
                    "default": 0,
                    "tooltip": "Change this value to re-trigger the node"
                }),
            },
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("results", "output_directory")
    FUNCTION = "run_scenes"
    CATEGORY = "Video Generation"
    OUTPUT_NODE = True  # Added OUTPUT_NODE = True for clarity

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Sets up file logging for the node."""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        # Configure a local logger just for this node instance
        log_name = f"scene_video_iterator_{id(self)}.log"
        handler = logging.FileHandler(log_path / log_name, mode='a')
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)

        logger = logging.getLogger(f"SceneVideoWanIteratorNode_{id(self)}")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            logger.addHandler(handler)
        
        # Add a stream handler for console output during execution
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        return logger

    def _inject_scene_into_workflow(self, workflow, scenario: str):
        """Deep copies the workflow and injects the scenario into a 'TextPrompt' node."""
        wf_copy = json.loads(json.dumps(workflow))  # deep copy
        injected = False
        for node_data in wf_copy["nodes"].values():
            # Customize this check if your prompt node has a different class_type or custom title
            if node_data["class_type"] == "CLIPTextEncode" or node_data.get("title") == "TextPrompt":
                # Assuming the prompt is always in the 'text' input field
                if "text" in node_data["inputs"]:
                    # Safely replace the prompt text
                    node_data["inputs"]["text"] = scenario
                    injected = True
                    break
        
        if not injected:
            self.logger.warning("Could not find a suitable node (e.g., CLIPTextEncode or titled 'TextPrompt') to inject the scenario. The default prompt will be used.")

        return wf_copy

    def _poll_for_completion(self, comfy_api_url: str, prompt_id: str, scene_id: int, poll_interval: int = 3):
        """Polls the ComfyUI API history for the prompt's completion."""
        self.logger.info(f"Scene {scene_id} - Polling for completion (ID: {prompt_id}).")
        
        # Simple backoff logic in case of network issues
        max_retries = 5
        retries = 0

        while True:
            time.sleep(poll_interval)
            try:
                response = requests.get(f"{comfy_api_url}/history/{prompt_id}", timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if prompt_id in data:
                    entry = data[prompt_id]
                    if entry.get("status", "").lower() == "completed":
                        return entry
                    elif entry.get("status", "").lower() == "failed":
                        raise RuntimeError(f"Workflow failed on ComfyUI for scene {scene_id}.")
                # If prompt_id not in data, it might not be in history yet, just continue polling
                
                retries = 0 # Reset retries on successful status check

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= max_retries:
                    raise ConnectionError(f"Polling failed after {max_retries} retries: {e}")
                self.logger.warning(f"Scene {scene_id} - Polling failed ({e}). Retrying in {poll_interval}s...")
                

    def _download_video(self, comfy_api_url, video_output_dir, result_data, scene_id):
        """Finds the output file info in the history result and downloads it."""
        try:
            video_output_dir.mkdir(parents=True, exist_ok=True)
            outputs = result_data.get("outputs", {})
            
            for node_output in outputs.values():
                for file_info in node_output.get("images", []): # ComfyUI often lists generated files under 'images'
                    if file_info.get("type", "").lower() == "output" and file_info.get("filename"):
                        # Build the download URL
                        file_url = f"{comfy_api_url}/view?filename={file_info['filename']}&subfolder={file_info.get('subfolder', '')}&type=output"
                        local_path = video_output_dir / f"scene_{scene_id}_{file_info['filename']}"
                        
                        self.logger.info(f"Scene {scene_id} - Downloading from {file_url}")
                        r = requests.get(file_url, stream=True, timeout=120)
                        r.raise_for_status()

                        # Check if file is actually a video (simple check for now)
                        if local_path.suffix.lower() not in ['.mp4', '.mov', '.webm', '.gif']:
                            local_path = local_path.with_suffix('.mp4') # Force .mp4 extension if not specified

                        with open(local_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        return local_path
            
            raise FileNotFoundError("No output files (images/videos) found in workflow result.")
        except Exception as e:
            self.logger.exception(f"Scene {scene_id} - Video download failed: {e}")
            raise

    def _run_scene(self, comfy_api_url: str, video_output_dir: Path, workflow_data , scene):
        """Submits, polls, and downloads the video for a single scene."""
        scene_id = scene.get("scene", "N/A")
        scenario = scene.get("scenario", "No scenario provided")
        start_time = time.time()
        
        self.logger.info(f"\n--- Scene {scene_id} ---")
        self.logger.info(f"Scenario: {scenario[:80]}...")
        
        try:
            # 1. Inject Prompt
            workflow = self._inject_scene_into_workflow(workflow_data, scenario)
            
            # 2. Queue Prompt
            response = requests.post(f"{comfy_api_url}/prompt", json={"prompt": workflow})
            response.raise_for_status()
            prompt_id = response.json().get("prompt_id")
            
            if not prompt_id:
                raise ValueError("ComfyUI did not return a prompt_id.")
            
            # 3. Poll for Result
            result = self._poll_for_completion(comfy_api_url, prompt_id, scene_id)
            
            # 4. Download Video
            video_path = self._download_video(comfy_api_url, video_output_dir, result, scene_id)
            
            # 5. Success
            duration = round(time.time() - start_time, 2)
            self.logger.info(f"Scene {scene_id} - Completed in {duration}s. Saved to {video_path}")
            
            return {"scene": scene_id, "path": str(video_path), "status": "done", "duration_s": duration}
        
        except Exception as e:
            # Log failure and return error result
            duration = round(time.time() - start_time, 2)
            error_message = f"{e.__class__.__name__}: {str(e)}"
            self.logger.error(f"Scene {scene_id} - FAILED after {duration}s. Error: {error_message}")
            return {"scene": scene_id, "error": error_message, "status": "failed", "duration_s": duration}

    def run_scenes(self, scenes_json: str, comfy_api_url: str, video_output_dir: str, workflow_path: str, max_workers: int = 3, trigger: int = 0):
        """Main execution function."""
        try:
            scenes = json.loads(scenes_json)
        except json.JSONDecodeError:
            error_msg = "Invalid JSON in 'scenes_json' input. Please check the format."
            self.logger.error(error_msg)
            return (
                [{"scene": "N/A", "error": error_msg, "status": "failed"}],
                video_output_dir
            )

        if not scenes:
            self.logger.info("No scenes provided in JSON input.")
            return ([], video_output_dir)

        try:
            video_output_dir_path = Path(video_output_dir)
            video_output_dir_path.mkdir(parents=True, exist_ok=True)
            
            with open(workflow_path, "r") as f:
                workflow_data = json.load(f)
        except FileNotFoundError:
            error_msg = f"Workflow file not found at: {workflow_path}"
            self.logger.error(error_msg)
            return (
                [{"scene": "N/A", "error": error_msg, "status": "failed"}],
                video_output_dir
            )
        except Exception as e:
            error_msg = f"Error preparing node: {e}"
            self.logger.error(error_msg)
            return (
                [{"scene": "N/A", "error": error_msg, "status": "failed"}],
                video_output_dir
            )

        self.logger.info(f"\n--- Starting Video Generation for {len(scenes)} Scenes ---")
        self.logger.info(f"ComfyUI URL: {comfy_api_url}")
        self.logger.info(f"Max Concurrent Workers: {max_workers}")

        results = []
        
        # Use ThreadPoolExecutor to limit concurrency to max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scenes
            futures = {
                executor.submit(self._run_scene, comfy_api_url, video_output_dir_path, workflow_data, scene): scene 
                for scene in scenes
            }
            
            # Retrieve results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    scene_failed = futures[future]
                    self.logger.exception(f"An exception occurred during execution of scene {scene_failed.get('scene', 'N/A')}: {exc}")
                    results.append({"scene": scene_failed.get('scene', 'N/A'), "error": str(exc), "status": "failed"})

        self.logger.info(f"\n--- All Scenes Processed. Total Results: {len(results)} ---\n")
        
        return (results, video_output_dir)