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
                "scenes_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": "JSON array of scenes with 'scene', 'start', 'end', 'dialogue', and 'scenario' keys"
                }),
                "comfy_api_url": ("STRING", {"default": "http://localhost:8188", "tooltip": "Base URL of the ComfyUI API (e.g., http://localhost:8188)"}),
                "workflow_path": ("STRING", {"default": "./workflow.json", "tooltip": "Path to the ComfyUI workflow template JSON file."}),
                "video_output_dir": ("STRING", {"default": "output/comfy_videos", "tooltip": "Directory to save the final MP4 videos (NOTE: Download is commented out)."},),
            },
            "optional": {
                "max_workers": ("INT", {"default": 3, "min": 1, "max": 20, "tooltip": "Maximum number of concurrent scenes to process (API calls/downloads)."}),
                "trigger": ("INT", { 
                    "default": 0,
                    "tooltip": "Change this value to re-trigger the node"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING") 
    RETURN_NAMES = ("results", "output_directory")
    FUNCTION = "run_scenes"
    CATEGORY = "Video Generation"
    OUTPUT_NODE = True 

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Sets up file logging for the node."""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        # Unique logger name
        log_name = f"scene_video_iterator_{int(time.time())}.log" 
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

    def _inject_scene_into_workflow(self, workflow, scene):
        """Deep copies the workflow and injects the scenario and sets the filename prefix."""
        scenario = scene.get("scenario", "No scenario provided")
        scene_id = scene.get("scene", "N/A")

        wf_copy = json.loads(json.dumps(workflow)) 
        
        # Inject Scenario into node 89 (CLIPTextEncode/Prompt)
        if "6" in wf_copy:
            wf_copy["6"]["inputs"]["text"] = scenario
            self.logger.info(f"Scene {scene_id} - Injected scenario into node 6.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node 6 not found for scenario injection.")

        # Set Filename Prefix into node 80 (Save Image/Video)
        if "58" in wf_copy:
            prefix = f"video/test/scene_{scene_id}"
            wf_copy["58"]["inputs"]["filename_prefix"] = prefix
            self.logger.info(f"Scene {scene_id} - Set filename prefix to '{prefix}' in node 58.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node 58 not found for filename prefix setting.")
        return wf_copy

    def _poll_for_completion(self, comfy_api_url, prompt_id, scene_id, poll_interval = 5):
        """Polls the ComfyUI API history for the prompt's completion."""
        self.logger.info(f"Scene {scene_id} - Starting poll loop for prompt ID: {prompt_id}.")
        
        max_retries = 500
        retries = 0

        while True:
            time.sleep(poll_interval)
            try:
                response = requests.get(f"{comfy_api_url}/history/{prompt_id}", timeout=10)
                response.raise_for_status() # Catches HTTPError if status code is bad
                data = response.json()
                self.logger.info(f"Scene {scene_id} - Polling response status code: {data}")
                self.logger.info(f"Scene {scene_id} - Polling response: {json.dumps(data, indent=2)}")
                
                if prompt_id in data:
                    entry = data[prompt_id]
                    status = entry.get("status", {"status_str": ""}).get("status_str", "").lower()
                    
                    if status == "success":
                        self.logger.info(f"Scene {scene_id} - Polling successful. Status: COMPLETED.")
                        return entry
                    elif status == "error":
                        self.logger.error(f"Scene {scene_id} - Polling failed. Status: FAILED.")
                        raise RuntimeError(f"Workflow failed on ComfyUI for scene {scene_id}.")
                
                retries += 1
                self.logger.error(f"Scene {scene_id} - retries: {retries}")

            except requests.exceptions.HTTPError as e:
                self.logger.error(f"Scene {scene_id} - Polling HTTP error: {e}. Status code: {e.response.status_code}")
                raise ConnectionError(f"Polling HTTP failure: {e}")
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= max_retries:
                    self.logger.error(f"Scene {scene_id} - Polling failed after {max_retries} retries.")
                    raise ConnectionError(f"Polling connection failed after {max_retries} retries: {e}")
                self.logger.warning(f"Scene {scene_id} - Polling connection error ({e}). Retrying in {poll_interval}s...")
                

    # def _download_video(self, comfy_api_url, video_output_dir, result_data, scene_id):
    #     """Finds the output file info in the history result and downloads it."""
    #     self.logger.info(f"Scene {scene_id} - Starting video download process.")
    #     try:
    #         video_output_dir.mkdir(parents=True, exist_ok=True)
    #         outputs = result_data.get("outputs", {})
            
    #         for node_output in outputs.values():
    #             for file_info in node_output.get("images", []): 
    #                 if file_info.get("type", "").lower() == "output" and file_info.get("filename"):
    #                     filename = file_info['filename']
    #                     subfolder = file_info.get('subfolder', '')
                        
    #                     file_url = f"{comfy_api_url}/view?filename={filename}&subfolder={subfolder}&type=output"
    #                     local_path = video_output_dir / f"scene_{scene_id}_{filename}"
                        
    #                     self.logger.info(f"Scene {scene_id} - Identified output file: {filename}. Starting download.")
                        
    #                     r = requests.get(file_url, stream=True, timeout=120)
    #                     r.raise_for_status() # Catches HTTPError if status code is bad
                        
    #                     # Ensure .mp4 extension for output
    #                     if local_path.suffix.lower() not in ['.mp4', '.mov', '.webm', '.gif']:
    #                         local_path = local_path.with_suffix('.mp4')

    #                     with open(local_path, "wb") as f:
    #                         for chunk in r.iter_content(chunk_size=8192):
    #                             f.write(chunk)
                        
    #                     self.logger.info(f"Scene {scene_id} - Download complete. Saved to: {local_path.resolve()}")
    #                     return local_path
            
    #         self.logger.warning(f"Scene {scene_id} - No video file information found in ComfyUI history output.")
    #         raise FileNotFoundError("No output files (images/videos) found in workflow result.")
        
    #     except requests.exceptions.HTTPError as e:
    #         self.logger.error(f"Scene {scene_id} - Download HTTP error: {e}. Status code: {e.response.status_code}")
    #         raise ConnectionError(f"Download HTTP failure: {e}")
    #     except Exception as e:
    #         self.logger.exception(f"Scene {scene_id} - Video download failed: {e}")
    #         raise

    def _run_scene(self, comfy_api_url, video_output_dir, workflow_data, scene):
        """Submits, polls, and downloads the video for a single scene."""
        scene_id = scene.get("scene", "N/A")
        scenario = scene.get("scenario", "No scenario provided")
        start_time = time.time()
        
        self.logger.info(f"\n--- Scene {scene_id} START ---")
        self.logger.info(f"Scenario Preview: {scenario[:80]}...")
        
        try:
            # 1. Inject Prompt
            self.logger.info(f"Scene {scene_id} (Step 1/3): Injecting prompt into workflow.")
            workflow = self._inject_scene_into_workflow(workflow_data, scene)
            
            # 2. Queue Prompt
            self.logger.info(f"Scene {scene_id} (Step 2/3): Sending prompt to ComfyUI API: {comfy_api_url}/prompt")
            response = requests.post(f"{comfy_api_url}/prompt", json={"prompt": workflow})
            response.raise_for_status() # Catches HTTPError on bad status code
            
            response_json = response.json()
            prompt_id = response_json.get("prompt_id")
            
            if not prompt_id:
                raise ValueError(f"ComfyUI did not return a prompt_id. Response: {response_json}")
            self.logger.info(f"Scene {scene_id} - Prompt successfully queued. Prompt ID: {prompt_id}")
            
            # 3. Poll for Result
            self.logger.info(f"Scene {scene_id} (Step 3/3): Polling for completion.")
            result = self._poll_for_completion(comfy_api_url, prompt_id, scene_id)
            
            # 4. Download Video (COMMENTED OUT)
            # self.logger.info(f"Scene {scene_id} (Step 4/4): Downloading video (SKIPPING).")
            # video_path = self._download_video(comfy_api_url, video_output_dir, result, scene_id)
            # self.logger.info(f"Scene {scene_id} - Output saved on ComfyUI server. Path: {video_path}")
            
            # 5. Success
            duration = round(time.time() - start_time, 2)
            # Since download is skipped, we only confirm generation completed on server
            self.logger.info(f"Scene {scene_id} - SUCCESS. Generation completed on server in {duration}s.")
            
            return {"scene": scene_id, "status": "done", "duration_s": duration} # Note: 'path' is removed as download is skipped
        
        except requests.exceptions.HTTPError as e:
            # Catch HTTP errors specific to the /prompt endpoint
            duration = round(time.time() - start_time, 2)
            error_message = f"HTTPError ({e.response.status_code}) during prompt submission: {str(e)}"
            self.logger.error(f"Scene {scene_id} - FAILED. Total time: {duration}s. Error: {error_message}")
            return {"scene": scene_id, "error": error_message, "status": "failed", "duration_s": duration}
        except Exception as e:
            # Catch all other exceptions (ValueError, RuntimeError, etc.)
            duration = round(time.time() - start_time, 2)
            error_message = f"{e.__class__.__name__}: {str(e)}"
            self.logger.error(f"Scene {scene_id} - FAILED. Total time: {duration}s. Error: {error_message}")
            return {"scene": scene_id, "error": error_message, "status": "failed", "duration_s": duration}

    def run_scenes(self, scenes_json, comfy_api_url, video_output_dir, workflow_path, max_workers = 3, trigger = 0):
        """Main execution function."""
        try:
            scenes = json.loads(scenes_json)
        except json.JSONDecodeError:
            error_msg = "Invalid JSON in 'scenes_json' input. Please check the format."
            self.logger.error(error_msg)
            return (
                json.dumps([{"scene": "N/A", "error": error_msg, "status": "failed"}]),
                video_output_dir
            )

        if not scenes:
            self.logger.info("No scenes provided in JSON input. Returning empty results.")
            return (json.dumps([]), video_output_dir)

        # Pre-execution checks and setup
        try:
            video_output_dir_path = Path(video_output_dir)
            video_output_dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory confirmed: {video_output_dir_path}")
            
            with open(workflow_path, "r") as f:
                workflow_data = json.load(f)
            self.logger.info(f"Workflow template loaded from: {workflow_path}")
            
            # --- MODIFICATION 1: Print loaded workflow JSON ---
            print("\n--- LOADED WORKFLOW JSON START ---")
            print(json.dumps(workflow_data, indent=2))
            print("--- LOADED WORKFLOW JSON END ---\n")
            # ----------------------------------------------------
            
        except FileNotFoundError:
            error_msg = f"Workflow file not found at: {workflow_path}"
            self.logger.error(error_msg)
            return (
                json.dumps([{"scene": "N/A", "error": error_msg, "status": "failed"}]),
                video_output_dir
            )
        except Exception as e:
            error_msg = f"Error preparing node (loading workflow or creating directory): {e}"
            self.logger.error(error_msg)
            return (
                json.dumps([{"scene": "N/A", "error": error_msg, "status": "failed"}]),
                video_output_dir
            )

        self.logger.info(f"\n--- Starting Video Generation for {len(scenes)} Scenes ---")
        self.logger.info(f"ComfyUI URL: {comfy_api_url} | Max Concurrent Workers: {max_workers}")

        results = []
        
        # Concurrently process scenes
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_scene, comfy_api_url, video_output_dir_path, workflow_data, scene): scene 
                for scene in scenes
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    scene_failed = futures[future]
                    error_msg = f"An unexpected exception occurred during execution: {exc}"
                    self.logger.exception(f"Scene {scene_failed.get('scene', 'N/A')} - {error_msg}")
                    results.append({"scene": scene_failed.get('scene', 'N/A'), "error": error_msg, "status": "failed"})

        self.logger.info(f"\n--- All Scenes Processed. Total Results: {len(results)} ---\n")
        
        final_results_json = json.dumps(results)
        
        # --- MODIFICATION 2: Print final results JSON ---
        print("\n--- FINAL RESULTS JSON START ---")
        print(final_results_json)
        print("--- FINAL RESULTS JSON END ---\n")
        # --------------------------------------------------
        
        # Return results as a JSON string
        return (final_results_json, video_output_dir)