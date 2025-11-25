import json
import time
import math
import logging
import requests
from pathlib import Path
import concurrent.futures
import copy # Import copy for safer deep copying

class SceneImage2VideoIterator:
    """
    Base class for ComfyUI scene-based video generation. 
    Handles logging, threading, API communication, and polling.
    
    The derived class MUST implement the abstract method:
    _inject_scene_into_workflow(self, workflow, scene, next_scene, video_output_dir)
    """
    # NOTE: The OUTPUT_NODE_ID should be defined in the derived classes based on the workflow
    # It is used here as a placeholder for the polling logic.
    OUTPUT_NODE_ID = "108" 
    POSITIVE_PROMPT_NODE_ID = "93"
    NEGATIVE_PROMPT_NODE_ID = "89"
    VIDEO_SETTINGS_NODE_ID = "98"
    FPS_SETTING_NODE_ID = "94"
    IMAGE_PROMPT_NODE_ID = "6"
    IMAGE_OUTPUT_NODE_ID = "60"
    IMAGE_PLACEHOLDER_NODE_ID = "121"

    @classmethod
    def INPUT_TYPES(cls):
        return  {
            "required": {
                "scenes_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": "JSON array of scenes with 'scene', 'start', 'end', 'dialogue', and 'video_prompt', 'image_prompt','negative_prompt' keys"
                }),
                "comfy_api_url": ("STRING", {"default": "http://localhost:8188", "tooltip": "Base URL of the ComfyUI API (e.g., http://localhost:8188)"}),
                "video_workflow_path": ("STRING", {"default": "./workflow.json", "tooltip": "Path to the ComfyUI video workflow template JSON file."}),
                "image_workflow_path": ("STRING", {"default": "./workflow.json", "tooltip": "Path to the ComfyUI image workflow template JSON file."}),
                "video_output_dir": ("STRING", {"default": "output/comfy_videos", "tooltip": "Directory to save the final videos."},),
            },
            "optional": {
                "max_workers": ("INT", {"default": 3, "min": 1, "max": 20, "tooltip": "Maximum number of concurrent scenes to process (API calls/downloads)."}),
                "trigger": ("INT", { 
                    "default": 0,
                    "tooltip": "Change this value to re-trigger the node"
                }),
            },
        }

    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("results",)
    FUNCTION = "run_scenes"
    CATEGORY = "Video Generation (Base)"
    OUTPUT_NODE = True 

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Sets up file logging for the node."""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        # Use a generic name for the base class logger
        log_name = f"{self.__class__.__name__}_{int(time.time())}.log" 
        handler = logging.FileHandler(log_path / log_name, mode='a')
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        # Prevent adding multiple stream handlers if the node is re-run
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            # Add a stream handler for console output during execution
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
             logger.addHandler(handler)

        return logger
    
    def _run_image_generation(self, comfy_api_url, scene, image_workflow_data):
        """Generates the initial image keyframe for the scene."""
        scene_id = scene.get("scene", "N/A")
        positive_prompt = scene.get("image_prompt")
        start_time = time.time() 

        self.logger.info(f"Scene {scene_id} - **Image Generation Start** (Node {self.IMAGE_OUTPUT_NODE_ID}).") # New log

        try:
            # Inject prompt into image workflow
            image_wf_copy = copy.deepcopy(image_workflow_data)
            
            if self.IMAGE_PROMPT_NODE_ID in image_wf_copy:
                image_wf_copy[self.IMAGE_PROMPT_NODE_ID]["inputs"]["text"] = positive_prompt
                self.logger.debug(f"Scene {scene_id} - Injected positive prompt for image.")
            else:
                self.logger.warning(f"Scene {scene_id} - Image prompt node {self.IMAGE_PROMPT_NODE_ID} not found.")

            # Queue Prompt
            self.logger.info(f"Scene {scene_id} - Sending image prompt to {comfy_api_url}/prompt...") # New log
            response = requests.post(f"{comfy_api_url}/prompt", json={"prompt": image_wf_copy}, timeout=30)
            response.raise_for_status() 
            response_json = response.json()
            prompt_id = response_json.get("prompt_id")
            
            if not prompt_id:
                raise ValueError(f"ComfyUI did not return a prompt_id for image generation. Response: {response_json}")
            self.logger.info(f"Scene {scene_id} - Image prompt successfully queued. ID: {prompt_id}") # Modified log
            
            result_path = self._poll_for_completion(
                comfy_api_url, prompt_id, scene_id, 
                output_node_id=self.IMAGE_OUTPUT_NODE_ID
            )
            
            duration = round(time.time() - start_time, 2)
            self.logger.info(f"Scene {scene_id} - **Image SUCCESS** in {duration}s. File: {result_path}") # Modified log
            return result_path
        
        except Exception as e:
            error_message = f"Image Generation FAILED: {e.__class__.__name__}: {str(e)}"
            self.logger.error(f"Scene {scene_id} - {error_message}")
            raise RuntimeError(error_message)

    def _inject_scene_into_workflow(self, workflow, scene, next_scene, video_output_dir, image_path):
        """
        ABSTRACT METHOD: MUST BE OVERRIDDEN BY DERIVED CLASSES.
        Deep copies the workflow, injects the scene variables, and returns the modified workflow.
        
        The 'next_scene' dictionary is now available for transition logic.
        """
        """Deep copies the workflow and injects the scenario and sets the filename prefix."""
        positive_prompt = scene.get("video_prompt")
        # negative_prompt = scene.get("negative_prompt") # Keeping this commented as in the original code
        scene_id = scene.get("scene", "N/A")
        start = scene.get("start", 0)
        end = scene.get("end", 0)

        next_scene_start = next_scene.get("start", None) if next_scene else None

        # Deep copy the workflow template for modification
        wf_copy = copy.deepcopy(workflow)
        
        # Positive Prompt Injection 
        if self.POSITIVE_PROMPT_NODE_ID in wf_copy:
            wf_copy[self.POSITIVE_PROMPT_NODE_ID]["inputs"]["text"] = positive_prompt
            self.logger.info(f"Scene {scene_id} - Injected positive prompt into node {self.POSITIVE_PROMPT_NODE_ID}.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node {self.POSITIVE_PROMPT_NODE_ID} not found for scenario injection.")

        # Negative Prompt Injection 
        # if self.NEGATIVE_PROMPT_NODE_ID in wf_copy:
        #     wf_copy[self.NEGATIVE_PROMPT_NODE_ID]["inputs"]["text"] = negative_prompt 
        #     self.logger.info(f"Scene {scene_id} - Injected negative into node {self.NEGATIVE_PROMPT_NODE_ID}.")
        # else:
        #     self.logger.warning(f"Scene {scene_id} - Node {self.NEGATIVE_PROMPT_NODE_ID} not found for scenario injection.")

        # Filename Prefix Setting 
        if self.OUTPUT_NODE_ID in wf_copy:
            prefix = Path(video_output_dir).joinpath(f"scene_{scene_id}").as_posix()
            wf_copy[self.OUTPUT_NODE_ID]["inputs"]["filename_prefix"] = prefix
            self.logger.info(f"Scene {scene_id} - Set filename prefix to '{prefix}' in node {self.OUTPUT_NODE_ID}.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node {self.OUTPUT_NODE_ID} not found for filename prefix setting.")
        
        # Video Length Calculation and Setting
        if self.VIDEO_SETTINGS_NODE_ID in wf_copy and self.FPS_SETTING_NODE_ID in wf_copy:
            try:
                # Get FPS value from a connected node (assuming it's node 57)
                fps_value = wf_copy.get(self.FPS_SETTING_NODE_ID, {}).get("inputs", {}).get("fps")
                fps = float(fps_value) if fps_value is not None else 24.0
            except (ValueError, TypeError):
                fps = 24.0
                self.logger.warning(f"Scene {scene_id} - Node {self.FPS_SETTING_NODE_ID} FPS value is invalid. Defaulting to {fps} FPS.")

            duration = (next_scene_start if next_scene_start is not None else end) - start
            # Calculate total required frames, rounding up to ensure the full duration is covered
            required_length = math.ceil(duration * fps)
            
            # Set the length (frames) into the video generating node (assuming it's node 55)
            wf_copy[self.VIDEO_SETTINGS_NODE_ID]["inputs"]["length"] = int(required_length)
            self.logger.info(f"Scene {scene_id} - Set length to {required_length} frames in node {self.VIDEO_SETTINGS_NODE_ID} (Duration: {duration}s).")
        else:
            self.logger.warning(f"Scene {scene_id} - Node {self.VIDEO_SETTINGS_NODE_ID}/{self.FPS_SETTING_NODE_ID} not found for length setting.")
            
        # Image Path Injection (for video workflow)
        if self.IMAGE_PLACEHOLDER_NODE_ID in wf_copy:
            # Note: The logic here assumes the Image Placeholder node expects the bare filename and 
            # might handle file type/subfolder internally, or the ComfyUI API implicitly handles it.
            # Using the format provided in your original _inject_scene_into_workflow method.
            wf_copy[self.IMAGE_PLACEHOLDER_NODE_ID]["inputs"]["image_path"] = f"output/{image_path}"
            self.logger.info(f"Scene {scene_id} - Injected image path '{image_path}' into node {self.IMAGE_PLACEHOLDER_NODE_ID}.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node {self.IMAGE_PLACEHOLDER_NODE_ID} not found for image path injection.")
        return wf_copy


    def _poll_for_completion(self, comfy_api_url, prompt_id, scene_id, poll_interval = 5, output_node_id = None):
        """Polls the ComfyUI API history for the prompt's completion."""
        target_node_id = output_node_id or self.OUTPUT_NODE_ID
        is_image_poll = output_node_id is not None
        step_name = "Image Generation" if is_image_poll else "Video Generation"

        self.logger.info(f"Scene {scene_id} - Starting poll loop for **{step_name}** prompt ID: {prompt_id} (Target Node: {target_node_id}).") # Modified log
        
        max_retries = 150
        retries = 0

        while True:
            time.sleep(poll_interval)
            try:
                response = requests.get(f"{comfy_api_url}/history/{prompt_id}", timeout=10)
                response.raise_for_status() # Catches HTTPError if status code is bad
                data = response.json()
                self.logger.debug(f"Scene {scene_id} - Polling response data: {data}")
                
                if prompt_id in data:
                    entry = data[prompt_id]
                    
                    # Check if the designated output node ran and produced an image/file path
                    outputs_for_output_node = entry.get("outputs", {}).get(target_node_id, {}).get("images", [])
                    
                    if outputs_for_output_node:
                        # Assuming success if the save node ran and produced a file info
                        output_info = outputs_for_output_node[0]
                        
                        if is_image_poll:
                            # For image generation, the path is often just the filename for later reuse
                            output_path = output_info.get('filename', '')
                            log_path = f"Filename: {output_path}"
                        else:
                            # For video generation, it's the full path including subfolder/type
                            output_path = f"{output_info.get('type')}/{output_info.get('subfolder', '')}/{output_info.get('filename', '')}"
                            log_path = f"Full Path: {output_path}"
                            
                        self.logger.info(f"Scene {scene_id} - **{step_name}** completed. Path: {log_path}") # Modified log
                        return output_path
                    
                    # Check for explicit failure state reported by the API
                    status_str = entry.get("status", {}).get("status_str", "").lower()
                    if status_str == "error":
                         self.logger.error(f"Scene {scene_id} - Polling failed. Status: FAILED (API reported error).")
                         raise RuntimeError(f"Workflow failed on ComfyUI for scene {scene_id}.")

                retries += 1
                if retries >= max_retries:
                    self.logger.error(f"Scene {scene_id} - Polling failed after {max_retries} retries.")
                    raise ConnectionError(f"Polling connection failed after {max_retries} retries.")
                self.logger.warning(f"Scene {scene_id} - {'Image' if is_image_poll else 'Video'} Retries count: {retries}/{max_retries}. Retrying in {poll_interval}s...")

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= max_retries:
                    self.logger.error(f"Scene {scene_id} - Polling failed after {max_retries} retries.")
                    raise ConnectionError(f"Polling connection failed after {max_retries} retries: {e}")
                self.logger.warning(f"Scene {scene_id} - Polling connection error ({e}). Retrying in {poll_interval}s...")


    def _run_scene(self, comfy_api_url, video_output_dir, video_workflow_data, image_workflow_data, scene, next_scene):
        """
        Submits, polls, and tracks the video generation for a single scene.
        The `next_scene` dictionary is passed for transition logic.
        """
        scene_id = scene.get("scene", "N/A")
        scenario = scene.get("video_prompt", "No scenario provided") # Use positive prompt for preview
        start_time = time.time() 
        
        self.logger.info(f"\n--- Scene {scene_id} START ---")
        self.logger.info(f"Scenario Preview: {scenario[:80]}...")
        
        try:
            # 1. RUN IMAGE GENERATION
            image_path = self._run_image_generation(comfy_api_url, scene, image_workflow_data)
            
            # 2. Inject Prompt and Image Path (Step 1/3 of Video Process)
            self.logger.info(f"Scene {scene_id} (Video Step 1/3): Injecting prompt into workflow.")
            workflow = self._inject_scene_into_workflow(video_workflow_data, scene, next_scene, video_output_dir, image_path=image_path)
            
            # 3. Queue Video Prompt (Step 2/3 of Video Process)
            self.logger.info(f"Scene {scene_id} (Video Step 2/3): Sending video prompt to ComfyUI API: {comfy_api_url}/prompt")
            response = requests.post(f"{comfy_api_url}/prompt", json={"prompt": workflow}, timeout=30)
            response.raise_for_status() 
            
            response_json = response.json()
            prompt_id = response_json.get("prompt_id")
            
            if not prompt_id:
                raise ValueError(f"ComfyUI did not return a prompt_id. Response: {response_json}")
            self.logger.info(f"Scene {scene_id} - Video prompt successfully queued. Prompt ID: {prompt_id}")
            
            # 4. Poll for Video Result (Step 3/3 of Video Process)
            self.logger.info(f"Scene {scene_id} (Video Step 3/3): Polling for completion.")
            result_path = self._poll_for_completion(comfy_api_url, prompt_id, scene_id)
            
            # 5. Success
            duration = round(time.time() - start_time, 2)
            self.logger.info(f"Scene {scene_id} - **Video SUCCESS**. Total Time: {duration}s.")
            
            return {"scene": scene_id, "video_path": result_path, "status": "done"}
        
        except requests.exceptions.RequestException as e:
            duration = round(time.time() - start_time, 2)
            error_message = f"RequestError ({e.__class__.__name__}): {str(e)}"
            self.logger.error(f"Scene {scene_id} - FAILED. Total time: {duration}s. Error: {error_message}")
            return {"scene": scene_id, "error": error_message, "status": "failed"}
        except Exception as e:
            duration = round(time.time() - start_time, 2)
            error_message = f"{e.__class__.__name__}: {str(e)}"
            self.logger.exception(f"Scene {scene_id} - FAILED. Total time: {duration}s. Error: {error_message}")
            return {"scene": scene_id, "error": error_message, "status": "failed"}

    def run_scenes(self, scenes_json, comfy_api_url, video_output_dir, video_workflow_path, image_workflow_path, max_workers = 3, trigger = 0):
        """
        Main execution function, running scenes concurrently.
        Now correctly determines and passes the `next_scene` object.
        """
        total_start_time = time.time()
        print(f"{'='*30}Scenes START{'='*30}")
        print(scenes_json)
        print(f"{'='*30}Scenes END{'='*30}")
        try:
            scenes = json.loads(scenes_json)
        except json.JSONDecodeError:
            error_msg = "Invalid JSON in 'scenes_json' input. Please check the format."
            self.logger.error(error_msg)
            return (json.dumps([{"scene": "N/A", "error": error_msg, "status": "failed"}]),)

        if not scenes:
            self.logger.info("No scenes provided in JSON input. Returning empty results.")
            return (json.dumps([]),)

        # --- MANDATORY VALIDATION CHECK ---
        for i, scene in enumerate(scenes):
            scene_id = scene.get("scene", f"Index {i+1}")
            if "image_prompt" not in scene or not scene["image_prompt"]:
                error_msg = f"Scene {scene_id} is missing the required key: 'image_prompt' or its value is empty. Cannot proceed."
                self.logger.error(error_msg)
                return (json.dumps([{"scene": scene_id, "error": error_msg, "status": "failed"}]),)
            

            if "video_prompt" not in scene or not scene["video_prompt"]:
                error_msg = f"Scene {scene_id} is missing the required key: 'video_prompt' or its value is empty. Cannot proceed."
                self.logger.error(error_msg)
                return (json.dumps([{"scene": scene_id, "error": error_msg, "status": "failed"}]),)
            
            if "negative_prompt" not in scene or not scene["negative_prompt"]:
                error_msg = f"Scene {scene_id} is missing the required key: 'negative_prompt' or its value is empty. Cannot proceed."
                self.logger.error(error_msg)
                return (json.dumps([{"scene": scene_id, "error": error_msg, "status": "failed"}]),)
        # --- END VALIDATION CHECK ---
        
        total_duration = scenes[-1].get("end", 0) 
        total_fps = float(total_duration * 24.0)

        self.logger.info(f"\n--- Total Duration: {total_duration} | Total FPS: {total_fps} ---")
        # Pre-execution checks and setup
        try:
            video_output_dir_path = Path(video_output_dir)
            video_output_dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory confirmed: {video_output_dir_path}")
            
            with open(video_workflow_path, "r") as f:
                video_workflow_data = json.load(f)
            self.logger.info(f"Video workflow template loaded from: {video_workflow_path}")

            with open(image_workflow_path, "r") as f:
                image_workflow_data = json.load(f)
            self.logger.info(f"Image workflow template loaded from: {image_workflow_path}")
            
        except Exception as e:
            error_msg = f"Error preparing node (loading workflow or creating directory): {e.__class__.__name__}: {str(e)}"
            self.logger.error(error_msg)
            return (json.dumps([{"scene": "N/A", "error": error_msg, "status": "failed"}]),)

        self.logger.info(f"\n--- Starting Video Generation for {len(scenes)} Scenes ---")
        self.logger.info(f"ComfyUI URL: {comfy_api_url} | Max Concurrent Workers: {max_workers}")

        results = []
        
        # Concurrently process scenes
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, scene in enumerate(scenes):
                # Determine the next scene object for transition logic
                next_scene = scenes[i+1] if i + 1 < len(scenes) else None
                
                future = executor.submit(
                    self._run_scene, 
                    comfy_api_url, 
                    video_output_dir_path, 
                    video_workflow_data, 
                    image_workflow_data,
                    scene, 
                    next_scene # Pass the next scene object
                )
                futures[future] = scene # Map future back to the current scene for logging
            
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
        
        total_end_time = time.time()
        total_duration = round(total_end_time - total_start_time, 2)
        self.logger.info(f"⏰ Total Execution Time for All {len(scenes)} Scenarios: {total_duration} seconds.")

        # Return results as a JSON string
        return (final_results_json,)