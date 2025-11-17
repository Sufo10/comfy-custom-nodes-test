import json
import time
import math
import logging
import requests
from pathlib import Path
import concurrent.futures
import copy # Import copy for safer deep copying

class BaseSceneIteratorNode:
    """
    Base class for ComfyUI scene-based video generation. 
    Handles logging, threading, API communication, and polling.
    
    The derived class MUST implement the abstract method:
    _inject_scene_into_workflow(self, workflow, scene, video_output_dir)
    """
    # NOTE: The OUTPUT_NODE_ID should be defined in the derived classes based on the workflow
    # It is used here as a placeholder for the polling logic.
    OUTPUT_NODE_ID = "58" 

    @classmethod
    def INPUT_TYPES(cls):
        return  {
            "required": {
                "scenes_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": "JSON array of scenes with 'scene', 'start', 'end', 'dialogue', and 'positive_prompt'/'negative_prompt' keys"
                }),
                "comfy_api_url": ("STRING", {"default": "http://localhost:8188", "tooltip": "Base URL of the ComfyUI API (e.g., http://localhost:8188)"}),
                "workflow_path": ("STRING", {"default": "./workflow.json", "tooltip": "Path to the ComfyUI workflow template JSON file."}),
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

    def _inject_scene_into_workflow(self, workflow, scene, video_output_dir):
        """
        ABSTRACT METHOD: MUST BE OVERRIDDEN BY DERIVED CLASSES.
        Deep copies the workflow, injects the scene variables, and returns the modified workflow.
        """
        raise NotImplementedError("Derived class must implement _inject_scene_into_workflow()")


    def _poll_for_completion(self, comfy_api_url, prompt_id, scene_id, poll_interval = 5):
        """Polls the ComfyUI API history for the prompt's completion."""
        self.logger.info(f"Scene {scene_id} - Starting poll loop for prompt ID: {prompt_id}.")
        
        max_retries = 100
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
                    outputs_for_output_node = entry.get("outputs", {}).get(self.OUTPUT_NODE_ID, {}).get("images", [])
                    
                    if outputs_for_output_node:
                        # Assuming success if the save node ran and produced a file info
                        output_info = outputs_for_output_node[0]
                        # ComfyUI file structure includes type (e.g., 'output')
                        output_path = f"{output_info.get('type')}/{output_info.get('subfolder', '')}/{output_info.get('filename', '')}"
                        self.logger.info(f"Scene {scene_id} - Polling successful. Status: COMPLETED. File: {output_path}")
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
                self.logger.warning(f"Scene {scene_id} - Retries count: {retries}/{max_retries}. Retrying in {poll_interval}s...")

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= max_retries:
                    self.logger.error(f"Scene {scene_id} - Polling failed after {max_retries} retries.")
                    raise ConnectionError(f"Polling connection failed after {max_retries} retries: {e}")
                self.logger.warning(f"Scene {scene_id} - Polling connection error ({e}). Retrying in {poll_interval}s...")


    def _run_scene(self, comfy_api_url, video_output_dir, workflow_data, scene):
        """Submits, polls, and tracks the video generation for a single scene."""
        scene_id = scene.get("scene", "N/A")
        scenario = scene.get("positive_prompt", "No scenario provided") # Use positive prompt for preview
        start_time = time.time() 
        
        self.logger.info(f"\n--- Scene {scene_id} START ---")
        self.logger.info(f"Scenario Preview: {scenario[:80]}...")
        
        try:
            # 1. Inject Prompt (Calls the derived class's implementation)
            self.logger.info(f"Scene {scene_id} (Step 1/3): Injecting prompt into workflow.")
            workflow = self._inject_scene_into_workflow(workflow_data, scene, video_output_dir)
            
            # 2. Queue Prompt
            self.logger.info(f"Scene {scene_id} (Step 2/3): Sending prompt to ComfyUI API: {comfy_api_url}/prompt")
            response = requests.post(f"{comfy_api_url}/prompt", json={"prompt": workflow}, timeout=30)
            response.raise_for_status() 
            
            response_json = response.json()
            prompt_id = response_json.get("prompt_id")
            
            if not prompt_id:
                raise ValueError(f"ComfyUI did not return a prompt_id. Response: {response_json}")
            self.logger.info(f"Scene {scene_id} - Prompt successfully queued. Prompt ID: {prompt_id}")
            
            # 3. Poll for Result
            self.logger.info(f"Scene {scene_id} (Step 3/3): Polling for completion.")
            # Result is the output file path on the server
            result_path = self._poll_for_completion(comfy_api_url, prompt_id, scene_id)
            
            # 4. Success
            duration = round(time.time() - start_time, 2)
            self.logger.info(f"Scene {scene_id} - SUCCESS. Generation completed on server in {duration}s.")
            
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

    def run_scenes(self, scenes_json, comfy_api_url, video_output_dir, workflow_path, max_workers = 3, trigger = 0):
        """Main execution function, running scenes concurrently."""
        total_start_time = time.time()
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
            
            if "positive_prompt" not in scene or not scene["positive_prompt"]:
                error_msg = f"Scene {scene_id} is missing the required key: 'positive_prompt' or its value is empty. Cannot proceed."
                self.logger.error(error_msg)
                return (json.dumps([{"scene": scene_id, "error": error_msg, "status": "failed"}]),)
            
            if "negative_prompt" not in scene or not scene["negative_prompt"]:
                error_msg = f"Scene {scene_id} is missing the required key: 'negative_prompt' or its value is empty. Cannot proceed."
                self.logger.error(error_msg)
                return (json.dumps([{"scene": scene_id, "error": error_msg, "status": "failed"}]),)
        # --- END VALIDATION CHECK ---
        
        # Pre-execution checks and setup
        try:
            video_output_dir_path = Path(video_output_dir)
            video_output_dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory confirmed: {video_output_dir_path}")
            
            with open(workflow_path, "r") as f:
                workflow_data = json.load(f)
            self.logger.info(f"Workflow template loaded from: {workflow_path}")
            
        except Exception as e:
            error_msg = f"Error preparing node (loading workflow or creating directory): {e.__class__.__name__}: {str(e)}"
            self.logger.error(error_msg)
            return (json.dumps([{"scene": "N/A", "error": error_msg, "status": "failed"}]),)

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
        
        total_end_time = time.time()
        total_duration = round(total_end_time - total_start_time, 2)
        self.logger.info(f"⏰ Total Execution Time for All {len(scenes)} Scenarios: {total_duration} seconds.")

        # Return results as a JSON string
        return (final_results_json,)
    

class SceneVideoWan5BIteratorNode(BaseSceneIteratorNode):
    """
    Concrete implementation extending BaseSceneIteratorNode.
    Specializes in injecting scene data into a specific workflow structure.
    
    Assumes: 
    - Node 6: Positive Prompt (Text node)
    - Node 7: Negative Prompt (Text node)
    - Node 58: Save Image/Video Node (Filename Prefix)
    - Node 55: Video Generation Node (Length/Frames)
    - Node 57: FPS setting node (used for calculation)
    """

    OUTPUT_NODE_ID = "58"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        return inputs

    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("results",)
    FUNCTION = "run_scenes"
    CATEGORY = "Custom Video Generation" 
    OUTPUT_NODE = True 

    def _inject_scene_into_workflow(self, workflow, scene, video_output_dir):
        """Deep copies the workflow and injects the scenario and sets the filename prefix."""
        positive_prompt = scene.get("positive_prompt")
        negative_prompt = scene.get("negative_prompt")
        scene_id = scene.get("scene", "N/A")
        start = scene.get("start", 0)
        end = scene.get("end", 0)

        # Deep copy the workflow template for modification
        wf_copy = copy.deepcopy(workflow)
        
        # Positive Prompt Injection (Node 6)
        if "6" in wf_copy:
            wf_copy["6"]["inputs"]["text"] = positive_prompt
            self.logger.info(f"Scene {scene_id} - Injected scenario into node 6.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node 6 not found for scenario injection.")

        # Negative Prompt Injection (Node 7)
        if "7" in wf_copy:
            wf_copy["7"]["inputs"]["text"] = negative_prompt
            self.logger.info(f"Scene {scene_id} - Injected negative into node 7.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node 7 not found for scenario injection.")

        # Filename Prefix Setting (Node 58)
        if self.OUTPUT_NODE_ID in wf_copy:
            prefix = Path(video_output_dir).joinpath(f"scene_{scene_id}").as_posix()
            wf_copy[self.OUTPUT_NODE_ID]["inputs"]["filename_prefix"] = prefix
            self.logger.info(f"Scene {scene_id} - Set filename prefix to '{prefix}' in node {self.OUTPUT_NODE_ID}.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node {self.OUTPUT_NODE_ID} not found for filename prefix setting.")
        
        # Video Length Calculation and Setting
        if "55" in wf_copy and '57' in wf_copy:
            try:
                # Get FPS value from a connected node (assuming it's node 57)
                fps_value = wf_copy.get("57", {}).get("inputs", {}).get("fps")
                fps = float(fps_value) if fps_value is not None else 24.0
            except (ValueError, TypeError):
                fps = 24.0
                self.logger.warning(f"Scene {scene_id} - Node 57 FPS value is invalid. Defaulting to {fps} FPS.")
            
            duration = end - start
            # Calculate total required frames, rounding up to ensure the full duration is covered
            required_length = math.ceil(duration * fps)
            
            # Set the length (frames) into the video generating node (assuming it's node 55)
            wf_copy["55"]["inputs"]["length"] = int(required_length)
            self.logger.info(f"Scene {scene_id} - Set length to {required_length} frames in node 55 (Duration: {duration}s).")
        else:
            self.logger.warning(f"Scene {scene_id} - Node 55/57 not found for length setting.")
            
        return wf_copy
    

class SceneVideoWan14BIteratorNode(BaseSceneIteratorNode):
    """
    Concrete implementation extending BaseSceneIteratorNode.
    Specializes in injecting scene data into a specific workflow structure.
    
    Assumes: 
    - Node 89: Positive Prompt (Text node)
    - Node 72: Negative Prompt (Text node)
    - Node 80: Save Image/Video Node (Filename Prefix)
    - Node 74: Video Generation Node (Length/Frames)
    - Node 88: FPS setting node (used for calculation)
    """

    OUTPUT_NODE_ID = "80"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        return inputs

    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("results",)
    FUNCTION = "run_scenes"
    CATEGORY = "Custom Video Generation" 
    OUTPUT_NODE = True 

    def _inject_scene_into_workflow(self, workflow, scene, video_output_dir):
        """Deep copies the workflow and injects the scenario and sets the filename prefix."""
        positive_prompt = scene.get("positive_prompt")
        negative_prompt = scene.get("negative_prompt")
        scene_id = scene.get("scene", "N/A")
        start = scene.get("start", 0)
        end = scene.get("end", 0)

        # Deep copy the workflow template for modification
        wf_copy = copy.deepcopy(workflow) 
        
        # Positive Prompt Injection (Node 89)
        if "89" in wf_copy:
            wf_copy["89"]["inputs"]["text"] = positive_prompt
            self.logger.info(f"Scene {scene_id} - Injected scenario into node 89.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node 89 not found for scenario injection.")

        # Negative Prompt Injection (Node 72)
        if "72" in wf_copy:
            wf_copy["72"]["inputs"]["text"] = negative_prompt
            self.logger.info(f"Scene {scene_id} - Injected negative into node 72.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node 72 not found for scenario injection.")

        # Filename Prefix Setting (Node 80)
        if self.OUTPUT_NODE_ID in wf_copy:
            prefix = Path(video_output_dir).joinpath(f"scene_{scene_id}").as_posix()
            wf_copy[self.OUTPUT_NODE_ID]["inputs"]["filename_prefix"] = prefix
            self.logger.info(f"Scene {scene_id} - Set filename prefix to '{prefix}' in node {self.OUTPUT_NODE_ID}.")
        else:
            self.logger.warning(f"Scene {scene_id} - Node {self.OUTPUT_NODE_ID} not found for filename prefix setting.")
        
        # Video Length Calculation and Setting
        if "74" in wf_copy and '88' in wf_copy:
            try:
                # Get FPS value from a connected node (assuming it's node 88)
                fps_value = wf_copy.get("88", {}).get("inputs", {}).get("fps")
                fps = float(fps_value) if fps_value is not None else 24.0
            except (ValueError, TypeError):
                fps = 24.0
                self.logger.warning(f"Scene {scene_id} - Node 88 FPS value is invalid. Defaulting to {fps} FPS.")
            
            duration = end - start
            # Calculate total required frames, rounding up to ensure the full duration is covered
            required_length = math.ceil(duration * fps) # Use ceil instead of floor for full duration
            
            # Set the length (frames) into the video generating node (assuming it's node 74)
            wf_copy["74"]["inputs"]["length"] = int(required_length)
            self.logger.info(f"Scene {scene_id} - Set length to {required_length} frames in node 74 (Duration: {duration}s).")
        else:
            self.logger.warning(f"Scene {scene_id} - Node 74/88 not found for length setting.")
            
        return wf_copy