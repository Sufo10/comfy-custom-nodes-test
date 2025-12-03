import json
import time
import math
import logging
import requests
from pathlib import Path
import concurrent.futures
import copy

class SceneImage2VideoFirstLastIterator:
    """
    Base class for ComfyUI scene-based video generation. 
    Handles logging, threading, API communication, and polling.
    """
    
    # --- Node ID Constants ---
    OUTPUT_NODE_ID = "14"                     
    POSITIVE_PROMPT_NODE_ID = "20"            
    NEGATIVE_PROMPT_NODE_ID = "7"             
    VIDEO_SETTINGS_NODE_ID = "22"             
    FPS_SETTING_NODE_ID = "12"                
    
    FIRST_IMAGE_PROMPT_NODE_ID = "6"          
    FIRST_IMAGE_SAVE_NODE_ID = "60"           
    
    LAST_IMAGE_PROMPT_NODE_ID = "16"          
    LAST_IMAGE_SAVE_NODE_ID = "10"            
    LAST_IMAGE_PLACEHOLDER_NODE_ID = "18"     
    
    VIDEO_FIRST_IMAGE_PLACEHOLDER_NODE_ID = "23" 
    VIDEO_LAST_IMAGE_PLACEHOLDER_NODE_ID = "24"  

    @classmethod
    def INPUT_TYPES(cls):
        return  {
            "required": {
                "scenes_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": "JSON array of scenes with 'scene', 'start', 'end', 'dialogue', 'first_frame_prompt', 'last_frame_prompt', 'video_prompt', and 'negative_prompt' keys"
                }),
                "comfy_api_url": ("STRING", {"default": "http://localhost:8188", "tooltip": "Base URL of the ComfyUI API (e.g., http://localhost:8188)"}),
                "video_workflow_path": ("STRING", {"default": "./workflow.json", "tooltip": "Path to the ComfyUI video workflow template JSON file."}),
                "first_image_workflow_path": ("STRING", {"default": "./workflow.json", "tooltip": "Path to the ComfyUI image workflow template JSON file for the first keyframe."}),
                "last_image_workflow_path": ("STRING", {"default": "./workflow.json", "tooltip": "Path to the ComfyUI image workflow template JSON file for the last keyframe."}),
                "video_output_dir": ("STRING", {"default": "output/comfy_videos", "tooltip": "Directory to save the final videos."},),
            },
            "optional": {
                "max_workers": ("INT", {"default": 3, "min": 1, "max": 20, "tooltip": "Maximum number of concurrent scenes to process (API calls/downloads)."}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 60, "tooltip": "Interval in seconds to poll the ComfyUI API for completion."}),
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
        """Sets up file logging for the node, ensuring unique file handlers."""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        log_name = f"{self.__class__.__name__}_{int(time.time())}.log" 
        log_file = log_path / log_name
        
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [TID:%(thread)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        if logger.handlers:
            logger.handlers = []

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
    
    def _run_image_generation(self, comfy_api_url, scene, image_workflow_data, is_first_image, first_image_path=None, poll_interval=5):
        """Generates either the first or last image keyframe for the scene."""
        scene_id = scene.get("scene", "N/A")
        
        if is_first_image:
            image_type = "FIRST"
            prompt_node_id = self.FIRST_IMAGE_PROMPT_NODE_ID
            output_node_id = self.FIRST_IMAGE_SAVE_NODE_ID
            image_placeholder_node_id = None
            # --- UPDATED KEY ---
            positive_prompt = scene.get("first_frame_prompt") or scene.get("positive_prompt")
        else:
            image_type = "LAST"
            prompt_node_id = self.LAST_IMAGE_PROMPT_NODE_ID
            output_node_id = self.LAST_IMAGE_SAVE_NODE_ID
            image_placeholder_node_id = self.LAST_IMAGE_PLACEHOLDER_NODE_ID
            # --- UPDATED KEY ---
            positive_prompt = scene.get("last_frame_prompt") or scene.get("positive_prompt")

        start_time = time.time() 

        step_name = f"{image_type} Image Generation"
        self.logger.info(f"[Scene {scene_id}] **{step_name} Start** (Output Node: {output_node_id}). Prompt: {positive_prompt[:50]}...")

        try:
            image_wf_copy = copy.deepcopy(image_workflow_data)
            
            # 1. Inject Positive Prompt
            if prompt_node_id in image_wf_copy:
                if not positive_prompt:
                     raise ValueError(f"Prompt for {image_type} image is missing or empty.")
                image_wf_copy[prompt_node_id]["inputs"]["text"] = positive_prompt
                self.logger.debug(f"[Scene {scene_id}] Injected positive prompt for {image_type} image into node {prompt_node_id}.")
            else:
                self.logger.warning(f"[Scene {scene_id}] Prompt node {prompt_node_id} not found in {image_type} image workflow.")

            # 2. Inject Previous Image Path (Only for LAST image generation)
            if not is_first_image and image_placeholder_node_id and first_image_path:
                full_path_for_comfy = f"output/{first_image_path}" 
                if image_placeholder_node_id in image_wf_copy:
                    image_wf_copy[image_placeholder_node_id]["inputs"]["image_path"] = full_path_for_comfy
                    self.logger.debug(f"[Scene {scene_id}] Injected previous image path '{full_path_for_comfy}' into node {image_placeholder_node_id}.")
                else:
                    self.logger.warning(f"[Scene {scene_id}] Image placeholder node {image_placeholder_node_id} not found for path injection.")

            # 3. Queue Prompt
            self.logger.info(f"[Scene {scene_id}] Sending {image_type} image prompt to {comfy_api_url}/prompt...")
            response = requests.post(f"{comfy_api_url}/prompt", json={"prompt": image_wf_copy}, timeout=30)
            response.raise_for_status() 
            response_json = response.json()
            prompt_id = response_json.get("prompt_id")
            
            if not prompt_id:
                raise ValueError(f"ComfyUI did not return a prompt_id for {image_type} image generation. Response: {response_json}")
            self.logger.info(f"[Scene {scene_id}] {image_type} image prompt successfully queued. Prompt ID: {prompt_id}")
            
            # 4. Poll for Completion
            result_path = self._poll_for_completion(
                comfy_api_url, prompt_id, scene_id, 
                poll_interval=poll_interval,
                output_node_id=output_node_id,
                step_name=step_name 
            )
            
            duration = round(time.time() - start_time, 2)
            self.logger.info(f"[Scene {scene_id}] **{step_name} SUCCESS** in {duration}s. File Path: {result_path}")
            return result_path
        
        except Exception as e:
            error_message = f"{step_name} FAILED: {e.__class__.__name__}: {str(e)}"
            self.logger.exception(f"[Scene {scene_id}] {error_message}")
            raise RuntimeError(error_message)

    def _run_first_image_generation(self, comfy_api_url, scene, image_workflow_data, poll_interval=5):
        return self._run_image_generation(comfy_api_url, scene, image_workflow_data, is_first_image=True, poll_interval=poll_interval)

    def _run_last_image_generation(self, comfy_api_url, scene, image_workflow_data, first_image_path, poll_interval=5):
        return self._run_image_generation(comfy_api_url, scene, image_workflow_data, is_first_image=False, first_image_path=first_image_path, poll_interval=poll_interval)


    def _inject_scene_into_workflow(self, workflow, scene, next_scene, video_output_dir, first_image_path, last_image_path):
        """
        Injects all scene-specific parameters (prompts, length, image paths, output prefix) 
        into a deep copy of the video workflow template.
        """
        positive_prompt = scene.get("video_prompt") or scene.get("positive_prompt")
        negative_prompt = scene.get("negative_prompt") 
        scene_id = scene.get("scene", "N/A")
        start = scene.get("start", 0)
        end = scene.get("end", 0)
        next_scene_start = next_scene.get("start", None) if next_scene else None

        wf_copy = copy.deepcopy(workflow)
        
        # 1. Prompt Injection 
        if self.POSITIVE_PROMPT_NODE_ID in wf_copy:
            wf_copy[self.POSITIVE_PROMPT_NODE_ID]["inputs"]["text"] = positive_prompt
            self.logger.debug(f"[Scene {scene_id}] Injected POSITIVE prompt into node {self.POSITIVE_PROMPT_NODE_ID}.")
        
        if self.NEGATIVE_PROMPT_NODE_ID in wf_copy:
            wf_copy[self.NEGATIVE_PROMPT_NODE_ID]["inputs"]["text"] = negative_prompt 
            self.logger.debug(f"[Scene {scene_id}] Injected NEGATIVE prompt into node {self.NEGATIVE_PROMPT_NODE_ID}.")

        # 2. Filename Prefix Setting 
        if self.OUTPUT_NODE_ID in wf_copy:
            prefix = Path(video_output_dir).joinpath(f"scene_{scene_id}").as_posix()
            wf_copy[self.OUTPUT_NODE_ID]["inputs"]["filename_prefix"] = prefix
            self.logger.debug(f"[Scene {scene_id}] Set output prefix to '{prefix}' in node {self.OUTPUT_NODE_ID}.")
        
        # 3. Video Length Calculation and Setting
        if self.VIDEO_SETTINGS_NODE_ID in wf_copy and self.FPS_SETTING_NODE_ID in wf_copy:
            try:
                fps_value = wf_copy.get(self.FPS_SETTING_NODE_ID, {}).get("inputs", {}).get("fps")
                fps = float(fps_value) if fps_value is not None else 24.0
            except (ValueError, TypeError):
                fps = 24.0
                self.logger.warning(f"[Scene {scene_id}] Invalid FPS value in node {self.FPS_SETTING_NODE_ID}. Defaulting to {fps} FPS.")

            duration = (next_scene_start if next_scene_start is not None else end) - start
            required_length = math.ceil(duration * fps)
            
            wf_copy[self.VIDEO_SETTINGS_NODE_ID]["inputs"]["length"] = int(required_length)
            self.logger.info(f"[Scene {scene_id}] Calculated duration: {duration}s @ {fps} FPS. Set frame length to {required_length} in node {self.VIDEO_SETTINGS_NODE_ID}.")
        
        # 4. Image Path Injection (for video workflow)
        full_first_image_path = f"output/{first_image_path}"
        full_last_image_path = f"output/{last_image_path}"

        if self.VIDEO_FIRST_IMAGE_PLACEHOLDER_NODE_ID in wf_copy:
            wf_copy[self.VIDEO_FIRST_IMAGE_PLACEHOLDER_NODE_ID]["inputs"]["image_path"] = full_first_image_path
            self.logger.debug(f"[Scene {scene_id}] Injected FIRST image path '{full_first_image_path}' into node {self.VIDEO_FIRST_IMAGE_PLACEHOLDER_NODE_ID}.")

        if self.VIDEO_LAST_IMAGE_PLACEHOLDER_NODE_ID in wf_copy:
            wf_copy[self.VIDEO_LAST_IMAGE_PLACEHOLDER_NODE_ID]["inputs"]["image_path"] = full_last_image_path
            self.logger.debug(f"[Scene {scene_id}] Injected LAST image path '{full_last_image_path}' into node {self.VIDEO_LAST_IMAGE_PLACEHOLDER_NODE_ID}.")
        
        return wf_copy


    def _poll_for_completion(self, comfy_api_url, prompt_id, scene_id, poll_interval, output_node_id=None, step_name="Video Generation"):
        """Polls the ComfyUI API history for the prompt's completion."""
        target_node_id = output_node_id or self.OUTPUT_NODE_ID

        self.logger.info(f"[Scene {scene_id}] Starting poll loop for **{step_name}** (Prompt ID: {prompt_id}, Target Node: {target_node_id}).")
        
        max_retries = 150
        retries = 0

        while retries < max_retries:
            time.sleep(poll_interval)
            try:
                response = requests.get(f"{comfy_api_url}/history/{prompt_id}", timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if prompt_id in data:
                    entry = data[prompt_id]
                    
                    # 1. Check for success (output node has run)
                    outputs_for_output_node = entry.get("outputs", {}).get(target_node_id, {}).get("images", [])
                    
                    if outputs_for_output_node:
                        output_info = outputs_for_output_node[0]
                        
                        output_path = f"{output_info.get('type')}/{output_info.get('subfolder', '')}/{output_info.get('filename', '')}".strip('/')
                            
                        self.logger.info(f"[Scene {scene_id}] Polling successful. **{step_name}** completed. Result Path: {output_path}")
                        return output_path
                    
                    # 2. Check for explicit failure state reported by the API
                    if entry.get("status", {}).get("status_str", "").lower() == "error":
                         self.logger.error(f"[Scene {scene_id}] Polling failed. Status: FAILED (API reported error in workflow for {step_name}).")
                         raise RuntimeError(f"Workflow failed on ComfyUI for scene {scene_id}.")

                retries += 1
                self.logger.info(f"[Scene {scene_id}] {step_name} Retries count: {retries}/{max_retries}. Polling in {poll_interval}s...")

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= max_retries:
                    self.logger.error(f"[Scene {scene_id}] Polling failed after {max_retries} retries due to connection error.")
                    raise ConnectionError(f"Polling connection failed after {max_retries} retries: {e}")
                self.logger.warning(f"[Scene {scene_id}] Polling connection error ({e.__class__.__name__}). Retrying in {poll_interval}s...")
        
        self.logger.error(f"[Scene {scene_id}] Polling timed out after {max_retries} retries.")
        raise TimeoutError(f"Polling timed out for scene {scene_id} after {max_retries} retries.")


    def _run_scene(self, comfy_api_url, video_output_dir, video_workflow_data, first_image_workflow_data, last_image_workflow_data, scene, next_scene, poll_interval):
        """
        Submits, polls, and tracks the video generation for a single scene, 
        including keyframe image generation.
        """
        scene_id = scene.get("scene", "N/A")
        start_time = time.time() 
        
        self.logger.info(f"\n{'='*10} [Scene {scene_id}] START Processing {'='*10}")
        self.logger.info(f"[Scene {scene_id}] Scenario Preview: {(scene.get('video_prompt') or scene.get('positive_prompt'))[:80]}...")
        
        try:
            # 1. RUN FIRST IMAGE GENERATION
            first_image_path = self._run_first_image_generation(comfy_api_url, scene, first_image_workflow_data, poll_interval)
            
            # 2. RUN LAST IMAGE GENERATION
            last_image_path = self._run_last_image_generation(comfy_api_url, scene, last_image_workflow_data, first_image_path, poll_interval)

            # 3. Inject Parameters into Video Workflow
            self.logger.info(f"[Scene {scene_id}] Injecting parameters into video workflow (first image: {first_image_path}, last image: {last_image_path}).")
            video_workflow = self._inject_scene_into_workflow(
                video_workflow_data, scene, next_scene, 
                video_output_dir, first_image_path, last_image_path
            )
            
            # 4. Queue Video Prompt
            self.logger.info(f"[Scene {scene_id}] Sending video prompt to ComfyUI API.")
            response = requests.post(f"{comfy_api_url}/prompt", json={"prompt": video_workflow}, timeout=30)
            response.raise_for_status() 
            
            response_json = response.json()
            prompt_id = response_json.get("prompt_id")
            
            if not prompt_id:
                raise ValueError(f"ComfyUI did not return a prompt_id for video. Response: {response_json}")
            self.logger.info(f"[Scene {scene_id}] Video prompt successfully queued. Prompt ID: {prompt_id}")
            
            # 5. Poll for Video Result
            result_path = self._poll_for_completion(comfy_api_url, prompt_id, scene_id, poll_interval)
            
            # 6. Success
            duration = round(time.time() - start_time, 2)
            self.logger.info(f"{'='*10} [Scene {scene_id}] **Video SUCCESS**. Total Time: {duration}s. {'='*10}\n")
            
            return {"scene": scene_id, "video_path": result_path, "status": "done"}
        
        except Exception as e:
            duration = round(time.time() - start_time, 2)
            error_message = f"Execution FAILED after {duration}s. Error: {e.__class__.__name__}: {str(e)}"
            self.logger.exception(f"[Scene {scene_id}] {error_message}")
            return {"scene": scene_id, "error": error_message, "status": "failed"}

    def run_scenes(self, scenes_json, comfy_api_url, video_output_dir, video_workflow_path, first_image_workflow_path, last_image_workflow_path, max_workers=3, poll_interval=5, trigger=0):
        """
        Main execution function, running scenes concurrently.
        """
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

        self.logger.info(f"\n{'='*50}\n--- Starting Workflow Setup for {len(scenes)} Scenes ---\n{'='*50}")
        
        # --- 1. MANDATORY VALIDATION CHECK (Updated for new keys) ---
        for i, scene in enumerate(scenes):
            scene_id = scene.get("scene", f"Index {i+1}")
            
            # Option A: Check for the new explicit frame prompts (first_frame_prompt AND last_frame_prompt)
            has_explicit_frame_prompts = (
                "first_frame_prompt" in scene and scene["first_frame_prompt"] and
                "last_frame_prompt" in scene and scene["last_frame_prompt"]
            )
            # Option B: Check for the legacy fallback (positive_prompt)
            has_positive_prompt_fallback = ("positive_prompt" in scene and scene["positive_prompt"])

            # Video Prompt is always required if we are generating a video (this logic remains the same)
            has_video_prompt = ("video_prompt" in scene and scene["video_prompt"])
            
            # Negative prompt is still required
            has_negative_prompt = ("negative_prompt" in scene and scene["negative_prompt"])
    
            if not (has_explicit_frame_prompts or has_positive_prompt_fallback):
                error_msg = (
                    f"Scene {scene_id} missing required image prompt configuration. "
                    f"Requires ('first_frame_prompt' AND 'last_frame_prompt') OR ('positive_prompt')."
                )
                self.logger.error(f"[Scene {scene_id}] Validation Failed: {error_msg}")
                return (json.dumps([{"scene": scene_id, "error": error_msg, "status": "failed"}]),)
            
            if not has_video_prompt:
                 error_msg = f"Scene {scene_id} is missing the required key: 'video_prompt' or its value is empty."
                 self.logger.error(f"[Scene {scene_id}] Validation Failed: {error_msg}")
                 return (json.dumps([{"scene": scene_id, "error": error_msg, "status": "failed"}]),)

            if not has_negative_prompt:
                error_msg = f"Scene {scene_id} is missing the required key: 'negative_prompt' or its value is empty."
                self.logger.error(f"[Scene {scene_id}] Validation Failed: {error_msg}")
                return (json.dumps([{"scene": scene_id, "error": error_msg, "status": "failed"}]),)
        # --- END VALIDATION CHECK ---
        
        # --- 2. Load Workflows and Setup Directory ---
        try:
            video_output_dir_path = Path(video_output_dir)
            video_output_dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory confirmed: {video_output_dir_path}")
            
            with open(video_workflow_path, "r") as f:
                video_workflow_data = json.load(f)
            self.logger.info(f"Video workflow template loaded from: {video_workflow_path}")

            with open(first_image_workflow_path, "r") as f:
                first_image_workflow_data = json.load(f)
            self.logger.info(f"First Image workflow template loaded from: {first_image_workflow_path}")

            with open(last_image_workflow_path, "r") as f:
                last_image_workflow_data = json.load(f)
            self.logger.info(f"Last Image workflow template loaded from: {last_image_workflow_path}")
            
        except Exception as e:
            error_msg = f"Error during setup (loading workflow or creating directory): {e.__class__.__name__}: {str(e)}"
            self.logger.error(error_msg)
            return (json.dumps([{"scene": "N/A", "error": error_msg, "status": "failed"}]),)

        self.logger.info("--- Starting Concurrent Video Generation ---")
        self.logger.info(f"ComfyUI URL: {comfy_api_url} | Max Workers: {max_workers} | Polling Interval: {poll_interval}s")

        # --- 3. Concurrently Process Scenes ---
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, scene in enumerate(scenes):
                next_scene = scenes[i+1] if i + 1 < len(scenes) else None
                
                future = executor.submit(
                    self._run_scene, 
                    comfy_api_url, 
                    video_output_dir_path.as_posix(), 
                    video_workflow_data, 
                    first_image_workflow_data,
                    last_image_workflow_data,
                    scene, 
                    next_scene,
                    poll_interval
                )
                futures[future] = scene 
            
            for future in concurrent.futures.as_completed(futures):
                scene_info = futures[future]
                scene_id = scene_info.get('scene', 'N/A')
                try:
                    results.append(future.result())
                except Exception as exc:
                    error_msg = f"Unexpected exception in thread for scene {scene_id}: {exc}"
                    self.logger.exception(error_msg)
                    results.append({"scene": scene_id, "error": error_msg, "status": "failed"})

        # --- 4. Finalization and Return ---
        
        final_results_json = json.dumps(results, indent=2)
        
        total_end_time = time.time()
        total_duration = round(total_end_time - total_start_time, 2)
        
        self.logger.info(f"\n{'='*50}\n✅ All Scenes Processed. Total Time: {total_duration} seconds.\n{'='*50}")

        return (final_results_json,)