from .whisper import WhisperTranscribeNode
from .iterator import SceneVideoIteratorNode
from .test import SceneVideoWanIteratorNode
from .json import JSONArrayPathMultilineConcatenator
from .wan import SceneVideoWan14BIteratorNode, SceneVideoWan5BIteratorNode
from .video import CustomVideoConcatenator
from .wani2v import SceneImage2VideoIterator

NODE_CLASS_MAPPINGS = { 
    "Whisper Transcribe" : WhisperTranscribeNode,
    "Scene Video Iterator" : SceneVideoIteratorNode,
    "Scene Video WAN Iterator" : SceneVideoWanIteratorNode,
    "Scene Video WAN 14B Iterator" : SceneVideoWan14BIteratorNode,
    "Scene Video WAN 5B Iterator" : SceneVideoWan5BIteratorNode,
    "JSON Array Path Multiline Concatenator" : JSONArrayPathMultilineConcatenator,
    "Custom Pymovie Video Concatenator" : CustomVideoConcatenator,
    "Scene Video WAN 14B Image to Video Iterator" : SceneImage2VideoIterator
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "Whisper Transcribe" : "Whisper Transcribe", 
     "Scene Video Iterator" : "Scene Video Custom Iterator",
     "Scene Video WAN Iterator": "Scene Video WAN Iterator",
     "Scene Video WAN 14B Iterator": "Scene Video WAN 14B Iterator",
     "Scene Video WAN 5B Iterator": "Scene Video WAN 5B Iterator",
     "JSON Array Path Multiline Concatenator": "JSON Array Path Multiline Concatenator",
     "Custom Pymovie Video Concatenator": "Custom Pymovie Video Concatenator",
     "Scene Video WAN 14B Image to Video Iterator": "Scene Video WAN 14B Image to Video Iterator"
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']