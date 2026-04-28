"""Scene-cut detection via PySceneDetect's ContentDetector.

Returns the set of source-frame indices that START a new scene. We use this
in the schedule to AVOID interpolating across cuts (which would produce
ghastly morph artefacts) and instead duplicate the previous frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Set

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scene_cuts(path: Path, threshold: float = 27.0) -> Set[int]:
    """Return frame indices where a new scene begins (i.e. cut points)."""
    video = open_video(str(path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video=video, show_progress=False)
    scenes = manager.get_scene_list()
    # Each scene is (start, end) FrameTimecode. The first scene starts at 0
    # which is not a cut; subsequent starts ARE cuts.
    cuts: Set[int] = set()
    for i, (start, _end) in enumerate(scenes):
        if i == 0:
            continue
        cuts.add(start.get_frames())
    return cuts
