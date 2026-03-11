"""
Kineti-AI Pose Engine
MediaPipe pose detection, angle calculation, form analysis, and vision processing.
"""

import os
import json
import math
import numpy as np
import cv2
import mediapipe as mp

from app.config import (
    LANDMARK_MAP, SQUAT_GOOD_DEPTH, SQUAT_PARTIAL_DEPTH, SQUAT_STANDING
)
from app import session

# --- MediaPipe State ---
pose_detector = None
USE_TASKS_API = True
_debug_frame_saved = False
_frame_count = 0  # For reducing log spam
_video_timestamp_ms = 0  # Monotonic timestamp for VIDEO mode
_warmup_frames = 3  # Skip form analysis for first N frames (VIDEO mode warmup)

# --- Debounce State (no heavy smoothing — just consecutive-frame filtering) ---
_down_consecutive = 0        # Consecutive frames in squat-down position
_up_consecutive = 0          # Consecutive frames in standing position
_MIN_DOWN_FRAMES = 2         # Must be down for 2+ consecutive frames to register squat
_MIN_UP_FRAMES = 3           # Must be standing for 3+ consecutive frames to count rep (prevents flicker)


def reset_angle_tracking():
    """Reset debounce state when workout starts/stops."""
    global _down_consecutive, _up_consecutive
    _down_consecutive = 0
    _up_consecutive = 0


def init_mediapipe():
    """Initialize MediaPipe pose detector using Tasks API."""
    global pose_detector, USE_TASKS_API
    if pose_detector is not None:
        return

    # Find model file
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'pose_landmarker.task'
    )
    if not os.path.exists(model_path):
        model_path = 'pose_landmarker.task'

    if not os.path.exists(model_path):
        print(f"❌ MediaPipe model not found at {model_path}")
        return

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,  # VIDEO mode enables temporal smoothing to prevent jitter
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(options)
    USE_TASKS_API = True
    print(f"✅ MediaPipe Pose initialized (Tasks API, model: {model_path}, Mode: VIDEO)")


def _try_detect_pose(rgb_frame):
    """Attempts MediaPipe pose detection using VIDEO mode for temporal smoothing.
    Requires monotonic timestamps to track movement across frames."""
    global _video_timestamp_ms
    
    # Simulate roughly 30fps delta (33ms) to keep temporal tracking smooth
    _video_timestamp_ms += 33 
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = pose_detector.detect_for_video(mp_image, _video_timestamp_ms)  
    
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        return results.pose_landmarks[0]
    return None


def _landmarks_to_json(landmarks):
    """Converts MediaPipe landmarks to skeleton JSON (backend) + unity JSON."""
    joints = []
    landmarks_for_unity = []

    for idx, name in LANDMARK_MAP.items():
        lm = landmarks[idx]
        # Raw coords for backend angle calculation
        joints.append({
            "name": name,
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
            "z": round(lm.z, 4)
        })
        # Unity coords: flip Y for Unity coordinate system
        landmarks_for_unity.append({
            "name": name,
            "x": round(lm.x, 4),
            "y": round(1.0 - lm.y, 4),  # Flip Y for Unity
            "z": round(lm.z, 4)
        })

    skeleton_json = json.dumps({"joints": joints})
    unity_json = json.dumps({"joints": landmarks_for_unity})
    
    # Debug: log specific joint coords to verify they change between frames
    global _frame_count
    if _frame_count % 10 == 0:
        for j in joints:
            if j["name"] == "RightKnee":
                print(f"🦵 RightKnee: x={j['x']:.4f} y={j['y']:.4f} z={j['z']:.4f}")
                break
    
    return skeleton_json, unity_json


def detect_pose_from_frame(image_bytes, rotation_angle=0, is_mirrored=False):
    """
    Runs MediaPipe Pose on a camera frame.
    Uses EXPLICIT rotation from Unity's videoRotationAngle — no guessing.
    """
    global _debug_frame_saved, _frame_count
    _frame_count += 1
    try:
        init_mediapipe()
        if pose_detector is None:
            return "{}", "{}"

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return "{}", "{}"

        h, w = frame.shape[:2]
        if w < 50 or h < 50:
            return "{}", "{}"

        if _frame_count % 10 == 1:
            print(f"📐 Frame: {w}x{h} rot={rotation_angle}° mirror={is_mirrored}")

        # Apply EXACT rotation from Unity's videoRotationAngle
        # videoRotationAngle = degrees CLOCKWISE to match display orientation
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if is_mirrored:
            frame = cv2.flip(frame, 0)

        h, w = frame.shape[:2]

        # Save debug frames periodically to verify camera sends different images
        if _frame_count % 50 == 1:
            debug_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            debug_path = os.path.join(debug_dir, f'debug_frame_{_frame_count}.jpg')
            cv2.imwrite(debug_path, frame)
            print(f"💾 Debug frame saved: {debug_path} ({w}x{h})")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = _try_detect_pose(rgb_frame)

        if landmarks is None:
            return "{}", "{}"

        skeleton_json, unity_json = _landmarks_to_json(landmarks)
        if _frame_count % 10 == 1:
            print(f"✅ Pose OK ({len(LANDMARK_MAP)} joints)")
        return skeleton_json, unity_json

    except Exception as e:
        print(f"⚠️ Pose detection error: {e}")
        import traceback
        traceback.print_exc()
        return "{}", "{}"


def calculate_angle(p1, p2, p3):
    """Calculates 3D angle between three joints."""
    try:
        ax, ay, az = p1['x']-p2['x'], p1['y']-p2['y'], p1['z']-p2['z']
        bx, by, bz = p3['x']-p2['x'], p3['y']-p2['y'], p3['z']-p2['z']
        dot_product = ax*bx + ay*by + az*bz
        mag_a = math.sqrt(ax**2 + ay**2 + az**2)
        mag_b = math.sqrt(bx**2 + by**2 + bz**2)
        if mag_a * mag_b == 0:
            return 180.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot_product / (mag_a * mag_b)))))
    except Exception:
        return 180.0


def check_dynamic_rule(joints, rules):
    """Checks if the user's skeleton meets the exercise rule."""
    joint_name = rules.get("joint", "RightKnee")
    target = rules.get("target", 90)
    mode = rules.get("mode", "min")

    angle = 180
    try:
        if joint_name == "RightKnee" and all(k in joints for k in ['RightHip', 'RightKnee', 'RightAnkle']):
            angle = calculate_angle(joints['RightHip'], joints['RightKnee'], joints['RightAnkle'])
        elif joint_name == "RightHip" and all(k in joints for k in ['RightShoulder', 'RightHip', 'RightKnee']):
            angle = calculate_angle(joints['RightShoulder'], joints['RightHip'], joints['RightKnee'])
    except Exception:
        pass

    passed = False
    if mode == "min" and angle < target:
        passed = True
    elif mode == "max" and angle > target:
        passed = True
    return passed, angle


def analyze_squat_form(joints):
    """Analyze squat form and return form_status, knee_angle, feedback."""
    try:
        required = ['RightHip', 'RightKnee', 'RightAnkle', 'LeftHip', 'LeftKnee', 'LeftAnkle']
        if not all(k in joints for k in required):
            print(f"⚠️ Missing joints for squat: have {list(joints.keys())}")
            return "unknown", 180, ""

        # Calculate both knee angles
        r_angle = calculate_angle(joints['RightHip'], joints['RightKnee'], joints['RightAnkle'])
        l_angle = calculate_angle(joints['LeftHip'], joints['LeftKnee'], joints['LeftAnkle'])
        
        # Use AVERAGE of both knees. When viewing from the side, the far leg
        # often reads incorrectly. Average + 2-frame debounce gives the best
        # balance between catching real squats and filtering glitches.
        knee_angle = (r_angle + l_angle) / 2.0

        # DEBUG: Print every frame so we can see the angles
        global _frame_count
        if _frame_count % 3 == 0:  # Every 3rd frame
            print(f"🦵 Knee angles: R={r_angle:.1f}° L={l_angle:.1f}° avg={knee_angle:.1f}° | Thresholds: good<{SQUAT_GOOD_DEPTH} partial<{SQUAT_PARTIAL_DEPTH} standing>{SQUAT_STANDING}")

        if knee_angle < SQUAT_GOOD_DEPTH:
            return "good", knee_angle, "Great depth!"
        elif knee_angle < SQUAT_PARTIAL_DEPTH:
            return "bad", knee_angle, "Go deeper! Bend your knees more."
        else:
            return "neutral", knee_angle, ""  # Standing position

    except Exception as e:
        print(f"⚠️ Squat analysis error: {e}")
        return "unknown", 180, ""


def process_vision_data(skeleton_json):
    """
    Analyzes skeleton data for exercise form and rep counting.
    Uses debounced state transitions to prevent phantom reps from flickering angles.
    """
    global _down_consecutive, _up_consecutive
    cs = session.current_session

    # Guard: skip if paused for voice (user is talking to AI)
    if cs.get("paused_for_voice", False):
        return None

    try:
        data = json.loads(skeleton_json)
        if "joints" not in data or len(data["joints"]) == 0:
            return None

        joints = {j['name']: j for j in data['joints']}

        # ALWAYS calculate angles (for debug display and /debug endpoint)
        form_status, knee_angle, form_tip = analyze_squat_form(joints)
        cs["current_angle"] = knee_angle
        cs["form_status"] = form_status

        # DEBUG: Log state every 3rd frame
        if _frame_count % 3 == 0:
            print(f"📊 state={cs['state']} form={form_status} angle={knee_angle:.1f}° is_moving={cs['is_moving']} reps={cs['reps_count']} down_frames={_down_consecutive} up_frames={_up_consecutive}")

        # GUARD: Skip form analysis during warmup (VIDEO mode gives bad angles on first frames)
        if _frame_count <= _warmup_frames:
            if _frame_count == _warmup_frames:
                print(f"🔥 Warmup complete ({_warmup_frames} frames), form analysis active")
            return None

        # GUARD: Only do rep counting and feedback during active workout
        if cs["state"] != "active_workout":
            return None

        # ---------- DEBOUNCED Rep Counting Logic ----------
        # Phase 1: Detect SQUAT DOWN — angle must go below SQUAT_GOOD_DEPTH for _MIN_DOWN_FRAMES
        # Phase 2: Detect STANDING UP — angle must go above SQUAT_PARTIAL_DEPTH for _MIN_UP_FRAMES
        # This counts as one complete rep.
        
        is_deep_squat = knee_angle < SQUAT_GOOD_DEPTH        # Below 100° = full squat
        is_rising = knee_angle > SQUAT_PARTIAL_DEPTH          # Above 130° = coming back up
        
        if is_deep_squat:
            # User is in deep squat position
            _down_consecutive += 1
            _up_consecutive = 0  # Reset standing counter
            
            if not cs["is_moving"] and _down_consecutive >= _MIN_DOWN_FRAMES:
                cs["is_moving"] = True
                print(f"⬇️ SQUAT DOWN confirmed! angle={knee_angle:.1f}° (held for {_down_consecutive} frames)")
                from app.conversation import _msg
                return _msg("good_depth")
        
        elif is_rising:
            # User is standing back up (angle above partial depth)
            if cs["is_moving"]:
                # Only count up frames when user WAS in a squat
                _up_consecutive += 1
                _down_consecutive = 0
                
                if _up_consecutive >= _MIN_UP_FRAMES:
                    # REP COMPLETE!
                    cs["is_moving"] = False
                    cs["reps_count"] += 1
                    count = cs['reps_count']
                    print(f"⬆️ REP {count} COUNTED! angle={knee_angle:.1f}° (stood for {_up_consecutive} frames)")
                    _up_consecutive = 0
                    from app.conversation import _msg
                    if count == 5:
                        return _msg("halfway", count=count)
                    elif count == cs['target_reps']:
                        return _msg("set_complete", count=count)
                    else:
                        return _msg("rep_count", count=count)
            else:
                # Not squatting, just standing normally
                _down_consecutive = 0
                _up_consecutive = 0
        
        else:
            # In between (100° to 130°) — transitional zone
            # Don't reset counters — allow smooth transition through this zone
            if not cs["is_moving"] and _down_consecutive == 0:
                # User hasn't squatted yet and is in partial zone
                from app.conversation import _msg
                return _msg("go_deeper")
                
    except Exception as e:
        print(f"⚠️ Vision processing error: {e}")
        import traceback
        traceback.print_exc()
        return None
    return None
