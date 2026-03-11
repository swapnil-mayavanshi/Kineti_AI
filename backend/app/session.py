"""
Kineti-AI Session State Management
Manages the current user session state.
"""


def create_default_session():
    """Returns a fresh session state dictionary."""
    return {
        "state": "check_in",          # check_in -> propose -> active_workout
        "pain_level": 0,
        "user_confirmed": False,
        "paused_for_voice": False,
        "active_exercise": None,
        "active_rules": None,
        "reps_count": 0,
        "target_reps": 10,
        "is_moving": False,
        "last_response": "Welcome back to Kineti AI.",
        "form_status": "neutral",      # good/bad/neutral
        "current_angle": 180,          # Current knee angle
        "language": "en-IN"            # en-IN or hi-IN (user can switch)
    }


# Global session state
current_session = create_default_session()
