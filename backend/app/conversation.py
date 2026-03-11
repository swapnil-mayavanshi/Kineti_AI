"""
Kineti-AI Conversation Engine
State machine logic for managing the therapy conversation flow.
Includes RAG-powered general fitness Q&A.
"""

import re
import json
import os
from datetime import datetime

from app import config
from app import session
from app.pose_engine import check_dynamic_rule, reset_angle_tracking


def _user_data_path():
    """Returns the absolute path to user_data.json."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user_data.json")


def get_user_context():
    """Loads user data from JSON. Auto-resets daily reps when a new day starts."""
    default_data = {
        "user_name": "Swapnil",
        "current_day": 1,
        "reps_today": 0,
        "last_exercise": "Squat",
        "last_date": datetime.now().strftime("%Y-%m-%d"),
        "exercise_history": []
    }
    try:
        path = _user_data_path()
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Auto-reset if it's a new day
            today = datetime.now().strftime("%Y-%m-%d")
            last_date = data.get("last_date", "")
            if last_date and last_date != today:
                data["reps_today"] = 0
                data["current_day"] = data.get("current_day", 1) + 1
                data["last_date"] = today
                # Auto-save the reset
                save_user_data(data)
                print(f"🗓️ New day detected! Day {data['current_day']}, reps reset to 0.")
            
            # Ensure all fields exist (backward compat)
            for key, val in default_data.items():
                if key not in data:
                    data[key] = val
            return data
    except Exception as e:
        print(f"⚠️ Could not load user data: {e}")
    return default_data


def save_user_data(data):
    """Saves user data to JSON file."""
    try:
        data["last_date"] = datetime.now().strftime("%Y-%m-%d")
        with open(_user_data_path(), 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save user data: {e}")


# --- Bilingual Messages ---
# ALL messages in both English and Hindi for dynamic language switching
_MESSAGES = {
    # --- Rep counting ---
    "rep_count": {
        "en-IN": "Rep {count}. Keep going!",
        "hi-IN": "Rep {count}. Bahut acche, aage badho!"
    },
    "halfway": {
        "en-IN": "{count}! Halfway there, you're doing great!",
        "hi-IN": "{count}! Aadha ho gaya, bahut badhiya kar rahe ho!"
    },
    "set_complete": {
        "en-IN": "{count}! Amazing work, you completed your set!",
        "hi-IN": "{count}! Shaandaar! Aapka set poora ho gaya!"
    },

    # --- Form feedback ---
    "good_depth": {
        "en-IN": "Good depth! Hold it.",
        "hi-IN": "Bahut acchi depth! Ruko."
    },
    "go_deeper": {
        "en-IN": "Go deeper! Bend your knees more.",
        "hi-IN": "Aur neeche jao! Ghutne aur moro."
    },
    "form_good": {
        "en-IN": "Yes! You're doing fantastic. Keep it up!",
        "hi-IN": "Haan! Bahut accha kar rahe ho. Aise hi karo!"
    },

    # --- Language switch ---
    "switched_hindi": {
        "en-IN": "Zaroor! Ab main Hindi mein baat karungi. Aap kaise hain?",
        "hi-IN": "Zaroor! Ab main Hindi mein baat karungi. Aap kaise hain?"
    },
    "switched_english": {
        "en-IN": "Sure! I'll talk in English now. How are you doing?",
        "hi-IN": "Sure! Ab main English mein baat karungi."
    },

    # --- Greeting ---
    "greeting": {
        "en-IN": "Hello {name}! Dr. Kineti here. How are you feeling today?",
        "hi-IN": "Namaste {name}! Dr. Kineti yahan hai. Aaj aap kaisa mehsoos kar rahe hain?"
    },

    # --- Check-in state ---
    "checkin_welcome": {
        "en-IN": "Welcome back! How is your pain level today? You can say low, medium, or high.",
        "hi-IN": "Wapas swagat hai! Aaj dard ka level kya hai? Low, medium, ya high bolo."
    },
    "checkin_good": {
        "en-IN": "That's wonderful to hear, {name}!{progress} Say 'ready' or 'start' when you want to begin.",
        "hi-IN": "Yeh sunke bahut khushi hui, {name}!{progress} Jab chahein 'ready' ya 'start' bolo."
    },
    "checkin_good_progress_has": {
        "en-IN": " You've completed {reps} reps today! Let's do some more {exercise}.",
        "hi-IN": " Aaj aapne {reps} reps kiye hain! Chalo aur {exercise} karte hain."
    },
    "checkin_good_progress_none": {
        "en-IN": " Let's do some {exercise} today to build strength.",
        "hi-IN": " Aaj {exercise} karte hain taaki taakat bane."
    },
    "checkin_no_pain": {
        "en-IN": "Great to hear you're pain-free, {name}! Let's do some {exercise}. Say 'ready' when you want to begin.",
        "hi-IN": "Bahut accha, koi dard nahi! Chalo {exercise} karte hain, {name}. 'Ready' bolo jab tayyar ho."
    },
    "checkin_medium": {
        "en-IN": "Thanks for letting me know, {name}. Let's take it easy with some Heel Slides. Say 'ready' when you want to start.",
        "hi-IN": "Batane ke liye shukriya, {name}. Aaram se Heel Slides karte hain. 'Ready' bolo jab tayyar ho."
    },
    "checkin_pain": {
        "en-IN": "I'm sorry to hear that, {name}. Since you're experiencing pain, let's try gentle Heel Slides instead of {exercise}. They're easy on your knee. Just say 'ready' when you want to start.",
        "hi-IN": "Yeh sunke dukh hua, {name}. Dard hai toh {exercise} ki jagah gentle Heel Slides karte hain. Ghutne par zyada pressure nahi padega. 'Ready' bolo jab tayyar ho."
    },

    # --- Propose state ---
    "propose_start": {
        "en-IN": "Great! Let's start {exercise}. Aim for {target} degrees. {cue} Begin when you're ready!",
        "hi-IN": "Bahut accha! Chalo {exercise} shuru karte hain. {target} degree ka goal rakhein. {cue} Jab tayyar ho shuru karo!"
    },
    "propose_reject": {
        "en-IN": "No problem! Take all the time you need. Just say 'ready' whenever you want to start.",
        "hi-IN": "Koi baat nahi! Apna time lo. Jab tayyar ho 'ready' bol dena."
    },
    "propose_unclear": {
        "en-IN": "I need a clear 'yes' or 'ready' before we start. Are you ready to begin the exercise?",
        "hi-IN": "Shuru karne se pehle 'yes' ya 'ready' bolo. Kya aap tayyar hain exercise ke liye?"
    },
    "propose_fasttrack": {
        "en-IN": "Let's do Squats! Aim for {target} degrees. {cue} Begin when you're ready!",
        "hi-IN": "Chalo Squats karte hain! {target} degree ka goal rakhein. {cue} Jab tayyar ho shuru karo!"
    },

    # --- Workout messages ---
    "workout_complete": {
        "en-IN": "Great work, {name}! You completed {reps} reps of {exercise}. That's {total} total reps today on Day {day}! I've saved your progress. Take a breather!",
        "hi-IN": "Bahut badhiya, {name}! Aapne {exercise} ke {reps} reps poore kiye. Aaj ka total {total} reps hai, Day {day} par! Progress save ho gaya. Thoda aaram karo!"
    },
    "workout_stop": {
        "en-IN": "No problem, {name}. We can stop here. Let me know if you need anything else!",
        "hi-IN": "Koi baat nahi, {name}. Yahaan ruk jaate hain. Kuch aur chahiye toh batao!"
    },
    "pain_stop": {
        "en-IN": "Let's stop right there. Your safety comes first. Can you describe where it hurts?",
        "hi-IN": "Ruko! Aapki safety sabse pehle hai. Batao kahaan dard ho raha hai?"
    },
    "no_reps_yet": {
        "en-IN": "No reps counted yet in this set. Make sure to do full squats — bend your knees deep and stand back up!",
        "hi-IN": "Abhi koi rep count nahi hua. Poore squat karo — ghutne acche se moro aur wapas khade ho jao!"
    },
    "rep_count_total": {
        "en-IN": "You've done {count} reps in this set, {total} total today. Keep going!",
        "hi-IN": "Is set mein {count} reps hue, aaj total {total}. Aise hi karo!"
    },
    "rep_count_global": {
        "en-IN": "You've done {count} reps in this set, and {total} total reps today! Great job!",
        "hi-IN": "Is set mein {count} reps hue, aur aaj total {total} reps! Shaandaar!"
    },
    "rep_count_none_today": {
        "en-IN": "You haven't done any reps in this set yet, but you've completed {total} total reps today. Keep it up!",
        "hi-IN": "Is set mein abhi reps nahi hue, lekin aaj total {total} reps hue hain. Aise hi karo!"
    },
    "rep_count_zero": {
        "en-IN": "No reps counted yet today. Make sure your full body is visible to the camera when doing squats!",
        "hi-IN": "Aaj abhi koi rap nahi hua. Camera mein poora sharir dikhna chahiye squats karte waqt!"
    },

    # --- Finished workout ---
    "finished_start_another": {
        "en-IN": "Let's do another set! How are you feeling?",
        "hi-IN": "Ek aur set karte hain! Kaisa feel ho raha hai?"
    },
}


def _msg(key, **kwargs):
    """Get a message in the current session language."""
    lang = session.current_session.get("language", "en-IN")
    msg_dict = _MESSAGES.get(key, {})
    template = msg_dict.get(lang, msg_dict.get("en-IN", key))
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        return template


def _get_lang_instruction():
    """Returns LLM instruction for current language."""
    lang = session.current_session.get("language", "en-IN")
    if lang == "hi-IN":
        return "\n\nCRITICAL LANGUAGE RULE: You MUST respond ENTIRELY in Hindi using Roman script (NOT Devanagari). Every single word of your response must be in Hindi/Hinglish. Do NOT use any English sentences. Example: 'Aap bahut accha kar rahe ho, Swapnil! Aaj hum squats karenge.'"
    return "\n\nRespond in English."


def ask_general_question(user_text):
    """
    Uses RAG + LLM to answer any fitness/PT/health question.
    Falls back to LLM-only if RAG is unavailable.
    """
    if not config.llm:
        return "I'm sorry, I can't answer that right now. Let me know how you're feeling so we can start your exercises!"

    try:
        context = ""
        if config.vector_store:
            retriever = config.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(user_text)
            context = "\n".join([d.page_content for d in docs]) if docs else ""

        user_data = get_user_context()
        u_name = user_data.get('user_name', 'Swapnil')
        u_day = user_data.get('current_day', 1)
        u_last = user_data.get('last_exercise', 'None recorded')
        u_reps = user_data.get('reps_today', 0)

        lang_instruction = _get_lang_instruction()
        prompt = f"""{config.SYSTEM_INSTRUCTION}

Patient Name: {u_name}
Current Therapy Day: {u_day}
Last Exercise Completed: {u_last}
Reps Today: {u_reps}

Context from PT protocol:
{context}

User question/statement: {user_text}
{lang_instruction}
Respond as Dr. Kineti. Be highly personalized, referencing his name and past progress if relevant. Keep it under 3 sentences."""

        response = config.llm.invoke([("system", prompt)]).content
        print(f"🧠 AI answered general question: {response[:80]}...")
        return response.strip()
    except Exception as e:
        print(f"⚠️ General Q&A error: {e}")
        return "That's a great question, Swapnil! I'd recommend discussing it with your physical therapist for personalized advice."


def extract_exercise_rules(exercise_name):
    """Gets the correct angle targets from the RAG/LLM using the Persona."""
    print(f"🔍 AI Reading Protocol for: {exercise_name}...")

    # Default fallback if RAG is off
    default_rules = {"joint": "RightKnee", "target": 90, "mode": "min", "cue": "Control your motion."}
    if not config.vector_store:
        return default_rules

    try:
        retriever = config.vector_store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(exercise_name)
        context = docs[0].page_content if docs else ""

        extraction_prompt = f"""
        {config.SYSTEM_INSTRUCTION}
        
        Context: {context}
        Task: Extract biomechanics for '{exercise_name}'.
        Return ONLY valid JSON: {{"joint": "RightKnee", "target": 90, "mode": "min", "cue": "Keep back straight"}}
        """
        response = config.llm.invoke([("system", extraction_prompt)]).content
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        print(f"⚠️ Rule Extraction Failed: {e}")

    return default_rules


def _is_question(text):
    """Check if user text looks like a general question."""
    question_starters = ["what", "how", "why", "when", "where", "who", "can",
                         "should", "is", "are", "do", "does", "will", "tell me",
                         "explain", "describe", "help me understand"]
    t = text.lower().strip()
    return any(t.startswith(q) for q in question_starters) or "?" in t


def manage_conversation(user_text):
    """
    Determines AI response based on conversation state.
    Now with general fitness Q&A support via RAG + LLM.
    """
    cs = session.current_session
    user_lower = user_text.lower()
    lang = cs.get("language", "en-IN")
    
    # Load user data for personalization
    user_data = get_user_context()
    u_name = user_data.get("user_name", "Swapnil")
    r_today = user_data.get("reps_today", 0)
    l_ex = user_data.get("last_exercise", "Squat")

    # --- GLOBAL: Language switch detection (works in ANY state) ---
    hindi_triggers = ["hindi", "hindi mein", "hindi me", "talk in hindi", "speak hindi",
                      "hindi boliye", "hindi mein baat", "hindi me baat", "talk hindi",
                      "baat hindi", "switch to hindi", "change to hindi", "in hindi"]
    english_triggers = ["talk in english", "speak english", "talk english", "baat english",
                        "switch to english", "change to english", "in english", "angrezi"]
    
    # Check Hindi first (more specific checks to avoid false triggers)
    wants_hindi = any(t in user_lower for t in hindi_triggers)
    wants_english = any(t in user_lower for t in english_triggers)
    
    if wants_hindi and not wants_english:
        cs["language"] = "hi-IN"
        print(f"🌐 Language switched to Hindi (detected in: '{user_text}')")
        return _msg("switched_hindi")
    
    if wants_english and not wants_hindi:
        cs["language"] = "en-IN"
        print(f"🌐 Language switched to English (detected in: '{user_text}')")
        return _msg("switched_english")

    # GLOBAL: Answer rep count questions in ANY state
    rep_question_words = ["how many", "tell me", "reps", "rep count", "total reps", "kitne"]
    if any(w in user_lower for w in rep_question_words) and ("reps" in user_lower or "kitne" in user_lower):
        count = cs['reps_count']
        if count > 0:
            return _msg("rep_count_global", count=count, total=r_today + count)
        elif r_today > 0:
            return _msg("rep_count_none_today", total=r_today)
        else:
            return _msg("rep_count_zero")

    # -------------------------------
    # STATE 1: CHECK IN (Assess pain)
    # -------------------------------
    if cs["state"] == "check_in":
        cs["user_confirmed"] = False

        # Fast-track: user directly asks to start exercise → skip propose, go to active_workout
        start_words = ["start", "ready", "begin", "let's go", "exercise", "workout", "squat"]
        if any(w in user_lower for w in start_words):
            cs["pain_level"] = 2
            cs["user_confirmed"] = True
            cs["active_exercise"] = "Squat"
            cs["active_rules"] = extract_exercise_rules("Squat")
            cs["reps_count"] = 0
            cs["is_moving"] = False
            cs["state"] = "active_workout"
            reset_angle_tracking()  # Clear smoothing buffer for fresh workout
            cue = cs["active_rules"].get("cue", "Control your motion.")
            target = cs["active_rules"].get("target", 90)
            return _msg("propose_fasttrack", target=target, cue=cue)

        # Negation words that flip meaning
        negation_words = ["no", "not", "nothing", "dont", "don't", "without", "zero", "none", "never"]
        has_negation = any(neg in user_lower for neg in negation_words)

        pain_words = ["pain", "hurt", "sore", "bad", "terrible", "awful", "worse", "ache"]
        good_words = ["good", "fine", "okay", "great", "better", "amazing", "fantastic", "perfect", "excellent", "well"]
        medium_words = ["medium", "moderate", "so-so", "alright", "okay-ish", "manageable"]

        has_pain_word = any(x in user_lower for x in pain_words)
        has_good_word = any(x in user_lower for x in good_words)
        has_medium_word = any(x in user_lower for x in medium_words)

        # LOGIC: "no pain" / "nothing hurts" = GOOD
        if has_good_word and not (has_pain_word and not has_negation):
            cs["pain_level"] = 2
            cs["state"] = "propose"
            progress_msg = _msg("checkin_good_progress_has", reps=r_today, exercise=l_ex) if r_today > 0 else _msg("checkin_good_progress_none", exercise=l_ex)
            return _msg("checkin_good", name=u_name, progress=progress_msg)

        elif has_pain_word and has_negation:
            cs["pain_level"] = 1
            cs["state"] = "propose"
            return _msg("checkin_no_pain", name=u_name, exercise=l_ex)

        elif has_medium_word:
            cs["pain_level"] = 5
            cs["state"] = "propose"
            return _msg("checkin_medium", name=u_name)

        elif has_pain_word and not has_negation:
            cs["pain_level"] = 7
            cs["state"] = "propose"
            return _msg("checkin_pain", name=u_name, exercise=l_ex)

        # General question during check-in? Answer it!
        elif _is_question(user_text):
            return ask_general_question(user_text)

        else:
            return _msg("checkin_welcome")

    # ------------------------------------------------
    # STATE 2: PROPOSE EXERCISE (The Gatekeeper)
    # ------------------------------------------------
    elif cs["state"] == "propose":

        confirmation_phrases = ["yes", "start", "sure", "okay", "go", "ready",
                                "let's do it", "begin", "i'm ready", "let's go",
                                "yep", "yeah", "absolutely", "definitely"]
        rejection_phrases = ["no", "wait", "stop", "not yet", "hold on",
                            "give me a moment", "not ready", "one second"]

        if any(phrase in user_lower for phrase in confirmation_phrases):
            cs["user_confirmed"] = True

            exercise = "Squat"
            if (cs["pain_level"] > 4 or "heel" in user_lower) and "squat" not in user_lower:
                exercise = "Heel Slides"

            cs["active_exercise"] = exercise
            cs["active_rules"] = extract_exercise_rules(exercise)
            cs["reps_count"] = 0
            cs["is_moving"] = False
            cs["state"] = "active_workout"
            reset_angle_tracking()  # Clear smoothing buffer for fresh workout

            cue = cs["active_rules"].get("cue", "Control your motion.")
            target = cs["active_rules"].get("target", 90)

            return _msg("propose_start", exercise=exercise, target=target, cue=cue)

        elif any(phrase in user_lower for phrase in rejection_phrases):
            cs["user_confirmed"] = False
            return _msg("propose_reject")

        # General question during propose state? Answer via AI!
        elif _is_question(user_text):
            return ask_general_question(user_text)

        else:
            return _msg("propose_unclear")

    # -----------------------------------
    # STATE 3: ACTIVE WORKOUT
    # -----------------------------------
    elif cs["state"] == "active_workout":

        # Stop workout — now includes simple "stop" and "done" as standalone triggers
        stop_phrases = ["stop", "done", "stop workout", "i'm done", "finished", 
                        "too tired", "stop now", "take a break", "end workout", "that's enough",
                        "bas", "ruko", "band karo"]  # Hindi stop words too
        if any(x in user_lower for x in stop_phrases):
            reps = cs['reps_count']
            cs["state"] = "finished_workout"
            cs["user_confirmed"] = False
            
            # Save progress to user_data.json with exercise history
            user_data = get_user_context()
            user_data["reps_today"] = user_data.get("reps_today", 0) + reps
            user_data["last_exercise"] = cs.get("active_exercise", "Squat")
            
            # Log this set in exercise history
            history = user_data.get("exercise_history", [])
            history.append({
                "exercise": cs.get("active_exercise", "Squat"),
                "reps": reps,
                "day": user_data.get("current_day", 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            user_data["exercise_history"] = history[-50:]
            save_user_data(user_data)

            total_today = user_data["reps_today"]
            if reps > 0:
                return _msg("workout_complete", name=u_name, reps=reps, exercise=cs.get('active_exercise', 'Squats'), total=total_today, day=user_data.get('current_day', 1))
            else:
                return _msg("workout_stop", name=u_name)

        if any(x in user_lower for x in ["pain", "hurt", "ouch", "sharp", "dard", "takleef"]):
            cs["state"] = "check_in"
            cs["user_confirmed"] = False
            return _msg("pain_stop")

        # General question during workout? 
        if _is_question(user_text) and len(user_text) > 5:
            # Special handling for rep count questions
            rep_words = ["how many", "reps", "count", "rep"]
            if any(w in user_lower for w in rep_words):
                count = cs['reps_count']
                total = r_today + count
                if count > 0:
                    return _msg("rep_count_total", count=count, total=total)
                else:
                    return _msg("no_reps_yet")
            
            # Disable general Q&A during active workout to prevent TV noise interruptions
            return None

        if any(x in user_lower for x in ["am i doing right", "am i doing well", "is this good", "is my form good", "sahi kar raha", "theek hai"]):
            return _msg("form_good")

        return None  # Let vision handle normal workout flow

    # -----------------------------------
    # STATE 4: FINISHED WORKOUT
    # -----------------------------------
    elif cs["state"] == "finished_workout":
        # If they explicitly ask to start another exercise, loop back to check_in
        start_words = ["start", "another", "more", "continue", "again", "exercise", "workout", "ready", "aur", "phir se"]
        if any(w in user_lower for w in start_words):
            cs["state"] = "check_in"
            return _msg("finished_start_another")
        
        # Otherwise just answer general questions and let them rest
        if _is_question(user_text):
            return ask_general_question(user_text)
            
        return "I'm right here if you need anything. Just say 'start another exercise' when you want to go again!"

    return "I'm here and listening. How can I help you?"

