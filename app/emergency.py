EMERGENCY_PHRASES = [
    # Breathing emergencies
    "difficulty breathing",
    "shortness of breath",
    "cannot breathe",
    "can't breathe",
    "not breathing",
    "breathing problem",

    # Chest-related emergencies
    "chest pain",
    "chest pressure",
    "tightness in chest",
    "pain in chest",

    # Stroke-like symptoms
    "face drooping",
    "slurred speech",
    "weakness on one side",
    "one side weakness",
    "sudden confusion",
    "sudden vision loss",

    # Severe bleeding / injury
    "severe bleeding",
    "heavy bleeding",
    "bleeding heavily",
    "blood not stopping",

    # Pregnancy emergencies
    "severe abdominal pain during pregnancy",
    "bleeding during pregnancy",
    "pregnancy bleeding",

    # Seizure / unconsciousness
    "having a seizure",
    "seizure for more than 5 minutes",
    "loss of consciousness",
    "passed out and not waking",
    "not waking up",

    # Urdu / Roman Urdu common phrases
    "saans nahi aa rahi",
    "saans lene mein mushkil",
    "seenay mein dard",
    "chest mein dard",
    "bohat zyada bleeding",
    "khoon band nahi ho raha",
    "behosh ho gaya",
    "behosh hai",
]


EMERGENCY_COMBINATIONS = [
    # These require both terms to appear together somewhere in the message
    ("chest", "pain"),
    ("chest", "pressure"),
    ("breath", "shortness"),
    ("breathing", "difficulty"),
    ("bleeding", "severe"),
    ("bleeding", "heavy"),
    ("pregnancy", "bleeding"),
    ("pregnant", "bleeding"),
    ("speech", "slurred"),
    ("face", "drooping"),
    ("one side", "weakness"),
    ("saans", "mushkil"),
    ("seenay", "dard"),
    ("chest", "dard"),
    ("khoon", "band nahi"),
]


def check_emergency(message: str) -> bool:
    """
    Checks whether the user's message contains high-risk emergency symptoms.
    Returns True if emergency is detected, otherwise False.
    """

    if not message:
        return False

    message = message.lower().strip()

    # Check full emergency phrases first
    for phrase in EMERGENCY_PHRASES:
        if phrase in message:
            return True

    # Check emergency combinations
    for term1, term2 in EMERGENCY_COMBINATIONS:
        if term1 in message and term2 in message:
            return True

    return False


def get_emergency_response() -> str:
    """
    Returns emergency alert message.
    """

    return (
        "⚠️ Emergency symptoms detected.\n\n"
        "Your symptoms may require immediate medical attention. "
        "Please call Rescue 1122 or go to the nearest emergency hospital immediately.\n\n"
        "This assistant is not a replacement for a doctor. "
        "Do not delay emergency care."
    )