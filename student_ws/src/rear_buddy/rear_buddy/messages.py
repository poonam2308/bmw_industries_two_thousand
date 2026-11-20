"""What the car says, per scene. Edit freely.

Placeholders: {name}, {skills}, {languages}, {linkedin}, {pitch} come from PROFILE, {flirt_name} from FLIRT_PROFILE;
{battery} is the car's battery voltage (texts using it are skipped while the voltage is unknown, e.g. in the
simulation or on your laptop).
No emojis in the texts: the OpenCV preview cannot draw them. The phone display adds its own per scene.

Your real details: put PROFILE, QR_URL and FLIRT_PROFILE into rear_buddy/profile_local.py (git-ignored, so they are
never pushed); it overrides the placeholders below.
"""

PROFILE = {
    'name': 'Your Name',
    'skills': 'Your Role | Your Field',
    'languages': 'Your Languages',
    'linkedin': 'linkedin.com/in/your-profile',
    'pitch': 'an engineer',  # "In front of you: {pitch} looking for a job."
}

# The rear display shows this link as a QR code on these scenes (needs `pip install segno`).
QR_URL = 'https://www.linkedin.com/in/your-profile'
QR_SCENES = {'fancy_car', 'person_pitch'}

# Who the demo-only flirt mode speaks for.
FLIRT_PROFILE = {
    'flirt_name': 'Alex',
}

try:  # local, git-ignored overrides with your real details
    from .profile_local import FLIRT_PROFILE, PROFILE, QR_URL  # noqa: F401,F811
except ImportError:
    pass

# Brand check on the closest car (fancy.py, CLIP, runs offline). 'premium' brands get the job pitch, 'budget' brands
# get "old is gold". 'other' brands are there so the model has alternatives to choose from: they get the normal hello,
# as does every car the model is unsure about.
CAR_BRANDS = {
    'premium': ['Porsche', 'Audi', 'BMW'],
    'budget': ['Toyota', 'Fiat', 'Mini'],
    'other': ['Mercedes-Benz', 'Volkswagen', 'Ford', 'Opel', 'Renault', 'Skoda', 'Tesla', 'Hyundai'],
}
# Descriptions that also count as budget, for small cars of any brand.
BUDGET_DESCRIPTIONS = ['a small cheap city car', 'a small old hatchback']

MESSAGES = {
    # A premium car (Porsche, Audi, BMW) behind us: job pitch, with name and LinkedIn.
    'fancy_car': [
        'Nice car! You clearly make good decisions.\nMake another one: hire {name}.\n{skills}\n{languages}\n'
        '{linkedin}',
        'Behind you: a great car. In front of you: {pitch} looking for a job.\n'
        "Let's both move forward.\n{name} - {linkedin}",
        'I built the brain of this car.\nImagine what I could build for your company.\n{name} - {linkedin}',
    ],
    # A person behind us, with --pitch-people: the job pitch, spoken to a person who can scan the QR code.
    'person_pitch': [
        'Hi! Looking for {pitch}?\nYou just found one: {name}.\n{skills}\n{languages}\n{linkedin}',
        '{name} built the brain of this car.\nImagine what {name} could build for your team.\nScan me: {linkedin}',
        'Hiring? {name} is available.\n{skills}\n{linkedin}',
    ],
    # A budget or small car (Toyota, small cars) behind us.
    'budget_car': [
        'Hi :) Old is gold.',
    ],
    # An ordinary car behind us (another brand, or the brand check is unsure).
    'car': [
        'Hello back there! Thanks for keeping your distance.',
        'Fun fact: this car sees you with a neural network. You look great.',
        'If you can read this, you are a very attentive driver.',
    ],
    # A car behind us while we have not moved for a while.
    'stuck': [
        'Stuck again? Joke of the day: why did the robot cross the road?\nIt was programmed by a chicken.',
        'We have been standing here so long, my battery dropped to {battery:.1f} V.',
        'Traffic level: even my wheel encoders fell asleep.',
        'Rate this traffic jam: 1 star. Would not queue again.',
    ],
    'cyclist': [
        "You're beating me in traffic. On a bicycle. I've never felt so slow.",
        "Respect for pedalling! My battery is at {battery:.1f} V and I'm already tired.",
        'Race you to the next traffic light? (I will lose.)',
    ],
    # Demo-only flirt mode (Director(flirt=True), off by default): shown to any cyclist, on behalf of
    # FLIRT_PROFILE's name. Deliberately not based on guessing gender: the camera cannot tell.
    'cyclist_flirt': [
        "Hi! I'm {flirt_name}. Nice, single and not a creep.\nCoffee at the next red light?",
        '{flirt_name} here: nice guy, single, excellent at waiting in traffic.\nWant to grab a coffee?',
        'This car is driven by AI. {flirt_name} is single.\nCoffee? {flirt_name} is buying.',
        'Single, friendly, great sense of humour, parks well.\nCoffee with {flirt_name}?',
    ],
    'pedestrian': [
        'Hi there! Wave and I will wave back with my lights.',
        'Walking is faster than this traffic. Good choice.',
    ],
    'nobody': [
        '',
    ],
}

# Rear LED animation per scene (see animations.py).
ANIMATIONS = {
    'fancy_car': 'gold_shine',
    'person_pitch': 'gold_shine',
    'budget_car': 'breathing',
    'car': 'breathing',
    'stuck': 'sleepy',
    'cyclist': 'green_wave',
    'cyclist_flirt': 'hearts',
    'pedestrian': 'rainbow',
    'nobody': 'scanner',
}
