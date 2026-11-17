"""Phrase bank for the synthetic maintenance-log corpus.

Hand-written, SYNTHETIC wording: no real InGen ticket or operator text was
used. A ticket is ``[prefix] core [suffix]`` where the *core* phrase carries
the category wording. Cores are grouped; each has a stable ``phrase_id``
(``group:index``) so that the lexical-shift test can hold out whole phrases.

Slots: {side} {pct} {pct2} {temp} {d} {min} {place} {hh} {hh2}.
"""

from __future__ import annotations

# --- operator symptom notes, one group per category -------------------------
SYMPTOM: dict[str, list[str]] = {
    "battery": [
        "battery draining faster than usual, came back to dock early",
        "did not finish the full patrol loop, low battery warning at {pct}%",
        "runtime is down, it used to do the whole shift on one charge",
        "battery percentage drops quickly on the uphill section",
        "voltage sag warning popped up when it accelerated",
        "charge level jumped from {pct2}% to {pct}% within a few minutes",
        "takes much longer to reach full charge than the other units",
        "battery pack feels warm after charging",
        "state of health on the dashboard went down again",
        "returned to dock twice this shift for charging",
        "power cuts out for a second under heavy load then recovers",
        "range per charge noticeably shorter this week",
        "low battery alarm halfway through the route",
        "battery not holding, loses charge sitting idle overnight",
        "cells look unbalanced on the service screen, one group reads low",
    ],
    "drivetrain": [
        "grinding noise from the {side} wheel when turning",
        "unit vibrates more than normal on smooth pavement",
        "rattling sound from the drive unit at low speed",
        "pulls slightly to the {side} on straight sections",
        "{side} wheel feels rough when spun by hand",
        "squeaking from the wheel hub, gets louder over the shift",
        "mast is shaking visibly while driving",
        "droning noise from the drivetrain that was not there last week",
        "wheel has play in it and wobbles a bit",
        "rides rough even on the flat car park",
        "clicking noise on every wheel rotation",
        "seems to struggle on the ramp, slower than usual",
        "noticeable shudder when starting from standstill",
        "metallic scraping sound near the {side} axle",
        "bearing on the {side} side sounds dry",
    ],
    "thermal": [
        "motor housing very hot to the touch after patrol",
        "overheating warning on the display near the end of the shift",
        "fan sounds weak and is not spinning up like it used to",
        "unit stays hot for a long time after docking",
        "motor temperature climbing higher than the other rovers on the same route",
        "hot smell coming from the motor bay",
        "thermal throttle kicked in on the hill and it slowed down",
        "cooling fan making a ticking noise and airflow is low",
        "temperature alarm cleared after a ten minute rest",
        "runs hotter every day this week",
        "vents feel blocked, barely any air coming out",
        "motor temperature reached {temp}C on a mild day",
        "had to pause the patrol to let it cool down",
        "heat warning even on the easy flat route",
        "cooling does not keep up under load any more",
    ],
    "sensor_drift": [
        "motor temperature reading shows {temp}C while docked overnight but the housing is cold",
        "temperature on the dashboard does not match the handheld IR gun, reads about {d}C high",
        "temperature sensor value creeping up even when the unit is parked",
        "motor temperature reading seems off, no heat felt on the housing",
        "thermistor reading differs from ambient after idling for hours",
        "temperature readout is offset, same error every morning before the first patrol",
    ],
    "sensor_drift_as_thermal": [   # drift described as if the motor were hot
        "motor temperature high on the dashboard again",
        "temperature warning showing although the route was easy",
        "dashboard says it is running hot",
    ],
    "sensor_minor": [
        "lidar lens dirty, false obstacle detections",
        "bumper sensor stuck triggered with nothing touching it",
        "wheel encoder count jumps, odometry looks wrong on the map",
        "IMU needs recalibration, heading drifts while stationary",
        "ultrasonic sensor reports an obstacle that is not there",
        "GPS position jumps around near the building",
        "sensor fault code flickers on and off, readings look frozen",
        "vibration sensor reads zero while moving",
        "proximity sensor on the {side} side gives no reading",
        "current sensor reading noisy, spikes with the unit parked",
    ],
    "connectivity": [
        "lost connection to the unit for about {min} minutes",
        "telemetry dropped out and the dashboard showed offline",
        "unit went offline near the {place} and came back by itself",
        "no data uplink during part of the patrol",
        "wifi handover failed at the {place}",
        "LTE signal weak, commands delayed",
        "remote control lagging, connection keeps dropping",
        "gap in the dashboard data between {hh}:00 and {hh2}:00",
        "modem rebooted itself in the middle of the patrol",
        "could not reach the unit from the control room, timeout",
        "SIM not registering on the network after restart",
        "missed heartbeat alerts from the fleet server",
        "data link unstable all morning, reconnects every few minutes",
    ],
    "user_or_operational_error": [
        "e-stop was left pressed after cleaning, unit would not start",
        "charger was unplugged by cleaning staff, unit did not charge overnight",
        "unit not docked properly, contacts misaligned, no charge",
        "blocked by a parked van, patrol aborted",
        "wrong route loaded by the operator, it went to the old zone",
        "driven manually into a kerb and got stuck",
        "gate was locked, could not complete the loop",
        "pre-shift check was skipped and the patrol started late",
        "someone moved the dock, unit could not find it",
        "sent off the path onto wet grass, wheels spinning, had to be pushed",
        "left in manual mode over the weekend, battery ran flat",
        "pallet left on the route, unit stopped and waited",
        "schedule entered in local time instead of UTC, patrol ran at the wrong hour",
        "service cover left open after inspection",
        "new guard did not know how to release the brake, called support",
    ],
    "routine": [
        "shift handover, nothing to report",
        "all good today",
        "weekly visual inspection done, no issues",
        "wiped down sensors and body, all ok",
        "firmware update applied, unit back in service",
        "normal patrol, no incidents",
        "heavy rain today, patrol completed as planned",
        "wheels and fasteners checked ok",
        "dock area swept",
        "unit cleaned after a muddy shift, running fine",
        "routine check complete, battery, temperatures and wheels fine",
        "no alarms during the night shift",
        "visitors asked about the robot, no issues",
        "monthly inventory check, unit present and working",
        "quiet shift",
        "checked for noise and vibration, nothing unusual",
        "charging normally and ran the full shift",
        "temperatures normal all day",
        "connection stable, no dropouts",
        "handover done, keys and charger in place",
    ],
    "vague": [
        "something is off with this unit",
        "not running right today",
        "acting weird again",
        "please have a look when you can",
        "same problem as yesterday",
        "does not feel right, not sure what it is",
        "issue again, see earlier note",
        "behaving strangely since this morning",
        "needs attention",
        "not happy with how it is running",
    ],
}

# --- text-only minor issues (exist in the corpus, not in telemetry) ----------
MINOR: dict[str, list[str]] = {
    "battery": [
        "charge contacts dirty, cleaned them and it charges again",
        "dock pins corroded, intermittent charging",
        "battery connector was loose, reseated",
        "charger cable damaged near the plug",
        "battery door seal worn, moisture warning",
    ],
    "drivetrain": [
        "tyre tread worn on the {side} wheel",
        "plastic wrap tangled around the axle",
        "wheel nut loose on the {side} side, tightened",
        "small stone stuck in the wheel well",
        "drive belt cover cracked",
    ],
    "thermal": [
        "fan intake filter clogged with leaves, cleaned",
        "sticker was covering the vent",
        "dust build-up on the heat sink, blown out",
        "air duct clip broken, duct hanging loose",
        "intake grille bent after a bump",
    ],
}

# --- field-service tickets for a simulated fault ----------------------------
SERVICE_OPEN: dict[str, list[str]] = {
    "battery": [
        "Work order opened: reduced runtime and voltage sag under load. Suspect weak cell group in the battery pack.",
        "Work order opened: battery capacity complaint. Run pack diagnostics and capacity test.",
        "Work order opened: unit returns to dock early. Check battery internal resistance.",
    ],
    "drivetrain": [
        "Work order opened: abnormal vibration and noise from the drive unit. Suspect wheel bearing.",
        "Work order opened: grinding from wheel hub. Inspect bearings and axle.",
        "Work order opened: vibration alarm on drivetrain. Check {side} bearing play.",
    ],
    "thermal": [
        "Work order opened: motor running hot under normal load. Suspect cooling fan.",
        "Work order opened: repeated overheating warnings. Inspect fan, duct and heat sink.",
        "Work order opened: motor temperature rise above fleet average. Check cooling path.",
    ],
}
SERVICE_CLOSE: dict[str, list[str]] = {
    "battery": [
        "Replaced battery pack. Capacity test passed. Unit returned to service.",
        "Replaced weak cell module. Internal resistance back in range. Returned to service.",
        "Battery pack swapped, charge and discharge test ok.",
    ],
    "drivetrain": [
        "Replaced {side} wheel bearing. Test drive ok, vibration back to normal.",
        "Replaced drive hub bearing. No noise on test loop. Returned to service.",
        "Bearing replaced and axle regreased, vibration check passed.",
    ],
    "thermal": [
        "Replaced cooling fan. Motor temperature normal on test patrol.",
        "Replaced fan and cleaned duct. Thermal test passed. Returned to service.",
        "Cooling fan swapped, heat sink cleaned, temperatures back in range.",
    ],
}
DRIFT_SERVICE = [
    "Motor temperature sensor reads high against reference thermometer. Logged for recalibration.",
    "Thermistor offset confirmed with IR gun while docked. Sensor flagged for replacement at next service.",
    "Temperature sensor drift confirmed, motor itself is fine. Calibration ticket raised.",
]
NFF_CLOSE = [
    "Inspected for reported {topic} issue, could not reproduce. No fault found.",
    "Bench test passed, all readings in spec. No fault found, returned to service.",
    "Checked {topic}, everything within limits. No fault found.",
    "Followed up on the complaint about {topic}, unit runs normally. No fault found.",
]
NFF_TOPIC = {"battery": "battery", "drivetrain": "noise and vibration", "thermal": "overheating"}
NFF_HEDGE = [
    "guard says {core}, could not see it myself",
    "{core}? only happened once",
    "thought {core} but it seems fine now",
    "customer reports {core}, not confirmed",
    "possibly {core}, hard to tell",
]

# --- misattribution: a real fault described with another subsystem's wording --
# (source fault category -> wording); each has a physical reason
MISATTRIBUTED: dict[str, list[str]] = {
    # bearing friction raises drive power, so the first thing noticed is shorter runtime
    "drivetrain": [
        "battery seems to drain faster, runtime shorter than last week",
        "comes back to dock earlier than it used to",
    ],
    # degraded cooling shows up as a high reading that is easy to blame on the sensor
    "thermal": [
        "temperature reading high, maybe the sensor",
        "motor temperature number looks wrong, probably a bad sensor",
    ],
    # a weak cell fails to reach full charge, which looks like a docking mistake
    "battery": [
        "did not charge fully, maybe it was not docked right",
        "found at {pct}% in the morning, someone may have unplugged the charger",
    ],
}

PREFIX = [
    "{unit}: ",
    "{unit} - ",
    "{shift} shift: ",
    "FYI ",
    "during patrol ",
    "after the morning round ",
    "note for maintenance: ",
    "again today ",
]
SUFFIX = [
    "please check",
    "will monitor",
    "logging for maintenance",
    "second time this week",
    "not urgent",
    "needs a look asap",
    "no error code shown",
    "reported by the guard on site",
    "same as last shift",
    "unit kept in service",
]
SLOT_VALUES: dict[str, list[str]] = {
    "side": ["left", "right", "front left", "front right", "rear left", "rear right"],
    "place": ["loading bay", "north gate", "car park", "east fence", "main entrance", "warehouse corner"],
    "shift": ["morning", "day", "evening", "night"],
}

# --- rule-based paraphrase (stand-in for LLM paraphrase, see docs/W03_Log_Corpus.md) --
SYNONYMS: dict[str, list[str]] = {
    "unit": ["unit", "rover", "robot"],
    "noise": ["noise", "sound"],
    "very": ["very", "really"],
    "faster": ["faster", "quicker"],
    "than usual": ["than usual", "than normal", "than before"],
    "noticeably": ["noticeably", "clearly"],
    "dashboard": ["dashboard", "display", "screen"],
    "came back": ["came back", "returned"],
    "seems": ["seems", "looks"],
    "issue": ["issue", "problem"],
}

ABBREVIATIONS: dict[str, str] = {
    "battery": "batt",
    "temperature": "temp",
    "temperatures": "temps",
    "charging": "chg",
    "charge": "chg",
    "charger": "chgr",
    "motor": "mtr",
    "connection": "conn",
    "vibration": "vib",
    "vibrates": "vibs",
    "please": "pls",
    "with": "w/",
    "without": "w/o",
    "about": "abt",
    "minutes": "min",
    "inspection": "insp",
    "maintenance": "maint",
    "operator": "op",
    "firmware": "fw",
    "replaced": "repl",
    "checked": "chkd",
    "sensor": "snsr",
    "bearing": "brg",
    "and": "&",
    "something": "smth",
    "overheating": "o/heat",
    "telemetry": "telem",
    "dashboard": "dash",
    "signal": "sig",
    "second": "2nd",
    "through": "thru",
    "because": "bc",
    "normal": "nrml",
    "problem": "prob",
    "reading": "rdg",
    "percentage": "pct",
}
PHRASE_ABBREVIATIONS: dict[str, str] = {
    "no fault found": "NFF",
    "work order": "WO",
    "returned to dock": "RTD",
    "returned to service": "RTS",
    "state of health": "SoH",
}


def phrase_ids() -> list[str]:
    """Every core phrase id (``group:index``) in a stable order."""
    ids = [f"{g}:{i}" for g, items in SYMPTOM.items() for i in range(len(items))]
    ids += [f"minor_{g}:{i}" for g, items in MINOR.items() for i in range(len(items))]
    ids += [f"misattributed_{g}:{i}" for g, items in MISATTRIBUTED.items() for i in range(len(items))]
    return ids
