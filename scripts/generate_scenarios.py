"""Generate DRAFT change scenarios from templates (fixed seed).

    python scripts/generate_scenarios.py                      # writes data/scenarios/drafts.jsonl
    python scripts/generate_scenarios.py --paraphrase         # also rewrites turn wording with the LLM

Drafts have reviewed=false. Check them with scripts/review_scenarios.py, which
copies accepted ones into data/scenarios/change_scenarios.jsonl.

The first batch (COUNTS, seed 42) is the original 34 drafts and never changes. A second batch
(EXTRA_COUNTS, seed 43) adds more, including cancelled bookings, numbered after the first.
With the 6 hand-written scenarios that gives 210. Drafts already in the file keep their text and
reviewed mark; only new ids are added, so hand edits are never overwritten.
--paraphrase only rewrites turn text; a rewrite that loses any fact value is discarded.
"""

import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.scenarios import DRAFTS, Scenario, read_jsonl, with_past_question, write_jsonl  # noqa: E402

SEED = 42
COUNTS = {"change": 21, "additive": 7, "restate": 3, "conflict": 3}
EXTRA_SEED = 43
EXTRA_COUNTS = {"change": 60, "cancel": 30, "additive": 40, "restate": 30, "conflict": 10}

NAMES = ["Ali", "Hina", "Bilal", "Zoya", "Ayesha", "Danish", "Usman", "Fatima", "Sana", "Omar",
         "Maryam", "Hamza", "Iqra", "Saad", "Noor", "Raza", "Amna", "Faisal", "Mehwish", "Kamran"]

CITIES = ["Karachi", "Lahore", "Islamabad", "Peshawar", "Multan", "Quetta", "Faisalabad", "Dubai", "London", "Toronto"]
COMPANIES = [("Meezan Bank", ["meezan"]), ("Systems Ltd", ["systems limited"]), ("Engro", []),
             ("Careem", []), ("Jazz", []), ("Daraz", []), ("Habib Bank", ["hbl"]), ("Netsol", [])]
TITLES = ["teacher", "nurse", "accountant", "software engineer", "designer", "pharmacist", "lawyer", "chef"]
UNIS = [("LUMS", []), ("NUST", []), ("FAST", []), ("Karachi University", ["university of karachi"]),
        ("IBA", []), ("GIKI", [])]
PARTNERS = ["Hassan", "Rabia", "Junaid", "Sadia", "Tariq", "Nida", "Waleed", "Anam"]
DRINKS = ["coffee", "green tea", "chai", "orange juice", "lassi", "hot chocolate"]
FOODS = ["biryani", "nihari", "sushi", "pizza", "haleem", "pasta", "karahi"]
SPORTS = ["cricket", "football", "tennis", "badminton", "squash", "swimming"]
HOBBIES = ["hiking", "painting", "photography", "gardening", "chess", "baking"]
PETS = ["dog", "cat", "parrot", "rabbit", "hamster", "turtle"]
SKILLS = ["guitar", "python", "urdu calligraphy", "photoshop", "sign language", "public speaking"]
ALLERGIES = ["peanuts", "shellfish", "dust", "pollen", "gluten", "penicillin"]
KIDS = ["Ahmed", "Zara", "Ibrahim", "Hania", "Musa", "Alina"]
BOOKINGS = ["flight to Dubai", "hotel room in Murree", "table at Monal", "dentist appointment", "trip to Hunza",
            "concert tickets", "car rental", "train to Lahore", "wedding hall booking", "gym membership"]
# A cancelled booking is current when the answer says so; it is stale when it says the booking still stands.
CANCELLED = ("cancelled", ["canceled", "called off"])
STILL_ON = ["still booked", "still on", "is booked", "confirmed", "going ahead", "is scheduled"]

OPENERS = ["Hey {name}, how have you been?", "Hi {name}! What's new?", "Good to see you, {name}. How are things?"]
REPLIES = ["That's great to hear!", "Oh nice, tell me more.", "Wow, good for you.", "Interesting!"]
FILLERS = [
    [("o", "Did you watch the match last night?"), ("p", "Yes, what a finish, I couldn't sleep after it.")],
    [("o", "The weather has been crazy this week."), ("p", "Tell me about it, it rained all day yesterday.")],
    [("o", "Any plans for the weekend?"), ("p", "Not really, just resting at home.")],
    [("o", "Have you seen the new season of that drama?"), ("p", "Not yet, no spoilers please!")],
    [("o", "Traffic was terrible this morning."), ("p", "Same here, it took me an hour to get anywhere.")],
]

# Each template: kind, predicate, a value picker, session-1 lines, later lines, questions.
# {v} is the first value, {w} the later one. Later lines are grouped by wording style.
TEMPLATES = [
    dict(kind="change", predicate="lives_in", pick=lambda r: _two(r, CITIES),
         first=["I've been living in {v} for three years now.", "I live in {v}, near my parents."],
         later={"explicit": ["I finally moved to {w} last month!", "We relocated to {w} a few weeks ago."],
                "implicit": ["My new flat in {w} is tiny but cozy.", "These days I live in {w}, it's a change of pace."]},
         q=["Where does {name} live now?"], q_alt=["Which city is {name} currently based in?"]),
    dict(kind="change", predicate="works_at", pick=lambda r: _two(r, COMPANIES),
         first=["I work at {v}, it keeps me busy.", "I've been at {v} for two years."],
         later={"explicit": ["I switched jobs, I'm at {w} now.", "I left {v} and joined {w}."],
                "implicit": ["My new manager at {w} is really kind.", "These days I work at {w}, the hours are better."]},
         q=["Where does {name} work now?"], q_alt=["Which company is {name} employed at these days?"]),
    dict(kind="change", predicate="job_title", pick=lambda r: _two(r, TITLES),
         first=["I work as a {v}.", "I'm a {v}, I really enjoy it."],
         later={"explicit": ["I'm no longer a {v}, I retrained and now work as a {w}.", "I switched careers, I'm a {w} now."],
                "implicit": ["My new job as a {w} is exciting.", "These days I work as a {w}."]},
         q=["What is {name}'s job now?"], q_alt=["What does {name} currently do for a living?"]),
    dict(kind="change", predicate="studies_at", pick=lambda r: _two(r, UNIS),
         first=["I study at {v}, the campus is lovely.", "I'm a student at {v}."],
         later={"explicit": ["I transferred to {w} this semester.", "I moved from {v} to {w} for my degree."],
                "implicit": ["My new classes at {w} are tough.", "These days I study at {w}."]},
         q=["Where does {name} study now?"], q_alt=["Which university is {name} attending these days?"]),
    dict(kind="change", predicate="relationship_status",
         pick=lambda r: r.choice([("single", "engaged"), ("engaged", "married"), ("single", "married")]),
         first=["I'm {v} at the moment.", "Right now I'm {v}."],
         later={"explicit": ["I'm no longer {v}, I'm {w} now!", "Big news, I got {w}!"],
                "implicit": ["These days I'm {w}, life feels different.", "Being {w} now is wonderful."]},
         q=["What is {name}'s relationship status now?"], q_alt=["Is {name} single, engaged or married at present?"]),
    dict(kind="change", predicate="partner", pick=lambda r: _two(r, PARTNERS),
         first=["My partner {v} and I went to the beach.", "I'm dating {v}, we met at work."],
         later={"explicit": ["{v} and I broke up; I'm with {w} now.", "I'm no longer with {v}, I'm seeing {w}."],
                "implicit": ["My new partner {w} cooks amazing food.", "These days I'm with {w}, we're very happy."]},
         q=["Who is {name}'s partner now?"], q_alt=["Who is {name} currently in a relationship with?"]),
    dict(kind="change", predicate="age", pick=lambda r: (lambda n: (str(n), str(n + 1)))(r.randint(22, 45)),
         first=["I just turned {v} last week.", "I'm {v} years old."],
         later={"explicit": ["I turned {w} yesterday, time flies!", "I'm {w} now, we just celebrated."],
                "implicit": ["These days I'm {w}, feeling older already.", "My new age is {w}, can you believe it?"]},
         q=["How old is {name} now?"], q_alt=["What is {name}'s current age?"]),
    dict(kind="change", predicate="likes", pick=lambda r: _two(r, DRINKS),
         first=["I love {v}, I have it every morning.", "{v} is my favourite drink."],
         later={"explicit": ["I no longer drink {v}; I switched to {w}.", "I used to love {v}, now it's {w} for me."],
                "implicit": ["These days I drink {w} instead of {v}.", "My new favourite drink is {w}, not {v}."]},
         q=["What drink does {name} like now?"], q_alt=["These days, which drink does {name} enjoy?"]),
    dict(kind="change", predicate="hobby", pick=lambda r: _two(r, SPORTS),
         first=["I play {v} every weekend.", "{v} is my favourite way to relax."],
         later={"explicit": ["I quit {v} and took up {w}.", "I stopped playing {v}, I switched to {w}."],
                "implicit": ["These days I play {w} instead of {v}.", "My new hobby is {w}; I gave up {v}."]},
         q=["What sport does {name} play now?"], q_alt=["Which sport is {name} into these days?"]),
    dict(kind="additive", predicate="has_pet", pick=lambda r: _two(r, PETS),
         first=["I have a {v}, it's so playful.", "My {v} keeps me company."],
         later={"tricky": ["I also got a {w} as well!", "We adopted a {w} too, the {v} is still adjusting."]},
         q=["What pets does {name} have?"], q_alt=["Which animals does {name} keep at home?"]),
    dict(kind="additive", predicate="likes", pick=lambda r: _two(r, FOODS),
         first=["I love {v}, I could eat it every day.", "{v} is my favourite food."],
         later={"tricky": ["I started eating {w} as well, it's great.", "I've also become a big fan of {w}."]},
         q=["What foods does {name} like?"], q_alt=["Which dishes does {name} enjoy?"]),
    dict(kind="additive", predicate="has_skill", pick=lambda r: _two(r, SKILLS),
         first=["I can do {v} pretty well.", "I learned {v} years ago."],
         later={"tricky": ["I also learned {w} this year.", "These days I'm also good at {w}."]},
         q=["What skills does {name} have?"], q_alt=["What is {name} able to do well?"]),
    dict(kind="additive", predicate="hobby", pick=lambda r: _two(r, HOBBIES),
         first=["I love {v} on weekends.", "{v} is my favourite hobby."],
         later={"tricky": ["Lately I've also been doing {w}.", "I picked up {w} as well."]},
         q=["What are {name}'s hobbies?"], q_alt=["What does {name} do for fun?"]),
    dict(kind="additive", predicate="allergic_to", pick=lambda r: _two(r, ALLERGIES),
         first=["I'm allergic to {v}, so I have to be careful.", "{v} gives me a bad allergy."],
         later={"tricky": ["Turns out I'm also allergic to {w}.", "The doctor said I'm allergic to {w} too."]},
         q=["What is {name} allergic to?"], q_alt=["Which allergies does {name} have?"]),
    dict(kind="additive", predicate="has_child", pick=lambda r: _two(r, KIDS),
         first=["My kid {v} started school this year.", "My child {v} is learning to read."],
         later={"tricky": ["We just welcomed a new baby, {w}!", "Our second child {w} was born last week."]},
         q=["What are the names of {name}'s children?"], q_alt=["Who are {name}'s kids?"]),
    dict(kind="restate", predicate="lives_in", pick=lambda r: (lambda c: (c, c))(r.choice(CITIES)),
         first=["I live in {v}, near my parents."],
         later={"restate": ["{v} is still home for me, I love it.", "Living in {v} is still great."]},
         q=["Where does {name} live now?"], q_alt=["Which city is {name}'s home at the moment?"]),
    dict(kind="restate", predicate="works_at", pick=lambda r: (lambda c: (c, c))(r.choice(COMPANIES)),
         first=["I work at {v}, it keeps me busy."],
         later={"restate": ["Work at {v} was busy again today.", "I'm still at {v}, it's going well."]},
         q=["Where does {name} work now?"], q_alt=["Which company does {name} work for these days?"]),
    dict(kind="restate", predicate="likes", pick=lambda r: (lambda c: (c, c))(r.choice(FOODS)),
         first=["I love {v}, I could eat it every day."],
         later={"restate": ["Had {v} again yesterday, still my favourite.", "Nothing beats {v} for me."]},
         q=["What food does {name} love?"], q_alt=["Which dish is {name}'s favourite?"]),
    dict(kind="conflict", predicate="lives_in", pick=lambda r: _two(r, CITIES),
         first=["I live in {v}, near the old city."],
         later={"conflict": ["I love my home in {w}, it is so peaceful.", "My house in {w} has a lovely garden."]},
         q=["Where does {name} live now?"], q_alt=["Which city is {name}'s home at the moment?"]),
    dict(kind="conflict", predicate="works_at", pick=lambda r: _two(r, COMPANIES),
         first=["I work at {v}, it keeps me busy."],
         later={"conflict": ["Work at {w} was exhausting today.", "My desk at {w} is right by the window."]},
         q=["Where does {name} work now?"], q_alt=["Which company does {name} work for?"]),
    dict(kind="conflict", predicate="job_title", pick=lambda r: _two(r, TITLES),
         first=["I'm a {v}, I really enjoy it."],
         later={"conflict": ["As a {w}, my shifts are long.", "Being a {w} is tiring some days."]},
         q=["What is {name}'s job now?"], q_alt=["What does {name} do for a living?"]),
]

for template in (
    dict(kind="cancel", predicate="has_booking", pick=lambda r: (r.choice(BOOKINGS), CANCELLED),
         first=["I booked a {v} for next Friday.", "I just made a booking: a {v} for the weekend."],
         later={"explicit": ["I cancelled the {v}, plans changed.", "The {v} is cancelled, I couldn't make it work."],
                "implicit": ["Bad news, the {v} got cancelled.", "Sadly they cancelled my {v} at the last minute."]},
         q=["What is the status of {name}'s {v}?"], q_alt=["Is {name}'s {v} still going ahead?"]),
):
    TEMPLATES.append(template)

LABELS = {"change": "corrective", "cancel": "corrective", "additive": "additive", "restate": "redundant",
          "conflict": "contradictory"}


def _two(rng, pool):
    a, b = rng.sample(pool, 2)
    return a, b


def _value(v):
    """(value, aliases) from a pool entry."""
    return (v[0], v[1]) if isinstance(v, tuple) and isinstance(v[1], list) else (v, [])


def _fmt_date(d: date, rng) -> str:
    hour, minute = rng.randint(1, 12), rng.choice(range(0, 60, 5))
    return f"{hour}:{minute:02d} {rng.choice(['am', 'pm'])} on {d.day} {d.strftime('%B')}, {d.year}"


def make(template, idx, rng, avoid=None) -> Scenario:
    for _ in range(50):  # avoid repeating a person and values already used with this template
        name, other = rng.sample(NAMES, 2)
        (v, v_alias), (w, w_alias) = map(_value, template["pick"](rng))
        if avoid is None or (id(template), name, v, w) not in avoid:
            break
    if avoid is not None:
        avoid.add((id(template), name, v, w))
    style = rng.choice(sorted(template["later"]))
    fill = lambda text: text.format(v=v, w=w, name=name)  # noqa: E731
    first, later = fill(rng.choice(template["first"])), fill(rng.choice(template["later"][style]))
    first, later = first[0].upper() + first[1:], later[0].upper() + later[1:]
    speakers = {"p": name, "o": other}

    day = date(2023, 1, 1) + timedelta(days=rng.randint(0, 200))
    sessions = [{"date": _fmt_date(day, rng), "turns": [
        {"speaker": other, "text": rng.choice(OPENERS).format(name=name)},
        {"speaker": name, "text": first},
        {"speaker": other, "text": rng.choice(REPLIES)}]}]
    for filler in rng.sample(FILLERS, rng.randint(0, 2)):
        day += timedelta(days=rng.randint(7, 30))
        sessions.append({"date": _fmt_date(day, rng),
                         "turns": [{"speaker": speakers[s], "text": t} for s, t in filler]})
    day += timedelta(days=rng.randint(7, 30))
    sessions.append({"date": _fmt_date(day, rng), "turns": [
        {"speaker": other, "text": rng.choice(OPENERS).format(name=name)},
        {"speaker": name, "text": later},
        {"speaker": other, "text": rng.choice(REPLIES)}]})

    kind, pred, last = template["kind"], template["predicate"], len(sessions) - 1
    facts = [{"speaker": name, "predicate": pred, "value": v, "session": 0, "expected_label": "additive"},
             {"speaker": name, "predicate": pred, "value": w, "session": last,
              "expected_label": LABELS[kind], "replaces": v if kind in ("change", "cancel") else None}]
    lower = lambda xs: [x.lower() for x in xs]  # noqa: E731
    if kind == "change":
        current, old = lower([w] + w_alias), lower([v] + v_alias)
    elif kind == "cancel":  # naming the booking is fine; saying it still stands is stale
        current, old = lower([w] + w_alias), STILL_ON
    elif kind == "additive":
        current, old = lower([v, w]), []
    elif kind == "restate":
        current, old = lower([v] + v_alias), []
    else:  # conflict: scored on labels only
        current, old = lower([v, w]), []
    return Scenario(id=f"gen-{kind}-{idx:02d}", kind=kind, reviewed=False, sessions=sessions, facts=facts,
                    question=fill(rng.choice(template["q"])), question_alt=fill(rng.choice(template["q_alt"])),
                    current=current, old=old)


def generate(counts=COUNTS, seed=SEED, start=None, avoid=None):
    """Drafts for `counts`, numbered per kind from start[kind] + 1."""
    rng = random.Random(seed)
    out = []
    for kind, n in counts.items():
        templates = [t for t in TEMPLATES if t["kind"] == kind]
        first = (start or {}).get(kind, 0)
        for i in range(n):
            out.append(make(templates[(first + i) % len(templates)], first + i + 1, rng, avoid))
    return out


def generate_all():
    """The original batch, unchanged, then the extra batch numbered after it."""
    avoid = set()
    first = generate(avoid=avoid)
    return first + generate(EXTRA_COUNTS, EXTRA_SEED, start=COUNTS, avoid=avoid)


def merge(existing, new):
    """Existing drafts as they are (hand edits, reviewed marks), then drafts with new ids."""
    ids = {s.id for s in existing}
    return existing + [s for s in new if s.id not in ids]


def paraphrase(scenarios, llm):
    """Rewrite turn wording with the LLM. Facts never change; a rewrite that drops a value is discarded."""
    for sc in scenarios:
        values = [f.value for f in sc.facts] + [f.replaces for f in sc.facts if f.replaces]
        for session in sc.sessions:
            for turn in session.turns:
                keep = [v for v in values if v.lower() in turn.text.lower()]
                prompt = ("Rewrite this chat message in different words. Keep the same meaning, including any "
                          "sign that something changed or did not change."
                          + (f" Keep these exact words: {', '.join(keep)}." if keep else "")
                          + f"\nOutput only the rewritten message.\n\nMessage: {turn.text}")
                try:
                    new = llm.get_completion(prompt, temperature=0.7).strip().strip('"')
                except Exception:
                    continue
                if new and "\n" not in new and all(v.lower() in new.lower() for v in keep):
                    turn.text = new
    return scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DRAFTS)
    parser.add_argument("--paraphrase", action="store_true")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    args = parser.parse_args()

    existing = read_jsonl(args.out)
    new = [s for s in generate_all() if s.id not in {e.id for e in existing}]
    if args.paraphrase:
        from tsom.llm import LLM
        llm = LLM(args.backend, args.model)
        llm.check()
        new = paraphrase(new, llm)
    scenarios = merge(existing, [with_past_question(s) for s in new])
    write_jsonl(args.out, scenarios)
    print(f"{len(scenarios)} drafts in {args.out} ({len(new)} new) from {len(TEMPLATES)} templates")


if __name__ == "__main__":
    main()
