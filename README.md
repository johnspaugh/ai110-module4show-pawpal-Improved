# 🐾 PawPal+ — AI-Assisted Pet Care Planner (Module 4)

PawPal+ is a Python and Streamlit app that turns a pet owner's task list into a realistic, explained daily plan. This Module 4 version adds an **AI planning assistant**: it retrieves pet-care guidance, proposes tasks grounded in that guidance, **checks its own work** (time budget, essential tasks, overlaps), retries when the check fails, and only saves anything after the owner accepts.

## Original Project (Modules 1–3)

This project builds on my **Music Recommender Simulation** from Module 3. That project loaded an 18-song catalog from a CSV file, scored every song against a listener's taste profile (genre and mood matches plus how close each song's energy, acousticness, valence, danceability and tempo were to the listener's targets), and printed a ranked top 5 with the reasons behind each score. PawPal+ reuses the same core idea (score and rank items against constraints, then explain the result in plain language) and Module 4 adds an AI layer on top.

## Why It Matters

Busy owners miss walks, medicine and grooming because nothing tells them what fits in the day. PawPal+ answers that directly: given a time budget and every pet's tasks, which tasks happen, in what order, which are left out, and why. The AI assistant removes the blank-page problem (what does a senior dog actually need each day?) while keeping a human in control and every suggestion traceable to a source.

## Features

**Scheduling core (Modules 1–3 style logic, in `pawpal_system.py`)**

- **Multiple pets, one owner:** all pets' tasks are pooled so the scheduler plans across every pet at once.
- **Priority-based daily plan:** pending tasks are sorted high to low priority, shortest first within a priority, then added while they fit the daily minutes. Tasks that don't fit are listed as skipped, never dropped silently.
- **Plan explanation:** `explain_plan()` reports minutes used, the order chosen, what was skipped and, for AI suggestions, the source of each task. `explain_schedule()` is the plain-language version shown for AI plans: tasks in time order with start times, a reason for each, and numbered footnotes that quote what the guides say.
- **Sorting, filtering and conflict warnings:** by `HH:MM` start time, by pet and status, and overlap detection that names both tasks, both pets and both time ranges.
- **Recurring tasks:** completing a daily, weekly or monthly task creates the next occurrence.

**AI layer (Module 4)**

- **Retrieval-Augmented Generation (RAG):** before the model proposes anything, `rag.py` searches a small knowledge base (`knowledge/`) for guidance matching each pet's species and life stage, plus the owner's notes. The model is told to build tasks **only** from that guidance, and every task must cite the guide section it came from. If nothing relevant is found, no tasks are suggested.
- **Agentic workflow:** `agent.py` runs plan, retrieve, act, check and retry. It adds proposed tasks to a sandbox copy of the owner, runs the real `Scheduler`, and tests its own result. Problems are sent back to the model for up to 3 attempts.
- **Guardrails:** strict output validation (all-or-nothing), an iteration limit, a sandbox so a bad proposal cannot touch real data, human accept/reject before saving, medication doses never chosen by the model, and a non-AI fallback if the model call fails.
- **Logging:** every step, tool call and retry is written to `pawpal.log` and the console.
- **Confidence score:** each proposal gets a 0 to 1 heuristic confidence (from retries, missing guidance and unresolved problems), shown in the app. It is not the model rating itself.
- **Demo mode:** with no API key, a rule-based stand-in reads the same retrieved guidance, so the whole app runs and is testable without paying for or exposing a key.

## Architecture Overview

The system diagram is in [`diagrams/architecture.mmd`](diagrams/architecture.mmd) (Mermaid source; paste it into [mermaid.live](https://mermaid.live) to view it). Class diagrams are in [`diagrams/uml_final.mmd`](diagrams/uml_final.mmd) and [`diagrams/uml_draft.mmd`](diagrams/uml_draft.mmd).

**Input → Process → Output, with checks:**

1. **Input.** The owner enters pets and tasks through Streamlit forms and may type free-text notes for the assistant.
2. **Process.** For each pet the agent builds a query (species plus life stage) and the **retriever** returns the best-matching guide sections. The agent sends the guidance and the owner's time budget to the **LLM** and gets back JSON. The **validator** rejects anything malformed or any task that cites a source that was not retrieved. Valid tasks are added to a **sandbox copy** of the owner and the real **Scheduler** builds a plan. The **evaluator** then checks the result: no essential task skipped for lack of time, and no overlaps among planned tasks. If it finds problems, the agent sends them back to the model and tries again.
3. **Output.** The owner sees a proposal with its sources, the checker's findings, the agent's step-by-step trace and the retrieved guidance. Nothing is saved until the owner clicks **Accept**.
4. **Where humans and tests check the AI.** The human accept/reject step, the evaluator, the validator, the fallback path and the pytest suite (which drives the agent with a scripted fake model) are all shown in the diagram.

| File | Role |
|---|---|
| `app.py` | Streamlit UI, including the AI Plan Assistant with accept/reject |
| `agent.py` | Planning agent, output validator, evaluator, `commit_proposal` |
| `rag.py` | Knowledge loading, chunking and keyword retrieval |
| `llm.py` | Prompt format, live Anthropic client, keyless demo model |
| `logging_setup.py` | Logging to `pawpal.log` and the console |
| `pawpal_system.py` | `Task`, `Pet`, `Owner`, `Scheduler` |
| `knowledge/` | Pet-care guides (dog, cat, senior pets, medication, other pets) |
| `agent_demo.py`, `main.py` | Terminal demos of the agent and the scheduler |
| `evaluate.py` | Reliability evaluation: labeled retrieval set, agent scenarios, consistency |
| `app_walkthrough.py` | Drives the real Streamlit app headlessly through the human-evaluation scenarios |
| `tests/` | 129 pytest tests |

## Setup Instructions

Requires Python 3.10 or newer (developed and tested on 3.13).

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd ai110-module4show-pawpal-Improved

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (streamlit, pytest, anthropic, python-dotenv)
pip install -r requirements.txt

# 4. (Optional) use the live model instead of demo mode
cp .env.example .env             # Windows: copy .env.example .env
#    then open .env and set ANTHROPIC_API_KEY=your-key

# 5. Run the web app
streamlit run app.py

# 6. Or run the terminal demos
python agent_demo.py             # the AI agent
python main.py                   # the scheduler

# 7. Run the tests and the reliability evaluation
python -m pytest
python evaluate.py
python app_walkthrough.py        # drives the app through the evaluation scenarios
```

**Demo mode vs. live mode.** With no `ANTHROPIC_API_KEY` the app runs in demo mode, and the app caption says so. With a key it uses the live model (default `claude-opus-5-5`; set `PAWPAL_MODEL` in `.env` to use another). Set `PAWPAL_DEMO=1` to force demo mode. `.env` and `pawpal.log` are git-ignored.

**Using the assistant in the app:** add an owner and pets, optionally type notes (for example "Biscuit takes arthritis medicine every morning"), press **Generate AI plan**, review the proposal, then **Accept** or **Reject**.

## Sample Interactions

All outputs below were produced by running the code in demo mode (no API key) on this machine. The demo model is a rule-based stand-in, so these show the retrieval, validation, checking and retry logic working; they are not output from the live model.

### 1. Agent plans for two pets, finds an overlap, retries and passes

**Input:** owner Sally with 90 minutes a day, Biscuit (dog, age 10), Mochi (cat, age 5) with one existing task, and the note "Biscuit takes arthritis medicine every morning. Keep walks gentle." Run with `PAWPAL_DEMO=1 python agent_demo.py`.

```
Owner notes: Biscuit takes arthritis medicine every morning. Keep walks gentle.

Model mode: demo   attempts: 2   checks passed: True   confidence: 0.85 (high)

What the agent did:
  - PLAN: 2 pet(s), 90 min/day, model mode=demo
  - RETRIEVE: Biscuit (dog, 10) -> senior_pets.md: Senior pets, dog_care.md: Senior dogs, medication_basics.md: Medication
  - RETRIEVE: Mochi (cat, 5) -> cat_care.md: Adult cats
  - VALIDATE (attempt 1): accepted 11 task(s)
  - CHECK (attempt 1): 1 problem(s): Biscuit's Feeding (08:00-08:10) overlaps Mochi's Feeding (08:00-08:10)
  - RETRY: sending the problems back to the model
  - VALIDATE (attempt 2): accepted 11 task(s)
  - CHECK (attempt 2): passed (time budget, essential tasks, no overlaps)

Proposed tasks:
  07:30  Biscuit: Gentle morning walk (15 min, high)  [dog_care.md: Senior dogs]
  08:00  Biscuit: Feeding (10 min, high)  [dog_care.md: Senior dogs]
  08:10  Mochi: Feeding (10 min, high)  [cat_care.md: Adult cats]
  08:20  Biscuit: Give prescribed medication (5 min, high)  [medication_basics.md: Medication]
  08:30  Mochi: Litter box cleaning (10 min, high)  [cat_care.md: Adult cats]
  18:00  Biscuit: Gentle evening walk (15 min, medium)  [dog_care.md: Senior dogs]
  18:30  Mochi: Interactive play (15 min, medium)  [cat_care.md: Adult cats]
  19:30  Biscuit: Joint comfort check (5 min, medium)  [dog_care.md: Senior dogs]
  19:35  Mochi: Brushing (5 min, low)  [cat_care.md: Adult cats]
  20:00  Biscuit: Comfort and mobility check (5 min, medium)  [senior_pets.md: Senior pets]
  20:30  Biscuit: Fresh water and bed check (5 min, medium)  [senior_pets.md: Senior pets]

Why this plan?
Plan for Sally: 90 of 90 minutes used.
The list is in time order. Tasks were chosen by priority (high first), then shortest first, and added while they still fit in the day.

07:30-07:45  Biscuit: Gentle morning walk (15 min)
    Why: high priority; essential, so it should not be skipped; suggested by guide [1]
08:00-08:10  Biscuit: Feeding (10 min)
    Why: high priority; essential, so it should not be skipped; suggested by guide [1]
08:10-08:20  Mochi: Feeding (10 min)
    Why: high priority; essential, so it should not be skipped; suggested by guide [2]
08:20-08:25  Biscuit: Give prescribed medication (5 min)
    Why: high priority; essential, so it should not be skipped; suggested by guide [3]
08:30-08:40  Mochi: Litter box cleaning (10 min)
    Why: high priority; essential, so it should not be skipped; suggested by guide [2]
12:00-12:10  Mochi: Vet visit reminder (10 min)
    Why: medium priority; a task you added yourself
18:00-18:15  Biscuit: Gentle evening walk (15 min)
    Why: medium priority; suggested by guide [1]
19:30-19:35  Biscuit: Joint comfort check (5 min)
    Why: medium priority; suggested by guide [1]
20:00-20:05  Biscuit: Comfort and mobility check (5 min)
    Why: medium priority; suggested by guide [4]
20:30-20:35  Biscuit: Fresh water and bed check (5 min)
    Why: medium priority; suggested by guide [4]

What the guides say:
  [1] dog_care.md: Senior dogs: "Senior dogs (about 8 years and older) usually do better with shorter, gentler walks spread across the day, and with extra attention to joints and weight."
  [2] cat_care.md: Adult cats: "Healthy adult cats (about 2 to 9 years old) need daily meals, a clean litter box, and play to stay active."
  [3] medication_basics.md: Medication: "Medicine, pills and other prescribed treatments (for example arthritis medicine, heartworm pills or flea medication) must follow the vet's instructions exactly."
  [4] senior_pets.md: Senior pets: "Older pets of any species (a senior dog, senior cat or other older animal) benefit from a short daily comfort check: how they move, whether they eat and drink normally, and whether their bed is easy to reach."

Left out because there was not enough time left:
  - Mochi: Interactive play (15 min, medium priority)
  - Mochi: Brushing (5 min, low priority)
```

**What it shows:** the senior dog got senior-dog and medication guidance while the adult cat got only adult-cat guidance, because the note was matched to Biscuit only. The first proposal put both pets' feeding at 08:00; the evaluator caught the overlap, the agent retried, and the second proposal passed (confidence 0.85, lowered by the one retry). Every task cites its guide section.

### 2. Guardrail: no guidance means no guessing

**Input:** one pet, a goldfish named Bubbles, which the knowledge base does not cover.

```
PLAN: 1 pet(s), 60 min/day, model mode=demo
RETRIEVE: no guidance found for Bubbles (fish); no tasks will be suggested for this pet
STOP: nothing retrieved, so the model was not called
Problems shown to the owner: ['No guidance was found for any of your pets, so nothing was suggested.']
```

**What it shows:** when retrieval finds nothing, the model is never called and nothing is invented.

### 3. Fallback: the model call fails

**Input:** Biscuit with one existing task, and a simulated model outage.

```
PLAN: 1 pet(s), 60 min/day, model mode=live
RETRIEVE: Biscuit (dog, 10) -> senior_pets.md: Senior pets, dog_care.md: Senior dogs
ERROR: model call failed (RuntimeError: simulated outage); using fallback
FALLBACK: scheduled existing tasks without AI suggestions
Problems shown to the owner: ['The AI step failed (RuntimeError). Showing your existing tasks scheduled normally instead.']
Plan: ['Morning walk']
```

**What it shows:** a failed AI step never crashes the app; the owner's existing tasks are scheduled the normal way and the reason is shown.

### 4. Core scheduling: conflicts, sorting and recurrence

Output of `python main.py` (seven tasks across two pets in a 90-minute day; excerpt):

```
====================================================
  Today's Schedule for Sally
  Pets: Robert, Mochi
  Time used: 75 of 90 min
====================================================
#  Pet      Task               Min  Priority Repeats
----------------------------------------------------
1  Robert   Heartworm pill       5  high     daily
2  Mochi    Flea medication      5  high     monthly
3  Mochi    Feeding             10  high     daily
4  Robert   Morning walk        30  high     daily
5  Mochi    Playtime            20  medium   daily
6  Mochi    Brushing             5  low      daily
----------------------------------------------------

Sorted by time:
  07:30  Mochi: Playtime
  08:00  Robert: Morning walk
  08:00  Robert: Heartworm pill
  08:15  Mochi: Feeding
  08:15  Mochi: Brushing
  17:00  Robert: Grooming
  --:--  Mochi: Flea medication


Conflict check:
  Conflict (same pet): Robert's Morning walk (08:00-08:30) overlaps Robert's Heartworm pill (08:00-08:05)
  Conflict (different pets): Robert's Morning walk (08:00-08:30) overlaps Mochi's Feeding (08:15-08:25)
  Conflict (different pets): Robert's Morning walk (08:00-08:30) overlaps Mochi's Brushing (08:15-08:20)
  Conflict (same pet): Mochi's Feeding (08:15-08:25) overlaps Mochi's Brushing (08:15-08:20)
```

## Design Decisions and Trade-offs

- **RAG over fine-tuning or "just ask the model".** The guidance is in small text files anyone can read and correct, and the model is required to cite them. The trade-off is a small knowledge base: it can miss, and it is general guidance, not veterinary advice.
- **Plain keyword retrieval, not a vector database.** No extra services or libraries, and results are identical every run, which makes tests reliable. It gives up semantic matching: it relies on shared words, so I added a species filter, a synonym list ("medicine" = "medication"), extra weight for life-stage words, and a rule that only distinctive words from the owner's notes steer the search. Without those rules a senior dog also got adult-dog guidance, and one pet's medicine note leaked into another pet's plan.
- **The agent works on a sandbox copy.** The real owner is never changed until a human accepts, so a bad or rejected proposal costs nothing. The trade-off is an extra deep copy per attempt, which is trivial at this size.
- **Validation is all-or-nothing.** One bad task rejects the whole reply and triggers a retry with the errors listed, so partial garbage can never be applied. Sources must match the retrieved guide labels, which enforces that the answer is grounded.
- **Confidence is a heuristic, not a self-rating.** Asking a model how sure it is tends to give unreliable numbers, so the score is built from things the agent can observe (retries, missing guidance, failed checks). The trade-off is that it measures process, not correctness of the advice.
- **The checker tests what I coded.** It checks the time budget, essential tasks and overlaps among planned tasks. A plan can pass and still be unrealistic in ways I did not encode.
- **Greedy plan by priority, then shortest first.** Simple and explainable, but not an optimal packing. The plan also ignores start times, so the conflict check warns rather than rearranges.
- **Demo mode as a first-class path.** It lets anyone run and grade the project without a key and makes the agent loop testable. The trade-off is that its output is only as smart as its rules, which is why the sample interactions label it clearly.
- **Medication is a placeholder.** The knowledge base marks doses and times as the vet's decision, and the system prompt forbids the model from choosing a dose.

## Testing Summary

**129 of 129 automated tests pass. On the evaluation harness (demo model, no API key), 13 of 13 labeled retrieval cases match exactly and 12 of 12 agent scenarios pass. Probing the code with edge cases found 4 real failures (3 serious); all 4 are fixed and now have tests. The live model has not been run against the real API.**

Reliability is measured four ways: automated tests, a confidence score, logging with error handling, and a human-evaluation table (to be completed by a reviewer). Everything below is text and can be re-generated with `python -m pytest` and `python evaluate.py`.

### 1. Automated tests (129)

| Suite | Tests | Covers |
|---|---|---|
| `tests/test_pawpal.py` | 58 | Scheduling core: sorting, recurrence, conflict detection, `generate_daily_plan`, `filter_tasks`, `explain_schedule`, input validation |
| `tests/test_ai_layer.py` | 45 | Retrieval, validation, the agent loop with a scripted fake model, the demo model, the Streamlit accept/reject flow |
| `tests/test_reliability.py` | 26 | Bugs found by probing, confidence scoring, logging, and the evaluation harness itself |

### 2. Confidence score

Each proposal gets a 0 to 1 confidence, shown in the app and in `agent_demo.py`. It is a **heuristic built from signals the agent can observe, not the model rating itself**: start at 1.0, subtract 0.15 per retry needed and 0.2 per pet with no retrieved guidance, cap at 0.4 if the checks still fail, and score 0 for a fallback or no suggestions. Labels: high at 0.8 and up, medium at 0.5 and up, otherwise low.

| Situation | Confidence | Label |
|---|---|---|
| Passes on the first attempt | 1.00 | high |
| Needed one retry (the sample run) | 0.85 | high |
| One of two pets has no guidance | 0.80 | high |
| Checks still failing after the last attempt | 0.40 | low |
| Model failed (fallback) or nothing suggested | 0.00 | low |

Limitation: the score reflects how smoothly the agent got there, not whether the advice is good. A plan can score 1.00 and still be wrong where the guidance is wrong.

### 3. Logging and error handling

Every step, tool call, retrieval, rejected reply, retry, fallback and the final confidence is written to `pawpal.log` (and the console). Tests assert that a model failure is logged with its reason, that rejected replies and retries are logged, that retrievals and tool calls are logged, that a duplicate accept is logged as a warning, that long prompts and the owner's notes are not dumped into the log, and that the log file is written without duplicate handlers. Failures never crash the app: a failed model call falls back to normal scheduling, a malformed reply is rejected and retried, and nothing is saved without the owner's accept.

### 4. Evaluation log (output of `python evaluate.py`)

#### Retrieval accuracy (labeled set)

| Test Input | Expected guides | Returned | Result |
|---|---|---|---|
| senior dog (age 10) | dog_care.md: Senior dogs, senior_pets.md: Senior pets | dog_care.md: Senior dogs, senior_pets.md: Senior pets | Pass |
| adult dog (age 3) | dog_care.md: Adult dogs | dog_care.md: Adult dogs | Pass |
| puppy (age 0) | dog_care.md: Puppies | dog_care.md: Puppies | Pass |
| adult cat (age 5) | cat_care.md: Adult cats | cat_care.md: Adult cats | Pass |
| senior cat (age 12) | cat_care.md: Senior cats, senior_pets.md: Senior pets | cat_care.md: Senior cats, senior_pets.md: Senior pets | Pass |
| kitten (age 0) | cat_care.md: Kittens | cat_care.md: Kittens | Pass |
| small pet (rabbit-type) | other_pets.md: Small and other pets | other_pets.md: Small and other pets | Pass |
| adult dog + 'flea pills' | dog_care.md: Adult dogs, medication_basics.md: Medication | dog_care.md: Adult dogs, medication_basics.md: Medication | Pass |
| senior dog + 'arthritis medicine' | dog_care.md: Senior dogs, medication_basics.md: Medication, senior_pets.md: Senior pets | dog_care.md: Senior dogs, medication_basics.md: Medication, senior_pets.md: Senior pets | Pass |
| adult dog + common words only | dog_care.md: Adult dogs | dog_care.md: Adult dogs | Pass |
| species not covered (fish) | (none) | (none) | Pass |
| species typed 'Dog' (capital) | dog_care.md: Adult dogs | dog_care.md: Adult dogs | Pass |
| adult cat + 'insulin shots' | cat_care.md: Adult cats | cat_care.md: Adult cats | Pass |

Retrieval: 13 of 13 exact matches; 0 wrong-species results.

#### Agent scenarios

| Test Input | Evaluation Criteria | Result | Detail |
|---|---|---|---|
| 2 pets + medicine note (demo model) | converges in <= 2 attempts; every task cites a retrieved guide | Pass | 11 tasks, 2 attempt(s), confidence 0.85 |
| model overlaps two tasks, then fixes it | checker catches overlap, retry converges | Pass | 2 attempt(s) |
| model never fixes the overlap | stops at the iteration limit and reports it | Pass | stopped after 3 calls, 1 problem(s) reported |
| model replies with prose, not JSON | rejected, retried, recovers | Pass | 2 calls |
| model cites a source that was never retrieved | first reply rejected as ungrounded | Pass | 2 attempt(s) |
| model call raises an error | falls back to normal scheduling, no crash | Pass | fallback plan used, confidence 0.00 |
| pet with no guidance (fish) | model is never called, nothing invented | Pass | model calls: 0 |
| whitespace-only notes | handled like no notes | Pass | 10 tasks |
| two pets with the same name | refused with a clear message (was: silently proposed 0 tasks) | Pass | Two pets share the same name. Rename one so tasks can be ass |
| run() before the owner accepts | real owner is not changed | Pass | owner has 0 tasks after run() |
| owner accepts the same proposal twice | tasks added once (was: duplicated) | Pass | 6 tasks after 2 accepts |
| task from 23:30 for 90 minutes | end time wraps to 01:00 (was: 25:00) | Pass | 23:30 + 90 min -> 01:00 |

Agent scenarios: 12 of 12 passed.

Consistency: 10 identical requests produced the same proposal every time (demo model; the live model is not deterministic).

How to read it: the retrieval set is labeled with the guide sections that *should* come back (no more, no fewer). The agent scenarios use scripted fake models to force specific failures (overlaps, prose instead of JSON, an ungrounded source, a model error). Caveats: I wrote this labeled set and tuned the retriever while looking at similar queries, so 13 of 13 is optimistic and not an independent accuracy figure. The consistency check uses the deterministic demo model; the live model will vary between runs.

### 5. Failures found and fixed

These were found by probing the code with inputs no test covered, then fixed and covered by tests.

| Test Input | Evaluation Criteria | Result before | Fix | Result now |
|---|---|---|---|---|
| Two pets with the same name | Tasks are assigned to the right pet | **Fail**: proposed 0 tasks with no explanation | The app refuses duplicate names; the agent stops with a clear message | Pass |
| Task from 23:30 for 90 minutes | End time is a valid time | **Fail**: showed "25:00" | End time wraps past midnight ("01:00") | Pass |
| Owner accepts the same proposal twice | Tasks are added once | **Fail**: every task was duplicated | `commit_proposal` ignores a second accept and logs a warning | Pass |
| Species typed as "Dog" | Guidance is found | **Fail**: no guidance | Species is matched case-insensitively | Pass |
| A proposal that fails its own checks | Confidence should not read as medium | **Fail**: scored 0.55 (medium) | Capped at 0.40 (low) | Pass |
| Task created with start time "25:99" (or "8:30", "24:00") | Bad times are rejected in the logic layer | **Fail**: accepted silently; only the UI form checked | `Task` validates `HH:MM` on creation and raises `ValueError` (negative durations too) | Pass |
| 30-minute day with 45 minutes of essential tasks | The owner is told what to change | **Fail**: only said which task did not fit | The agent adds "The essential tasks alone need N minutes but the day has M. Raise the daily minutes to at least N", only when the day is truly too short (not when other tasks crowded the essential one out); the app hides internal codes | Pass |
| A tool-call log line | Notes and prompts are not dumped into the log | **Fail**: the whole prompt was logged | Arguments truncated to 60 characters | Pass |

### 6. Human evaluation

**Run 1.** The project owner ran the Streamlit app in demo mode with Biscuit (dog, age 10) and Mochi (cat, age 5), a 120-minute day, no notes typed, and pasted the proposal. Result shown by the app: 10 tasks after 2 attempts, checks passed, confidence 0.85 (high). The Pass/Fail verdicts below were drafted by Claude from that pasted output and are **pending the owner's confirmation**. In the same run the agent's trace showed attempt 1 failing with two overlaps (both feedings at 08:00, and Biscuit's Joint comfort check against Mochi's Brushing at 19:30) and attempt 2 passing, so the retry fixed both at once. The medication, 30-minute and no-guidance rows come from `python app_walkthrough.py`, an automated run of the real app (headless, demo model), so they record what the app did and still need a person's judgment. Rows marked _not yet run_ need another run.

| Test Input | Evaluation Criteria | Result (Pass / Fail + note) |
|---|---|---|
| Senior dog, age 10 | Suggestions suit an older dog (short walks, joint check) | **Pass**: two 15-minute gentle walks, feeding, a joint comfort check, plus a general senior comfort check. No long walk. |
| Adult cat, age 5 | Feeding, litter box and play are reasonable | **Pass**: feeding, litter box cleaning, interactive play and brushing, all from the adult-cat guide. |
| Each task's cited source | The cited guide matches the task | **Pass**: all 10 tasks cite a section that fits (dog tasks cite the dog guide, cat tasks the cat guide, the two senior check tasks cite the senior-pets guide). |
| Any proposal | Tasks do not overlap and fit the day | **Pass**: no overlaps; the 10 tasks total 95 minutes against 120. Joint comfort check (19:30-19:35) ends exactly as Mochi's Brushing starts, which is not counted as a conflict. |
| Quality note | Suggestions are not redundant | **Minor issue**: Biscuit gets two near-duplicate checks 30 minutes apart ("Joint comfort check" 19:30 and "Comfort and mobility check" 20:00), because the senior-dog and general senior-pet guides both say to check how the pet moves. Not wrong, but repetitive. |
| Note: "Biscuit takes arthritis medicine every morning." | Medication task appears for that pet only, with no dose given | **Pass** (automated walkthrough): exactly one medication task, "Give prescribed medication" for Biscuit, none for Mochi. No dose or "mg" appears in any task or in the explanation. 11 tasks, 2 attempts, confidence 0.85. |
| 30-minute day, two pets | Essential tasks come first; skipped tasks are listed | **Partial** (automated walkthrough): the app does not hide the problem. The essential tasks alone need 45 minutes, so no plan can fully pass. After 3 attempts the app shows "Proposed 4 task(s), but the checks still found problems", names the task that did not fit, lists it under "Would not fit in the day", gives confidence 0.40 (low), and now says what to do: "The essential tasks alone need 45 minutes but the day has 30. Raise the daily minutes to at least 45, or remove an essential task." Issue: it drops the senior dog's 15-minute essential walk but keeps the cat's 10-minute tasks, because the plan adds shortest tasks first within a priority. A person may disagree with that choice. |
| Species with no guidance | System says it has no guidance and suggests nothing | **Not reachable in the app**: the species dropdown offers only dog, cat and other, and all three have guidance. "other" (a pet named Bubbles) gets 2 sensible tasks from the small-pets guide (daily check, food and fresh water; habitat cleaning) with confidence 1.00. The no-guidance behavior (model never called, nothing invented) is covered by the automated tests and `evaluate.py` using a species the guides do not cover (fish). |
| Read "Why this plan?" | A non-expert can understand why each task is there | **Run 1: Partial** (before the change): it explained the ordering and the budget, but gave only a file-name tag as the reason per task, omitted start times, and its numbered order differed from the time order above it. **Changed since:** the explanation now lists tasks in time order with start times, a "Why:" line per task (priority, essential, who suggested it), and numbered footnotes that quote the guide in plain language. **Needs re-review by the owner** to confirm it now reads clearly. |

### 7. What is not covered

- **The live model.** It has not been run against the real Anthropic API in development (no key was available). I verified the request shape against the installed SDK and exercised the success, error and thinking-block handling over a mocked HTTP connection, but real model output may differ. Validation, retry and the confidence score exist for that reason. The demo model results above are not evidence about the live model.
- **Retrieval beyond the cases I wrote.** It has no independent accuracy measure. One labeled case shows a real limitation: a note about a topic the guides do not cover ("insulin shots") is silently ignored, with no warning to the owner.
- **Scheduling core, known limits (pinned by tests, not fixed):** zero-duration tasks never conflict, times are assumed to be on one day (a task running past midnight is not compared with early-morning tasks), and `filter_tasks` matches pet names case-sensitively. `generate_daily_plan` and `filter_tasks` are now covered in depth, and a mutation check (deliberately breaking the code) showed the new tests catch 8 of 8 injected bugs.
- **Quality of advice.** The knowledge base is general guidance I wrote, not vetted veterinary advice. Passing checks means the plan is consistent, not that it is medically right.

**What I learned:** an AI feature is only as trustworthy as the checks around it. The most valuable tests were not about the model working, but about what happens when it is wrong, absent or unhelpful. Probing with edge cases found three serious bugs that 48 passing tests had missed, which is a good reminder that a passing suite only covers the cases someone thought of.

## Reflection

PawPal+ taught me that most of the value in an AI project lives in the ordinary code around the model: clear data classes, grounding the answer in retrieved sources, validation, a sandbox, tests, logging, and a human who accepts or rejects the result. Because the deterministic scheduler existed first, the agent could check the model's output against rules I already trust instead of taking it on faith. My responsible-AI reflection (how I worked with AI, a suggestion I accepted and one I rejected or revised, and the system's limitations) goes in `model_card.md`.
