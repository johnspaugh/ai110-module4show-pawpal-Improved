import re

import streamlit as st
from agent import MAX_NOTES_CHARS, PlanningAgent, commit_proposal, confidence_label
from llm import get_llm
from logging_setup import setup_logging
from pawpal_system import Owner, Pet, Task, Scheduler

setup_logging()

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="wide")

# --- Session "vault": create the Owner once, reuse it on every rerun ---
if "owner" not in st.session_state:
    st.session_state.owner = Owner(name="Jordan", available_minutes_per_day=120)
owner = st.session_state.owner

# Colored section panels: each section holds an invisible marker span, and the CSS paints the
# container that directly holds that marker.
SECTION_COLORS = {"pet": "#0b2a5b", "tasks": "#b34700", "plan": "#588505"}  # dark blue, dark orange, dark gray
_panel_css = "".join(
    f'[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .pp-{name}) '
    f"{{background:{color}; padding:1rem 1.25rem; border-radius:0.75rem;}}\n"
    f'[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .pp-{name}) '
    f"h1,h2,h3,p,label,span,li,[data-testid=\"stCaptionContainer\"] {{color:#fff;}}\n"
    for name, color in SECTION_COLORS.items()
)
st.markdown(f"<style>{_panel_css}</style>", unsafe_allow_html=True)

if "show_intro" not in st.session_state:
    st.session_state.show_intro = True


def render_intro() -> None:
    st.title("🐾 PawPal+")
    st.markdown(
        """
PawPal+ plans a pet owner's day: it fits care tasks for all of your pets into your
time budget, warns about overlaps, repeats recurring tasks, and can suggest tasks
from pet-care guidance with an AI assistant that checks its own work.
"""
    )


def render_owner() -> None:
    st.subheader("Owner")
    owner.name = st.text_input("Owner name", value=owner.name)
    owner.available_minutes_per_day = int(
        st.number_input("Minutes available per day", min_value=1, max_value=1440,
                        value=owner.available_minutes_per_day)
    )


# PawPal+ intro and Owner sit side by side. When the intro is hidden, Owner fills the whole width.
if st.session_state.show_intro:
    intro_col, owner_col = st.columns(2, gap="large")
    with intro_col:
        render_intro()
        if st.button("Hide intro"):
            st.session_state.show_intro = False
            st.rerun()
    with owner_col:
        render_owner()
else:
    render_owner()
    if st.button("Show PawPal+ intro"):
        st.session_state.show_intro = True
        st.rerun()

st.divider()

with st.container():
    st.markdown('<span class="pp-pet"></span>', unsafe_allow_html=True)
    st.subheader("Add a Pet")
    with st.form("add_pet_form", clear_on_submit=True):
        pet_name = st.text_input("Pet name", value="Mochi")
        species = st.selectbox("Species", ["dog", "cat", "other"])
        age = st.number_input("Age", min_value=0, max_value=40, value=1)
        if st.form_submit_button("Add pet"):
            clean_name = pet_name.strip()
            if not clean_name:
                st.error("Give your pet a name.")
            elif any(p.name.strip().lower() == clean_name.lower() for p in owner.pets):
                st.error(f"You already have a pet named {clean_name}. Use a different name.")
            else:
                owner.add_pet(Pet(name=clean_name, species=species, age=int(age)))
                st.success(f"Added {clean_name}!")

    if owner.pets:
        st.write("Your pets:")
        st.table([{"name": p.name, "species": p.species, "age": p.age, "tasks": len(p.tasks)}
                  for p in owner.pets])
    else:
        st.info("No pets yet. Add one above.")


with st.container():
    st.markdown('<span class="pp-tasks"></span>', unsafe_allow_html=True)
    st.markdown("### Tasks")
    if owner.pets:
        with st.form("add_task_form", clear_on_submit=False):
            target_name = st.selectbox("For which pet?", [p.name for p in owner.pets])
            col1, col2, col3 = st.columns(3)
            with col1:
                task_title = st.text_input("Task title", value="Morning walk")
            with col2:
                category = st.selectbox("Category", ["exercise", "feeding", "meds", "grooming", "other"])
            with col3:
                duration = st.number_input("Duration (min)", min_value=1, max_value=240, value=20)
            col5, col6, col7 = st.columns(3)
            with col5:
                priority = st.selectbox("Priority", ["low", "medium", "high"], index=2)
            with col6:
                frequency = st.selectbox("Repeats", ["daily", "weekly", "monthly", "once"])
            with col7:
                start_time = st.text_input("Start time (HH:MM, optional)", value="")
            if st.form_submit_button("Add task"):
                start_time = start_time.strip()
                if start_time and not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", start_time):
                    st.error("Start time must look like 08:30 (24-hour HH:MM).")
                else:
                    pet = next(p for p in owner.pets if p.name == target_name)
                    pet.add_task(Task(title=task_title, category=category,
                                      duration_minutes=int(duration), priority=priority,
                                      frequency=frequency, start_time=start_time))
                    st.success(f"Added '{task_title}' to {target_name}.")

        scheduler = Scheduler(owner)

        # --- Conflict warnings: shown first so the owner sees problems before the list ---
        conflicts = scheduler.detect_conflicts()
        if conflicts:
            st.warning(
                f"{len(conflicts)} scheduling conflict(s) found. "
                "Change one task's start time or duration to fix each one."
            )
            for message in conflicts:
                st.warning(message, icon="⚠️")
        elif any(t.start_time for t in owner.get_all_tasks()):
            st.success("No time conflicts between your scheduled tasks.")

        all_tasks = owner.get_all_tasks()
        if all_tasks:
            st.write("Your tasks:")
            f1, f2 = st.columns(2)
            with f1:
                pet_filter = st.selectbox("Filter by pet", ["All"] + [p.name for p in owner.pets])
            with f2:
                status_filter = st.selectbox("Filter by status", ["All", "Pending", "Completed"])

            shown = scheduler.filter_tasks(
                pet_name=None if pet_filter == "All" else pet_filter,
                completed={"All": None, "Pending": False, "Completed": True}[status_filter],
            )
            shown = scheduler.sort_by_time(shown)  # earliest first, unscheduled last

            if shown:
                st.table([{
                    "time": f"{t.start_time}-{t.end_time}" if t.start_time else "unscheduled",
                    "pet": t.pet_name,
                    "title": t.title,
                    "minutes": t.duration_minutes,
                    "priority": t.priority,
                    "repeats": t.frequency,
                    "due": t.due_date.isoformat() if t.due_date else "",
                    "done": "✅" if t.completed else "",
                } for t in shown])
            else:
                st.info("No tasks match these filters.")

            pending = scheduler.filter_tasks(completed=False)
            if pending:
                st.write("Mark a task complete:")
                labels = {id(t): f"{t.pet_name}: {t.title} ({t.start_time or 'unscheduled'})"
                          for t in pending}
                chosen = st.selectbox("Task", pending, format_func=lambda t: labels[id(t)])
                if st.button("Complete task"):
                    pet = next(p for p in owner.pets if p.name == chosen.pet_name)
                    next_task = pet.complete_task(chosen)
                    if next_task:
                        st.success(f"Done! Next '{chosen.title}' is due {next_task.due_date}.")
                    else:
                        st.success(f"Done! '{chosen.title}' won't repeat.")
                    st.rerun()
        else:
            st.info("No tasks yet. Add one above.")
    else:
        st.caption("Add a pet first, then you can add tasks for it.")

st.divider()

st.subheader("AI Plan Assistant")
if "llm" not in st.session_state:
    st.session_state.llm = get_llm()
llm = st.session_state.llm
if llm.mode == "live":
    st.caption(f"Live model: {llm.provider} / {llm.model}. "
               "Suggestions come from retrieved pet-care guidance.")
else:
    st.caption(
        "Demo mode (no ANTHROPIC_API_KEY set): a rule-based stand-in reads the same "
        "retrieved guidance. Add a key to use the live model."
    )

st.warning(
    "General pet-care guidance, not veterinary advice. For medication, use only the doses "
    "and times your vet gave you; the assistant never decides a dose.",
    icon="🩺",
)

if st.session_state.get("flash"):
    st.success(st.session_state.pop("flash"))

notes = st.text_area(
    "Notes for the assistant (optional)",
    max_chars=MAX_NOTES_CHARS,
    placeholder="e.g. Biscuit takes arthritis medicine every morning. Keep walks gentle.",
)
if st.button("Generate AI plan", disabled=not owner.pets):
    with st.spinner("Retrieving guidance, proposing tasks and checking the result..."):
        st.session_state.proposal = PlanningAgent(owner, llm).run(notes)

proposal = st.session_state.get("proposal")
if proposal:
    if proposal.used_fallback:
        for message in proposal.problems:
            st.warning(message)
    elif not proposal.tasks:
        for message in proposal.problems or ["The assistant had no suggestions."]:
            st.warning(message)
    else:
        if proposal.converged:
            st.success(
                f"Proposed {len(proposal.tasks)} task(s) after {proposal.iterations} attempt(s). "
                "The agent's own checks passed: time budget, essential tasks, no overlaps."
            )
            st.caption(f"Confidence: {proposal.confidence:.2f} "
                       f"({confidence_label(proposal.confidence)}). A heuristic from retries "
                       "and missing guidance, not the model rating itself.")
        else:
            st.warning(
                f"Proposed {len(proposal.tasks)} task(s), but the checks still found "
                f"problems after {proposal.iterations} attempt(s). Review them before accepting."
            )
            for message in proposal.problems:
                # Drop the internal code ("SKIPPED_ESSENTIAL: ") and show the readable part.
                st.warning(message.split(": ", 1)[-1], icon="⚠️")
            st.caption(f"Confidence: {proposal.confidence:.2f} "
                       f"({confidence_label(proposal.confidence)}), capped because the "
                       "checks did not pass.")
        st.table([{
            "pet": t.pet, "task": t.title, "start": t.start_time or "unscheduled",
            "minutes": t.duration_minutes, "priority": t.priority,
            "essential": "yes" if t.essential else "", "source": t.source,
        } for t in proposal.tasks])
        if proposal.skipped:
            st.info("Would not fit in the day: " + ", ".join(
                f"{t.pet_name}'s {t.title} ({t.duration_minutes} min)" for t in proposal.skipped))
        accept_col, reject_col = st.columns(2)
        with accept_col:
            if st.button("Accept and add these tasks"):
                added = commit_proposal(owner, proposal)
                st.session_state.pop("proposal", None)
                st.session_state.flash = f"Added {len(added)} task(s) to your pets."
                st.rerun()
        with reject_col:
            if st.button("Reject"):
                st.session_state.pop("proposal", None)
                st.rerun()
    if proposal.no_guidance:
        st.info("No guidance found for: " + ", ".join(proposal.no_guidance)
                + ". No tasks were suggested for them.")
    with st.expander("What the agent did"):
        st.text("\n".join(proposal.trace))
    with st.expander("Retrieved guidance"):
        for pet_name, hits in proposal.guidance.items():
            st.write(f"**{pet_name}:** " + (", ".join(f"{label} (score {score})"
                                                     for label, score in hits) or "nothing found"))
    if proposal.explanation:
        with st.expander("Why this plan? (in plain language)"):
            st.text(proposal.explanation)

st.divider()

with st.container():
    st.markdown('<span class="pp-plan"></span>', unsafe_allow_html=True)
    st.subheader("Build Schedule")
    if st.button("Generate schedule"):
        scheduler = Scheduler(owner)
        plan = scheduler.generate_daily_plan()
        if plan:
            used = sum(t.duration_minutes for t in plan)
            st.success(f"Planned {len(plan)} task(s): {used} of "
                       f"{owner.available_minutes_per_day} minutes used.")
            st.table([{"#": i, "pet": t.pet_name, "task": t.title,
                       "minutes": t.duration_minutes, "priority": t.priority}
                      for i, t in enumerate(plan, start=1)])
            if scheduler.skipped:
                st.warning("Not enough time for: " + ", ".join(
                    f"{t.pet_name}'s {t.title} ({t.duration_minutes} min)"
                    for t in scheduler.skipped))
        else:
            st.warning("Nothing to schedule. Add pets and tasks first.")
        with st.expander("Why this plan?"):
            st.text(scheduler.explain_plan())
