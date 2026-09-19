"""Streamlit UI:  streamlit run ui/app.py

Login -> role-based view
  * admin : upload / manage policies, ask policies, audit logs, login sessions, observability, evaluation, users
  * user  : policy selector + chat box, previous conversations (history), feedback
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))
from api_client import ApiClient, ApiError  # noqa: E402

st.set_page_config(page_title="Policy Assistant", page_icon="📘", layout="wide")

CONF_BADGE = {"high": "🟢 High confidence", "medium": "🟡 Medium confidence", "low": "🔴 Low confidence"}


# ------------------------------------------------------------------ helpers
def api() -> ApiClient:
    return ApiClient(st.session_state.get("token"))


def reset_state() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def handle(err: ApiError) -> None:
    """Show a friendly error; send the user back to login when the session is gone."""
    if err.status == 401 and st.session_state.get("token"):
        reset_state()
        st.session_state["flash"] = err.message
        st.rerun()
    msg = err.message + (f"  \n_Reference: {err.trace_id}_" if err.trace_id else "")
    st.error(msg)


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ApiError as e:
        handle(e)
        return None


# ------------------------------------------------------------------ login
def login_page() -> None:
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.title("📘 Policy Assistant")
        st.caption("Ask questions about company policies. Please sign in.")
        if st.session_state.get("flash"):
            st.warning(st.session_state.pop("flash"))
        with st.form("login"):
            username = st.text_input("User ID")
            password = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in", use_container_width=True, type="primary")
        if ok:
            if not username or not password:
                st.error("Please enter both your user ID and password.")
                return
            try:
                data = ApiClient().login(username, password)
            except ApiError as e:
                st.error(e.message)
                return
            st.session_state.update(token=data["access_token"], role=data["role"], username=data["username"],
                                    full_name=data["full_name"], login_session_id=data["login_session_id"])
            st.rerun()


def logout() -> None:
    try:
        api().logout()
    except ApiError:
        pass
    reset_state()
    st.session_state["flash"] = "You have been logged out."
    st.rerun()


# ------------------------------------------------------------------ chat (shared)
def render_assistant(msg: dict) -> None:
    st.markdown(msg.get("content") or msg.get("answer", ""))
    for w in msg.get("warnings", []) or []:
        st.warning(w, icon="⚠️")
    meta = []
    if msg.get("confidence"):
        meta.append(CONF_BADGE.get(msg["confidence"], msg["confidence"]))
    if msg.get("answer_type") in ("no_context", "blocked", "withheld", "fallback_extractive"):
        meta.append({"no_context": "ℹ️ Not found in policy", "blocked": "🛡️ Blocked by safety check",
                     "withheld": "🛡️ Answer withheld", "fallback_extractive": "⚙️ Fallback mode"}[msg["answer_type"]])
    if msg.get("self_eval", {}).get("overall") is not None:
        meta.append(f"Self-eval {msg['self_eval']['overall']:.0%}")
    if meta:
        st.caption(" · ".join(meta))
    sources = msg.get("sources") or []
    if sources:
        with st.expander(f"📄 Sources ({len(sources)})"):
            for s in sources:
                score = f" · relevance {s['score']:.2f}" if s.get("score") is not None else ""
                st.markdown(f"**[{s['ref']}] {s.get('policy_name', '')}** — {s.get('section', '')} ({s.get('pages', '')}){score}")
                st.caption(s.get("snippet", "")[:400])
    mid = msg.get("id") or msg.get("message_id")
    if mid and msg.get("answer_type") in ("answered", "no_context"):
        c1, c2, _ = st.columns([1, 1, 10])
        if c1.button("👍", key=f"up_{mid}", help="Helpful"):
            if call(api().feedback, mid, 1):
                st.toast("Thanks for your feedback!")
        if c2.button("👎", key=f"down_{mid}", help="Not helpful"):
            if call(api().feedback, mid, -1):
                st.toast("Thanks - we'll use this to improve.")


def load_session(session_id: str) -> None:
    msgs = call(api().messages, session_id)
    if msgs is not None:
        st.session_state["chat_session_id"] = session_id
        st.session_state["messages"] = msgs


def new_chat() -> None:
    st.session_state["chat_session_id"] = None
    st.session_state["messages"] = []


def chat_sidebar(policies: list[dict]) -> str | None:
    """Policy selector + conversation history. Returns selected policy id (None = all)."""
    st.sidebar.subheader("Policy")
    options = {"all": "🔎 All policies"} | {p["id"]: f"{p['name']} (v{p['version']})" for p in policies}
    current = st.session_state.get("policy_id", next(iter(options)))
    if current not in options:
        current = "all"
    choice = st.sidebar.selectbox("Select a policy", list(options.keys()), index=list(options.keys()).index(current),
                                  format_func=lambda k: options[k])
    if choice != st.session_state.get("policy_id"):
        if st.session_state.get("messages"):
            new_chat()  # a conversation is bound to one policy scope
        st.session_state["policy_id"] = choice

    st.sidebar.divider()
    if st.sidebar.button("➕ New conversation", use_container_width=True):
        new_chat()
        st.rerun()
    st.sidebar.subheader("Previous conversations")
    sessions = call(api().sessions) or []
    if not sessions:
        st.sidebar.caption("No conversations yet.")
    for s in sessions[:30]:
        c1, c2 = st.sidebar.columns([5, 1])
        active = s["id"] == st.session_state.get("chat_session_id")
        if c1.button(("▶ " if active else "") + s["title"][:40], key=f"s_{s['id']}", use_container_width=True,
                     help=s["updated_at"][:16].replace("T", " ")):
            load_session(s["id"])
            if s.get("policy_id") in options:
                st.session_state["policy_id"] = s["policy_id"]
            st.rerun()
        if c2.button("🗑", key=f"d_{s['id']}", help="Delete conversation"):
            if call(api().delete_session, s["id"]):
                if active:
                    new_chat()
                st.rerun()
    return None if choice == "all" else choice


def chat_page(policies: list[dict]) -> None:
    policy_id = chat_sidebar(policies)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_session_id", None)
    name = next((p["name"] for p in policies if p["id"] == policy_id), "all policies")
    st.subheader(f"💬 Ask about {name}")

    if not policies:
        st.info("No policies are available yet. Please ask an administrator to upload the policy documents.")
        return
    if not st.session_state["messages"]:
        st.info("Hello! Ask me anything about the selected policy — e.g. *How many days of annual leave do I get?*")

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "📘"):
            if m["role"] == "assistant":
                render_assistant(m)
            else:
                st.markdown(m["content"])

    question = st.chat_input("Type your question about the policy...")
    if question:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📘"):
            with st.spinner("Searching the policy and preparing an answer..."):
                try:
                    res = api().ask(question, policy_id, st.session_state["chat_session_id"])
                except ApiError as e:
                    handle(e)
                    return
        st.session_state["chat_session_id"] = res["session_id"]
        st.session_state["messages"].append({"role": "user", "content": question})
        st.session_state["messages"].append({"role": "assistant", "content": res["answer"], **res,
                                             "id": res["message_id"]})
        st.rerun()


# ------------------------------------------------------------------ admin pages
def admin_upload() -> None:
    st.subheader("⬆️ Upload policy")
    with st.form("upload", clear_on_submit=True):
        f = st.file_uploader("Policy PDF", type=["pdf"])
        c1, c2 = st.columns(2)
        name = c1.text_input("Policy name", placeholder="e.g. Leave Policy")
        dept = c2.text_input("Department", value="HR")
        c3, c4 = st.columns(2)
        version = c3.text_input("Version", value="1.0")
        eff = c4.date_input("Effective date", value=None)
        desc = st.text_area("Description", height=70)
        tags = st.text_input("Tags (comma separated)", placeholder="leave, attendance")
        submitted = st.form_submit_button("Upload & index", type="primary")
    if submitted:
        if not f:
            st.error("Please choose a PDF file.")
            return
        res = call(api().upload_policy, f.name, f.getvalue(),
                   {"policy_name": name or Path(f.name).stem, "department": dept, "version": version,
                    "effective_date": eff.isoformat() if eff else "", "description": desc, "tags": tags})
        if res:
            st.success(res["message"])
    st.divider()
    st.markdown("**Bulk load** — copy PDFs into `data/inbox/` on the server, then:")
    if st.button("📥 Ingest all PDFs in inbox"):
        res = call(api().ingest_inbox)
        if res:
            st.success(f"{res['queued']} policies queued for indexing.")
            for s in res["skipped"]:
                st.warning(f"Skipped {s['file']}: {s['reason']}")


def admin_policies() -> None:
    st.subheader("📚 Policies")
    if st.button("🔄 Refresh"):
        st.rerun()
    pols = call(api().admin_policies) or []
    if not pols:
        st.info("No policies uploaded yet.")
        return
    df = pd.DataFrame(pols)[["name", "version", "department", "status", "page_count", "chunk_count", "uploaded_by",
                             "uploaded_at", "indexed_at", "error", "id"]]
    st.dataframe(df, use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns([3, 1, 1])
    target = c1.selectbox("Policy", pols, format_func=lambda p: f"{p['name']} v{p['version']} [{p['status']}]")
    if c2.button("Re-index") and target:
        r = call(api().reindex_policy, target["id"])
        if r:
            st.success(r["message"])
    if c3.button("Delete", type="secondary") and target:
        r = call(api().delete_policy, target["id"])
        if r:
            st.success(r["message"])
            st.rerun()
    st.markdown("**Ingestion jobs**")
    jobs = call(api().jobs) or []
    if jobs:
        st.dataframe(pd.DataFrame(jobs), use_container_width=True, hide_index=True)


def admin_audit() -> None:
    st.subheader("🧾 Audit trail")
    c1, c2, c3 = st.columns(3)
    action = c1.selectbox("Action", ["", "LOGIN_SUCCESS", "LOGIN_FAILED", "LOGIN_LOCKED", "LOGOUT", "SESSION_EXPIRED",
                                     "CHAT_QUERY", "GUARDRAIL_BLOCK", "POLICY_UPLOAD", "POLICY_INGESTED",
                                     "POLICY_INGEST_FAILED", "POLICY_DELETE", "POLICY_REINDEX", "ACCESS_DENIED",
                                     "FEEDBACK", "USER_CREATE", "EVAL_RUN", "CHAT_SESSION_DELETE"])
    username = c2.text_input("Username")
    outcome = c3.selectbox("Outcome", ["", "success", "failure", "blocked"])
    rows = call(api().audit_logs, action=action, username=username, outcome=outcome, limit=500) or []
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "audit_logs.csv", "text/csv")
    else:
        st.info("No audit events match the filter.")


def admin_sessions() -> None:
    st.subheader("🔐 Login sessions")
    c1, c2 = st.columns(2)
    username = c1.text_input("Username", key="ls_user")
    status = c2.selectbox("Status", ["", "active", "logged_out", "expired"])
    rows = call(api().login_sessions, username=username, status=status, limit=500) or []
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "login_sessions.csv", "text/csv")


def admin_observability() -> None:
    st.subheader("📈 Observability")
    s = call(api().stats)
    if s:
        c = st.columns(5)
        c[0].metric("Active policies", s["active_policies"])
        c[1].metric("Indexed chunks", s["indexed_chunks"])
        c[2].metric("Questions (24h)", s["questions_24h"])
        c[3].metric("Avg latency", f"{s['avg_latency_ms'] / 1000:.1f}s")
        c[4].metric("Avg self-eval", f"{s['avg_self_eval']:.0%}" if s["avg_self_eval"] is not None else "—")
        c = st.columns(5)
        c[0].metric("Total questions", s["total_questions"])
        c[1].metric("👍 / 👎", f"{s['feedback_up']} / {s['feedback_down']}")
        c[2].metric("Guardrail blocks", s["guardrail_blocks"])
        c[3].metric("Active sessions", s["active_login_sessions"])
        c[4].metric("Failed logins (24h)", s["logins_24h"].get("LOGIN_FAILED", 0))
        if s["answer_types"]:
            st.markdown("**Answer outcomes**")
            st.bar_chart(pd.Series(s["answer_types"], name="count"))
    st.markdown("**Recent request traces** (pipeline stage timings)")
    traces = call(api().traces, 50) or []
    if traces:
        df = pd.DataFrame(traces)[["created_at", "name", "username", "status", "total_ms", "trace_id"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
        tid = st.selectbox("Inspect trace", [t["trace_id"] for t in traces])
        t = next(t for t in traces if t["trace_id"] == tid)
        spans = pd.DataFrame(t["spans"])
        if not spans.empty:
            st.bar_chart(spans.set_index("name")["duration_ms"])
            st.dataframe(spans, use_container_width=True, hide_index=True)
    st.caption("Prometheus metrics are exposed at `/metrics`; structured JSON logs are in `logs/`.")


def admin_evaluation() -> None:
    st.subheader("🧪 Evaluation (DeepEval)")
    st.markdown("Runs the golden dataset (`eval/golden_dataset.yaml`) through the full RAG pipeline and scores it "
                "with retrieval, generation, LLM-quality and prompt-alignment metrics.")
    c1, c2 = st.columns([1, 3])
    max_cases = c1.number_input("Max cases (0 = all)", min_value=0, value=0)
    if c2.button("▶ Run evaluation", type="primary"):
        r = call(api().run_evaluation, int(max_cases) or None)
        if r:
            st.success("Evaluation started. It can take several minutes with local models - refresh to see results.")
    if st.button("🔄 Refresh results"):
        st.rerun()
    runs = call(api().evaluations) or []
    for r in runs:
        with st.expander(f"{r['started_at'][:19].replace('T', ' ')} · {r['status']} · by {r['triggered_by']}",
                         expanded=r is runs[0]):
            if r["error"]:
                st.error(r["error"])
            summ = r.get("summary") or {}
            if summ.get("metrics"):
                st.dataframe(pd.DataFrame(summ["metrics"]).T, use_container_width=True)
            if summ.get("retrieval"):
                st.json(summ["retrieval"])
            if r["report_path"]:
                st.caption(f"Full report: `{r['report_path']}`")


def admin_users() -> None:
    st.subheader("👥 Users")
    users = call(api().users) or []
    if users:
        st.dataframe(pd.DataFrame(users), use_container_width=True, hide_index=True)
    with st.form("new_user", clear_on_submit=True):
        c1, c2 = st.columns(2)
        u = c1.text_input("Username")
        n = c2.text_input("Full name")
        c3, c4 = st.columns(2)
        p = c3.text_input("Password (min 8 chars)", type="password")
        r = c4.selectbox("Role", ["user", "admin"])
        if st.form_submit_button("Create user"):
            if call(api().create_user, {"username": u, "password": p, "role": r, "full_name": n}):
                st.success(f"User '{u}' created.")


# ------------------------------------------------------------------ router
def header() -> None:
    st.sidebar.markdown(f"### 📘 Policy Assistant\nSigned in as **{st.session_state.get('full_name') or st.session_state['username']}**"
                        f"  \nRole: `{st.session_state['role']}`")
    if st.sidebar.button("🚪 Log out", use_container_width=True):
        logout()
    st.sidebar.divider()


def main() -> None:
    if not st.session_state.get("token"):
        login_page()
        return
    header()
    if st.session_state["role"] == "admin":
        pages = {"Upload policy": admin_upload, "Manage policies": admin_policies, "Ask policies": None,
                 "Audit trail": admin_audit, "Login sessions": admin_sessions, "Observability": admin_observability,
                 "Evaluation": admin_evaluation, "Users": admin_users}
        page = st.sidebar.radio("Admin menu", list(pages.keys()))
        st.sidebar.divider()
        if page == "Ask policies":
            chat_page(call(api().policies) or [])
        else:
            pages[page]()
    else:
        policies = call(api().policies)
        if policies is not None:
            chat_page(policies)


main()
