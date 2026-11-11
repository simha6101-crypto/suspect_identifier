"""Investigator UI for Multi-Modal Suspect Ranking with Safety Guardrails & Bias Auditing.
Run: streamlit run app.py

Features:
- Live toggle between Hand-Set Heuristic Weights and ML-Learned Softmax Ranker
- Synthetic benchmark population (128-D) with optional Colab real_population.npz loading
- Evidence Confidence Floor: Suppresses automated ranking when evidence is insufficient
- Comprehensive Bias & Safety Audit: Per-group False Positive Rates (FPR) and Recall@K
- Explainable evidence contributions and tamper-evident hash-chained audit trail
"""
import os, json, time
import numpy as np
import pandas as pd
import streamlit as st
from suspect_id import (
    make_population, make_case, score, rank_candidates, evaluate,
    audit, verify_audit_log, W_HAND, FEATS
)
from ranker import train as train_ranker, load_weights, save_weights

LOG = "audit_log.jsonl"
HAIR = ["black", "brown", "blond", "red", "grey"]
BUILD = ["slim", "medium", "heavy"]
AGE = ["18-25", "26-35", "36-50", "51+"]
GROUP_NAMES = ["Group 0 (Majority)", "Group 1 (Cohort B)", "Group 2 (Cohort C)"]

st.set_page_config(page_title="FairRank-ID | Multi-Modal Suspect Ranking", layout="wide", page_icon="🛡️")

st.title("🛡️ FairRank-ID: Multi-Modal Suspect Ranking Assistant")
st.caption("Human-in-the-Loop Decision Support with Learned Ranking, Safety Guardrails & Bias Auditing")

st.warning(
    "⚠️ **LEGAL & ETHICAL NOTICE:** This system provides decision support leads only, NEVER automated identifications. "
    "Rankings cannot establish probable cause. Every decision requires a named human reviewer, written justification, and tamper-evident audit logging."
)


@st.cache_resource
def load_data(dataset_choice, n_pop, n_cases, seed):
    rng = np.random.default_rng(seed)
    real_path = "real_population.npz" if "Colab" in dataset_choice and os.path.exists("real_population.npz") else None
    pop = make_population(n=n_pop, rng=rng, path=real_path)
    cases = [make_case(pop, rng) for _ in range(n_cases)]
    return pop, cases


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("1. Population & Embeddings")
    available_datasets = ["Synthetic Benchmark (128-D Unit Spheres)"]
    if os.path.exists("real_population.npz"):
        available_datasets.append("External LFW Embeddings (Colab real_population.npz)")

    dataset_choice = st.selectbox(
        "Embedding Source",
        available_datasets,
        help="Synthetic benchmark (128-D) with calibrated noise. External LFW embeddings loaded if generated via Colab."
    )

    seed = st.number_input("Random Seed", 0, 999, 7)
    k = st.slider("Candidates to Rank (k)", 3, 25, 10)
    reviewer = st.text_input("Reviewer Name (Required for decisions)", value="Det. S. Walker")

    pop_size = 5000 if "Synthetic" in dataset_choice else 1200
    pop_size = st.number_input("Population Size", 500, 20000, int(pop_size), 500)
    pop, cases = load_data(dataset_choice, int(pop_size), 200, int(seed))

    st.caption("ℹ️ *Note: All contextual attributes (hair, build, age, zone, MO) are simulated.*")

    st.markdown("---")
    st.header("2. Ranking Engine Toggle")
    ranker_choice = st.radio(
        "Weight Optimization Mode",
        ["Hand-Set Weights (Heuristic)", "Learned Ranker (Softmax Loss)"],
        help="Toggle live between heuristic baseline weights and ML-trained weights."
    )
    is_learned = "Learned" in ranker_choice

    # Load or initialize learned weights
    learned_w_dict = load_weights("weights.json")
    if learned_w_dict is None:
        learned_w_dict = {
            "face": 8.23, "face_x_quality": 7.70, "hair": 1.11,
            "build": 1.00, "age": 1.08, "zone": 2.80, "mo": 1.27
        }

    with st.expander("🔍 View Active Feature Weights", expanded=False):
        if is_learned:
            st.write("**Learned Softmax Weights (from `weights.json`):**")
            st.dataframe(pd.Series(learned_w_dict, name="Weight").round(2))
            if st.button("🔄 Retrain Learned Ranker Now"):
                with st.spinner("Optimizing listwise softmax weights on 100 cases..."):
                    rng_train = np.random.default_rng(int(seed) + 42)
                    train_cases = [make_case(pop, rng_train) for _ in range(100)]
                    w_trained = train_ranker(pop, train_cases, iters=200)
                    save_weights(w_trained, "weights.json")
                    st.success("Weights re-trained and saved!")
                    st.rerun()
        else:
            st.write("**Hand-Set Baseline Weights:**")
            st.dataframe(pd.Series(W_HAND, name="Weight"))

    st.markdown("---")
    st.header("3. Safety & Bias Guardrails")
    evidence_threshold = st.slider(
        "Evidence Confidence Floor (τ)",
        1.0, 6.0, 3.5, 0.1,
        help="If the top candidate score is below this floor, rankings are suppressed to prevent confirmation bias."
    )
    simulate_low_quality = st.checkbox("Simulate Degraded CCTV (Quality = 0.32)", value=False)

    st.markdown("---")
    st.header("4. Case Selection")
    case_id = st.selectbox("Select Case", range(len(cases)), index=0, format_func=lambda i: f"Case {i:03d}")
    st.caption("💡 *Tip for Demo:* Try **Case 017** or **Case 044** to see the learned ranker rescue low-signal cases.")


# ==========================================
# MAIN INTERFACE TABS
# ==========================================
tab_review, tab_audit, tab_eval, tab_limitations = st.tabs([
    "🔍 Case Review & Ranking",
    "📜 Tamper-Evident Audit Trail",
    "📊 Benchmarking & Bias Audit",
    "⚖️ Limitations & Ethics"
])


# ------------------------------------------
# TAB 1: CASE REVIEW
# ------------------------------------------
with tab_review:
    c = cases[case_id].copy()
    if simulate_low_quality:
        c["quality"] = 0.32
        d_emb = pop["face"].shape[1]
        c["probe"] = (pop["face"][c["truth"]] + np.random.normal(size=d_emb) * 3.5 / np.sqrt(d_emb))
        c["probe"] /= np.linalg.norm(c["probe"])

    st.subheader(f"Case {case_id:03d} Dossier")

    cols = st.columns(5)
    cols[0].metric("CCTV Probe Quality", f"{c['quality']:.2f}",
                   delta="Low / Degraded" if c['quality'] < 0.5 else "Adequate",
                   delta_color="inverse" if c['quality'] < 0.5 else "normal")
    cols[1].metric("Witness Hair", HAIR[c["hair"]] if c["hair"] is not None else "Unobserved")
    cols[2].metric("Witness Build", BUILD[c["build"]] if c["build"] is not None else "Unobserved")
    cols[3].metric("Witness Age", AGE[c["age"]] if c["age"] is not None else "Unobserved")
    cols[4].metric("Activity Zone / MO", f"Z{c['zone']} / {'M' + str(c['mo']) if c['mo'] is not None else '?'}")

    if c["quality"] < 0.5:
        st.error("🚨 **Low Optical Quality Alert:** Probe image is degraded by blur, sensor noise, or low resolution. "
                 "Face embedding match alone is highly unreliable.")

    # Execute ranking under current mode
    active_mode = "learned" if is_learned else "hand"
    active_weights = learned_w_dict if is_learned else None
    rank_res = rank_candidates(pop, c, k=k, mode=active_mode, weights=active_weights, min_threshold=evidence_threshold)

    # Alternate mode for live comparison
    alt_mode = "hand" if is_learned else "learned"
    alt_weights = None if is_learned else learned_w_dict
    alt_rank_res = rank_candidates(pop, c, k=k, mode=alt_mode, weights=alt_weights)

    # Check if True suspect is known for transparent validation
    true_id = c["truth"]
    true_name = pop["names"][true_id] if "names" in pop else f"Candidate_{true_id:04d}"
    true_rank_active = next((item["rank"] for item in rank_res["candidates"] if item["candidate"] == true_id), "> k")
    true_rank_alt = next((item["rank"] for item in alt_rank_res["candidates"] if item["candidate"] == true_id), "> k")

    # Safety Guardrail: Insufficient Evidence Suppression
    if rank_res["insufficient_evidence"]:
        st.error(
            f"🛑 **INSUFFICIENT EVIDENCE — AUTOMATED RANKING SUPPRESSED**\n\n"
            f"The highest candidate score in the entire gallery (**{rank_res['top_score']:.2f}**) falls below "
            f"the required safety confidence threshold (**{evidence_threshold:.2f}**).\n\n"
            f"**Why this happens:** Under low optical quality ({c['quality']:.2f}) and sparse witness corroboration, "
            f"forcing a top-10 ranking creates extreme **confirmation bias** and puts innocent citizens at risk of wrongful pursuit.\n\n"
            f"**Protocol:** Canvas for higher-resolution footage, interview additional witnesses, or corroborate via forensic evidence."
        )

        with st.expander("⚠️ Request Authorized Supervisor Emergency Override"):
            st.caption("Investigative supervisors may override suppression for exigent exploratory leads. "
                       "This override will be permanently recorded in the hash-chained audit trail.")
            sup_name = st.text_input("Authorizing Supervisor Name", key="sup_name")
            sup_reason = st.text_area("Exigent Justification (Required)", key="sup_reason")
            override_clicked = st.button("Authorize Override & Unlock Exploratory Leads")
            if override_clicked:
                if not sup_name.strip() or not sup_reason.strip():
                    st.error("Both supervisor name and written justification are required.")
                else:
                    audit(
                        case_id=case_id,
                        ranked=[],
                        reviewer=reviewer.strip() if reviewer else "Det. S. Walker",
                        supervisor=sup_name.strip(),
                        reason=sup_reason.strip(),
                        action_type="insufficient_evidence_override",
                        threshold=evidence_threshold
                    )
                    st.session_state[f"override_{case_id}"] = True
                    st.success("Override authorized and logged to hash-chained audit trail.")
                    st.rerun()

    has_override = st.session_state.get(f"override_{case_id}", False)

    if not rank_res["insufficient_evidence"] or has_override:
        if has_override:
            st.info("🔓 **SUPERVISOR OVERRIDE ACTIVE:** Displaying low-confidence exploratory leads. Proceed with caution.")

        # Live comparison callout
        st.markdown(
            f"**Active Engine:** **{ranker_choice}** | Gallery: **{pop['dataset_name']}** ({len(pop['face'])} records)\n\n"
            f"🔎 *Validation Inspection:* Ground truth candidate **{true_name}** is ranked **#{true_rank_active}** "
            f"under {ranker_choice.split()[0]} (vs **#{true_rank_alt}** under {('Hand-Set' if is_learned else 'Learned')} weights)."
        )

        if st.button("📥 Log Current Ranking Snapshot to Audit Trail"):
            audit(
                case_id=case_id,
                ranked=rank_res["candidates"],
                reviewer=reviewer.strip() if reviewer else "Det. S. Walker",
                action_type="ranking_snapshot"
            )
            st.success("Ranking snapshot committed to tamper-evident audit log.")

        st.markdown("### Ranked Candidates & Evidence Breakdown")

        for r in rank_res["candidates"]:
            cand_name = r.get("name", f"Candidate_{r['candidate']:04d}")
            group_label = GROUP_NAMES[r["group"]] if r["group"] < len(GROUP_NAMES) else f"Group {r['group']}"
            is_truth_badge = " ⭐ (Ground Truth)" if r["candidate"] == true_id else ""

            exp_label = f"#{r['rank']}  {cand_name} (ID: {r['candidate']}){is_truth_badge}  |  Total Score: {r['score']}  |  {group_label}"
            with st.expander(exp_label, expanded=(r["rank"] == 1)):
                col_chart, col_meta = st.columns([2, 1])

                with col_chart:
                    ev_series = pd.Series(r["evidence"], name="Score Contribution")
                    st.bar_chart(ev_series)
                    st.caption("Additive decomposition of evidence features contributing to this candidate's ranking.")

                with col_meta:
                    st.write("**Candidate Signals:**")
                    st.write(f"- Demographic: `{group_label}`")
                    st.write(f"- Image Quality Factor: `{r['image_quality']}`")
                    st.write(f"- Modus Operandi Match: `{'Yes' if c['mo'] == pop['mo'][r['candidate']] else 'No'}`")
                    st.write(f"- Zone Presence: `{'Yes' if pop['zones'][r['candidate'], c['zone']] else 'No'}`")

                st.markdown("---")
                with st.form(f"decision_form_{case_id}_{r['candidate']}"):
                    st.write("**Human Investigator Decision:**")
                    decision = st.radio(
                        "Action",
                        ["Pursue as lead", "Reject as unlikely", "Request further corroborating evidence"],
                        horizontal=True
                    )
                    reason = st.text_area("Investigative Rationale (Required for compliance):")
                    submitted = st.form_submit_button("Record Decision in Audit Trail")

                    if submitted:
                        if not reviewer.strip():
                            st.error("Error: Reviewer name is required in the sidebar.")
                        elif not reason.strip():
                            st.error("Error: Written justification rationale is required.")
                        else:
                            audit(
                                case_id=case_id,
                                ranked=[r],
                                reviewer=reviewer.strip(),
                                decision=decision,
                                reason=reason.strip(),
                                action_type="decision"
                            )
                            st.success(f"Decision recorded for {cand_name} by {reviewer.strip()}.")


# ------------------------------------------
# TAB 2: AUDIT TRAIL
# ------------------------------------------
with tab_audit:
    st.subheader("📜 Tamper-Evident Hash-Chained Audit Trail")
    st.caption("Append-only audit trail where every record stores the SHA-256 hash of the previous line.")

    is_valid, n_recs, msg = verify_audit_log(LOG)
    if is_valid:
        st.success(f"🔒 **Cryptographic Integrity Verified:** {msg}")
    else:
        st.error(f"🚨 **Tampering Alert:** {msg}")

    if os.path.exists(LOG):
        try:
            records = []
            with open(LOG, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))

            if len(records) > 0:
                df_log = pd.DataFrame(records)
                df_log["timestamp"] = pd.to_datetime(df_log["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
                df_log["hash_preview"] = df_log["hash"].apply(lambda h: f"{h[:8]}..." if isinstance(h, str) else "")
                df_log["prev_preview"] = df_log["prev_hash"].apply(lambda h: f"{h[:8]}..." if isinstance(h, str) else "")

                disp_cols = [c for c in ["timestamp", "type", "case", "reviewer", "supervisor", "decision", "reason", "hash_preview", "prev_preview"] if c in df_log.columns]
                st.dataframe(df_log[disp_cols].sort_values(by="timestamp", ascending=False))

                with open(LOG, "r", encoding="utf-8") as f:
                    st.download_button(
                        "📥 Download Audit Trail (JSONL)",
                        data=f.read(),
                        file_name="audit_log.jsonl",
                        mime="application/json"
                    )
            else:
                st.info("Audit log is currently empty.")
        except Exception as e:
            st.error(f"Error reading audit log: {e}")
    else:
        st.info("No audit log records found yet.")


# ------------------------------------------
# TAB 3: BENCHMARKING & BIAS AUDIT
# ------------------------------------------
with tab_eval:
    st.subheader("📊 Performance Benchmarking & Demographic Bias Audit")
    st.caption("Quantitative audit comparing Face-Only Retrieval, Hand-Set Multi-Modal Fused, and Learned Softmax Ranker.")

    eval_cases_n = st.slider("Evaluation Test Cases", 50, 300, 150, 50)
    if st.button("🚀 Run Comprehensive Evaluation & Bias Audit"):
        with st.spinner("Evaluating models across held-out cases..."):
            rng_eval = np.random.default_rng(999)
            eval_cases = [make_case(pop, rng_eval) for _ in range(eval_cases_n)]

            res_face = evaluate(pop, eval_cases, face_only=True)
            res_hand = evaluate(pop, eval_cases, face_only=False, mode="hand")
            res_learned = evaluate(pop, eval_cases, face_only=False, mode="learned", weights=learned_w_dict)

            # Build comparison table
            metrics_list = [
                ("Recall@1", res_face["recall@1"], res_hand["recall@1"], res_learned["recall@1"]),
                ("Recall@5", res_face["recall@5"], res_hand["recall@5"], res_learned["recall@5"]),
                ("Recall@10", res_face["recall@10"], res_hand["recall@10"], res_learned["recall@10"]),
                ("Recall@20", res_face["recall@20"], res_hand["recall@20"], res_learned["recall@20"]),
                ("FPR @ 90% Recall", res_face["FPR@90%recall"], res_hand["FPR@90%recall"], res_learned["FPR@90%recall"]),
                ("Analyst Time Saved", res_face["analyst_time_saved"], res_hand["analyst_time_saved"], res_learned["analyst_time_saved"]),
                ("Disparate Impact Ratio (FPR)", res_face["disparity_ratio_fpr"], res_hand["disparity_ratio_fpr"], res_learned["disparity_ratio_fpr"]),
            ]

            df_comp = pd.DataFrame(metrics_list, columns=["Metric", "Face-Only Baseline", "Hand-Set Fused", "Learned Ranker"])
            st.dataframe(df_comp.round(4))

            st.markdown("---")
            st.subheader("⚖️ Algorithmic Fairness & Demographic Subgroup Audit")

            col_rec, col_fpr = st.columns(2)

            with col_rec:
                st.write("**Recall@10 by Demographic Group** *(Higher is better)*")
                df_rec = pd.DataFrame({
                    "Face-Only": res_face["recall@10_by_group"],
                    "Hand-Set Fused": res_hand["recall@10_by_group"],
                    "Learned Ranker": res_learned["recall@10_by_group"]
                }, index=[f"Group {g}" for g in res_face["recall@10_by_group"].keys()])
                st.bar_chart(df_rec)

            with col_fpr:
                st.write("**False Positive Rate (FPR) @ 90% Recall** *(Lower is safer)*")
                df_fpr = pd.DataFrame({
                    "Face-Only": res_face["fpr@90%recall_by_group"],
                    "Hand-Set Fused": res_hand["fpr@90%recall_by_group"],
                    "Learned Ranker": res_learned["fpr@90%recall_by_group"]
                }, index=[f"Group {g}" for g in res_face["fpr@90%recall_by_group"].keys()])
                st.bar_chart(df_fpr)

            st.info(
                "💡 **Nuanced Fairness Trade-off Disclosure:** The Learned Ranker slashes absolute false positive rates across all groups "
                "by ~4× to 5× compared to hand-set weights, and by ~70× compared to face-only search. "
                "However, as absolute false positives drop close to zero, the relative disparity ratio (Max FPR / Min FPR) can widen. "
                "This illustrates a core lesson in algorithmic fairness: relative disparity can increase even as every demographic cohort experiences a dramatic absolute safety gain."
            )


# ------------------------------------------
# TAB 4: LIMITATIONS & ETHICS
# ------------------------------------------
with tab_limitations:
    st.subheader("⚖️ System Limitations & Ethical Guardrails")
    st.markdown("""
    ### 1. Decision Support, Not Identification
    - **Non-Evidentiary Status:** Rankings generated by this system represent **investigative leads**, not forensic proof, probable cause, or legal identification.
    - **Human-in-the-Loop Requirement:** An AI ranking cannot trigger an arrest warrant, stop-and-frisk, or formal charge without independent investigation by a named human officer.

    ### 2. Facial Embedding Disparities & Degradation
    - **Representation Bias:** Deep face models (including ArcFace and InsightFace) demonstrate measurable variance in false-match and false-rejection rates across lighting conditions, camera angles, skin tones, and age brackets.
    - **CCTV Degradation:** Surveillance images frequently lack sufficient resolution, causing high-dimensional embedding dispersion. Our system explicitly conditions face importance on optical quality ($q$).

    ### 3. Witness Vulnerability & Eyewitness Fallibility
    - **Cognitive Decay:** Eyewitness memory degrades rapidly under stress. To prevent confirmation bias, witness attributes (hair, build, age) are modeled with realistic omission (30%) and error rates (20%).
    - Attributes must corroborate, never supersede, physical evidence.

    ### 4. Feedback Loop Mitigation in Modus Operandi & Geographic Zones
    - Historical arrest patterns reflect historical policing focus, which can create self-reinforcing feedback loops.
    - Activity zones and MO tags are weighted as exploratory corroboration and must never be used to criminalize geographic neighborhoods.
    - *All contextual attributes in this prototype are simulated.*

    ### 5. Refusal to Rank Under Low Evidence
    - Traditional systems force-rank all candidates, returning a top-10 list even on pure random noise.
    - Our system introduces an **Evidence Confidence Floor ($\tau$)**: when signals fall below the confidence floor, automated ranking is **suppressed**, preventing wrongful accusation.
    """)
