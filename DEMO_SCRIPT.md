# ⏱️ 2-Minute Demo Script: FairRank-ID

> **Presentation Goal:** Convince judges within 120 seconds that FairRank-ID solves the real algorithmic, ethical, and legal failure modes of police facial recognition through multimodal fusion, learned ranking, evidence suppression guardrails, and cryptographic auditability.

---

## 🕒 Beat 1: The Hook & The Problem (0:00 – 0:25)

**On Screen:**
- Streamlit app open on `http://localhost:8501`.
- Sidebar showing `Reviewer Name: Det. S. Walker`, select `Case 017` (Image quality: 0.49).

**Speaker Dialogue:**
> *"Most law-enforcement AI suffers from a fatal flaw: tunnel vision. When surveillance footage is blurry, traditional facial recognition forces a top-10 list out of pure noise, leading detectives directly into confirmation bias and wrongful accusations.*
> 
> *This is FairRank-ID. It is strictly decision support: it ranks investigative leads, never identifies suspects, and refuses to output a ranking when evidence is insufficient."*

---

## 🕒 Beat 2: Live Toggle — Hand-Set vs. Learned Ranker (0:25 – 0:55)

**On Screen:**
- On `Case 017` with **Hand-Set Weights (Heuristic)** selected:
  - Point out that the ground-truth candidate is buried outside the top-10 (`> k`).
- In sidebar, click the radio switch to **Learned Ranker (Softmax Loss)**.
- The candidate ranking updates live on screen, bringing the ground truth suspect to **Rank #1**!

**Speaker Dialogue:**
> *"Look at our first major breakthrough: our **Ranking Engine Toggle**. 
> 
> Here in Case 017, under hand-set heuristic weights, the true candidate isn't even in the top 10 because CCTV quality was only 0.49. 
> 
> Now watch what happens when I flip the sidebar to our **Learned Softmax Ranker**.
> 
> Boom—instantly re-ranked. The true suspect jumps straight to **Rank #1**. 
> 
> Why? Because our learned ranker was trained with listwise softmax loss to discover non-linear feature interactions—specifically `Face × Quality`. When optical quality drops, it automatically down-weights the noisy face vector and leans on witness attributes and activity zones."*

---

## 🕒 Beat 3: Safety Guardrails & Insufficient Evidence Suppression (0:55 – 1:25)

**On Screen:**
- Check the sidebar box: **"Simulate Degraded CCTV (Quality = 0.32)"**.
- The candidate list vanishes, replaced by the prominent red warning:
  `🛑 INSUFFICIENT EVIDENCE — AUTOMATED RANKING SUPPRESSED`.
- Show the **Authorized Supervisor Override** expander.

**Speaker Dialogue:**
> *"Now for the safety story—the part judges weigh most heavily. 
> 
> What happens when an image is genuinely too degraded to trust? Let’s simulate degraded CCTV. 
> 
> Notice what the system does: **It refuses to rank.** 
> 
> Instead of guessing and putting 10 innocent people under suspicion, our **Evidence Confidence Floor** suppresses the ranking. It explicitly instructs investigators to canvas for better footage or forensic evidence. 
> 
> If an exigent circumstance exists, an authorized supervisor must input their name and a mandatory written justification to override the gate—and that override is permanently logged."*

---

## 🕒 Beat 4: Bias Auditing & Transparent Numbers (1:25 – 1:50)

**On Screen:**
- Uncheck "Simulate Degraded CCTV".
- Click over to **Tab 3: "Benchmarking & Bias Audit"**.
- Click **"Run Comprehensive Evaluation & Bias Audit"**.
- Show the results table and the **Per-Group False Positive Rate (FPR)** bar chart.

**Speaker Dialogue:**
> *"Look at our fairness audit. We don't just measure Recall—we audit **Per-Group False Positive Rates** at the 90% recall threshold. 
> 
> On Face-Only search, innocent people face a **4.2% false alarm rate**, and noisy cohorts suffer a recall drop to 75%. 
> 
> Our Learned Ranker slashes absolute false positives across all groups to **0.06%**—a 70-fold reduction in false accusations—while restoring cohort recall to **91.7%**. 
> 
> We also report the honest trade-off: because absolute false alarms drop close to zero, the relative disparity ratio widens slightly. That's a real-world fairness trade-off every judge appreciates seeing."*

---

## 🕒 Beat 5: Tamper-Evident Audit Trail & Conclusion (1:50 – 2:00)

**On Screen:**
- Switch to **Tab 2: "Tamper-Evident Audit Trail"**.
- Point out the green badge: `🔒 Cryptographic Integrity Verified: Hash chain valid`.

**Speaker Dialogue:**
> *"Finally, every decision and override is committed to an append-only, SHA-256 hash-chained audit trail. If a single byte is tampered with out-of-band, the system detects it immediately.
> 
> FairRank-ID turns facial AI from an unaccountable black box into an audited, legally defensible partner in public safety. Thank you!"*

---

## 🎯 Quick Demo Checklist for Presenters

1. [ ] App running: `streamlit run app.py`
2. [ ] Reviewer name entered: `Det. S. Walker`
3. [ ] Select **Case 017**: Switch between Hand-Set and Learned weights (shows rank boost from `>10` to `#1`).
4. [ ] Toggle **"Simulate Degraded CCTV"** (triggers Insufficient Evidence refusal gate).
5. [ ] Open **Tab 3: Benchmarking & Bias Audit** (show 70× drop in FPR to 0.06% and Group 2 recall boost to 91.7%).
6. [ ] Open **Tab 2: Audit Trail** (show SHA-256 hash chain verification).
