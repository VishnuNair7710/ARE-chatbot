# Sample transcripts

Recorded by `scripts/record_transcripts.py`: scripted visitors talk to the real bot; the text is its actual output, not hand-written. Re-run the script to regenerate them.

Recorded 2026-10-21 · mode: LLM `openai/gpt-oss-120b` on `groq`

| Transcript | What it shows |
|---|---|
| [Lane A: new-build buyer, qualified, FAQ question mid-flow](01_lane_a_new_build_qualified_with_faq.md) | Budget over £1M, postcode with an available new-build, matches shown, contact details, call-back. The head-office question is answered from the FAQ and the budget question is then asked again. |
| [Lane B: resale buyer, qualified (any budget)](02_lane_b_resale_qualified.md) | The budget is recorded but never filters resale. Natural wording at every step. |
| [Lane C: buyer with a specific property (available)](03_lane_c_specific_property_available.md) | The visitor names a listing in the first message. The bot still asks the name once, then shares the type, bedrooms and price and moves to contact details. |
| [Lane C: the property is sold, visitor is re-qualified](04_lane_c_property_sold_requalified.md) | A sold new-build: the bot offers a similar home, asks the budget once, then continues at the postcode step of lane A without asking the budget again. |
| [Lane C: two listings share a name](05_lane_c_several_matches.md) | 'The Old Rectory' exists twice. The bot asks which one, the visitor answers in words. |
| [Lane D: seller, postcode covered](06_lane_d_seller_qualified.md) | A seller's postcode is checked against the seller coverage list only. |
| [Closing messages: M1, M2, M3 (and no contact details asked)](07_closing_messages_m1_m2_m3.md) | Three short conversations. In each the visitor is not qualified: the bot gives the closing message, the office number and says goodbye, and never asks for a phone number or email. |
| [Change of mind, invalid input and a property question](08_robustness_change_of_mind_and_bad_input.md) | Invalid postcode, phone and email are re-asked politely. The visitor switches from buying to selling mid-flow. A question about a listing is answered and the flow resumes. |
| [Safety: prompt injection and off-topic requests](09_safety_injection_and_off_topic.md) | The £1M rule cannot be talked away. Off-topic requests are declined politely and the flow resumes. |
