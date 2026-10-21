# Lane B: resale buyer, qualified (any budget)

The budget is recorded but never filters resale. Natural wording at every step.

*Recorded 2026-10-21 against the running bot, mode: LLM `openai/gpt-oss-120b` on `groq`; embeddings: `BAAI/bge-small-en-v1.5`. Phone and email are fake test values.*

**Bot:** How may I help you?

**Visitor:** hi there

**Bot:** Can I get your name?

**Visitor:** Sam

**Bot:** Thanks, Sam. Are you looking to buy or sell a property?

**Visitor:** I want to buy

**Bot:** Do you have a specific property in mind?

**Visitor:** nothing specific

**Bot:** Are you looking for a new-build or a resale home?

**Visitor:** a resale home

**Bot:** What is your budget?

**Visitor:** about 650k

**Bot:** Postcode of your location of interest?

**Visitor:** TW9 5FL

**Bot:** Here are some homes that match what you're looking for:
> 1. 128 Sheen Road – 3 bed – £1,060,000
> 2. 78 Sheen Road – 3 bed – £1,110,000
> 3. The Old Rectory – 3 bed – £1,490,000
> 4. Ivy Cottage – 5 bed – £2,700,000
> 5. 141 Sheen Road – 6 bed – £2,860,000
> 
> Can I get your phone number?

**Visitor:** 07911 123456

**Bot:** Can I get your email address?

**Visitor:** sam@example.com

**Bot:** Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?

**Visitor:** that's all, thanks

**Bot:** Thank you for chatting with us. Goodbye.

---
Final state: `step=ENDED` · `outcome=QUALIFIED` · `qualified=True` · `ended=True`
