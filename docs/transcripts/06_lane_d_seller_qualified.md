# Lane D: seller, postcode covered

A seller's postcode is checked against the seller coverage list only.

*Recorded 2026-10-21 against the running bot, mode: LLM `openai/gpt-oss-120b` on `groq`; embeddings: `BAAI/bge-small-en-v1.5`. Phone and email are fake test values.*

**Bot:** How may I help you?

**Visitor:** hello

**Bot:** Can I get your name?

**Visitor:** Sam

**Bot:** Thanks, Sam. Are you looking to buy or sell a property?

**Visitor:** I'm thinking of putting my flat on the market

**Bot:** What is your postcode?

**Visitor:** gu1 3ad

**Bot:** Can I get your phone number?

**Visitor:** 07911 123456

**Bot:** Can I get your email address?

**Visitor:** sam@example.com

**Bot:** Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?

**Visitor:** yes please

**Bot:** How can I help you? Are you looking to buy or sell a property?

**Visitor:** actually I want to buy

**Bot:** Do you have a specific property in mind?

**Visitor:** no

**Bot:** Are you looking for a new-build or a resale home?

**Visitor:** resale

**Bot:** What is your budget?

**Visitor:** £500k

**Bot:** Postcode of your location of interest?

**Visitor:** GU1 3AD

**Bot:** Here are some homes that match what you're looking for:
> 1. 72 Shalford Road – 4 bed – £1,335,000
> 2. 136 Shalford Road – 5 bed – £1,450,000
> 3. 61 Shalford Road – 5 bed – £1,830,000
> 
> Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?

**Visitor:** no thanks

**Bot:** Thank you for chatting with us. Goodbye.

---
Final state: `step=ENDED` · `outcome=QUALIFIED` · `qualified=True` · `ended=True`
