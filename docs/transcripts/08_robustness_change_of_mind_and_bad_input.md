# Change of mind, invalid input and a property question

Invalid postcode, phone and email are re-asked politely. The visitor switches from buying to selling mid-flow. A question about a listing is answered and the flow resumes.

*Recorded 2026-10-21 against the running bot, mode: LLM `openai/gpt-oss-120b` on `groq`; embeddings: `BAAI/bge-small-en-v1.5`. Phone and email are fake test values.*

**Bot:** How may I help you?

**Visitor:** hi

**Bot:** Can I get your name?

**Visitor:** Sam

**Bot:** Thanks, Sam. Are you looking to buy or sell a property?

**Visitor:** buy

**Bot:** Do you have a specific property in mind?

**Visitor:** no

**Bot:** Are you looking for a new-build or a resale home?

**Visitor:** new-build

**Bot:** What is your budget?

**Visitor:** two million

**Bot:** Postcode of your location of interest?

**Visitor:** How many bedrooms does Apartment 1, Aldermoor Place have?

**Bot:** Apartment 1, Aldermoor Place has 2 bedrooms.
> 
> Postcode of your location of interest?

*side question (property) answered, flow resumed at `A3_POSTCODE`*

**Visitor:** actually I want to sell my home instead

**Bot:** No problem. What is your postcode?

**Visitor:** SW3

**Bot:** Sorry, that doesn't look like a full UK postcode (for example SW3 3RB). What is your postcode?

**Visitor:** GU1 3AD

**Bot:** Can I get your phone number?

**Visitor:** 12345

**Bot:** Sorry, that doesn't look like a valid phone number. Can I get your phone number?

**Visitor:** 07911 123456

**Bot:** Can I get your email address?

**Visitor:** not-an-email

**Bot:** Sorry, that doesn't look like a valid email address. Can I get your email address?

**Visitor:** sam@example.com

**Bot:** Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?

**Visitor:** no

**Bot:** Thank you for chatting with us. Goodbye.

---
Final state: `step=ENDED` · `outcome=QUALIFIED` · `qualified=True` · `ended=True`
