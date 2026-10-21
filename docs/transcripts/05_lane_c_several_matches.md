# Lane C: two listings share a name

'The Old Rectory' exists twice. The bot asks which one, the visitor answers in words.

*Recorded 2026-10-21 against the running bot, mode: LLM `openai/gpt-oss-120b` on `groq`; embeddings: `BAAI/bge-small-en-v1.5`. Phone and email are fake test values.*

**Bot:** How may I help you?

**Visitor:** hi

**Bot:** Can I get your name?

**Visitor:** Sam

**Bot:** Thanks, Sam. Are you looking to buy or sell a property?

**Visitor:** I want to buy

**Bot:** Do you have a specific property in mind?

**Visitor:** yes

**Bot:** Which property is it?

**Visitor:** The Old Rectory

**Bot:** I found a few matches:
> 1. The Old Rectory – The Old Rectory, Markham Square
> 2. The Old Rectory – The Old Rectory, Sheen Road
> Which one do you mean?

**Visitor:** the one on Sheen Road

**Bot:** The Old Rectory is a detached house with 3 bedrooms, priced at £1,490,000.
> 
> Can I get your phone number?

**Visitor:** 07911 123456

**Bot:** Can I get your email address?

**Visitor:** sam@example.com

**Bot:** Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?

**Visitor:** no

**Bot:** Thank you for chatting with us. Goodbye.

---
Final state: `step=ENDED` · `outcome=QUALIFIED` · `qualified=True` · `ended=True`
