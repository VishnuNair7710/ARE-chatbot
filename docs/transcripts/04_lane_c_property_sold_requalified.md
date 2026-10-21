# Lane C: the property is sold, visitor is re-qualified

A sold new-build: the bot offers a similar home, asks the budget once, then continues at the postcode step of lane A without asking the budget again.

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

**Visitor:** Townhouse 2, Riverside Quarter

**Bot:** Sorry, that one's gone. I can help you find a similar home. What's your budget?

**Visitor:** 1.8 million

**Bot:** Postcode of your location of interest?

**Visitor:** GU1 1FU

**Bot:** Here are some homes that match what you're looking for:
> 1. Apartment 3, Juniper Lane Mews – 3 bed – £1,150,000
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
