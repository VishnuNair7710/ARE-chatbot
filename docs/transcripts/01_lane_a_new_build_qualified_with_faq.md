# Lane A: new-build buyer, qualified, FAQ question mid-flow

Budget over £1M, postcode with an available new-build, matches shown, contact details, call-back. The head-office question is answered from the FAQ and the budget question is then asked again.

*Recorded 2026-10-21 against the running bot, mode: LLM `openai/gpt-oss-120b` on `groq`; embeddings: `BAAI/bge-small-en-v1.5`. Phone and email are fake test values.*

**Bot:** How may I help you?

**Visitor:** hello

**Bot:** Can I get your name?

**Visitor:** Sam

**Bot:** Thanks, Sam. Are you looking to buy or sell a property?

**Visitor:** I'd like to buy a home

**Bot:** Do you have a specific property in mind?

**Visitor:** no

**Bot:** Are you looking for a new-build or a resale home?

**Visitor:** a new-build please

**Bot:** What is your budget?

**Visitor:** What is your head office address?

**Bot:** Our head office is at 22 Halden Mews, Mayfair, London W1K 2PX.
> 
> What is your budget?

*side question (faq) answered, flow resumed at `A1_BUDGET`*

**Visitor:** around 2 million

**Bot:** Postcode of your location of interest?

**Visitor:** GU5 7AQ

**Bot:** Here are some homes that match what you're looking for:
> 1. Apartment 5, Fairview Terrace – 3 bed – £1,070,000
> 2. Apartment 1, Fairview Terrace – 4 bed – £1,340,000
> 3. Townhouse 3, Fairview Terrace – 3 bed – £1,365,000
> 4. Townhouse 4, Fairview Terrace – 4 bed – £1,690,000
> 5. Penthouse 2, Fairview Terrace – 3 bed – £1,705,000
> 
> Can I get your phone number?

**Visitor:** 07911 123456

**Bot:** Can I get your email address?

**Visitor:** sam@example.com

**Bot:** Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?

**Visitor:** no thanks

**Bot:** Thank you for chatting with us. Goodbye.

---
Final state: `step=ENDED` · `outcome=QUALIFIED` · `qualified=True` · `ended=True`
