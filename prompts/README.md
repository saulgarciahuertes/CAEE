# Included Prompt Templates

`llama3_chat_template.jinja` is the exact tokenizer chat-format template used by the four supplied training notebooks. It serializes role/content messages; it is not the prompt that generated the augmented study data.

`geval_scoring_rubric.txt` is the exact generic four-dimension rubric used by the GPT scorer. It requests coherence, consistency, fluency, and relevance ratings directly. The user-message template supplying question, reference answer, and generated answer is recorded in `configurations/evaluation.json` and in the notebook source.

The inference notebook retains the original generic career-trajectory system instruction and plain system/user/assistant formatting. No participant-specific prompt instances are included.

`preparation_prompts.json` contains the recovered original academic, professional, innovation, entrepreneurship, motivation, and CV prompts, the three narrative-style suffixes, both system messages, and the contrasting-answer prompt prefix from `emnlp.ipynb`. They are generic templates only, not participant-specific request instances. The corresponding models/settings and positional field slices are in `configurations/preparation.json`.

The narrative omission instruction applies to institution/company names in the output. It does not redact those strings before they are sent. The original upstream 90/10 split code and complete historical API request logs remain unrecovered.
