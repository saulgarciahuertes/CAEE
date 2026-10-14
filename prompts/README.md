# Included Prompt Templates

`llama3_chat_template.jinja` is the exact tokenizer chat-format template used by the four supplied training notebooks. It serializes role/content messages; it is not the prompt that generated the augmented study data.

`geval_scoring_rubric.txt` is the exact generic four-dimension rubric used by the GPT scorer. It requests coherence, consistency, fluency, and relevance ratings directly. The user-message template supplying question, reference answer, and generated answer is recorded in `configurations/evaluation.json` and in the notebook source.

The inference notebook retains the original generic career-trajectory system instruction and plain system/user/assistant formatting. No participant-specific prompt instances are included.

The original data-augmentation and rejected-answer-generation request templates were not recovered. They have not been reconstructed or presented as original prompts in this repository.
