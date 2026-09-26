# CLAUDE.md

Guidance for working in this repo.

- This is a workspace in which an app service will be quickly built and pushed to GitHub, so prepare to create a codbase with that in mind
- DO NOT over-engineer beyond what is described. Conventional code practices can be sacrificed in the name of lean code.
- Prioritise speed over exhaustive exploration when building and iteratively refining the build.
- If the service being built requires access to an LLM, use OpenRouter interface code to access this.
- Write code in a way that is intuitive and modular, where one solution can be tidily swapped out for another — e.g. if the approach to a TTS task needs to change, the relevant code should be easy to locate and swap out.
- Do not produce lengthy docstrings — one or two lines of comments per function, class, or script is enough.
- Don't be too clever with the code. Basic setups are fine.
- Every time you complete a task or adaptation, update the README accordingly