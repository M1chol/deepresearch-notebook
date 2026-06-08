# Deep Research Notebook v0.1
Early prototype of my ChatGPT-style deep research workflow.

Working:
- Research planning phase
- Topic expansion
- Question generation and optional question selection
- Final research plan output
- Human in the loop questioning
- Streaming / non streaming

Planned:
- Search phase
- Final report generation
- Review loop
- Textual UI
- Research effort selection
- Local integration with ollama

The UI is not yet working as the main workflow interface, use CLI instead.

Current run statistics:
planning phase: 12k-17k tokens or about 0.04$-0.06$ on KimiK2.6

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` with:

```bash
OPENROUTER_API_KEY=your_key_here
```

## Launch

Run the research planning CLI:

```bash
python -m research_pipeline.cli --help
```

If you want to install instead
```bash
pip install -e .
``` 
and then run with
```bash
research-pipeline --help
```

Useful options:

```bash
research-pipeline --help
research-pipeline "Your research topic" --max-questions 5
research-pipeline --topic-file topic.md --human-in-loop
research-pipeline "Your research topic" --output research_plan.md
```

Run the experimental UI module:

```bash
cd research_ui
python -m  research_ui.research_cli
```
## Configuration
Right now possible configurations include:
- changing research prompts in `prompts.yaml`
- modyfying default models in `config.py`