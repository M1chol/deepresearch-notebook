# Deep Research Notebook v0.2
Early version of my ChatGPT-style deep research workflow.

Working:
- Research planning phase
- Topic expansion
- Question generation and optional question selection
- Final research plan output
- YAML-defined pipeline orchestration
- Per-plan-point web search through SearxNG
- Automatic broad-query retries when a search has no useful results
- Cheap-model result selection and page summarization
- Mid-model step synthesis
- Final report generation from all collected step and source summaries
- Per-call token, cost, latency, and throughput statistics
- Human in the loop questioning

Planned:
- Review loop
- Textual UI
- Research effort selection
- Local integration with ollama

The UI is not yet working as the main workflow interface, use CLI instead.

## Current single run statistics:
- Calls: 147
- Prompt tokens: 552,599
- Completion tokens: 42,439
- Reasoning tokens: 21,551
- Total tokens: 595,038
- Cost: $0.170518
- LLM response time: 1772.67s
- Completion throughput: 23.9 tok/s

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
research-pipeline "Your research topic"
research-pipeline "Your research topic" --name custom-research-name
research-pipeline "Your research topic" --skip-search
```

The default folder name is derived from the generated `#` research title:

```text
results/generated-research-title/
├── research_plan.md
├── final_report.md
├── nerd_stats.json
├── nerd_stats.md
├── planning/
└── searches/
```

Search requires a SearxNG instance at `http://localhost:8080/search`. Use
`--searxng-url` to select another JSON endpoint.

Run the experimental UI module:

```bash
cd research_ui
python -m  research_ui.research_cli
```
## Configuration
Right now possible configurations include:
- changing the complete pipeline workflow and prompts in `prompts.yaml`
- modifying default cheap, mid, and reasoning models in `config.py`
- configuring query retries and final report output under `research:` in
  `prompts.yaml`
