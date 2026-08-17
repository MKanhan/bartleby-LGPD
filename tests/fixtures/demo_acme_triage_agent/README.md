# ACME Seguros — Triagem de Sinistros (FICTÍCIO)

This is a fictional reference repository used for Bartleby (agent-lgpd) demos. **No real personal data, no real API keys, no real customers.** Every CPF, CNPJ, email, and phone in this tree is synthetic and was generated for training/demo purposes.

The repo simulates a typical Brazilian insurance triage agent:

- FastAPI backend exposing `/triage` for new claim submissions
- Dual LLM providers (OpenAI primary, Anthropic fallback)
- Pinecone vector store for retrieving historical similar claims
- Redis-backed conversation memory (per claimant CPF, 24h TTL)
- LangSmith telemetry on every LLM call
- Tool calls with side effects (`request_human_review`, `notify_claimant_email`, `lookup_policy_status`)
- Next.js admin panel under `web/` to triage queue management

## Why this fixture exists

Bartleby's pipeline becomes informative when the input exercises ≥10 of the 18 AI-specific risks in `app/templates/catalogs/riscos.yml`. Real client repos may not be available for early demos; minimalist test fixtures (e.g. `openai_triage_agent/`) prove unit correctness but produce thin reports. ACME sits in between: realistic enough to demo, fictional enough to ship in-tree.

## Run with Bartleby

```bash
python scripts/run_demo.py --provider anthropic
```

See `docs/demos/canonical_demo.md` for talking points and expected output.
