# Bartleby — gate de LGPD para agentes de IA

Linter estático de LGPD que roda no seu CI. A cada pull request, ele varre o código do seu agente
de IA, detecta tratamento de dados pessoais e emite **achados nomeados** (regra · severidade ·
artigo da LGPD · `arquivo:linha` · remediação) — **determinístico, sem LLM, sem chave de API, sem
licença**. O código **nunca sai do runner**.

Grátis e open-core (MIT). É o tier Gate do [Bartleby](https://bartleby.com.br).

## GitHub Action

```yaml
# .github/workflows/lgpd-gate.yml
name: LGPD gate
on:
  pull_request:
    branches: [main]
jobs:
  bartleby:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write   # para o upload-sarif anotar o PR
    steps:
      - uses: actions/checkout@v4
      - name: Bartleby LGPD gate
        id: gate
        uses: MKanhan/bartleby-LGPD@v1
        with:
          path: "."
          fail-on: "alto"        # none | baixo | medio | alto | critico
      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: ${{ steps.gate.outputs.sarif-file }}
```

`fail-on: none` só relata (nunca reprova) — bom para os primeiros PRs. `alto` reprova o build
quando há achado de severidade Alto ou Crítico. Com o `upload-sarif`, cada achado aparece anotado
na linha exata do PR (GitHub code-scanning).

## CLI

```bash
pip install bartleby-lgpd
bartleby check .                    # só relatório
bartleby check . --fail-on alto     # reprova em Alto/Crítico
bartleby check . --sarif out.sarif  # emite SARIF 2.1.0
bartleby check . --scan-out scan.json [--redact]   # exporta o scan (sem enviar o código)
```

Exit codes: `0` passou · `2` fonte/limite inválido · `3` scan vazio · `4` gate reprovou.

## Veja funcionando

Demonstração com o gate rodando num pull request real (agente de IA fictício):
**[MKanhan/bartleby-gate-demo](https://github.com/MKanhan/bartleby-gate-demo)** →
[PR #1](https://github.com/MKanhan/bartleby-gate-demo/pull/1) — o gate reprova o PR e anota cada
achado na linha exata (SARIF / code-scanning).

## O gate é o alarme; o RIPD é a remediação

Quando o gate acende, gere a documentação completa (RIPD Art. 38 + ROPA Art. 37 + Mapa de Riscos)
com o Bartleby: <https://bartleby.com.br>.

**Sem entregar o repositório.** `bartleby check . --scan-out scan.json` grava o resultado do scan
num JSON indentado que você lê antes de enviar; `--redact` tira dele qualquer linha de código
(estrutura, caminhos e severidades ficam — o risco calculado é o mesmo). É esse arquivo, e só ele,
que o Bartleby precisa para escrever os documentos.

---
Este pacote é derivado automaticamente do produto Bartleby. Licença MIT.
