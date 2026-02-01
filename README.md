# RAG-QA

**Retrieval-Augmented Generation (RAG) Q&A** application using Dockerized or locally hosted Ollama models.  
Supports both **basic** and **semantic** chunking strategies for PDF documents.

## Quick Start
### Ollama
**Recommended:** Dockerized Ollama 
- `docker compose up -d rag-qa-ollama` in the root folder

In any other case, user must ensure that ollama is running in the same network as rag / or that they somehow communicate. And the environment variable `OLLAMA_HOST` should be set accordingly. 

***Dockerized Ollama benefits:***
- Persistent models via Docker volumes - no need to re-pull models between runs
- Hardware agnostic - same setup works across machines without manual GPU/CPU tuning
- Clean separation between:
    - Model runtime (Ollama)
    - Application logic (rag-qa)

### RAG
- Build the application docker image: `docker compose build rag`
- Run the built image, providing:
    - the question (required), 
    - any optional CLI arguments 
-> See the examples below for full `docker run` commands.

#### Alternatives
1. **Use the image with the preloaded corpus only**
   - Embeddings for the three provided PDF files are precomputed and bundled.
   - This results in faster startup and query execution.

2. **Ingest a custom corpus**
   - Navigate to the directory containing the PDF files you want to ingest.
   - Run the built image from that directory.
   - When using the volume mount `-v rag-qa-storage:/app/storage`, embeddings for these PDFs are persisted and reused across runs.

#### Examples
- Local hosted Ollama with preloaded corpus only
    - Run from the project root directory

```bash
docker run \
  --name rag-qa \
  --network host \
  -e OLLAMA_HOST=http://localhost:11434 \
  --rm \
  rag-qa:latest \
  [question] 
```

- Dockerized Ollama with preloaded corpus only
    - Run from the project root directory

```bash
docker run \
  --name rag-qa \
  --network rag-qa-default \
  -e OLLAMA_HOST=http://rag-qa-ollama:11434 \
  --rm \
  rag-qa:latest \
  [question]
```
---
- Local hosted Ollama with custom documents
    - Run from the directory containing the PDFs you want to ingest
```bash
docker run \
  --name rag-qa \
  --network host \
  -e OLLAMA_HOST=http://localhost:11434 \
  -v rag-qa-storage:/app/storage \
  -v "$(pwd):/app/docs" \
  --rm \
  rag-qa:latest \
  [question]
```

- Dockerized Ollama with custom documents
    - Run from the directory containing the PDFs you want to ingest

```bash
docker run \
  --name rag-qa \
  --network rag-qa-default \
  -e OLLAMA_HOST=http://rag-qa-ollama:11434 \
  -v rag-qa-storage:/app/storage \
  -v "$(pwd):/app/docs" \
  --rm \
  rag-qa:latest \
  [question]
```
**Notes:**

- **Persistent embeddings:**  
  If the `-v rag-qa-storage:/app/storage` volume is omitted, newly ingested documents will **not** be persisted and their embeddings will be recomputed on every run.

- **Working directory matters:**  
    When running the last two run commands, the current working directory is treated as the document source. It must therefore contain both the original corpus and any additional PDFs. The `/app/docs` directory inside the container is ignored.

- **Changing chunking strategy:**  
  To switch the chunking strategy, remove the existing embeddings volume before re-running the container:
  ```bash
  docker volume rm rag-qa-storage
  ```
---

## General Notes

- **CLI arguments supported:**
  - `--chunking-strategy` (`basic`, `semantic`)
  - `--top-k`
  - `--chunk-size`
  - `--overlap-ratio`

- **Important:**  
  The question is a **positional argument** and must always be provided first, before any optional CLI flags.

---

## Limitations & Future Improvements

- **Embedding cache management:**  
  Currently, embeddings are persisted in a Docker volume. When ingesting new PDFs or changing the chunking strategy, stale `.pkl` files may remain. This can be addressed by:
  - Removing and recreating the entire volume
  - Manually deleting specific `.pkl` files
  - Automatically cleaning up unused or outdated cache files at the end of execution

- **Installation ergonomics:**  
  A shell script (e.g. `setup.sh`) could be added to streamline the setup process, including image builds, volume creation, and dependency checks.

---

## Sample queries
**1.** *Chunking strategy: semantic*
**Q:** How can I include data not mentioned in the ICH E3 text or appendices since the\nguidance predates the ICH M4 guidance associated with the CTD and Electronic\nCommon Technical Document (eCTD)?
**A:** To include additional data, you should create new headings within the Common Systematic Report (CSR) and add appropriate sections to accommodate topics such as pharmacokinetics, pharmacodynamics, pharmacogenomics, gene therapy, stem cells, biomarkers, devices, quality of life, assay validation, data monitoring/review committees, electrocardiogram results, other safety reports, images, pictures or scans, diagnostic tests for individualized therapy, and patient-reported outcomes. These should be well referenced in the CSR body with clear identification within the table of contents using Arabic numbers as per ICH E3 guidance endorsement by Step 4 on June 2012. The specific presentation can vary based on relevancy to your study, recognizing that not all subjects may need detailed sections due to variations in research design and goals. Remember to provide a rationale if some aspects of the original text are omitted or reordered according to ICH E3 guidance flexibility principles.

```json
{
  "trace_id": "bc492a23-28d8-4866-9cc4-293c70da2ac8",
  "question": "How can I include data not mentioned in the ICH E3 text or appendices since the\nguidance predates the ICH M4 guidance associated with the CTD and Electronic\nCommon Technical Document (eCTD)?",
  "plan": [
    "retrieve",
    "draft",
    "cite"
  ],
  "retrieval": [
    {
      "file": "docs/E3 Structure - Document 2.pdf",
      "chunk_id": 103,
      "score": 0.7678,
      "section_path": [
        "Q4: \t How can I include data not mentioned in the ICH E3 text or appendices since the"
      ]
    },
    {
      "file": "docs/E3 Structure - Document 2.pdf",
      "chunk_id": 97,
      "score": 0.6851,
      "section_path": [
        "II. \t QUESTIONS AND ANSWERS",
        "A. \t Content and Structure (1)",
        "Q1: \t Some in the pharmaceutical industry have expressed concern that the ICH E3"
      ]
    },
    {
      "file": "docs/E3 Structure - Document 2.pdf",
      "chunk_id": 98,
      "score": 0.6418,
      "section_path": []
    }
  ],
  "draft_tokens": 1098,
  "latency_ms": {
    "retrieve": 620,
    "draft": 19587,
    "total": 20207
  },
  "errors": []
}
```


**2.** *Chunking strategy: semantic*
**Q:** What are some Advantages of Placebo-controlled Trials?
**A:** a. Ability to Demonstrate Efficacy  
A placebo-controlled trial can provide strong internal evidence that a treatment is more effective than no treatment because it compares the outcomes between treated subjects and those given an inactive substance (placebo). If there's a significant difference, this suggests efficacy of the drug. 
b. Measures Absolute Efficacy and Safety  
Placebo-controlled trials evaluate both therapeutic benefit as well as adverse events directly attributable to the treatment itself rather than those due to underlying disease or other factors (background noise). This allows a clear distinction of drug effects from non-drug influences. 
c. Efficiency  
These trials often require fewer participants compared to others types, reducing costs and resources needed while maintaining power in detecting treatment differences
```json
{
  "trace_id": "3ed19842-2221-4ccd-a3eb-70980768bb3c",
  "question": "What are some Advantages of Placebo-controlled Trials?",
  "plan": [
    "retrieve",
    "draft",
    "cite"
  ],
  "retrieval": [
    {
      "file": "docs/E10 - Document 3.pdf",
      "chunk_id": 54,
      "score": 0.6744,
      "section_path": [
        "6. Advantages of Placebo-controlled Trials (2.1.6)"
      ]
    },
    {
      "file": "docs/E10 - Document 3.pdf",
      "chunk_id": 6,
      "score": 0.6592,
      "section_path": [
        "II. DETAILED CONSIDE RATION OF TYPES OF CONTROL (2.0) ................................ ............................. 14",
        "A. PLACEBO CONTROL (2.1) ................................ ................................ ................................ ................................ ....14",
        "6. Advantages of Placebo-controlled Trials (2.1.6) ................................ ................................ ......................... 19"
      ]
    },
    {
      "file": "docs/E10 - Document 3.pdf",
      "chunk_id": 55,
      "score": 0.6375,
      "section_path": []
    }
  ],
  "draft_tokens": 335,
  "latency_ms": {
    "retrieve": 4696,
    "draft": 14438,
    "total": 19136
  },
  "errors": []
}
```

**3.** *Chunking strategy: semantic*
**Q:** What is the clinical investigator?
**A:** A Clinical Investigator, according to Document 1 on page 40 and further explained in document 2 starting from Page 65 of "docs/ICD - Document 1.pdf," refers to a person responsible for ensuring subjects' rights, safety, and welfare during clinical investigations. They are also accountable for getting legally effective informed consent unless an exception applies or the study is exempted from such requirements due to its nature being minimal risk as detailed in 45 CFR part 46 (referenced at pages 27-30). If delegating this responsibility, it must be done by someone educated, trained, and experienced enough. They should also make sure that any consent discussions are appropriately supervised with adequate oversight from the Investigational Review Board (IRB), as mentioned on page 41 of Document 1 in "docs/ICD - Document 1.pdf." The IRB approval is necessary before starting a clinical investigation to ensure informed consent forms, reflecting any modifications suggested by FDA comments are reviewed and approved accordingly per the guidelines stated between pages 27-30 of Document 1 in "docs/ICD - Document 1.pdf."
```json
{
  "trace_id": "cd4841d7-e29c-4ee2-b28d-16ade29fc1ab",
  "question": "What is the clinical investigator?",
  "plan": [
    "retrieve",
    "draft",
    "cite"
  ],
  "retrieval": [
    {
      "file": "docs/ICD - Document 1.pdf",
      "chunk_id": 189,
      "score": 0.6214,
      "section_path": [
        "B. The Clinical Investigator"
      ]
    },
    {
      "file": "docs/ICD - Document 1.pdf",
      "chunk_id": 191,
      "score": 0.5389,
      "section_path": [
        "1. Delegation of Consent Discussion"
      ]
    },
    {
      "file": "docs/ICD - Document 1.pdf",
      "chunk_id": 194,
      "score": 0.5251,
      "section_path": []
    }
  ],
  "draft_tokens": 690,
  "latency_ms": {
    "retrieve": 768,
    "draft": 19112,
    "total": 19881
  },
  "errors": []
}
```