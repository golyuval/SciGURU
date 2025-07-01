# SciGURU Frontend

A modern **Wix Velo** web interface for interacting with the SciGURU LLM backend and delivering polished scientific explanations to end-users.

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Directory Structure](#directory-structure)
- [Core Concepts](#core-concepts)
  - [AsyncBot.jsw Service](#1-asyncbotjsw-service)
  - [Real-time Polling](#2-real-time-polling)
  - [Markdown Rendering](#3-markdown-rendering)
  - [Model Switching](#4-model-switching)
- [Local Development](#local-development)
- [Deployment](#deployment)

</details>

---
</br>

## Overview
The frontend provides an intuitive chat experience that allows users to:

1. Submit scientific questions.
2. Select a specific model version (*zero → phi*).
3. Receive formatted answers with rich Markdown support.

All heavy-lifting happens on **Wix Velo (serverless)** functions which proxy requests to the Python backend.


---
</br>

## Directory Structure
```text
Frontend/
├── Code/
│   ├── Pages/
│   │   ├── chat.js            # Main Q&A interface
│   │   ├── account.js         # Profile page wrappers
│   │   └── ...
│   ├── backend/
│   │   └── AsyncBot.jsw       # Server-less handler ➜ HF inference API
│   └── public/
│       └── header.js          # Shared site header component
├── animations/                # GIFs & visual effects
└── ReadMe.md                  # ← this file
```



---
</br>

## Core Concepts
### 1. `AsyncBot.jsw` Service
* Located under `Code/backend/` – executed on Wix back-end runtime.
* Handles **four** concerns:
  1. Securely fetches HuggingFace URL & token from Wix Secrets Manager.
  2. Builds request body depending on model type (Phi-3 chat, RLVR delta, etc.).
  3. Implements retry logic & exponential back-off for cold starts.
  4. Exposes two functions to client side: `initiate_guru_request` & `guru_request_status`.

### 2. Real-time Polling
* `chat.js` starts a JavaScript `setInterval` (3 s) to hit `guru_request_status`.
* Dots animation indicates backend progress; statuses displayed are *sending → queued → generating → done*.

### 3. Markdown Rendering
* `chat.js` converts LLM output into HTML via `parseMarkdownToHTML()` ensuring:
  * Headings, code-blocks, bold/italic, ordered / unordered lists.
  * Maintains SciGURU colour palette and monospace code fonts.

### 4. Model Switching
* Drop-down selects value `zero | alpha | beta | gamma | delta | phi`.
* Client maps to Wix Secret IDs (e.g. `URL_sciguru_beta`) ensuring **zero hard-coded endpoints**.

</br>

---
</br>

## Local Development

1. Clone repo & run Wix **VS Code Editor** or online editor.

2. Modify files under `Frontend/Code/…`; Wix hot-reloads sandbox.

3. To test serverless functions locally, use **Wix HTTP Functions** runner or deploy to a preview site.

> **Tip:** Keep `AsyncBot.jsw` under `backend/` to avoid exposing secrets in client JS.

---
</br>

## Deployment

* Publishing the Wix site triggers CI that pins latest JS modules.

* No additional build step is required – Wix bundles and serves assets automatically.

* Remember to update secrets (`URL_sciguru_*`, `TOKEN_huggingface`) via Wix Dashboard when backend URLs change.

---
