# Docker CLI Tooling Design

**Date:** 2026-04-27

## Goal

Make the KeySG development container ready to use for future agent-driven work by baking `npm`, OpenAI Codex CLI, and Claude Code into the Docker image at build time.

## Scope

This change only affects container tooling:

- `docker/Dockerfile`
- optional validation notes in the existing Docker workflow

It does not change:

- Python or Conda environment contents beyond what is already required for KeySG
- runtime authentication setup
- dataset mounts or visualizer networking

## Requirements

The final image must:

- install Node.js 20 and `npm`
- install `@openai/codex` globally with `npm`
- install `@anthropic-ai/claude-code` globally with `npm`
- verify the binaries during build with version checks
- avoid embedding API keys, login state, or user-specific config into image layers

## Chosen Approach

Use a single package manager path for all JavaScript-based CLI tooling:

- install Node.js 20 from the NodeSource Debian repository
- install both AI CLIs globally with `npm`
- verify with `npm --version`, `codex --version`, and `claude --version`

## Alternatives Considered

### 1. `npm` for both CLIs

Pros:

- simplest Dockerfile
- one install/update path
- easy to verify in a single build layer

Cons:

- Claude Code is not installed through Anthropic's OS package repository

### 2. `npm` for Codex, apt repository for Claude Code

Pros:

- follows Anthropic's Linux package-manager distribution path

Cons:

- adds repository key management and apt source maintenance
- increases Dockerfile complexity without a clear runtime benefit in this image

### 3. Install Node only and defer both CLIs to container runtime

Pros:

- smaller image

Cons:

- not ready to use after `docker compose run`
- shifts setup burden to every new container

## Decision

Choose option 1.

For this image, the main goal is reproducible developer tooling inside an immutable build artifact. Global `npm` installation is the smallest reliable solution that satisfies that goal for both CLIs.

## Implementation Notes

- keep the existing CUDA + Conda base untouched
- add Node.js installation before switching to the non-root runtime user
- keep CLI installation in the image build, not in `.bashrc`
- keep authentication interactive at runtime only
- fail the Docker build early if any CLI binary is missing

## Validation

The updated Docker build should succeed through these checks:

- `npm --version`
- `codex --version`
- `claude --version`

At runtime, the expected first-use flows are:

- `codex` prompts for ChatGPT login or API-key-based auth
- `claude` prompts for Anthropic-supported login flow

## Out of Scope

- pre-authenticating either CLI
- mounting host credential directories automatically
- adding project-specific wrapper scripts for either CLI
