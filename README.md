# podcode

Deploy an open-weight coding model to Runpod and connect local [OpenCode](https://opencode.ai) in one command.

`podcode up` shows the current Runpod GPU price, estimates the selected time window, and requires typed confirmation before it creates anything.

## Quick start

### 1. Install prerequisites

Install podcode:

```sh
git clone <THIS_REPOSITORY_URL>
cd self-hosted-open-code
./install.sh
```

On first use, `podcode` downloads Runpod CLI (`runpodctl`) automatically to
`~/.local/bin` if it is not already available. Set `RUNPODCTL_BIN_DIR` to use
a different location.

If `podcode` is not found afterward, add `~/.local/bin` to your shell path:

```sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Configure secrets

Create a local `.env` in the project you want OpenCode to use:

```sh
cp /path/to/self-hosted-open-code/.env.example .env
```

Set these values:

```dotenv
RUNPOD_API_KEY=rp_...
RUNPOD_VLLM_API_KEY=a-long-random-secret
```

Authenticate Runpod CLI once:

```sh
runpodctl config --apiKey "$RUNPOD_API_KEY"
```

`HF_TOKEN` is optional and only needed for gated Hugging Face models.

### 3. Start a coding agent

From your code repository:

```sh
podcode up qwen3-coder-next --estimate-hours 10 --terminate-after 10h --ephemeral
```

Type `DEPLOY` when shown the live cost. This command:

1. Selects the recommended GPU for the model.
2. Starts vLLM on Runpod with an API key.
3. Writes `opencode.json` locally, with the deployed Runpod model as the default.
4. Uses ephemeral model storage, so model weights are discarded when you delete the Pod.
5. Shows a loader until vLLM is ready to accept requests (up to 30 minutes by default; adjust with `--wait-timeout 45m`).

Start OpenCode in that directory; it uses the configured `runpod/...` model by default.

## Models

```sh
podcode models
podcode gpus                 # live Runpod availability and pricing
```

Recommended starting point: `qwen3-coder-next` on a 48 GB A6000/A40, using its verified 4-bit vLLM checkpoint. For a lower-cost alternative, try `qwen3.6-27b`.

## Daily commands

```sh
podcode status
podcode wait                    # live GPU/model-loading stage; uses the only Pod automatically
podcode logs                    # detailed live container logs; uses the only Pod automatically
podcode stop POD_ID
podcode start POD_ID
podcode destroy POD_ID
podcode swap OLD_POD_ID qwen3.6-27b
```

## Storage choices

Use `--ephemeral` for disposable sessions: no persistent volume, no retained model download after termination.

Without it, podcode creates a model-sized Pod volume (60 GB for Qwen3-Coder-Next) at `/workspace` so future starts avoid downloading weights again. For portable storage across replacement Pods, pass `--network-volume-id VOLUME_ID`.

Delete a network volume explicitly only when you are finished with its contents:

```sh
podcode destroy POD_ID --delete-network-volume VOLUME_ID
```

## Safety and costs

- `DEPLOY` and `REPLACE` confirmations cannot be bypassed.
- Timer flags depend on the installed Runpod CLI version. If unavailable, `podcode` stops before creation rather than launching a Pod without the requested cost guard; delete it explicitly with `podcode destroy POD_ID` when finished.
- Runpod’s displayed live GPU price and Billing page are the source of truth. Storage and bandwidth can be separate charges.
