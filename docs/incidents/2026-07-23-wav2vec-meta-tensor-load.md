# 2026-07-23 wav2vec2 meta-tensor cold-load failure

## Status

Fixed on non-production branch `codex/wav2vec-meta-init-hardening`.

No endpoint was deployed or mutated by this investigation. The deploy-wired
`main` branch was not touched. A live endpoint canary for this fix remains an
integration action after an explicit deployment through the normal branch
workflow.

## Incident evidence

The Holy Grale endpoint was serving worker commit
`dc00a76ce4b3ff156a74665766e9fb51b184f063`.

Two Starforge final-span canaries were running concurrently. On source
`pod-interview-01__0120-0180.wav`
(`sha256:85733435d45f801458d3e4def1cd8aa76dda257476a3dd4e7af78a1527780987`):

- flag-off job `f808fa51-7979-463d-a394-129f99d08f53-e2` completed on worker
  `qfo3qaxlic7mj6` after 1,014 ms queue delay and 8,035 ms execution;
- flag-on job `1f2b239c-df4a-4067-8cae-72ee9080a9df-e2` failed at
  `predict.py -> self.aligner.setup(device=device)` with:

  `RuntimeError: wav2vec2 bundle returned meta tensors`

The failed flag-on caller did not persist a provider output, worker ID, timing,
or cost receipt before throwing. Those fields are therefore
`MISSING_PROVIDER_OUTPUT` / `UNKNOWN`, not zero.

A sequential retry on the exact same source completed:

- flag-off `16eebcdc-eb1b-4ee3-9ff3-8dc854ed20cf-e1`, worker
  `qfo3qaxlic7mj6`, 131 ms delay, 7,975 ms execution;
- flag-on `6fcc23c7-48c6-4fb0-ae67-873c260a9ce9-e2`, worker
  `scdwu4ik2ryion`, 130 ms delay, 23,345 ms execution.

This rules out the source audio and request geometry as the cause.

Another concurrently submitted canary also completed on worker
`qfo3qaxlic7mj6`:

- flag-off `8265afcf-bd9a-4690-9431-ce1214935976-e2`;
- flag-on `aa52be0d-2338-4e10-9e74-c8b128b9c5d9-e2`.

### Preserved local receipts

| Evidence | SHA-256 | Bytes |
| --- | --- | ---: |
| `P:\Web2Labs\Code\.tmp\pyannote-spike\runpod-span-canary-pod-interview-0060-0120-r1\result.json` | `ea5d407c054a6d90ca825ac24dd20579d500d34911586e8f3d802aec5910850d` | 2,627 |
| `P:\Web2Labs\Code\.tmp\pyannote-spike\runpod-span-canary-pod-interview-0120-0180-r1\flag-off.provider-output.json` | `d8a61c80a887af45f680522ea76c5a38dba827b70e7fef009cb60df9498c10b6` | 101,989 |
| `P:\Web2Labs\Code\.tmp\pyannote-spike\runpod-span-canary-pod-interview-0120-0180-r2\result.json` | `e969d9b76d4632f9b91f441e0a83dbcb4de68ea0532f7d9f411f742c33e4cc31` | 2,629 |

## Root cause

The flag-on final-span path starts CLAP scoring in a background thread before
Whisper and forced alignment finish. CLAP, wav2vec2, and pyannote each had only
component-local lazy-load protection, so distinct model constructors could run
at the same time in one worker process.

The installed Transformers 4.57 loader states that `from_pretrained` model
construction always uses a meta-device context. Its
`transformers.integrations.accelerate.init_on_device` implementation
temporarily replaces the process-global
`torch.nn.Module.register_parameter`. That patch is visible to other Python
threads. A torchaudio wav2vec2 module constructed during this window receives
meta parameters even though the aligner requested a CPU map location.

A deterministic local probe held the real Transformers `init_empty_weights`
context in one thread and constructed `torch.nn.Linear` in another. Both
parameters were on `meta`. A second pre-fix simulation overlapped the worker's
CLAP and aligner loaders and reproduced the endpoint signature exactly:

`RuntimeError: wav2vec2 bundle returned meta tensors`

The RunPod concurrency increased the chance of a cold worker encountering this
condition, but cross-job concurrency is not required: the single flag-on job
already overlaps CLAP cold construction with foreground alignment.

## Fix

- Introduced one re-entrant process-wide cold-model construction lock.
- Applied it to CLAP, wav2vec2, and pyannote construction/device transfer.
- Kept resident-model inference outside the lock.
- Added pyannote instance-level setup locking to prevent duplicate publication.
- Made CLAP publication atomic; failed construction cannot expose only a
  processor or half-loaded model.
- Made wav2vec2 publication atomic and added one bounded serialized retry when
  the bundle returns meta parameters.
- After two meta results, wav2vec2 clears all setup state and fails the job.
  It deliberately does not use `to_empty`, because that would allocate
  uninitialized weights and silently corrupt alignment.
- Added contention timing to logs for cold-load diagnosis.

## Verification

Command, run from `src`:

`..\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py" -v`

Result: **16 tests passed**.

The regression suite uses the real Transformers meta-device context and proves
that CLAP and wav2vec2 cold construction no longer overlap. It also covers:

- one-meta-then-hydrated wav2vec2 recovery;
- repeated-meta fail-closed cleanup;
- concurrent pyannote setup publishing exactly one complete pipeline;
- existing diarization sidecar immutability and final-span routing contracts.

## Remaining boundary

This checkpoint proves the process lifecycle fix locally. It does not prove
CUDA/image behavior or live RunPod behavior until the checkpoint is integrated
into the Holy Grale deployment branch, built, and exercised with a protected
concurrent cold-worker canary. No such deploy was performed here.
