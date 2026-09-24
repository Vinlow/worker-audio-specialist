# German quiet-seam alignment repair

Status: local real-model diagnostic passed; image and deployed qualification pending.

Studio project `263f9659-506c-477f-84b0-7c37f040afe0` failed after the
full-source RunPod job `7ae39a27-7dbf-4702-88ba-5058fc99f9b5-e2` returned
unaligned native timestamps with 12 zero-duration words. Container logs bind
the alignment failure to `No coherent acoustic window join at word 2736`.
The serving worker used the intended `c975dde` image.

The exact returned German words and source audio at 1325–1438 seconds reproduce
the failure using the pinned German model and Torch/torchaudio 2.7.1 on CPU.
The UTF-8 fixture contains 200 words. Audio SHA256:
`41ab4c382d9ab55123c4c5a6407674ce9a01319f3685b401afa91277c5bfec3c`.
An initial Windows diagnostic read used the wrong encoding; that run is invalid
and is excluded from the committed regression fixture.

With regular seven-second overlap, the last word in the previous window has
a trailing blank envelope reaching 1385 seconds. The next window measures
speech before that boundary, so strict envelope ordering fails. Each word has
individual context, but neither observation includes both sides of the seam.

The planner retains regular windows and adds a centered window when adjacent
alignable words lack shared context and fit within a normal window. It bounds
additional inference to one source-duration of audio. Candidates remain actual
CTC measurements; the strict stitcher and explicit fallback rules are unchanged.
The full recording plans 34 rather than 31 windows: 1993.833333 versus
1813.833333 seconds of alignment audio, not another recognition pass.

The targeted candidate aligns all 200 excerpt words with four windows. A
diagnostic globally widened overlap also aligned them, but is not the shipped
approach. The small committed fixture retains the failed and successful
measurements and proves the old join rejects while new measured candidates
form an ordered path without mutation.

This is geometry evidence, not recognition or cut-quality acceptance. The
independent chunk transcript contains speech near this seam omitted by the
full-source recognizer. Studio's source reconciliation and limitation checks
must still run; alignment must not certify transcript completeness.

Local checks: 16 stitch/planner tests, three German lexical tests, seven real
torchaudio boundary tests, 17 deployment-lock tests, and the hosted lightweight
source-test groups passed. The Predictor authority suite could not import in
the local diagnostic environment because full worker dependencies are absent;
its required execution belongs to the exact-image suite, not a passing local
claim. Deployed canary and full Studio pipeline remain required.
