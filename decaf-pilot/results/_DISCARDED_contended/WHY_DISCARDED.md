# Discarded — measured while the machine was thrashing

These artifacts were generated at 14:03 on 2026-10-09 from 23 Phase-A scaling runs
taken while swap was 100% exhausted (4,092 / 4,095 MB).

Measured prefill was 62-98 ms/token against a known-good pilot value of 4.1 ms/token,
and TPOT 162-178 ms against 37 ms — roughly 15x inflated. The curve fits, the
break-even L*, and anything derived from them are meaningless.

Kept only as a record of the failure mode. Do not cite. The source runs are in
`cache/scaling_v2.CONTENDED.jsonl.bak`.

Causes: (1) num_ctx=20480 inflated llama-server to 3.0 GB RSS; (2) the dev stack
(ng serve, VS Code, JVMs) returned mid-run; (3) the preflight guard checked load and
free RAM but not swap pressure, so it admitted a machine already primed to thrash.
All three are now fixed.
