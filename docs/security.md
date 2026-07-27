# Security: access control and the audit trail

What is gated in this service, why those particular things, what gets written
down when someone uses them, and — the section that matters most — an honest
accounting of the distance between what is implemented here and what a
production deployment would actually require.

Implemented in [`app/serving/auth.py`](../app/serving/auth.py) and
[`app/observability/audit_log.py`](../app/observability/audit_log.py); exercised
by [`tests/test_auth.py`](../tests/test_auth.py).

---

## The model: two roles, one header

Every request carries its credential in an `X-API-Key` header. Keys are read
from `.env` as two comma-separated lists, and a key's membership *is* its role:

```bash
VIEWER_API_KEYS=line-station-a,line-station-b
OPERATOR_API_KEYS=ops-console
```

| Endpoint     | Role required | Cost of one call            | What it reveals                       |
| ------------ | ------------- | --------------------------- | ------------------------------------- |
| `/health`    | *none*        | none                        | liveness, names of resident models    |
| `/predict`   | `viewer`      | one frame (~150 ms)         | one score and one heatmap             |
| `/models`    | `operator`    | a few `stat` calls          | artifact paths, which categories exist |
| `/benchmark` | `operator`    | **minutes of CPU**          | metrics over the whole test split     |

Roles nest: `operator` grants `viewer`, so the operator key works everywhere and
is not listed twice. Two roles is the smallest split that separates *runs the
line* from *administers the service*; inventing more before a second consumer
exists would be fiction.

Three status codes carry the outcome, and the distinctions are deliberate:

| Condition                        | Status | `detail`              |
| -------------------------------- | ------ | --------------------- |
| No `X-API-Key` header            | 401    | `missing_api_key`     |
| Key not recognised               | 401    | `invalid_api_key`     |
| Key valid, role insufficient     | 403    | `insufficient_role`   |
| **No keys configured at all**    | 503    | `auth_not_configured` |

401 says *I do not know who you are*; 403 says *I know exactly who you are and
the answer is still no*. Collapsing them into one code sends a viewer who hits
`/benchmark` off hunting for a typo in a key that is perfectly correct.

The last row is the one worth arguing about. A server with **no** keys
configured refuses every gated request rather than serving them openly. A
missing environment variable is by far the most likely way this protection gets
switched off by accident, and failing open would mean a deployment with no access
control at all that still answers 200 to everything. `/health` stays up so an
orchestrator sees a live pod with a broken config rather than one it restarts
forever.

### Why `/health` is the exception

A kubelet liveness probe cannot hold a credential, and a health check that can
fail closed on an authentication problem will eventually restart a perfectly
healthy pod during a key rotation — a worse outage than the one the gate was
protecting against. So `/health` is open, and what it discloses is bounded to
match: that the process is alive, and the names of the models currently resident.
Both are strictly less than an anonymous caller learns from the port being open
at all.

---

## Why `/benchmark` is the most restricted endpoint

Two independent reasons, either of which would be sufficient.

### 1. Resource exhaustion

`POST /benchmark` runs every requested backend over a category's **entire** test
split. On `bottle` that is 83 images per backend: seconds for an ONNX graph,
roughly a minute for PatchCore, several minutes for WinCLIP. The request holds a
thread-pool worker for its whole duration, and there is no timeout, no queue and
no concurrency limit that will save the process from a caller who sends ten of
them.

That is a denial-of-service primitive with a JSON body, and the thing it starves
is the inference path an actual production line is waiting on. Worse, it does not
take malice: a dashboard that polls `/benchmark` on a 30-second refresh, written
by someone who assumed it was as cheap as `/models`, will do it by accident. An
endpoint whose cost is three orders of magnitude above its neighbours needs a
gate in front of it regardless of who is on the other side.

### 2. Inferring dataset properties

The response is not raw imagery, and it is tempting to treat it as harmless
aggregate statistics. It is not. Each result carries `num_images`,
`num_defective`, `num_normal`, `num_masked`, four accuracy metrics and the
timing, all computed over the customer's own inspection data. From those an
unauthenticated caller could enumerate:

- **How much data exists, and how imbalanced it is** — the counts are reported
  directly, per category. Probing `/benchmark` across category names maps the
  customer's product line and how much of each has been collected.
- **How hard the defects are** — image-AUROC and AU-PRO are a direct read on
  separability. A category sitting at 0.82 rather than 0.99 says that
  manufacturer has a subtle or highly variable defect mode, which is
  commercially sensitive information about a production process.
- **Which categories are in service at all** — a 503 `dataset_not_available`
  distinguishes "no such data" from "data present", and does so cheaply.
- **Membership-style inference over time.** Re-running the same benchmark after
  a data refresh and watching `num_images` and the metrics move is a channel for
  learning about individual additions to the test set — not a full membership
  inference attack, but the same shape of leak, and the reason aggregate metrics
  over private data are treated as sensitive rather than public.

None of this is catastrophic on its own. All of it is information a customer
would be surprised to learn was available to anyone who could reach the port.

`/models` is gated for the second reason only: it costs nothing to serve, but its
`artifact` and `detail` fields name absolute filesystem paths and the exact set
of trained categories, which is a free map of the deployment for anyone probing
it.

---

## What is logged, and why

Two separate destinations, because they answer two different questions and want
opposite properties.

### The application log — *what is the server doing?*

Standard `logging`, written for whoever is debugging right now. It records
rejected requests (with the **hashed** key identity, never the key), guard
failures with their metrics, model loads and their durations, scored frames, and
tracebacks for anything unhandled. It is verbose, rotated, and filtered by level
in production.

### The audit trail — *who ran the expensive thing, and what did they get?*

`results/audit.jsonl`, append-only, one JSON object per line, independent of log
level. Every `/benchmark` call — successful or failed — appends one entry:

```json
{"timestamp": "2026-07-27T18:44:12+00:00", "event": "benchmark",
 "caller": "hmac-sha256:5cd0a7e10b4f2ac9", "role": "operator",
 "category": "bottle", "models": ["onnx_efficientad"], "duration_seconds": 41.2,
 "outcome": "ok", "metrics": {"onnx_efficientad": {"image_auroc": 0.976, ...}}}
```

Each field earns its place:

- **`timestamp`** — UTC with an explicit offset. A naive local timestamp becomes
  unreadable the first time the file is opened in another timezone.
- **`caller` / `role`** — accountability. The identity is
  `HMAC-SHA-256(pepper, key)` truncated to 64 bits, never the key itself: audit
  files get copied, shipped to log stores and read by people who should not
  thereby acquire the ability to call the API.
- **`category` / `models`** — what was asked for, as requested rather than as
  resolved, so an ONNX fallback is visible as a mismatch against the keys of
  `metrics`.
- **`duration_seconds`** — the resource-consumption side of the record. This is
  what makes an abusive or badly-configured caller visible: one operator
  accounting for forty minutes of CPU a day is a conversation, and there is no
  way to have that conversation without the number.
- **`outcome`** — `ok`, or `failed:<ExceptionType>`. A run that 503s still burned
  the CPU it burned before failing, and "this key triggers minute-long failures
  over and over" is exactly the pattern the trail exists to surface.
- **`metrics`** — what the caller actually *received*. This is the field that
  makes the record worth keeping: "someone ran a benchmark" answers nothing
  during an incident review, whereas "this key obtained these numbers over this
  category" answers the question directly.

**Debugging value, not only accountability.** The same file explains a
performance mystery: a benchmark whose `duration_seconds` tripled between two
otherwise identical entries dates a regression to a window, and the `metrics`
alongside it say whether accuracy moved with it. That is a question the
application log cannot answer once it has rotated.

Three consequences of this design are worth stating plainly rather than leaving
to be discovered:

1. **The audit file is itself sensitive.** It contains the metrics whose
   disclosure is half the reason `/benchmark` is gated. It is created `0600`,
   `.gitignore`d, and should be treated as customer data — not shipped to a
   general-purpose log aggregator with looser access control than the API.
2. **The trail fails open.** If the write fails — full disk, read-only mount —
   the failure is logged at `ERROR` and the request is still served. That is the
   right trade-off for an evaluation endpoint on an inspection service and the
   wrong one for a regulated system, where "cannot audit" must mean "must not
   serve".
3. **Denied requests are not in it.** A 403 is rejected by the dependency before
   any handler runs, so it lands in the application log (at `WARNING`, with the
   same hashed identity) and not in the audit file. For an access-control audit
   that is a real gap; the fix is to record denials from the dependency itself.

---

## What this implementation does vs. what production requires

Everything above is a real access-control mechanism, and it is not a production
one. The gap is not an oversight to be discovered later — it is the following
specific list, and each item is work that was consciously not done at this scope.

**Secrets are in a file on disk, in plaintext.** `.env` is read at startup by
`python-dotenv`. There is no encryption at rest, no envelope key, and any process
running as the same user can read every key the service accepts. Production keeps
credentials in a secrets manager (AWS Secrets Manager, Vault, GCP Secret
Manager), injects them at runtime rather than baking them into an image, and
audits reads of the secret itself. The single most likely real-world failure of
the current scheme is not an attack at all — it is a `.env` committed to git.
`.gitignore` covers it; a pre-commit secret scanner would cover it properly.

**Keys are long-lived bearer secrets with no lifecycle.** They do not expire,
cannot be rotated without a restart, and cannot be revoked individually without
editing a file and redeploying. There is no issued-at, no per-key metadata, no
"this key was last used on…". Production replaces this with OAuth2/OIDC: an
identity provider issues short-lived JWTs, the service validates signatures
against a rotating JWKS, and revocation, expiry and rotation stop being this
codebase's problem. `app/serving/auth.py` is structured so that swap touches only
the credential check — the roles, the 401/403 split, the `Principal` handlers
receive, and the audit trail all survive it unchanged.

**Authorization is global, not per-resource.** A viewer key can `/predict`
against *every* category; an operator key can `/benchmark` *every* category. In a
multi-tenant deployment — one service inspecting parts for several
manufacturers, which is the obvious commercial shape of this system — that is
unacceptable: it is precisely the dataset-property leak described above, moved
inside the customer boundary. Production needs per-category (per-tenant) ACLs
carried on the identity and checked against `request.category` in the handler,
plus a default-deny on any category the caller is not explicitly granted.

**There is no rate limiting or concurrency control.** Gating `/benchmark` bounds
*who* can exhaust the service, not *how much*. A single compromised or careless
operator key can still saturate the process. Production wants a per-key rate
limit, a bounded concurrent-benchmark count (ideally one, with 429 for the rest),
a request timeout, and — better — moving benchmarking onto a job queue so the
endpoint returns a job id in milliseconds instead of holding a worker for
minutes.

**Log retention and integrity are undefined.** `results/audit.jsonl` grows
without bound, is never rotated, has no retention policy, and can be edited or
truncated by anything running as the service user. A production audit trail has a
documented retention period driven by policy rather than by disk space, ships to
append-only or WORM storage, and is tamper-evident (hash-chained entries, or a
store the service can write to but not rewrite). Related: the entries carry
customer-derived metrics, so retention is also a *deletion* obligation under
GDPR-style regimes, not merely a storage question.

**Transport is not addressed here.** An API key in a header is only as
confidential as the channel carrying it. This service must sit behind TLS
termination; over plain HTTP every key is recoverable by anyone on the path, and
nothing in this code can detect that.

**The key identity is reversible for a weak key.** `hash_api_key` is
`HMAC-SHA-256` truncated to 64 bits, with the pepper defaulting to empty. Against
a high-entropy random key that is fine. Against a key someone chose by hand,
whoever holds the audit file can hash candidates until one matches. Setting
`API_KEY_HASH_PEPPER` closes it; production would source that pepper from the
same secrets manager as the keys and would generate keys with a CSPRNG rather
than trusting an operator to pick one.

**Timing.** Key comparison uses `hmac.compare_digest` rather than `==`, so a
candidate cannot be recovered a byte at a time from response latency. The loop
does return early on a match, which leaks which position in a list of a handful
of keys matched — not a secret. This is the one item on the list that is
genuinely finished.

**A known asymmetry in the role split.** `/predict` is `viewer`-gated, and the
first request for a cold backend triggers that backend's load: tens of seconds
and up to ~1 GB resident for WinCLIP. So a viewer can indirectly cause a model
load, even though "triggering model loads" is conceptually an operator concern.
The alternative — a viewer key that only works once an operator has warmed the
process — is worse, and there is no separate load endpoint to gate. The
production answer is not a different role assignment but eager loading at
startup with a readiness probe, which removes the cold-load cost from the request
path entirely and makes the question moot.

---

## Running it

```bash
# 1. Configure keys (see .env.example)
cp .env.example .env
# edit VIEWER_API_KEYS / OPERATOR_API_KEYS — generate with:
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. Serve
make serve

# 3. Call it
curl -s localhost:8000/health                                    # open
curl -s localhost:8000/models -H "X-API-Key: $OPERATOR_KEY"      # operator
curl -s localhost:8000/models -H "X-API-Key: $VIEWER_KEY"        # 403

# 4. Read the trail
tail -n 1 results/audit.jsonl | python -m json.tool
```

Never commit `.env`. It is in `.gitignore`; the keys in `.env.example` are
placeholders and must not be used.
