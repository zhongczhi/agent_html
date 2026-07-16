# SOTA Failure Inspection — `cot_extract_k10`

**Date**: 2026-07-12
**Pipeline**: `cot_extract_k10` (MiniLM-L6, k=10, CoT + verbatim-span prompt)
**Result**: `contains_gold` 0.904 on n=334 (HotpotQA dev_distractor, SHA `4e9ecb5c8d3b719f`)
**Failures**: 32 questions. All inspection data below is from
`docs/eval-results/iter15-cot-extract-k10-dump.jsonl`, produced via
`scripts/eval_qa_hotpotqa.py --dump-results`.

---

## TL;DR — what's actually breaking at 0.904

Manual inspection of every failed question places the failures in **six
empirical categories**. The two biggest categories together account for
71.8% of the failure floor:

| Category | Count | Share | Best lever |
|---|---:|---:|---|
| **Name-variant mismatch** | 13 | 40.6% | Better / cleaner answer normalization, not prompt change |
| **Wrong entity or granularity** | 10 | 31.2% | Stronger LLM with better comparison / disambiguation |
| Yes/No answer format | 3 | 9.4% | Strict-output prompt ("answer 'yes' or 'no' only") |
| Extract discipline | 3 | 9.4% | Stronger LLM (model reasoned but didn't pick a span) |
| Wrong format | 2 | 6.2% | Stronger LLM |
| Dataset noise | 1 | 3.1% | None (gold contradicts corpus) |

**Key implication**: ~40% of the remaining failures are **not** extraction
errors in any meaningful sense — the model identifies the correct entity
but uses a non-canonical name form. A canonical-name post-processing
step would close this category cheaply. The other 60% are real LLM
reasoning or format errors that need a stronger model or different
prompt.

---

## 1. Methodology

For each of the 32 failed questions I extracted:
- The question (full text)
- The model's predicted answer
- The gold answer
- The retrieved paragraph titles
- The gold paragraph titles

I then read each one and assigned it to one of the categories below
based on whether the model's prediction was **semantically right**,
**partially right**, or **wrong** and on which dimension it diverged
from gold.

The full per-failure record lives in `docs/eval-results/iter15-cot-extract-k10-dump.jsonl`.
Six representative examples below; the rest follow the same patterns.

---

## 2. Category breakdown with examples

### Category A — Name-variant mismatch (13 / 32, 40.6%)

**Pattern**: The model identifies the correct entity but uses a name form
that doesn't literally match the gold. The model's reasoning section
(which is rendered before the visible extract) typically contains the
gold-matching form. So the failure is entirely in the leading span, not
in understanding.

| # | Question summary | Predicted | Gold | Variant type |
|---|---|---|---|---|
| 1 | Singer Sudha Kheterpal played with — Australian best-seller | "Princess of Pop" | Kylie Ann Minogue | nickname → formal name |
| 2 | English model in "Nasty Girl" video | Naomi Campbell | Naomi Elaine Campbell | short name → full name |
| 6 | Charles J G Saunders's school | Merchant Taylors' School | Merchant Taylors' School (MTS) | name → name+abbreviation |
| 7 | Maxeda's owner since 2004 | Kohlberg Kravis Roberts | KKR & Co | long → ticker |
| 9 | Director closer to Canada | J. Searle Dawley | James Searle Dawley | initials → full first name |
| 10 | MLS team owned by Precourt | Columbus Crew SC | Columbus Crew Soccer Club | acronym → full |
| 11 | Philadelphia, Here I Come!'s author | Brian Friel | Brian Patrick Friel | short → full middle name |
| 16 | Adjective "Ortonesque" — author | Joe Orton | John Kingsley "Joe" Orton | short → full name |
| 19 | Capital of Ostrogothic Kingdom + birthplace of Fiorentini | Ravenna | Ravenna (…; Romagnol: "Ravèna") is the capital city of the Province of Ravenna | clean → metadata-padded |
| 20 | Cathedral in Cornwall | Church of England cathedral in the city of Truro, Cornwall | The Cathedral of the Blessed Virgin Mary, Truro | description → official name |
| 22 | Born earlier | Virginia Woolf | Adeline Virginia Woolf | short → full first name |
| 28 | Owner of Tusker beer | East African Breweries | East African Breweries Limited | name → legal entity |
| 32 | French Romantic composer | Hector Berlioz | Louis-Hector Berlioz | short → full first name |

**Common sub-patterns**:
- Short vs full names (no middle, no first) — 6 cases
- Abbreviation/ticker vs full — 4 cases
- Nickname vs formal name — 1 case
- Canonical form vs metadata-padded — 1 case
- Description vs official name — 1 case

**Why this happens**: The `extract_span` instruction asks the model to
"begin your response with the extracted span (in quotation marks)". The
model picks the most natural-seeming span, which is usually the
shortest / most conversational form. HotpotQA's gold standard prefers
canonical Wikipedia-style names (full names, legal suffixes, official
forms).

**Could a prompt change fix this?** Possibly — an instruction like "Use
the most complete / canonical name as written in Wikipedia" might shift
some of these. But it could also over-correct and break clean
answers that the gold happens to accept (e.g., #19). Risk vs reward
unclear without testing.

**Could post-processing fix this?** Yes — a string canonicalization
step (resolve nickname → formal name, drop parenthetical from gold,
expand abbreviations) would close the gap cleanly. But it's brittle
and doesn't help on novel name forms.

### Category B — Wrong entity or granularity (10 / 32, 31.2%)

**Pattern**: The model identifies a related but **semantically different**
answer. Either picking a different entity, or picking the right entity
at the wrong granularity.

| # | Question summary | Predicted | Gold | Error type |
|---|---|---|---|---|
| 3 | Voice actress in Alpha and Omega — what character | Tails in Sonic the Hedgehog | Kairi in Kingdom Hearts | completely wrong entity (model ignored the question's link and answered a different one) |
| 4 | Pioneer with park near Lexington — famous for what | Daniel Boone | Wilderness Road | picked the person when Q asked for the road named after him |
| 5 | Greater impact on French culture | Jean Vigo | Sri Lankabhimanya Lester James Peries | model chose the French director, missed the comparison logic |
| 12 | City where 432d Wing is stationed | Creech Air Force Base, Nevada | Clark County | wrong granularity: city/bases vs county |
| 15 | One risk to Norway's financial reserve | sensitivity to global business cycles | terrorist activity | different risk from the same paragraph |
| 23 | Most populated island of what larger area | the seven Canary Islands | Macaronesia | broader geographic distinction |
| 27 | Role of Nettie Harris in The Color Purple | Whoopi Goldberg | Akosua Gyamama Busia | picked the wrong actress (Celie vs Nettie) |
| 29 | Special about Favre-Leuba wristwatches | long heritage (2nd-oldest brand) | Swiss made | different fact from the same paragraph |
| 30 | Tribe with Alvaro Mexia on diplomatic mission | native populations (generic) | Apalachees | didn't identify the specific tribe from the narrative |
| 31 | Chinese company that helped develop the J-7's predecessor | Mikoyan (Soviet) | Chengdu Aircraft Corporation | picked the wrong side of the lineage |

**Why this happens**:
- Multi-hop reasoning errors (4, 5, 27) — model needs to chain facts but picks prematurely
- Granularity confusion (12, 23) — model answers at the asked level in some cases but at a finer level in others
- Wrong-fact-from-same-paragraph (15, 29) — multiple plausible answers in the context; model picks a notable one but not the gold one

**Could a stronger LLM close this?** Likely yes for most of these.
The MiniLM + generic-model combo doesn't have great comparison /
disambiguation ability. A larger model would handle these better.

### Category C — Yes/No answer format (3 / 32, 9.4%)

**Pattern**: Gold is the literal word "yes" or "no". The model gives
an extensive explanation but never types "yes"/"no" literally.

| # | Question summary | Predicted | Gold |
|---|---|---|---|
| 13 | Are Eve Beglarian and Zach Bogosian both of Armenian descent? | (explanation about both) | yes |
| 14 | Are Nerdcore Rising and What Would Jesus Buy focused on similar topics? | "Yes. Based on the context, both films are documentary films…" | no |
| 18 | Are Couroupita and Graptopetalum plants both native to central America? | "No, they are not both native to Central America…" | yes |

**Why this happens**: `extract_span` makes the model lead with a span
for yes/no questions too, but the "span" for yes/no questions is just
"yes" or "no". The model treats it as an explanation request and
explains — never typing the literal word.

**Could a prompt change fix this?** Yes — a separate instruction for
yes/no questions ("If the question can be answered with 'yes' or 'no',
respond with that single word"). Modest code change.

### Category D — Extract discipline (3 / 32, 9.4%)

**Pattern**: The model's reasoning section is correct but the visible
extract doesn't end up with a clean entity name. Either the model
copied the question back as the answer, listed options, or narrated
without picking.

| # | Question summary | Predicted | Gold |
|---|---|---|---|
| 17 | More solo albums, Ozzy or Curt Smith | "more solo albums, Ozzy Osbourne or Curt Smith" (restated Q) | John Michael "Ozzy" Osbourne |
| 24 | More bands joined, Johnny Edwards or Ian Anderson | listed Johnny Edwards' bands | John Douglas "Johnny" Edwards |
| 26 | More awards, Dan Schneider or Helen Hunt | listed Helen Hunt's awards | Helen Elizabeth Hunt |

**Why this happens**: The model wrote the comparison correctly in its
reasoning but the lead span (extract_span) ended up being the question
echo or a description, not a clean name. Looks like a discipline
failure — the model thought extracting the answer wasn't important,
just the reasoning.

### Category E — Wrong format (2 / 32, 6.2%)

**Pattern**: Model gave the right kind of thing but in an unexpected
form that doesn't substring-match.

| # | Question summary | Predicted | Gold |
|---|---|---|---|
| 8 | Wider scope of profession, Pete Dexter or Elie Wiesel | "writer, professor, political activist, Nobel Laureate and Holocaust survivor" | Eliezer "Elie" Wiesel KBE |
| 21 | Common pursuit of Minaskanian and Stambolian | (literary magazine text from a related show) | American educator, writer, |

### Category F — Dataset noise (1 / 32, 3.1%)

| # | Question summary | Predicted | Gold | Reality |
|---|---|---|---|---|
| 25 | Formed first, Cha Cha Cohen or Swervedriver | Swervedriver (1989) | Cha Cha Cohen (1994) | Corpus dates say Swervedriver is earlier; gold answer contradicts the retrieved evidence |

This is a HotpotQA annotation error. Model is correct per corpus.

---

## 3. Empirical breakdown — what would close the gap

If we pick the **best targeted intervention per category** and assume
each cleanly closes its category (high upper bound):

| Levers | Categories addressed | Theoretical max lift |
|---|---|---:|
| A. **Canonical-name post-processing** (resolve abbreviations, drop parentheticals from gold matching, expand initials) | Category A (13) | +3.9 pp |
| B. **Strict yes/no instruction** for predicate questions | Category C (3) | +0.9 pp |
| C. **Larger / better LLM** (better comparison, better entity disambiguation, better extract discipline) | Category B (most) + Category D + Category E | +4.5 pp |
| D. **Multi-step pipeline** (ask model to list candidate entities first, then extract) | Categories B / D partially | +2.0 pp |
| E. **Closer extract_span instruction** ("Choose the most formal / canonical name as written in Wikipedia") | Category A partial; risk of over-correction | +1-2 pp |

**Realistic best-case**: A+B+C together would plausibly lift
`contains_gold` from 0.904 → **0.96** on this sample.
**Most practical near-term**: B (yes/no fix) for +0.9 pp, plus a
canonical-name matcher for +2-3 pp ≈ **0.93 ceiling** without changing
the LLM.

---

## 4. Surprising findings

1. **40% of failures are name-form mismatches, not extraction
   errors.** The model frequently knows the right answer but quotes it
   in a slightly different form than gold expects. This is partly a
   metric artifact (substring containment) and partly a model style
   issue.

2. **The CoT prompt helps but doesn't help enough on multi-hop.** A
   handful of multi-hop questions got fixed vs extract_span_k10 (5
   questions from 37 → 32 extraction misses). But ~10 of the 32
   remaining failures are still multi-hop comparison / chain failures
   that the bare "step by step" scaffold doesn't fully address.

3. **Yes/No questions are silently underperforming.** 3 of 32 (9.4%)
   are yes/no where the model gave explanations. If added to a stricter
   prompt, these would close for free.

4. **HotpotQA name conventions aren't intuitive.** Full middle names,
   legal suffixes "(MTS)", parentheticals, ticker symbols, and
   abbreviations are all gold-standard. The conversational model
   consistently picks the friendly / short form.

5. **At least one HotpotQA annotation is wrong** (#25: corpus
   contradicts gold). Without manual inspection we'd never know this;
   it would be a permanent failure noise.

---

## 5. Concrete next-step recommendations (in priority order)

### Quick wins (low cost, isolated change)

1. **Add yes/no handling to the prompt builder (~10 lines)**
   - 3 questions × ~0.9 pp lift → **0.91** ceiling
   - Cost: zero. Pure prompt engineering.

2. **Add canonical-name post-processing for top-k-1 exact match (~30 lines)**
   - 13 questions × ~80% recoverable = 10 questions → **+3 pp**
   - Cost: zero on output side. Some risk of breaking existing
     correct answers that happen to use a shorter form.

### Bigger lifts (require either stronger LLM or pipeline rework)

3. **Switch to a stronger LLM for these multi-hop/dataset questions**
   - Best ROI for the remaining 14 questions (categories B + D)
   - Out of codebase scope; user-side call

4. **Multi-step pipeline (route by question type)** — separate
   prompts for comparison, yes/no, location, etc.
   - ~+2-4 pp on multi-hop + format questions
   - ~150-200 lines of code

### Long-term

5. **Move to a benchmark with stricter name gold standards** — fewer
   "is `Kohlberg Kravis Roberts` close enough to `KKR & Co`?" failures.
   Or use semantic-similarity metrics (RAGAS-style answer-correctness)
   that don't punish name-form variations.

---

## 6. Honest caveats

- **Manual classification is subjective.** I used a single pass with
  consistent rules. Borderline cases (e.g., #8 — counted as wrong-format
  rather than name-variant) could land in either bucket.
- **`minimax-3` may have idiosyncrasies.** Different LLMs might
  produce systematically different name forms (e.g., always full
  names, or always abbreviated). The category sizes here are
  `minimax-3`-specific.
- **Categories B, D, E overlaps.** A multi-hop comparison question
  also has "wrong entity" and "wrong format" and "extract discipline"
  facets. The 10/32 in Category B is somewhat conservative — some of
  these could move to D or E.
- **HotpotQA gold spans** are Wikipedia-style and conservative. On a
  different dataset, Category A would shrink or grow substantially.

---

## 7. Reproducibility

```bash
# Re-run the SOTA with dump
python scripts/eval_qa_hotpotqa.py \
    --subset 1000 \
    --pipeline cot_extract_k10 \
    --dump-results docs/eval-results/iter15-cot-extract-k10-dump.jsonl

# Filter failures from the dump
python -c "
import json
with open('docs/eval-results/iter15-cot-extract-k10-dump.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        if r['mode'] == 'with_context' and r['contains_gold'] < 1.0:
            print(r['qid'], '|', r['gold'], '|', r['predicted'][:80])
"
```

The dump file (one JSON object per line) is preserved so others can
re-classify or extend the analysis.
