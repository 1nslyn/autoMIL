# Humanize Check Report

- Matrix path: `paper/preprint/NeurIPS26_autoMIL/paper_rewriting_output/humanize_matrix.md`
- Humanize tier: medium
- Matrix rows: 50
- Manuscript paragraphs: 82
- Coverage: 61%
- Sentence length stddev: 24.25
- Connector density: 0.34/1k chars
- Status: PASS

## Dimension Scores

### D1 sentence structure: WARNING [required]
- Metrics: sentence_count=906, length_stddev=24.36, sentence_length_cv=0.557, repeated_start_ratio=0.27, uniform_length_runs=45, short_sentence_ratio=0.2, long_sentence_ratio=0.02
- Affected units: S8-S10, S9-S11, S10-S12, S11-S13, S12-S14
- D1 consecutive sentences have near-identical lengths: ['S8-S10', 'S9-S11', 'S10-S12', 'S11-S13', 'S12-S14'].

### D2 paragraph similarity: PASS [required]
- Metrics: paragraph_count=82, max_4gram_count=4, repeated_4gram_ratio=0.0243, paragraph_length_stddev=318.48, repeated_opening_ratio=0.22, min_paragraph_length=51, max_paragraph_length=1521, adjacent_paragraph_similarity_mean=0.065, adjacent_paragraph_similarity_max=0.362
- No dimension-specific risk found.

### D3 information density: WARNING [required]
- Metrics: generic_phrase_density=0.0, information_anchor_density=9.12, generic_phrase_count=0, anchor_count=260, mechanism_term_count=115, ttr=0.2578, token_count=5602, unique_token_count=1444
- D3 TTR information density is low: 0.2578 < 0.32. Consider using more diverse vocabulary.

### D4 connector frequency: PASS [required]
- Metrics: connector_count=14, connector_density=0.34, max_paragraph_connector_density=3.88
- No dimension-specific risk found.

### D5 term-context matching: PASS [advisory]
- Metrics: frequent_terms_checked=12, contexts_checked=96, generic_context_ratio=0.0, mechanism_contexts=44, risky_terms=
- No dimension-specific risk found.

## Required Findings

- None

## Advisory Findings

- Advisory dimensions not covered: term-context matching
- D1 consecutive sentences have near-identical lengths: ['S8-S10', 'S9-S11', 'S10-S12', 'S11-S13', 'S12-S14'].
- D3 TTR information density is low: 0.2578 < 0.32. Consider using more diverse vocabulary.

## Threshold Profile

- adjacent_similarity_max_fail: 0.65
- adjacent_similarity_mean_warning: 0.45
- max_4gram_count_warning: 5
- max_connector_density: 8
- max_generic_density: 7
- max_paragraph_connector_density: 14
- max_repeated_start_ratio: 0.35
- max_term_generic_context_ratio: 0.45
- min_info_anchor_density: 2.5
- min_paragraph_length_stddev: 25
- min_sentence_length_stddev: 6
- repeated_4gram_ratio_fail: 0.15
- repeated_4gram_ratio_warning: 0.08
- sentence_length_cv_fail: 0.25
- sentence_length_cv_warning: 0.35
- ttr_fail_en: 0.25
- ttr_fail_zh: 0.35
- ttr_warning_en: 0.32
- ttr_warning_zh: 0.42
