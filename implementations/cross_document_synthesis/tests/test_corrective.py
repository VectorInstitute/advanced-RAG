"""Measure the shared corrective policy without changing generation settings."""

from synthesis_lab.methods import DenseAnswerer
from synthesis_lab.tutorial import QUESTION, source_answer, synthetic_corpus


def test_single_pass_and_corrective_retry_share_generation_and_thresholds():
    corpus, vectors, encode = synthetic_corpus()
    seen = []

    def complete(prompt):
        seen.append(prompt)
        if prompt.startswith("Rewrite a search query"):
            return "Where was Mira born?"
        if "Return one JSON object:" in prompt:
            return source_answer(prompt)
        return '{"score":0.4,"reason":"missing the other document"}'

    single = DenseAnswerer(corpus.passages, vectors, encode, complete)
    baseline = single.answer("q1", QUESTION)
    corrective = DenseAnswerer(corpus.passages, vectors, encode, complete, corrective=True)
    retried = corrective.answer("q2", QUESTION)
    assert len(baseline.attempts) == 1
    assert len(retried.attempts) == 2
    assert any(p.startswith("Rewrite a search query") for p in seen)
    assert corrective.spine.config.evidence_budget == single.spine.config.evidence_budget == 5
    assert corrective.spine.policy.thresholds.sufficient == 0.6
    assert corrective.spine.policy.thresholds.insufficient == 0
