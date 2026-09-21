from analysis.analyze_transfer_summary import TARGETS
from analysis.plot_regimes import METRIC_SCALE, regimes_summary_figure


def test_transfer_summary_declares_all_release_targets():
    labels = {target[0] for target in TARGETS}
    assert len(TARGETS) == 16
    assert {
        "FiQA-2018 (chunking)",
        "TREC-COVID (inj.)",
        "MuSiQue (inj.)",
        "2WikiMultiHopQA (inj.)",
    }.issubset(labels)


def test_paper_plot_scale_and_summary_entry_point():
    assert METRIC_SCALE == 100.0
    assert callable(regimes_summary_figure)
