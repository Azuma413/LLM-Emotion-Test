from llm_emotion_test.main import main


def test_help_runs(capsys) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "llm-emotion-test" in captured.out
    assert "prepare-data" in captured.out


def test_prepare_data_loads_config() -> None:
    exit_code = main(["prepare-data", "--config", "configs/base.yaml"])

    assert exit_code == 0
