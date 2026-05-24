from llm_emotion_test.main import main


def test_help_runs(capsys) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "llm-emotion-test" in captured.out
    assert "prepare-data" in captured.out


def test_prepare_data_loads_config(monkeypatch) -> None:
    def fake_prepare_wrime_dataset(config):
        return {
            "output_path": str(config.data.processed_path),
            "num_samples": 1,
            "text_length": {"min": 1, "max": 1, "mean": 1},
            "split_counts": {"train": 1},
            "label_counts": {"joy": 1},
            "input_latent_counts": {"0": 1},
            "target_latent_counts": {"0": 1},
        }

    monkeypatch.setattr(
        "llm_emotion_test.main.prepare_wrime_dataset", fake_prepare_wrime_dataset
    )

    exit_code = main(["prepare-data", "--config", "configs/base.yaml"])

    assert exit_code == 0


def test_train_sft_loads_config(monkeypatch) -> None:
    def fake_train_sft(config):
        return {
            "final_checkpoint": str(config.output.checkpoints_dir / "final"),
            "eval": {"eval_loss": 1.25},
            "sample_latent_marker_accuracy": 0.5,
        }

    monkeypatch.setattr("llm_emotion_test.main.train_sft", fake_train_sft)

    exit_code = main(["train-sft", "--config", "configs/sft.yaml"])

    assert exit_code == 0


def test_distill_loads_config(monkeypatch) -> None:
    def fake_train_distill(config):
        return {
            "distillation": {
                "output_path": str(config.output.run_dir / "distillation_data.jsonl"),
                "student_data_path": str(config.data.processed_path),
                "teacher_cache_path": str(config.output.run_dir / "teacher_generations.jsonl"),
                "num_distill_records": 1,
            },
            "final_checkpoint": str(config.output.checkpoints_dir / "final"),
            "eval": {"eval_loss": 1.0},
            "sample_latent_marker_accuracy": 1.0,
        }

    monkeypatch.setattr("llm_emotion_test.main.train_distill", fake_train_distill)

    exit_code = main(["distill", "--config", "configs/distill.yaml"])

    assert exit_code == 0
