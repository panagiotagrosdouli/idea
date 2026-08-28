from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_paper_tables(
    benchmark_summary: str | Path,
    forecast_summary: str | Path,
    ablation_summary: str | Path,
    output_path: str | Path,
) -> Path:
    benchmark = _read_json(benchmark_summary)
    forecast = _read_json(forecast_summary)
    ablation = _read_json(ablation_summary)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Communication scheduling results.}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Policy & Goodput & PDR & P95 latency & Before expiry "
            r"& Undelivered & Jain \\"
        ),
        r"\midrule",
    ]
    for name, metrics in benchmark["schedulers"].items():
        lines.append(
            f"{_latex_escape(name)} & "
            f"{metrics['goodput_mbps']['mean']:.3f} & "
            f"{metrics['packet_delivery_ratio']['mean']:.3f} & "
            f"{metrics['p95_latency_ms']['mean']:.1f} & "
            f"{metrics['delivered_before_expiry_ratio']['mean']:.3f} & "
            f"{metrics['undelivered_packets_at_disconnect']['mean']:.1f} & "
            f"{metrics['jain_fairness']['mean']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Motion and derived-link forecast errors.}",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            (
                r"Predictor & ADE & FDE & Range MAE & SNR MAE & F1 "
                r"& AUROC & Lifetime MAE \\"
            ),
            r"\midrule",
        ]
    )
    for name, metrics in forecast.items():
        lines.append(
            f"{_latex_escape(name)} & {metrics['ade_m']:.3f} & "
            f"{metrics['fde_m']:.3f} & {metrics['range_mae_m']:.3f} & "
            f"{metrics['snr_mae_db']:.3f} & "
            f"{metrics['outage_f1']:.3f} & "
            f"{metrics['outage_auroc']:.3f} & "
            f"{metrics['link_lifetime_error_steps']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Paper ablation summary.}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            (
                r"Ablation & Goodput & PDR & Deadline miss & Before expiry "
                r"& Undelivered \\"
            ),
            r"\midrule",
        ]
    )
    for name, metrics in ablation.items():
        lines.append(
            f"{_latex_escape(name)} & {metrics['goodput_mbps']:.3f} & "
            f"{metrics['packet_delivery_ratio']:.3f} & "
            f"{metrics['deadline_miss_ratio']:.3f} & "
            f"{metrics['delivered_before_expiry_ratio']:.3f} & "
            f"{metrics['undelivered_packets_at_disconnect']:.1f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def make_paper_figures(
    forecast_summary: str | Path,
    ablation_summary: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    forecast = _read_json(forecast_summary)
    ablation = _read_json(ablation_summary)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    names = list(forecast)
    ade = [forecast[name]["ade_m"] for name in names]
    snr = [forecast[name]["snr_mae_db"] for name in names]
    figure, first = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    second = first.twinx()
    positions = range(len(names))
    first.bar(positions, ade, color="#2563eb", alpha=0.78, label="ADE")
    second.plot(positions, snr, color="#d97706", marker="o", label="SNR MAE")
    first.set_xticks(list(positions), names, rotation=24, ha="right")
    first.set_ylabel("ADE (m)", color="#2563eb")
    second.set_ylabel("SNR MAE (dB)", color="#d97706")
    first.grid(axis="y", alpha=0.25)
    first.set_title("Trajectory accuracy and communication relevance")
    forecast_path = destination / "forecast_link_tradeoff.png"
    figure.savefig(forecast_path, dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for name in names:
        axis.scatter(
            forecast[name]["ade_m"],
            forecast[name]["outage_f1"],
            s=55,
        )
        axis.annotate(
            name,
            (forecast[name]["ade_m"], forecast[name]["outage_f1"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("ADE (m)")
    axis.set_ylabel("Outage F1")
    axis.set_title("Motion accuracy versus outage-prediction quality")
    axis.grid(alpha=0.25)
    outage_path = destination / "ade_vs_outage_f1.png"
    figure.savefig(outage_path, dpi=220)
    plt.close(figure)

    selected = [
        name
        for name in (
            "no_prediction",
            "cv_predictor",
            "kalman_predictor",
            "imm_predictor",
            "trajectory_predictive",
            "link_lifetime",
            "perfect_future",
        )
        if name in ablation
    ]
    goodput = [ablation[name]["goodput_mbps"] for name in selected]
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    axis.bar(range(len(selected)), goodput, color="#2563eb")
    axis.set_xticks(range(len(selected)), selected, rotation=24, ha="right")
    axis.set_ylabel("Goodput (Mbps)")
    axis.set_title("Reactive, predictive, and oracle-information comparison")
    axis.grid(axis="y", alpha=0.25)
    ablation_path = destination / "scheduler_ablation.png"
    figure.savefig(ablation_path, dpi=220)
    plt.close(figure)

    channel_names = [
        name
        for name in (
            "range_only_channel",
            "range_pointing_channel",
            "full_channel",
            "part_a_ber_lut",
        )
        if name in ablation
    ]
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.bar(
        range(len(channel_names)),
        [ablation[name]["goodput_mbps"] for name in channel_names],
        color="#2563eb",
    )
    axis.set_xticks(
        range(len(channel_names)), channel_names, rotation=22, ha="right"
    )
    axis.set_ylabel("Goodput (Mbps)")
    axis.set_title("Channel-model and BER-source ablation")
    axis.grid(axis="y", alpha=0.25)
    channel_path = destination / "channel_ablation.png"
    figure.savefig(channel_path, dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for prefix, label, color in (
        ("history_noise_", "history measurement noise", "#2563eb"),
        ("forecast_error_", "forecast degradation", "#d97706"),
    ):
        selected_noise = [name for name in ablation if name.startswith(prefix)]
        selected_noise.sort(key=_noise_level)
        levels = [0.0, *[_noise_level(name) for name in selected_noise]]
        values = [
            ablation["full_channel"]["goodput_mbps"],
            *[ablation[name]["goodput_mbps"] for name in selected_noise],
        ]
        axis.plot(levels, values, marker="o", color=color, label=label)
    axis.set_xlabel("Noise standard deviation (m)")
    axis.set_ylabel("Goodput (Mbps)")
    axis.set_title("Robustness to sensing and forecast errors")
    axis.grid(alpha=0.25)
    axis.legend()
    robustness_path = destination / "robustness_noise.png"
    figure.savefig(robustness_path, dpi=220)
    plt.close(figure)
    return {
        "forecast": forecast_path,
        "outage": outage_path,
        "ablation": ablation_path,
        "channel": channel_path,
        "robustness": robustness_path,
    }


def make_example_motion_figure(
    womd_export: str | Path, output_path: str | Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .data.womd_export import load_womd_motion_scenarios

    scenario = load_womd_motion_scenarios(womd_export, max_vehicles=5)[0]
    split = scenario.start_index
    origin = scenario.ego_positions_xy[split - 1]
    ego = scenario.ego_positions_xy - origin
    vehicles = scenario.vehicle_positions_xy - origin
    figure, axis = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    axis.plot(
        ego[:split, 0],
        ego[:split, 1],
        color="black",
        linewidth=2,
        marker="o",
        markersize=3,
        label="proxy ego history",
    )
    axis.plot(
        ego[split - 1 :, 0],
        ego[split - 1 :, 1],
        color="black",
        linestyle="--",
        linewidth=2,
        marker="s",
        markersize=3,
        label="proxy ego future",
    )
    colors = plt.cm.tab10(np.linspace(0, 1, scenario.vehicle_count))
    for vehicle, color in enumerate(colors):
        trajectory = vehicles[:, vehicle]
        axis.plot(
            trajectory[:split, 0],
            trajectory[:split, 1],
            color=color,
            marker="o",
            markersize=2.5,
        )
        axis.plot(
            trajectory[split - 1 :, 0],
            trajectory[split - 1 :, 1],
            color=color,
            linestyle="--",
            marker="s",
            markersize=2.5,
        )
    axis.set_xlabel("Translated global x (m)")
    axis.set_ylabel("Translated global y (m)")
    axis.set_title(f"Compact WOMD mobility example: {scenario.scenario_id}")
    axis.grid(alpha=0.25)
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(fontsize=8)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220)
    plt.close(figure)
    return destination


def _read_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _noise_level(name: str) -> float:
    tail = name.rsplit("_", maxsplit=1)[-1]
    if tail.endswith("m"):
        tail = tail[:-1]
    if "_" in name and "0_5m" in name:
        return 0.5
    return float(tail)
