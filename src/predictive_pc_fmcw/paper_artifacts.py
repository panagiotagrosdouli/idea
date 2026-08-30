from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .data.scenario import MotionScenario
from .geometry import heading_from_positions, range_and_bearing
from .link import LinkModel
from .simulation.engine import SimulationOutput


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


def make_corrected_result_figures(
    benchmark_summary: str | Path,
    episode_metrics: str | Path,
    staged_rows: str | Path,
    scenario_slices: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create exact charts from one isolated corrected-run directory."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmark = _read_json(benchmark_summary)
    episodes = _read_json(episode_metrics)
    staged = _read_json(staged_rows)
    slices = _read_json(scenario_slices)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    policies = (
        "reactive_greedy",
        "kalman_predictive",
        "link_lifetime",
        "oracle",
    )

    figure, first = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    second = first.twinx()
    positions = np.arange(len(policies))
    goodput = [benchmark["schedulers"][name]["goodput_mbps"] for name in policies]
    means = [item["mean"] for item in goodput]
    errors = np.asarray(
        [
            [item["mean"] - item["low"] for item in goodput],
            [item["high"] - item["mean"] for item in goodput],
        ]
    )
    first.bar(positions, means, yerr=errors, capsize=4, color="#2563eb", alpha=0.8)
    latency = [
        benchmark["schedulers"][name]["p95_latency_ms"]["mean"]
        for name in policies
    ]
    second.plot(positions, latency, color="#d97706", marker="o", linewidth=2)
    first.set_xticks(positions, policies, rotation=18, ha="right")
    first.set_ylabel("Goodput (Mbps), bootstrap 95% CI", color="#2563eb")
    second.set_ylabel("P95 latency (ms)", color="#d97706")
    first.set_title("Corrected controlled benchmark: gain and latency trade-off")
    first.grid(axis="y", alpha=0.25)
    benchmark_path = destination / "corrected_benchmark_tradeoff.png"
    figure.savefig(benchmark_path, dpi=240)
    plt.close(figure)

    by_key = {
        (row["scenario_id"], int(row["seed"]), row["scheduler"]): row
        for row in episodes
    }
    differences = []
    for (scenario, seed, scheduler), row in by_key.items():
        if scheduler != "link_lifetime":
            continue
        baseline = by_key[(scenario, seed, "reactive_greedy")]
        differences.append(float(row["goodput_mbps"]) - float(baseline["goodput_mbps"]))
    values = np.sort(np.asarray(differences, dtype=np.float64))
    ecdf = np.arange(1, values.size + 1) / values.size
    figure, axis = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    axis.step(values, ecdf, where="post", color="#2563eb", linewidth=2)
    axis.axvline(0.0, color="black", linewidth=1, alpha=0.65)
    axis.set_xlabel("Link-Lifetime − Reactive goodput (Mbps)")
    axis.set_ylabel("Empirical cumulative fraction")
    axis.set_title("Paired episode gains include positive and negative cases")
    axis.grid(alpha=0.25)
    ecdf_path = destination / "paired_goodput_difference_ecdf.png"
    figure.savefig(ecdf_path, dpi=240)
    plt.close(figure)

    staged_index = {
        (row["study"], row["setting"], int(row["seed"]), row["scheduler"]): row
        for row in staged
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, study, x_key, label in (
        (axes[0], "offered_load", "offered_load", "Offered load"),
        (axes[1], "prediction_horizon", "prediction_horizon_s", "Horizon (s)"),
    ):
        settings = sorted(
            {row["setting"] for row in staged if row["study"] == study},
            key=lambda setting: float(
                next(
                    row[x_key]
                    for row in staged
                    if row["study"] == study and row["setting"] == setting
                )
            ),
        )
        x_values = []
        means = []
        lows = []
        highs = []
        for setting in settings:
            selected = [
                row
                for row in staged
                if row["study"] == study
                and row["setting"] == setting
                and row["scheduler"] == "link_lifetime"
            ]
            gains = []
            for row in selected:
                reference = staged_index[
                    (study, setting, int(row["seed"]), "reactive_greedy")
                ]
                gains.append(
                    float(row["goodput_mbps"])
                    - float(reference["goodput_mbps"])
                )
            x_values.append(float(selected[0][x_key]))
            means.append(float(np.mean(gains)))
            lows.append(float(np.min(gains)))
            highs.append(float(np.max(gains)))
        axis.plot(x_values, means, color="#2563eb", marker="o", linewidth=2)
        axis.fill_between(x_values, lows, highs, color="#93c5fd", alpha=0.35)
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.65)
        axis.set_xlabel(label)
        axis.set_ylabel("Goodput difference vs Reactive (Mbps)")
        axis.grid(alpha=0.25)
    figure.suptitle("Staged diagnostic: operating-region sign changes")
    staged_path = destination / "staged_operating_region_diagnostic.png"
    figure.savefig(staged_path, dpi=240)
    plt.close(figure)

    label_counts: dict[str, int] = {}
    for row in slices:
        for label in row["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1
    labels = sorted(label_counts, key=label_counts.get, reverse=True)
    figure, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    axis.barh(
        labels[::-1],
        [label_counts[label] for label in labels[::-1]],
        color="#2563eb",
    )
    axis.set_xlabel("Actor-scenario count")
    axis.set_title("Deterministic mobility and optical-boundary slices")
    axis.grid(axis="x", alpha=0.25)
    slice_path = destination / "scenario_slice_counts.png"
    figure.savefig(slice_path, dpi=240)
    plt.close(figure)
    return {
        "benchmark": benchmark_path,
        "ecdf": ecdf_path,
        "staged": staged_path,
        "slices": slice_path,
    }


def make_required_diagnostic_figures(
    benchmark_summary: str | Path,
    episode_metrics: str | Path,
    forecast_metrics: str | Path,
    ber_lut: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create the diagnostic plots explicitly requested by the research plans."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmark = _read_json(benchmark_summary)
    episodes = _read_json(episode_metrics)
    forecasts = _read_json(forecast_metrics)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    policies = [
        name
        for name in (
            "reactive_greedy",
            "proportional_fair",
            "cv_predictive",
            "kalman_predictive",
            "imm_predictive",
            "link_lifetime",
            "oracle",
        )
        if name in benchmark["schedulers"]
    ]
    colors = dict(
        zip(
            policies,
            plt.cm.viridis(np.linspace(0.08, 0.92, len(policies))),
            strict=True,
        )
    )

    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for policy in policies:
        values = np.sort(
            np.asarray(
                [
                    float(row["goodput_mbps"])
                    for row in episodes
                    if row["scheduler"] == policy
                ]
            )
        )
        axis.step(
            values,
            np.arange(1, values.size + 1) / values.size,
            where="post",
            linewidth=1.8,
            label=policy,
            color=colors[policy],
        )
    axis.set_xlabel("Episode goodput (Mbps)")
    axis.set_ylabel("Empirical cumulative fraction")
    axis.set_title("Policy-level goodput distributions")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    goodput_path = destination / "policy_goodput_ecdf.png"
    figure.savefig(goodput_path, dpi=240)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for policy in policies:
        metrics = benchmark["schedulers"][policy]
        x_value = float(metrics["scheduled_outage_fraction"]["mean"])
        y_value = float(metrics["p95_latency_ms"]["mean"])
        axis.scatter(
            x_value,
            y_value,
            s=65,
            color=colors[policy],
            label=policy,
        )
    axis.set_xlabel("Scheduled outage fraction")
    axis.set_ylabel("P95 packet latency (ms)")
    axis.set_title("Reliability–latency operating points")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    outage_latency_path = destination / "outage_latency_tradeoff.png"
    figure.savefig(outage_latency_path, dpi=240)
    plt.close(figure)

    points = []
    for policy in policies:
        metrics = benchmark["schedulers"][policy]
        points.append(
            (
                policy,
                float(metrics["demand_normalized_jain_fairness"]["mean"]),
                float(metrics["goodput_mbps"]["mean"]),
            )
        )
    pareto = {
        name
        for name, fairness, goodput in points
        if not any(
            (other_fairness >= fairness and other_goodput >= goodput)
            and (other_fairness > fairness or other_goodput > goodput)
            for other_name, other_fairness, other_goodput in points
            if other_name != name
        )
    }
    figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for policy, fairness, goodput in points:
        axis.scatter(
            fairness,
            goodput,
            s=90 if policy in pareto else 55,
            marker="D" if policy in pareto else "o",
            color=colors[policy],
            edgecolor="black" if policy in pareto else "none",
            linewidth=0.8,
            label=policy,
        )
    axis.set_xlabel("Demand-normalized Jain fairness")
    axis.set_ylabel("Mean goodput (Mbps)")
    axis.set_title("Throughput–fairness Pareto diagnostic")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2, loc="lower right")
    pareto_path = destination / "throughput_fairness_pareto.png"
    figure.savefig(pareto_path, dpi=240)
    plt.close(figure)

    predictors = [
        name
        for name in (
            "last_position",
            "constant_velocity",
            "kalman_cv",
            "imm",
            "constant_acceleration",
        )
        if any(row["predictor"] == name for row in forecasts)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    maximum = 0.0
    for predictor in predictors:
        rows = [row for row in forecasts if row["predictor"] == predictor]
        actual = np.asarray(
            [float(row["actual_link_lifetime_s"]) for row in rows]
        )
        predicted = np.asarray(
            [float(row["predicted_link_lifetime_s"]) for row in rows]
        )
        errors = np.sort(np.abs(predicted - actual))
        maximum = max(maximum, float(actual.max()), float(predicted.max()))
        axes[0].scatter(actual, predicted, s=12, alpha=0.28, label=predictor)
        axes[1].step(
            errors,
            np.arange(1, errors.size + 1) / errors.size,
            where="post",
            linewidth=1.7,
            label=predictor,
        )
    axes[0].plot([0, maximum], [0, maximum], color="black", linestyle="--")
    axes[0].set_xlabel("Actual/censored link lifetime (s)")
    axes[0].set_ylabel("Predicted/censored link lifetime (s)")
    axes[0].set_title("Link-lifetime calibration")
    axes[1].set_xlabel("Absolute lifetime error (s)")
    axes[1].set_ylabel("Empirical cumulative fraction")
    axes[1].set_title("Link-lifetime error ECDF")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    lifetime_path = destination / "link_lifetime_calibration_ecdf.png"
    figure.savefig(lifetime_path, dpi=240)
    plt.close(figure)

    with Path(ber_lut).open("r", encoding="utf-8") as handle:
        ber_rows = list(csv.DictReader(handle))
    snr = np.asarray([float(row["ebn0_db"]) for row in ber_rows])
    observed = np.asarray([float(row["simulated_ber"]) for row in ber_rows])
    upper = np.asarray([float(row["ber_upper_95"]) for row in ber_rows])
    lut = np.asarray([float(row["ber_for_lut"]) for row in ber_rows])
    theory = np.asarray([float(row["theoretical_ber"]) for row in ber_rows])
    semantics = ber_rows[0].get("snr_semantics", "ebn0_db")
    receiver = ber_rows[0].get("receiver", "symbol_level_dbpsk")
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    axis.semilogy(snr, np.maximum(lut, 1e-15), marker="o", label="LUT value")
    axis.semilogy(
        snr,
        np.maximum(upper, 1e-15),
        linestyle=":",
        label="one-sided 95% upper bound",
    )
    if semantics == "ebn0_db":
        axis.semilogy(
            snr,
            np.maximum(theory, 1e-15),
            linestyle="--",
            label="DBPSK theory",
        )
    zero_errors = observed == 0
    if np.any(zero_errors):
        axis.scatter(
            snr[zero_errors],
            np.maximum(lut[zero_errors], 1e-15),
            marker="v",
            color="#dc2626",
            label="zero-error, confidence-limited",
        )
    axis.set_xlabel(
        "Waveform sample SNR (dB)"
        if semantics == "waveform_sample_snr_db"
        else "$E_b/N_0$ (dB)"
    )
    axis.set_ylabel("Bit error rate")
    axis.set_title(f"DPSK BER calibration: {receiver}")
    axis.grid(which="both", alpha=0.25)
    axis.legend(fontsize=7)
    ber_path = destination / "dpsk_ber_curve.png"
    figure.savefig(ber_path, dpi=240)
    plt.close(figure)

    uncensored = [
        row
        for row in forecasts
        if row["predictor"] != "oracle"
        and not bool(row["actual_link_lifetime_censored"])
    ]
    ranked = sorted(
        uncensored,
        key=lambda row: float(row["link_lifetime_error_s"]),
        reverse=True,
    )
    worst = ranked[:6]
    figure, axes = plt.subplots(
        1, 2, figsize=(11.2, 4.8), constrained_layout=True
    )
    predictor_colors = dict(
        zip(
            predictors,
            plt.cm.tab10(np.linspace(0.0, 0.8, len(predictors))),
            strict=True,
        )
    )
    for predictor in predictors:
        rows = [row for row in uncensored if row["predictor"] == predictor]
        axes[0].scatter(
            [float(row["ade_m"]) for row in rows],
            [float(row["link_lifetime_error_s"]) for row in rows],
            s=13,
            alpha=0.25,
            color=predictor_colors[predictor],
            label=predictor,
        )
    for index, row in enumerate(worst, start=1):
        axes[0].annotate(
            str(index),
            (
                float(row["ade_m"]),
                float(row["link_lifetime_error_s"]),
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )
    labels = [
        f"{index}: {row['predictor']}\n{row['actor_id']} @ {row['anchor_index']}"
        for index, row in enumerate(worst, start=1)
    ]
    positions = np.arange(len(worst))
    axes[1].barh(
        positions,
        [float(row["predicted_link_lifetime_s"]) for row in worst],
        color="#d97706",
        alpha=0.82,
        label="predicted",
    )
    axes[1].scatter(
        [float(row["actual_link_lifetime_s"]) for row in worst],
        positions,
        color="black",
        marker="D",
        s=32,
        label="actual",
        zorder=3,
    )
    axes[0].set_xlabel("Trajectory ADE (m)")
    axes[0].set_ylabel("Absolute link-lifetime error (s)")
    axes[0].set_title("All uncensored deployable forecasts")
    axes[1].set_yticks(positions, labels, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Link lifetime (s)")
    axes[1].set_title("Six largest lifetime errors")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    figure.suptitle("Forecast failure cases at optical-link boundaries")
    failure_path = destination / "forecast_failure_cases.png"
    figure.savefig(failure_path, dpi=240)
    plt.close(figure)
    return {
        "goodput_ecdf": goodput_path,
        "outage_latency": outage_latency_path,
        "pareto": pareto_path,
        "lifetime": lifetime_path,
        "ber": ber_path,
        "failure_cases": failure_path,
    }


def make_scheduler_timeline_figure(
    output: SimulationOutput,
    slot_duration_s: float,
    output_path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_s = np.arange(output.selected_vehicle.size) * slot_duration_s
    figure, axes = plt.subplots(
        3, 1, figsize=(10, 7.2), sharex=True, constrained_layout=True
    )
    for vehicle in range(output.actual_snr_db.shape[1]):
        axes[0].plot(
            time_s,
            output.actual_snr_db[:, vehicle],
            linewidth=1.2,
            label=f"vehicle {vehicle}",
        )
        axes[2].plot(
            time_s,
            output.queue_packets[:, vehicle],
            linewidth=1.2,
            label=f"vehicle {vehicle}",
        )
    axes[0].set_ylabel("Actual SNR (dB)")
    axes[0].set_title("Exact scheduler timeline for one controlled episode")
    selected = output.selected_vehicle.astype(np.float64)
    selected[selected < 0] = np.nan
    axes[1].step(time_s, selected, where="post", color="#dc2626", linewidth=1.4)
    axes[1].set_ylabel("Selected vehicle")
    axes[1].set_yticks(np.arange(output.actual_snr_db.shape[1]))
    axes[2].set_ylabel("Queued packets")
    axes[2].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=3)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=240)
    plt.close(figure)
    return destination


def make_probabilistic_calibration_figure(
    calibration_artifact: str | Path, output_path: str | Path
) -> Path:
    """Plot held-out empirical coverage for classical Gaussian baselines."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _read_json(calibration_artifact)
    rows = payload["metrics"]
    nominal = np.asarray([0.50, 0.90, 0.95])
    figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="ideal")
    for row in rows:
        empirical = np.asarray(
            [row["coverage_50"], row["coverage_90"], row["coverage_95"]]
        )
        axis.plot(
            nominal,
            empirical,
            marker="o",
            linewidth=2,
            label=(
                f"{row['predictor']} "
                f"(calibration error={row['calibration_error']:.3f})"
            ),
        )
    axis.set_xlim(0.45, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Nominal 2-D Gaussian coverage")
    axis.set_ylabel("Held-out empirical coverage")
    axis.set_title("Scenario-safe classical Gaussian calibration")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=240)
    plt.close(figure)
    return destination


def make_trajectory_link_trace_figure(
    scenario: MotionScenario,
    link_model: LinkModel,
    output_path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headings = heading_from_positions(scenario.ego_positions_xy)
    distance, bearing = range_and_bearing(
        scenario.vehicle_positions_xy,
        scenario.ego_positions_xy[:, None, :],
        headings[:, None],
    )
    distance = distance.T
    bearing = bearing.T
    link = link_model.evaluate_arrays(distance, bearing)
    vehicle = int(np.argmax(np.ptp(link["snr_db"], axis=1)))
    time_s = scenario.timestamps_s - scenario.timestamps_s[0]
    figure, axes = plt.subplots(
        3, 1, figsize=(9.5, 7.0), sharex=True, constrained_layout=True
    )
    axes[0].plot(time_s, distance[vehicle], color="#2563eb")
    axes[0].set_ylabel("Range (m)")
    axes[0].set_title("Trajectory-derived communication trace")
    axes[1].plot(time_s, link["snr_db"][vehicle], color="#d97706")
    axes[1].fill_between(
        time_s,
        np.min(link["snr_db"][vehicle]),
        np.max(link["snr_db"][vehicle]),
        where=link["outage"][vehicle],
        color="#fee2e2",
        alpha=0.65,
        label="outage",
    )
    axes[1].set_ylabel("SNR (dB)")
    axes[1].legend(fontsize=8)
    axes[2].semilogy(
        time_s,
        np.maximum(link["ber"][vehicle], 1e-15),
        color="#7c3aed",
        label="BER",
    )
    axes[2].semilogy(
        time_s,
        np.maximum(link["per"][vehicle], 1e-15),
        color="#dc2626",
        label="PER",
    )
    axes[2].set_ylabel("Error probability")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=240)
    plt.close(figure)
    return destination


def make_system_architecture_figure(output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")
    boxes = (
        (0.3, 2.8, 1.55, "Causal WOMD\nhistory"),
        (2.25, 2.8, 1.55, "Trajectory\nforecast"),
        (4.2, 2.8, 1.55, "Future relative\ngeometry"),
        (6.15, 2.8, 1.55, "PC-FMCW/DPSK\nlink model"),
        (8.1, 2.8, 1.55, "Finite-horizon\nscheduler"),
        (10.05, 2.8, 1.55, "Goodput, outage,\nPDR, latency, fairness"),
        (5.25, 0.65, 2.05, "Packet queues, deadlines\nand traffic workload"),
        (2.8, 0.65, 2.05, "Communication-aware\ntraining loss"),
    )
    for x, y, width, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            0.95,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#1f4e79",
            facecolor="#eaf2f8",
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + 0.475, label, ha="center", va="center", fontsize=9)
    for left, right in zip(boxes[:5], boxes[1:6], strict=True):
        axis.add_patch(
            FancyArrowPatch(
                (left[0] + left[2], left[1] + 0.475),
                (right[0], right[1] + 0.475),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#374151",
            )
        )
    axis.add_patch(
        FancyArrowPatch(
            (6.3, 1.6),
            (8.65, 2.8),
            arrowstyle="-|>",
            mutation_scale=12,
            color="#374151",
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (4.85, 1.15),
            (5.3, 2.8),
            arrowstyle="-|>",
            mutation_scale=12,
            color="#374151",
        )
    )
    axis.text(
        6.0,
        4.35,
        (
            "Prediction informs decisions; ground-truth geometry realizes "
            "packet outcomes"
        ),
        ha="center",
        fontsize=10,
        weight="bold",
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=240, bbox_inches="tight")
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
