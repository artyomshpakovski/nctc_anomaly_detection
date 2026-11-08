# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


ORG_IP = "org_ip"
ANON_PROTOCOL_PATTERN = r"(?:^tor\b|shadowsocks|vpn|psiphon|proxy|lantern|warp)"

CORE_FEATURES = [
    "log_flows",
    "log_flows_per_session",
    "log_n_hours",
    "log_packets_per_flow",
    "log_pkt_size",
    "log_median_duration_ms",
    "log_n_rports",
    "upload_ratio",
    "share_anon",
    "share_p2p",
    "share_icmp",
    "retrans_rate",
    "median_flow_retrans_ratio",
]

FEATURE_GROUPS = {
    "activity": [
        "log_flows",
        "log_flows_per_session",
        "log_n_hours",
    ],
    "flow_structure": [
        "log_packets_per_flow",
        "log_pkt_size",
        "log_median_duration_ms",
        "log_n_rports",
        "upload_ratio",
    ],
    "traffic_behavior": [
        "share_anon",
        "share_p2p",
        "share_icmp",
        "retrans_rate",
        "median_flow_retrans_ratio",
    ],
}

REQUIRED_COLUMNS = [
    "source_ipv4",
    "destination_ipv4",
    "octet_delta_count",
    "packet_delta_count",
    "net_protocol",
]
OPTIONAL_COLUMNS = [
    "source_port",
    "destination_port",
    "session_id",
    "repeat_delta_pack",
    "flow_start_time",
    "flow_end_time",
    "flow_start_millisecond",
    "flow_end_millisecond",
    "source_astitle",
    "destination_astitle",
    "app_protocol_group",
    "app_protocol",
]

PALETTE = {
    "blue": "#3977b9",
    "orange": "#e47a43",
    "green": "#31936e",
    "red": "#d95757",
    "purple": "#8064a2",
    "gray": "#b7bdc5",
    "dark": "#263238",
    "grid": "#e6e9ed",
    "background": "#ffffff",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["background"],
    "axes.facecolor": PALETTE["background"],
    "savefig.facecolor": PALETTE["background"],
    "axes.edgecolor": PALETTE["grid"],
    "axes.labelcolor": PALETTE["dark"],
    "xtick.color": PALETTE["dark"],
    "ytick.color": PALETTE["dark"],
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.frameon": False,
})


def _mode_or_unknown(values: pd.Series):
    values = values.dropna().astype("string").str.strip()
    values = values[values.ne("") & values.ne("<NA>")]
    if values.empty:
        return "Unknown"
    modes = values.mode()
    return str(modes.iloc[0] if not modes.empty else values.iloc[0])


def _read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [col for col in REQUIRED_COLUMNS if col not in header]
    if missing:
        raise ValueError(
            "В CSV отсутствуют обязательные столбцы: " + ", ".join(missing)
        )

    usecols = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in header]
    print(f"Чтение CSV: {path}")
    print(f"Загружаются поля: {len(usecols)} из {len(header)}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype("float64")


def prepare_flows(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if raw.empty:
        raise ValueError("Входной CSV не содержит записей.")

    src = raw["source_ipv4"].astype("string").str.strip().str.lower()
    dst = raw["destination_ipv4"].astype("string").str.strip().str.lower()
    org = ORG_IP.lower()

    source_is_org = src.eq(org)
    destination_is_org = dst.eq(org)

    is_outgoing = source_is_org & ~destination_is_org
    is_incoming = destination_is_org & ~source_is_org
    valid_direction = is_outgoing | is_incoming

    flows = raw.loc[valid_direction].copy()
    outgoing = is_outgoing.loc[valid_direction]
    flows["is_outgoing"] = outgoing.astype(bool)
    flows["remote_ip"] = np.where(
        flows["is_outgoing"],
        flows["destination_ipv4"].astype("string").str.strip(),
        flows["source_ipv4"].astype("string").str.strip(),
    )

    net_proto = flows["net_protocol"].astype("string").str.strip().str.upper()
    is_tcp_udp = net_proto.str.startswith(("TCP", "UDP"), na=False)
    source_port = _numeric(flows, "source_port", np.nan)
    destination_port = _numeric(flows, "destination_port", np.nan)
    source_port = source_port.where(source_port.between(1, 65535))
    destination_port = destination_port.where(destination_port.between(1, 65535))

    flows["remote_port"] = destination_port.where(
        flows["is_outgoing"], source_port
    ).where(is_tcp_udp)
    flows["local_port"] = source_port.where(
        flows["is_outgoing"], destination_port
    ).where(is_tcp_udp)

    if "session_id" not in flows.columns:
        flows["session_id"] = flows.index.astype(str)
    else:
        session = flows["session_id"].astype("string")
        missing_session = session.isna() | session.str.strip().eq("")
        session = session.copy()
        session.loc[missing_session] = "missing_session_" + flows.index[missing_session].astype(str)
        flows["session_id"] = session

    flows["bytes"] = _numeric(flows, "octet_delta_count", 0.0).clip(lower=0)
    flows["packets"] = _numeric(flows, "packet_delta_count", 0.0).clip(lower=0)
    flows["repeats"] = _numeric(flows, "repeat_delta_pack", 0.0).clip(lower=0)

    if "flow_start_time" in flows.columns:
        start_dt = pd.to_datetime(flows["flow_start_time"], errors="coerce", utc=True)
    else:
        start_dt = pd.Series(pd.NaT, index=flows.index, dtype="datetime64[ns, UTC]")

    if start_dt.isna().all() and "flow_start_millisecond" in flows.columns:
        start_ms = pd.to_numeric(flows["flow_start_millisecond"], errors="coerce")
        start_dt = pd.to_datetime(start_ms, unit="ms", errors="coerce", utc=True)

    flows["hour"] = start_dt.dt.hour.astype("float64")

    if "flow_start_millisecond" in flows.columns and "flow_end_millisecond" in flows.columns:
        start_ms = pd.to_numeric(flows["flow_start_millisecond"], errors="coerce")
        end_ms = pd.to_numeric(flows["flow_end_millisecond"], errors="coerce")
        duration = end_ms - start_ms
    elif "flow_start_time" in flows.columns and "flow_end_time" in flows.columns:
        start_t = pd.to_datetime(flows["flow_start_time"], errors="coerce", utc=True)
        end_t = pd.to_datetime(flows["flow_end_time"], errors="coerce", utc=True)
        duration = (end_t - start_t).dt.total_seconds() * 1000
    else:
        duration = pd.Series(np.nan, index=flows.index, dtype="float64")
    flows["duration_ms"] = pd.to_numeric(duration, errors="coerce").where(lambda x: x >= 0)

    if "source_astitle" in flows.columns and "destination_astitle" in flows.columns:
        flows["remote_as_title"] = flows["destination_astitle"].where(
            flows["is_outgoing"], flows["source_astitle"]
        )
    else:
        flows["remote_as_title"] = "Unknown"

    if "app_protocol" in flows.columns:
        app = flows["app_protocol"].astype("string").str.strip().str.lower().fillna("")
    else:
        app = pd.Series("", index=flows.index, dtype="string")
    if "app_protocol_group" in flows.columns:
        app_group = flows["app_protocol_group"].astype("string").str.strip().str.lower().fillna("")
    else:
        app_group = pd.Series("", index=flows.index, dtype="string")
    flows["app_protocol_clean"] = app
    flows["app_group_clean"] = app_group

    flows["is_anon"] = app_group.eq("vpn") | app.str.contains(
        ANON_PROTOCOL_PATTERN, regex=True, na=False
    )
    flows["is_tor"] = app.str.startswith("tor", na=False)
    flows["is_p2p"] = app_group.eq("p2p") | app.str.contains("bittorrent", regex=False, na=False)
    flows["is_icmp"] = net_proto.str.startswith("ICMP", na=False)
    flows["flow_retrans_ratio"] = flows["repeats"].div(
        flows["packets"].where(flows["packets"] > 0)
    )
    flows["repeat_exceeds_packets"] = flows["repeats"] > flows["packets"]

    flows = flows.loc[flows["remote_ip"].notna() & flows["remote_ip"].ne("")].copy()
    diagnostics = {
        "input_rows": int(len(raw)),
        "usable_rows": int(len(flows)),
        "rows_removed_not_exactly_one_org_side": int(len(raw) - int(valid_direction.sum())),
        "outgoing_rows": int(is_outgoing.sum()),
        "incoming_rows": int(is_incoming.sum()),
        "unique_remote_ips": int(flows["remote_ip"].nunique()),
        "timestamp_missing_rows": int(flows["hour"].isna().sum()),
        "duration_missing_rows": int(flows["duration_ms"].isna().sum()),
        "port_values_counted_only_for_tcp_udp": True,
    }
    if flows.empty:
        raise ValueError(
            "После определения стороны организации не осталось потоков. "
            f"Проверьте значение ORG_IP={ORG_IP!r} и поля source_ipv4/destination_ipv4."
        )
    return flows, diagnostics


def _group_mode(df: pd.DataFrame, column: str) -> pd.Series:
    part = df[["remote_ip", column]].dropna(subset=[column])
    if part.empty:
        return pd.Series(dtype="object")
    counts = (
        part.groupby(["remote_ip", column], observed=True, sort=False)
        .size().rename("count").reset_index()
    )
    counts = counts.sort_values(
        ["remote_ip", "count", column],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return counts.drop_duplicates("remote_ip").set_index("remote_ip")[column]


def build_profiles(flows: pd.DataFrame) -> pd.DataFrame:
    g = flows.groupby("remote_ip", sort=False, observed=True)
    out = flows.loc[flows["is_outgoing"]]
    inn = flows.loc[~flows["is_outgoing"]]

    profile = pd.DataFrame(index=g.size().index)
    profile.index.name = "remote_ip"
    profile["flows"] = g.size()
    profile["sessions"] = g["session_id"].nunique()
    profile["as_name"] = _group_mode(flows, "remote_as_title").reindex(profile.index).fillna("Unknown")
    profile["main_app"] = _group_mode(flows, "app_protocol_clean").reindex(profile.index).fillna("Unknown")
    profile["main_app_group"] = _group_mode(flows, "app_group_clean").reindex(profile.index).fillna("Unknown")
    profile["main_remote_port"] = _group_mode(flows, "remote_port").reindex(profile.index)

    profile["flows_out"] = out.groupby("remote_ip").size().reindex(profile.index, fill_value=0)
    profile["flows_in"] = inn.groupby("remote_ip").size().reindex(profile.index, fill_value=0)
    profile["bytes_out"] = out.groupby("remote_ip")["bytes"].sum().reindex(profile.index, fill_value=0)
    profile["bytes_in"] = inn.groupby("remote_ip")["bytes"].sum().reindex(profile.index, fill_value=0)
    profile["bytes_total"] = profile["bytes_out"] + profile["bytes_in"]
    profile["packets"] = g["packets"].sum()
    profile["retrans"] = g["repeats"].sum()

    profile["n_rports"] = g["remote_port"].nunique()
    profile["n_lports"] = g["local_port"].nunique()
    profile["share_1pkt"] = (flows["packets"] <= 1).groupby(flows["remote_ip"]).mean().reindex(profile.index)
    profile["n_hours"] = g["hour"].nunique()
    profile["median_duration_ms"] = g["duration_ms"].median()
    profile["p95_duration_ms"] = g["duration_ms"].quantile(0.95)

    for source, target in [
        ("is_anon", "share_anon"),
        ("is_tor", "share_tor"),
        ("is_p2p", "share_p2p"),
        ("is_icmp", "share_icmp"),
        ("repeat_exceeds_packets", "repeat_gt_packets_share"),
    ]:
        profile[target] = g[source].mean()

    is_bot = flows["app_group_clean"].str.contains("bot", regex=False, na=False)
    is_custom = flows["app_group_clean"].str.contains("custom", regex=False, na=False)
    profile["bot_flows"] = is_bot.groupby(flows["remote_ip"]).sum().reindex(profile.index, fill_value=0)
    profile["custom_protocol_flows"] = is_custom.groupby(flows["remote_ip"]).sum().reindex(profile.index, fill_value=0)
    profile["icmp_flows"] = g["is_icmp"].sum()
    profile["bytes_per_session"] = profile["bytes_total"] / profile["sessions"].replace(0, np.nan)
    profile["pkt_size"] = profile["bytes_total"] / profile["packets"].replace(0, np.nan)
    profile["upload_ratio"] = profile["bytes_out"] / profile["bytes_total"].replace(0, np.nan)
    profile["out_in_flow_ratio"] = (profile["flows_out"] + 1) / (profile["flows_in"] + 1)
    profile["retrans_rate"] = profile["retrans"] / profile["packets"].replace(0, np.nan)
    profile["median_flow_retrans_ratio"] = g["flow_retrans_ratio"].median()
    profile["max_bytes_out_hour"] = (
        out.groupby(["remote_ip", "hour"])["bytes"].sum()
        .groupby("remote_ip").max().reindex(profile.index).fillna(0)
    )

    share_columns = [
        "share_1pkt", "share_anon", "share_tor", "share_p2p", "share_icmp",
        "repeat_gt_packets_share",
    ]
    profile[share_columns] = profile[share_columns].fillna(0)
    profile = profile.replace([np.inf, -np.inf], np.nan)
    return profile.sort_index()


def make_core_matrix(profiles: pd.DataFrame, min_flows: int = 20):
    p = profiles.copy()
    p["flows_per_session"] = p["flows"] / p["sessions"].replace(0, np.nan)
    p["packets_per_flow"] = p["packets"] / p["flows"].replace(0, np.nan)

    raw_features = pd.DataFrame(index=p.index)
    raw_features["log_flows"] = np.log1p(p["flows"].clip(lower=0))
    raw_features["log_flows_per_session"] = np.log1p(p["flows_per_session"].clip(lower=0))
    raw_features["log_n_hours"] = np.log1p(p["n_hours"].clip(lower=0))
    raw_features["log_packets_per_flow"] = np.log1p(p["packets_per_flow"].clip(lower=0))
    raw_features["log_pkt_size"] = np.log1p(p["pkt_size"].clip(lower=0))
    raw_features["log_median_duration_ms"] = np.log1p(p["median_duration_ms"].clip(lower=0))
    raw_features["log_n_rports"] = np.log1p(p["n_rports"].clip(lower=0))
    raw_features["upload_ratio"] = p["upload_ratio"]
    raw_features["share_anon"] = p["share_anon"]
    raw_features["share_p2p"] = p["share_p2p"]
    raw_features["share_icmp"] = p["share_icmp"]
    raw_features["retrans_rate"] = p["retrans_rate"]
    raw_features["median_flow_retrans_ratio"] = p["median_flow_retrans_ratio"]

    raw_features = raw_features.replace([np.inf, -np.inf], np.nan)
    for column in raw_features:
        median = raw_features[column].median()
        raw_features[column] = raw_features[column].fillna(0.0 if pd.isna(median) else median)
    raw_features = raw_features.astype(float)

    scaler = RobustScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(raw_features[CORE_FEATURES]).astype("float64"),
        index=raw_features.index,
        columns=CORE_FEATURES,
    )

    group_variances = {
        name: float(scaled[columns].var(axis=0, ddof=0).sum())
        for name, columns in FEATURE_GROUPS.items()
    }
    positive = [v for v in group_variances.values() if v > 0]
    target = float(np.mean(positive)) if positive else 1.0
    group_weights = {}
    balanced = scaled.copy()
    for name, columns in FEATURE_GROUPS.items():
        variance = group_variances[name]
        weight = math.sqrt(target / variance) if variance > 0 else 1.0
        group_weights[name] = float(weight)
        balanced.loc[:, columns] = balanced[columns] * weight

    active_index = profiles.index[profiles["flows"] >= min_flows]
    if len(active_index) < 3:
        raise ValueError(
            f"После фильтра flows >= {min_flows} осталось только {len(active_index)} IP. "
            "Для сравнения трёх моделей требуется больше профилей."
        )
    return raw_features, balanced, group_weights, group_variances


def _top_set(scores: pd.Series, fraction: float) -> set:
    k = max(1, int(math.ceil(len(scores) * fraction)))
    return set(scores.sort_values(ascending=False, kind="mergesort").head(k).index)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def run_models(matrix: pd.DataFrame, seed: int, eps: float | None, min_samples: int):
    n = len(matrix)
    X = matrix.to_numpy(dtype=float)

    print(f"Обучение моделей на {n:,} активных IP-профилях...")
    iso = IsolationForest(
        n_estimators=300,
        max_samples=min(256, n),
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X)
    if_score = -iso.score_samples(X)
    if_label = iso.predict(X)

    n_neighbors = min(35, n - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric="minkowski", p=2)
    lof.fit_predict(X)
    lof_score = -lof.negative_outlier_factor_

    db_min_samples = min(min_samples, n)
    if eps is None:
        eps = 1.373  # базовая конфигурация DBSCAN, выбранная в исследовании
    eps_source = "экспериментальная базовая конфигурация" if eps == 1.373 else "значение из аргумента --eps"
    dbscan = DBSCAN(eps=eps, min_samples=db_min_samples, metric="euclidean", n_jobs=-1)
    db_labels = dbscan.fit_predict(X)

    result = pd.DataFrame(index=matrix.index)
    result["if_score"] = if_score
    result["if_outlier"] = if_label == -1
    result["lof_score"] = lof_score
    result["dbscan_cluster"] = db_labels
    result["dbscan_noise"] = db_labels == -1

    result["if_rank"] = result["if_score"].rank(ascending=False, method="first").astype(int)
    result["lof_rank"] = result["lof_score"].rank(ascending=False, method="first").astype(int)

    sets = {}
    for pct in (0.01, 0.02, 0.05):
        tag = f"top{int(pct * 100)}"
        sets[f"if_{tag}"] = _top_set(result["if_score"], pct)
        sets[f"lof_{tag}"] = _top_set(result["lof_score"], pct)
    sets["dbscan_noise"] = set(result.index[result["dbscan_noise"]])

    for model_name, ids in sets.items():
        result[model_name] = result.index.isin(ids)

    result["consensus_top5"] = result.index.isin(
        sets["if_top5"] & sets["lof_top5"] & sets["dbscan_noise"]
    )
    result["models_top5_count"] = (
        result["if_top5"].astype(int)
        + result["lof_top5"].astype(int)
        + result["dbscan_noise"].astype(int)
    )

    summary_rows = []
    for pct in (1, 2, 5):
        if_set = sets[f"if_top{pct}"]
        lof_set = sets[f"lof_top{pct}"]
        noise_set = sets["dbscan_noise"]
        summary_rows.append({
            "top_percent": pct,
            "n_if": len(if_set),
            "n_lof": len(lof_set),
            "n_dbscan_noise": len(noise_set),
            "if_lof_intersection": len(if_set & lof_set),
            "if_lof_jaccard": _jaccard(if_set, lof_set),
            "if_dbscan_intersection": len(if_set & noise_set),
            "lof_dbscan_intersection": len(lof_set & noise_set),
            "all_three_intersection": len(if_set & lof_set & noise_set),
        })
    summary = pd.DataFrame(summary_rows)
    model_info = {
        "n_active_profiles": int(n),
        "iforest": {"n_estimators": 300, "max_samples": min(256, n), "contamination": "auto", "random_state": seed},
        "lof": {"n_neighbors": n_neighbors, "score": "-negative_outlier_factor_"},
        "dbscan": {"eps": eps, "eps_source": eps_source, "min_samples": db_min_samples},
        "dbscan_noise_count": int(result["dbscan_noise"].sum()),
        "dbscan_noise_share": float(result["dbscan_noise"].mean()),
        "consensus_top5_count": int(result["consensus_top5"].sum()),
    }
    return result, summary, model_info, sets


def describe_profile(row: pd.Series) -> str:
    def mb(value):
        return f"{float(value) / 1e6:,.1f} МБ".replace(",", " ")

    as_name = str(row.get("as_name", "Unknown"))
    app = str(row.get("main_app", "Unknown"))
    parts = [
        f"AS: {as_name}",
        f"основное приложение: {app}",
        f"{int(row.get('flows', 0))} потоков / {int(row.get('sessions', 0))} сессий",
        f"отправлено {mb(row.get('bytes_out', 0))} / получено {mb(row.get('bytes_in', 0))}",
        f"upload ratio: {float(row.get('upload_ratio', 0)):.3f}",
    ]
    if float(row.get("n_rports", 0)) > 0:
        parts.append(f"удалённых TCP/UDP-портов: {int(row['n_rports'])}")
    if float(row.get("share_anon", 0)) > 0:
        parts.append(f"доля VPN/Tor/прокси: {float(row['share_anon']):.1%}")
    if float(row.get("share_p2p", 0)) > 0:
        parts.append(f"доля P2P: {float(row['share_p2p']):.1%}")
    return "; ".join(parts)


def save_visualisations(
    profiles: pd.DataFrame,
    matrix: pd.DataFrame,
    results: pd.DataFrame,
    raw_features: pd.DataFrame,
    output_dir: Path,
    top_n: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    noise = results["dbscan_noise"]
    ax.scatter(
        results.loc[~noise, "if_score"], results.loc[~noise, "lof_score"],
        s=12, alpha=0.45, color=PALETTE["gray"], label="DBSCAN: кластер",
    )
    ax.scatter(
        results.loc[noise, "if_score"], results.loc[noise, "lof_score"],
        s=16, alpha=0.55, color=PALETTE["orange"], label="DBSCAN: шум",
    )
    consensus = results["consensus_top5"]
    if consensus.any():
        ax.scatter(
            results.loc[consensus, "if_score"], results.loc[consensus, "lof_score"],
            s=65, facecolors="none", edgecolors=PALETTE["red"], linewidths=1.4,
            label="Консенсус трёх моделей", zorder=4,
        )
    ax.set_xlabel("Isolation Forest: anomaly score (больше — необычнее)")
    ax.set_ylabel("LOF: anomaly score (больше — локальнее выброс)")
    ax.set_title("Сопоставление оценок Isolation Forest и LOF")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "01_model_agreement.png", dpi=160)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix.to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    labels = results["dbscan_cluster"].to_numpy()
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        mask = labels == label
        color = PALETTE["gray"] if label == -1 else None
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=10, alpha=0.55,
            color=color, label="Шум DBSCAN" if label == -1 else f"Кластер {label}",
        )
    if consensus.any():
        mask = consensus.to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], s=55, facecolors="none",
                   edgecolors=PALETTE["red"], linewidths=1.3, label="Консенсус")
    variance = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({variance[0]:.1%} дисперсии)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%} дисперсии)")
    ax.set_title("PCA-проекция профилей и кластеризация DBSCAN")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "02_dbscan_pca.png", dpi=160)
    plt.close(fig)

    if consensus.any():
        active_raw = raw_features.loc[matrix.index, CORE_FEATURES]
        percentiles = active_raw.rank(pct=True)
        median_percentiles = percentiles.loc[consensus].median().sort_values()
        fig, ax = plt.subplots(figsize=(10, 6.5))
        vals = median_percentiles.values * 100
        ax.barh(median_percentiles.index, vals, color=PALETTE["blue"], height=0.68)
        ax.axvline(50, color=PALETTE["dark"], lw=1, ls="--", alpha=0.7)
        ax.axvline(90, color=PALETTE["orange"], lw=1, ls=":", alpha=0.8)
        ax.axvline(10, color=PALETTE["orange"], lw=1, ls=":", alpha=0.8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Медианный процентиль признака внутри активной популяции")
        ax.set_title("Положение консенсусных IP-профилей по основным признакам")
        fig.tight_layout()
        fig.savefig(output_dir / "03_consensus_feature_percentiles.png", dpi=160)
        plt.close(fig)

    top_if = results.nlargest(top_n, "if_score")["if_score"].sort_values()
    top_lof = results.nlargest(top_n, "lof_score")["lof_score"].sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.5, top_n * 0.24)))
    axes[0].barh(top_if.index.astype(str), top_if.values, color=PALETTE["blue"])
    axes[0].set_title(f"Топ-{len(top_if)} Isolation Forest")
    axes[0].set_xlabel("Anomaly score")
    axes[1].barh(top_lof.index.astype(str), top_lof.values, color=PALETTE["purple"])
    axes[1].set_title(f"Топ-{len(top_lof)} LOF")
    axes[1].set_xlabel("Anomaly score")
    fig.tight_layout()
    fig.savefig(output_dir / "04_top_model_scores.png", dpi=160)
    plt.close(fig)


def run_analysis(args) -> None:
    input_path = Path(args.csv).expanduser()
    output_dir = Path(args.out).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_input(input_path)
    flows, prep_info = prepare_flows(raw)
    print(
        f"Потоков после нормализации: {len(flows):,}; "
        f"внешних IP: {flows['remote_ip'].nunique():,}"
    )
    print(
        f"Исходящие: {prep_info['outgoing_rows']:,}; "
        f"входящие: {prep_info['incoming_rows']:,}; "
        f"исключено строк: {prep_info['rows_removed_not_exactly_one_org_side']:,}"
    )

    profiles = build_profiles(flows)
    raw_features, full_matrix, group_weights, group_variances = make_core_matrix(
        profiles, min_flows=args.min_flows
    )
    active_index = profiles.index[profiles["flows"] >= args.min_flows]
    matrix = full_matrix.loc[active_index].copy()
    print(f"Построено IP-профилей: {len(profiles):,}")
    print(f"Активных профилей (flows >= {args.min_flows}): {len(matrix):,}")
    print("Используемые признаки:", ", ".join(CORE_FEATURES))
    print("Коэффициенты балансировки групп:", group_weights)

    model_results, model_summary, model_info, sets = run_models(
        matrix, seed=args.seed, eps=args.eps, min_samples=args.min_samples
    )
    profiles = profiles.join(model_results, how="left")
    profiles["description"] = profiles.apply(describe_profile, axis=1)

    if_top = set(model_results.index[model_results["if_top5"]])
    lof_top = set(model_results.index[model_results["lof_top5"]])
    noise = set(model_results.index[model_results["dbscan_noise"]])
    candidate_groups = {
        "consensus_all_three": if_top & lof_top & noise,
        "if_specific_noise": (if_top & noise) - lof_top,
        "lof_specific_noise": (lof_top & noise) - if_top,
        "if_top_clustered": if_top - noise,
        "lof_top_clustered": (lof_top - noise) - if_top,
    }
    candidate_to_group = {
        ip: group for group, ids in candidate_groups.items() for ip in ids
    }
    candidates = profiles.loc[sorted(candidate_to_group)].copy()
    candidates["candidate_group"] = [candidate_to_group[ip] for ip in candidates.index]
    candidates["evidence"] = candidates["candidate_group"].map({
        "consensus_all_three": "top-5% IF + top-5% LOF + шум DBSCAN",
        "if_specific_noise": "top-5% IF + шум DBSCAN, вне top-5% LOF",
        "lof_specific_noise": "top-5% LOF + шум DBSCAN, вне top-5% IF",
        "if_top_clustered": "top-5% IF, кластер DBSCAN",
        "lof_top_clustered": "top-5% LOF, кластер DBSCAN, вне top-5% IF",
    })
    candidates = candidates.sort_values(
        ["candidate_group", "if_score", "lof_score"],
        ascending=[True, False, False],
    )
    consensus = profiles.loc[profiles["consensus_top5"].fillna(False)].copy()
    consensus = consensus.sort_values(["if_score", "lof_score"], ascending=False)

    model_summary.to_csv(output_dir / "model_agreement.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(output_dir / "ip_profiles.csv", encoding="utf-8-sig")
    candidates.to_csv(output_dir / "anomaly_candidates.csv", encoding="utf-8-sig")
    consensus.to_csv(output_dir / "consensus_ips.csv", encoding="utf-8-sig")
    matrix.to_csv(output_dir / "active_core_matrix.csv", encoding="utf-8-sig")

    feature_description = pd.DataFrame({
        "feature": CORE_FEATURES,
        "group": [next(k for k, v in FEATURE_GROUPS.items() if c in v) for c in CORE_FEATURES],
    })
    feature_description.to_csv(output_dir / "core_features.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_file": str(input_path),
        "preprocessing": prep_info,
        "n_ip_profiles": int(len(profiles)),
        "min_flows": int(args.min_flows),
        "n_active_profiles": int(len(matrix)),
        "core_features": CORE_FEATURES,
        "feature_groups": FEATURE_GROUPS,
        "group_variance_sums_before_balance": group_variances,
        "group_balance_weights": group_weights,
        "models": model_info,
        "agreement": model_summary.to_dict(orient="records"),
        "candidate_count": int(len(candidates)),
        "consensus_count": int(len(consensus)),
        "interpretation": (
            "Кандидаты являются статистически нетипичными профилями и требуют проверки; "
            "без размеченных данных их нельзя считать подтверждёнными атаками."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    save_visualisations(
        profiles=profiles,
        matrix=matrix,
        results=model_results,
        raw_features=raw_features,
        output_dir=output_dir,
        top_n=args.top,
    )

    print("\nСогласованность моделей (top-1/2/5%):")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(model_summary.to_string(index=False))
    print(f"\nDBSCAN noise: {model_info['dbscan_noise_count']:,} ({model_info['dbscan_noise_share']:.2%})")
    print(f"Консенсус трёх моделей в top-5%: {len(consensus):,} IP")
    print(f"Кандидатов для ручного просмотра: {len(candidates):,} IP")

    show_cols = [
        "if_rank", "lof_rank", "dbscan_cluster", "models_top5_count",
        "consensus_top5", "as_name", "main_app", "flows", "upload_ratio", "evidence",
    ]
    print(f"\nПервые {args.top} кандидатов:")
    with pd.option_context("display.max_colwidth", 70, "display.width", 240):
        print(candidates[[c for c in show_cols if c in candidates.columns]].head(args.top).to_string())

    print(f"\nРезультаты сохранены в: {output_dir.resolve()}")
    print("Основные файлы:")
    print("  ip_profiles.csv          — профили всех внешних IP")
    print("  anomaly_candidates.csv   — кандидаты по результатам моделей")
    print("  consensus_ips.csv        — пересечение трёх подходов")
    print("  model_agreement.csv      — согласованность моделей")
    print("  summary.json             — параметры и сводка анализа")
    print("  *.png                    — графики")


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Поиск нетипичных IP-профилей с помощью IF, LOF и DBSCAN."
    )
    parser.add_argument("csv", help="Путь к исходному DPI/NetFlow CSV")
    parser.add_argument("--out", default="output", help="Каталог для результатов (по умолчанию output)")
    parser.add_argument("--min-flows", type=int, default=20, help="Минимум потоков для моделирования IP (по умолчанию 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed для Isolation Forest")
    parser.add_argument("--eps", type=float, default=1.373, help="eps для DBSCAN (экспериментальная базовая конфигурация: 1.373)")
    parser.add_argument("--min-samples", type=int, default=20, help="min_samples для DBSCAN")
    parser.add_argument("--top", type=int, default=20, help="Число строк в консольном топе и графике")
    return parser.parse_args(argv)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    try:
        run_analysis(args)
    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
