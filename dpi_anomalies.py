import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ORG = "org_ip"

C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
}
GRAY = "#b9b8b2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "legend.frameon": False,
})

RULES = {
    "Утечка данных (exfiltration)": (5, C["red"],
        "Организация выгрузила на внешний адрес ≥200 МБ; исходящий трафик многократно превышает входящий"),
    "Сигнатура ботнета (DPI)": (5, C["red"],
        "DPI классифицировал взаимодействие как трафик ботнета"),
    "Адрес из блок-листа DPI": (4, C["orange"],
        "Трафик попал под пользовательские списки DPI (cp_spisok / cp_ipblock) — адрес в чёрном списке"),
    "Сканирование портов организации": (4, C["orange"],
        "С одного исходного порта перебираются сотни/тысячи портов организации короткими пакетами без полезной нагрузки"),
    "Маячок / высокочастотный канал (beaconing)": (4, C["violet"],
        "Тысячи коротких однотипных сессий к одному адресу — строго периодичных (beaconing) либо высокочастотных (C2 / туннель)"),
    "Односторонний трафик без ответов": (3, C["violet"],
        "Организация шлёт запросы, а ответов почти нет — блокировка, мёртвый C2-сервер или туннель"),
    "Крупная загрузка по нестандартному порту": (3, C["yellow"],
        "Гигабайты по порту вне 80/443 на хостинг — возможный обход политики или туннель"),
    "Доминирующий канал на нестандартный порт": (3, C["violet"],
        "Один адрес даёт ≥1% всех потоков организации, порт нестандартный — постоянный служебный или скрытый канал"),
    "Анонимайзер (Tor/VPN/прокси)": (3, C["magenta"],
        "Большая часть взаимодействия — Tor, Shadowsocks, коммерческие VPN/прокси: обход контроля трафика"),
    "ICMP-зондирование": (2, C["aqua"],
        "Серии входящих ICMP-пакетов пачками — ping-sweep или мониторинг доступности извне"),
    "Интенсивный P2P (BitTorrent)": (1, C["blue"],
        "Объёмный BitTorrent-обмен — нарушение политики, риск утечек и вредоносного ПО"),
}
STD_PORTS = {80, 443, 53, 123, 853, 993, 995, 465, 587, 25, 110, 143, 5222, 5228, 3478}
ANON_RE = r"tor(?!rent)|shadowsocks|vpn|psiphon|proxy|lantern|warp"
BLOCK_RE = r"cp_spisok|cp_ipblock"
CLOUD_AS_RE = re.compile(
    r"yandex|mail\.ru|google|amazon|microsoft|azure|cloud|digitalocean|hetzner|"
    r"ovh|vdsina|selectel|alexhost|onlanta|scaleway|qwarta|beget|firstcolo", re.I
)


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df["out"] = df["source_ipv4"].eq(ORG)
    df["remote"] = np.where(df.out, df.destination_ipv4, df.source_ipv4)
    df["rport"] = np.where(df.out, df.destination_port, df.source_port)
    df["lport"] = np.where(df.out, df.source_port, df.destination_port)
    df["ts"] = pd.to_datetime(df.flow_start_time)
    df["hour"] = df.ts.dt.hour
    df["bytes"] = df.octet_delta_count

    if "source_astitle" in df.columns:
        df["remote_as"] = np.where(df.out, df.destination_astitle, df.source_astitle)
    else:
        org_mode = df.loc[df.destination_ipv4 == ORG, "destination_astitle"].mode()
        org_as = org_mode.iloc[0] if len(org_mode) else None
        df["remote_as"] = df.destination_astitle
        if org_as is not None:
            df.loc[(~df.out) & df.destination_astitle.eq(org_as), "remote_as"] = None

    proto = df.app_protocol.fillna("").astype(str).str.lower()
    df["is_anon"] = df.app_protocol_group.eq("VPN") | proto.str.contains(ANON_RE, regex=True)
    df["is_tor"] = proto.str.match(r"^tor($|[^a-z])", case=False)
    df["is_p2p"] = df.app_protocol_group.eq("P2P")
    df["is_bot"] = df.app_protocol_group.eq("Botnet")

    df["is_block"] = proto.str.contains(BLOCK_RE, regex=True)
    if df["is_block"].sum() == 0:
        df["is_block"] = df.app_protocol_group.eq("Custom protocols")

    df["is_icmp"] = df.net_protocol.fillna("").astype(str).str.startswith("ICMP")
    return df[df.remote != ORG]


def features(df):
    g = df.groupby("remote")
    out, inn = df[df.out], df[~df.out]
    f = pd.DataFrame({
        "as_name": g.remote_as.first(),
        "main_app": g.app_protocol.agg(lambda s: s.value_counts().index[0] if len(s.value_counts()) else ""),
        "main_rport": g.rport.agg(lambda s: s.value_counts().index[0] if len(s.value_counts()) else -1),
        "flows": g.size(),
        "sessions": g.session_id.nunique(),
        "flows_out": out.groupby("remote").size(),
        "flows_in": inn.groupby("remote").size(),
        "bytes_out": out.groupby("remote").bytes.sum(),
        "bytes_in": inn.groupby("remote").bytes.sum(),
        "packets": g.packet_delta_count.sum(),
        "n_rports": g.rport.nunique(),
        "n_lports": g.lport.nunique(),
        "share_1pkt": g.packet_delta_count.agg(lambda s: (s <= 1).mean()),
        "n_hours": g.hour.nunique(),
        "share_night": g.hour.agg(lambda s: ((s < 8) | (s >= 20)).mean()),
        "retrans": g.repeat_delta_pack.sum(),
        "share_anon": g.is_anon.mean(),
        "share_tor": g.is_tor.mean(),
        "share_p2p": g.is_p2p.mean(),
        "bot": g.is_bot.sum(),
        "block": g.is_block.sum(),
        "icmp": g.is_icmp.sum(),
    }).fillna(0)

    f["flows_out"] = f["flows_out"].fillna(0)
    f["flows_in"] = f["flows_in"].fillna(0)
    f["bytes_out"] = f["bytes_out"].fillna(0)
    f["bytes_in"] = f["bytes_in"].fillna(0)

    f["bytes_total"] = f.bytes_out + f.bytes_in
    f["upload_ratio"] = f.bytes_out / f.bytes_total.replace(0, np.nan)
    f["bytes_per_session"] = f.bytes_total / f.sessions.replace(0, np.nan)
    f["flows_per_session"] = f.flows / f.sessions.replace(0, np.nan)
    f["pkt_size"] = f.bytes_total / f.packets.replace(0, np.nan)
    f["out_in_flow_ratio"] = (f.flows_out + 1) / (f.flows_in + 1)
    f["retrans_rate"] = f.retrans / f.packets.replace(0, np.nan)

    hourly = out.groupby(["remote", "hour"]).bytes.sum().groupby("remote").max()
    f["max_bytes_out_hour"] = hourly.reindex(f.index).fillna(0)

    st = (df[df.remote.isin(f.index[f.sessions >= 20])]
          .groupby(["remote", "session_id"]).flow_start_millisecond.min().reset_index()
          .sort_values(["remote", "flow_start_millisecond"]))
    st["dt"] = st.groupby("remote").flow_start_millisecond.diff() / 1000
    reg = st.groupby("remote").dt.agg(["median", "mean", "std"])
    f["iat_median_s"] = reg["median"]
    f["iat_cv"] = reg["std"] / reg["mean"]

    return f.fillna({"upload_ratio": 0, "pkt_size": 0, "retrans_rate": 0})


def apply_rules(f):
    MB, GB = 1e6, 1e9
    R = {
        "Утечка данных (exfiltration)":
            (f.bytes_out >= 200 * MB) & (f.upload_ratio >= 0.85) & (f.share_p2p < 0.5),
        "Сигнатура ботнета (DPI)": f.bot > 0,
        "Адрес из блок-листа DPI": f.block >= 3,
        "Сканирование портов организации":
            (f.n_lports >= 200) & (f.n_rports <= 3) & (f.share_1pkt >= 0.3)
            & (f.pkt_size < 120) & ~f.main_rport.isin(STD_PORTS),
        "Маячок / высокочастотный канал (beaconing)":
            (f.sessions >= 5000) & (f.bytes_per_session < 1000)
            & ((f.iat_cv.fillna(1.0) < 0.6) | ((f.n_lports >= 500) & (f.share_1pkt < 0.1))),
        "Односторонний трафик без ответов": (f.flows >= 500) & (f.out_in_flow_ratio >= 10),
        "Крупная загрузка по нестандартному порту":
            (f.bytes_total >= 1 * GB) & ~f.main_rport.isin(STD_PORTS) & (f.share_p2p < 0.5),
        "Доминирующий канал на нестандартный порт":
            (f.flows >= 0.01 * f.flows.sum()) & ~f.main_rport.isin(STD_PORTS)
            & (f.share_p2p < 0.5) & (f.share_1pkt < 0.3),
        "Анонимайзер (Tor/VPN/прокси)": (f.share_anon >= 0.5) & (f.flows >= 300),
        "ICMP-зондирование": (f.icmp >= 100) & (f.flows_out == 0),
        "Интенсивный P2P (BitTorrent)": (f.share_p2p >= 0.5) & (f.bytes_total >= 1 * GB),
    }
    flags = pd.DataFrame(R)
    f["rule_score"] = sum(flags[k] * RULES[k][0] for k in RULES)
    f["anomaly_types"] = flags.apply(lambda r: "; ".join(k for k in RULES if r[k]), axis=1)
    f["primary_type"] = flags.apply(
        lambda r: max((k for k in RULES if r[k]), key=lambda k: RULES[k][0], default=""), axis=1)
    return f, flags


def detect_twins(f, min_sessions=500, min_flows=50, tol=0.02):
    d = f[(f.sessions >= min_sessions) & (f.flows >= min_flows)]
    if len(d) < 2:
        return pd.Series(dtype="Int64")
    rows = list(d.iterrows())
    group_of = {}
    next_gid = 0
    for i in range(len(rows)):
        idx_a, a = rows[i]
        for j in range(i + 1, len(rows)):
            idx_b, b = rows[j]
            if a.main_rport != b.main_rport:
                continue
            if not (pd.notna(a.as_name) and pd.notna(b.as_name) and a.as_name == b.as_name):
                continue
            if a.sessions == 0 or b.sessions == 0:
                continue
            ms = max(a.sessions, b.sessions)
            if abs(a.sessions - b.sessions) / ms > tol:
                continue
            if a.bytes_out == 0 or b.bytes_out == 0:
                continue
            mb = max(a.bytes_out, b.bytes_out)
            if abs(a.bytes_out - b.bytes_out) / mb > tol:
                continue
            ga = group_of.get(idx_a)
            gb = group_of.get(idx_b)
            if ga is None and gb is None:
                next_gid += 1
                g = next_gid
            elif ga is None:
                g = gb
            elif gb is None:
                g = ga
            else:
                g = min(ga, gb)
            group_of[idx_a] = g
            group_of[idx_b] = g
    return pd.Series(group_of, dtype="Int64")


def isolation_forest(f, min_flows=20, seed=42):
    cols = ["flows", "sessions", "bytes_out", "bytes_in", "n_rports", "n_lports", "share_1pkt",
            "n_hours", "share_night", "upload_ratio", "bytes_per_session", "pkt_size",
            "out_in_flow_ratio", "retrans_rate", "share_anon", "share_p2p",
            "max_bytes_out_hour", "iat_cv"]
    X = f.loc[f.flows >= min_flows, cols].copy()
    X["iat_cv"] = X["iat_cv"].fillna(1.0)
    for c in ["flows", "sessions", "bytes_out", "bytes_in", "n_rports", "n_lports",
              "bytes_per_session", "pkt_size", "out_in_flow_ratio", "max_bytes_out_hour"]:
        X[c] = np.log1p(X[c])
    Xs = StandardScaler().fit_transform(X.fillna(0))
    iso = IsolationForest(n_estimators=400, contamination=0.01, random_state=seed).fit(Xs)
    s = -iso.score_samples(Xs)
    f["if_score"] = 0.0
    f.loc[X.index, "if_score"] = (s - s.min()) / (s.max() - s.min())
    f["if_outlier"] = False
    f.loc[X.index, "if_outlier"] = iso.predict(Xs) == -1
    return f


def _hbytes(x):
    if pd.isna(x):
        return "0 Б"
    if x >= 1e9:
        return f"{x / 1e9:.2f} ГБ"
    if x >= 1e6:
        return f"{x / 1e6:.1f} МБ"
    if x >= 1e3:
        return f"{x / 1e3:.1f} КБ"
    return f"{int(x)} Б"


def describe(r):
    as_display = r.as_name if pd.notna(r.as_name) and str(r.as_name).strip() else "AS неизвестен (входящий поток)"
    port = int(r.main_rport) if pd.notna(r.main_rport) else -1
    parts = [
        f"AS: {as_display}",
        f"осн. протокол: {r.main_app} (порт {port})",
        f"{int(r.sessions):,} сессий".replace(",", " "),
        f"↑{_hbytes(r.bytes_out)} / ↓{_hbytes(r.bytes_in)}",
    ]
    is_icmp_proto = str(r.main_app).lower().startswith("icmp")
    if not is_icmp_proto:
        if r.n_lports >= 200:
            parts.append(f"{int(r.n_lports)} портов организации")
        if r.n_rports >= 200:
            parts.append(f"{int(r.n_rports)} удалённых портов")
    if r.share_anon >= 0.5:
        parts.append(f"анонимайзеры {r.share_anon:.0%}")
    if r.share_night >= 0.5:
        parts.append(f"вечер/ночь {r.share_night:.0%}")
    if pd.notna(r.iat_cv) and r.iat_cv < 0.5 and pd.notna(r.iat_median_s) and r.iat_median_s > 0:
        parts.append(f"регулярные интервалы ~{r.iat_median_s:.0f} с (CV={r.iat_cv:.2f})")
    if getattr(r, "is_twin", False):
        parts.append("группа «близнецов» — идентичный профиль у нескольких IP")
    if r.sessions <= 2 and r.bytes_total > 1e9:
        parts.append(f"одна сессия объёмом {_hbytes(r.bytes_total)} — возможна выкачка БД или архива")
    if pd.notna(r.flows_per_session) and r.flows_per_session > 100:
        parts.append(f"подозрительно много потоков на сессию ({r.flows_per_session:.0f})")
    if isinstance(as_display, str) and CLOUD_AS_RE.search(as_display):
        parts.append("облачный провайдер — возможна легальная синхронизация")
    return "; ".join(parts)


def annotate_grouped(ax, d, x, y):
    key = pd.DataFrame({"kx": np.round(np.log10(x.astype(float) + 1), 1),
                        "ky": np.round(np.log10(y.astype(float) + 1), 1),
                        "x": x, "y": y}, index=d.index)
    for _, grp in key.groupby(["kx", "ky"]):
        ax.annotate(" / ".join(grp.index), (grp.x.iloc[0], grp.y.iloc[0]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK)


def fig_hourly(df, out):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    hours = np.arange(df.hour.min(), 24)
    h = df.groupby(["hour", "out"]).bytes.sum().unstack(fill_value=0) / 1e9
    h = h.reindex(hours, fill_value=0)
    inflow = h[False] if False in h.columns else pd.Series(0.0, index=hours)
    outflow = h[True] if True in h.columns else pd.Series(0.0, index=hours)
    ax.bar(hours - 0.21, inflow, 0.4, color=C["blue"], label="Входящий (↓)")
    ax.bar(hours + 0.21, outflow, 0.4, color=C["orange"], label="Исходящий (↑)")
    ax.set_xticks(hours)
    ax.set_xlabel("Час суток")
    ax.set_ylabel("ГБ")
    ax.set_title("Профиль трафика организации по часам")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out}/01_hourly_profile.png", dpi=150)
    plt.close(fig)


def fig_scatter_volume(f, out, top):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    base = f[f.bytes_total > 0]
    ax.scatter(base.bytes_in + 1, base.bytes_out + 1, s=8, color=GRAY, alpha=0.45, lw=0, label="Остальные IP")
    for t, (w, col, _) in RULES.items():
        d = top[top.primary_type == t]
        if len(d):
            ax.scatter(d.bytes_in + 1, d.bytes_out + 1, s=70, color=col,
                       edgecolor="#fcfcfb", lw=1.5, label=t, zorder=3)
    lab = top[top.primary_type != "Анонимайзер (Tor/VPN/прокси)"].head(20)
    annotate_grouped(ax, lab, lab.bytes_in + 1, lab.bytes_out + 1)
    lim = [1, base.bytes_total.max() * 3]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.text(2e9, 4e9, "↑ = ↓", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Получено организацией, байт (log)")
    ax.set_ylabel("Отправлено организацией, байт (log)")
    ax.set_title("Объём и асимметрия трафика по внешним IP")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/02_volume_asymmetry.png", dpi=150)
    plt.close(fig)


def fig_scatter_ports(f, out, top):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(f.n_rports, f.n_lports, s=np.clip(np.log1p(f.sessions) * 4, 4, 60),
               color=GRAY, alpha=0.45, lw=0, label="Остальные IP")
    for t, (w, col, _) in RULES.items():
        d = top[top.primary_type == t]
        if len(d):
            ax.scatter(d.n_rports, d.n_lports, s=70, color=col, edgecolor="#fcfcfb",
                       lw=1.5, label=t, zorder=3)
    lab = top[((top.n_lports > 1000) | (top.n_rports > 100))
              & (top.primary_type != "Анонимайзер (Tor/VPN/прокси)")]
    annotate_grouped(ax, lab, lab.n_rports, lab.n_lports)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Уникальных портов удалённой стороны")
    ax.set_ylabel("Уникальных портов организации")
    ax.set_title("Разнообразие портов: сканирование и туннели")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/03_ports.png", dpi=150)
    plt.close(fig)


def fig_heatmap(df, out, top):
    ips = top.head(20).index
    hours = list(range(df.hour.min(), 24))
    m = (df[df.remote.isin(ips)].groupby(["remote", "hour"]).size()
         .unstack(fill_value=0).reindex(index=ips, columns=hours, fill_value=0))
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = LinearSegmentedColormap.from_list("seq", ["#f1f6fd", "#9cc3ef", C["blue"], "#0d3a73"])
    im = ax.imshow(np.log1p(m.values), aspect="auto", cmap=cmap)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(m.columns)
    ax.set_yticks(range(len(ips)))
    ax.set_yticklabels([f"{ip}  ·  {top.loc[ip, 'primary_type'] or 'IF-выброс'}" for ip in ips], fontsize=8)
    ax.set_xlabel("Час суток")
    ax.grid(False)
    ax.set_title("Активность топ-20 аномальных IP по часам (число потоков, log)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label("log(1 + потоков)")
    fig.tight_layout()
    fig.savefig(f"{out}/04_heatmap_top.png", dpi=150)
    plt.close(fig)


def fig_score(top, out):
    d = top.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 9.5))
    cols = [RULES[t][1] if t else GRAY for t in d.primary_type]
    ax.barh(range(len(d)), d.total_score, color=cols, height=0.7)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{ip} ({str(a)[:28]})" for ip, a in zip(d.index, d.as_name)], fontsize=8)
    for i, (s, t) in enumerate(zip(d.total_score, d.anomaly_types)):
        txt = "\n".join(t.split("; ")) if t else "статистический выброс (Isolation Forest)"
        ax.text(s + 0.15, i, txt, va="center", fontsize=7, color=INK2, linespacing=0.95)
    ax.set_xlim(0, d.total_score.max() * 1.75)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Итоговый балл = сумма весов правил + 3 × скор Isolation Forest (0–1)")
    ax.set_title("Рейтинг аномальных IP-адресов")
    fig.tight_layout()
    fig.savefig(f"{out}/05_ranking.png", dpi=150)
    plt.close(fig)


def fig_anon(df, out):
    d = df[df.is_anon | df.is_p2p].copy()
    d["kind"] = np.select(
        [d.is_tor, d.is_p2p,
         d.app_protocol.fillna("").astype(str).str.contains("shadowsocks", case=False, regex=False)],
        ["Tor", "BitTorrent", "Shadowsocks"],
        "Другие VPN/прокси",
    )
    t = d.groupby([d.ts.dt.floor("15min"), "kind"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k, col in zip(["BitTorrent", "Shadowsocks", "Tor", "Другие VPN/прокси"],
                      [C["blue"], C["orange"], C["violet"], C["magenta"]]):
        if k in t:
            ax.plot(t.index, t[k], color=col, lw=2, label=k)
    ax.set_ylabel("Потоков за 15 мин")
    ax.legend(ncol=4, loc="upper right")
    ax.set_title("Средства обхода контроля и P2P во времени")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{out}/06_anonymizers_p2p.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="output")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load(a.csv)
    print(f"Записей: {len(df):,}, внешних IP: {df.remote.nunique():,}, "
          f"период {df.ts.min()} — {df.ts.max()}")

    f = features(df)
    twin_groups = detect_twins(f)
    f["is_twin"] = f.index.isin(twin_groups.index)
    f["twin_group"] = twin_groups.reindex(f.index)

    f, flags = apply_rules(f)
    f = isolation_forest(f)
    f["total_score"] = f.rule_score + 3 * f.if_score

    top = f[(f.rule_score > 0) | f.if_outlier].sort_values("total_score", ascending=False)
    top["description"] = top.apply(describe, axis=1)

    keep = ["total_score", "primary_type", "anomaly_types", "description", "as_name",
            "main_app", "main_rport", "flows", "sessions", "bytes_out", "bytes_in",
            "upload_ratio", "n_lports", "n_rports", "share_1pkt", "share_anon",
            "share_night", "iat_median_s", "iat_cv", "flows_per_session",
            "is_twin", "twin_group", "if_score"]
    top[keep].to_csv(f"{a.out}/anomalous_ips.csv", encoding="utf-8-sig")

    pd.DataFrame([(k, v[0], v[2], int(flags[k].sum())) for k, v in RULES.items()],
                 columns=["тип", "вес", "описание", "IP"]).to_csv(
        f"{a.out}/anomaly_types.csv", index=False, encoding="utf-8-sig")

    print("\nСработавшие правила:")
    print(flags.sum().to_string())

    if len(twin_groups):
        print(f"\nГрупп «близнецов»: {twin_groups.nunique()}, IP: {len(twin_groups)}")
        with pd.option_context("display.width", 220, "display.max_rows", 60):
            tbl = f.loc[twin_groups.index, ["twin_group", "as_name", "main_rport",
                                            "sessions", "bytes_out", "share_anon"]]
            print(tbl.sort_values(["twin_group", "sessions"]).to_string())

    print(f"\nТоп-{a.top} аномальных IP:")
    with pd.option_context("display.max_colwidth", 140, "display.width", 260):
        print(top[["total_score", "anomaly_types", "description"]].head(a.top).to_string())

    fig_hourly(df, a.out)
    fig_scatter_volume(f, a.out, top.head(a.top))
    fig_scatter_ports(f, a.out, top.head(a.top))
    fig_heatmap(df, a.out, top)
    fig_score(top, a.out)
    fig_anon(df, a.out)
    print(f"\nГрафики и таблицы сохранены в {a.out}/")


if __name__ == "__main__":
    main()