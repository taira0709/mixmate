from __future__ import annotations
import io, os, json, hashlib, tempfile, time
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, send_file, abort, jsonify

# DSP
from pedalboard import (
    Pedalboard,
    NoiseGate, HighpassFilter, LowpassFilter,
    PeakFilter, HighShelfFilter, LowShelfFilter,
    Compressor, Delay, Reverb, Limiter,
    Distortion  # サチュレーション用
)
import pyloudnorm as pyln

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configure for large file uploads and memory optimization
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit (reduced for stability)
app.config['UPLOAD_TIMEOUT'] = 600  # 10 minutes timeout
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300  # Cache timeout

# Error handler for large files
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large',
        'message': f'Maximum file size is {app.config["MAX_CONTENT_LENGTH"] / 1024 / 1024:.0f}MB'
    }), 413

# ======================================================
# Utils
# ======================================================
def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.concatenate([audio, audio], axis=1)
    return audio.astype(np.float32, copy=False)

def to_mono_mid(x: np.ndarray) -> np.ndarray:
    """
    等電力サム（Mid抽出）でモノ化。shape: (N,C)->(N,1)
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] == 1:
        return x
    m = (x[:, 0] + x[:, 1]) / np.sqrt(2.0)
    return m.reshape(-1, 1)

def peak_db(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x)) + 1e-12)
    return 20 * np.log10(p)

def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)

def gr_estimate_db(before: np.ndarray, after: np.ndarray) -> float:
    gr = max(0.0, peak_db(before) - peak_db(after))
    return float(np.clip(gr, 0.0, 24.0))

def _eq_nodes_summary(nodes: list) -> str:
    """EQノードリストを簡易文字列化してログに出す"""
    parts = []
    for n in nodes:
        cname = n.__class__.__name__
        # pedalboardの主要EQノードの代表的プロパティを拾う
        if hasattr(n, "cutoff_frequency_hz"):
            parts.append(f"{cname}({getattr(n, 'cutoff_frequency_hz', 0.0):.0f}Hz)")
        elif hasattr(n, "center_frequency_hz"):
            cf = getattr(n, "center_frequency_hz", 0.0)
            gain = getattr(n, "gain_db", 0.0)
            q = getattr(n, "q", 1.0)
            parts.append(f"{cname}({cf:.0f}Hz, {gain:+.1f}dB, Q={q:.1f})")
        elif hasattr(n, "frequency_hz"):
            f = getattr(n, "frequency_hz", 0.0)
            gain = getattr(n, "gain_db", 0.0)
            parts.append(f"{cname}({f:.0f}Hz, {gain:+.1f}dB)")
        else:
            parts.append(cname)
    return " / ".join(parts) if parts else "(none)"

# ======================================================
# Stages
# ======================================================
def make_noise_gate(mode: str, gender: str = 'male') -> NoiseGate:
    mode_clean = (mode or '').lower()
    g = (gender or 'male').lower()
    
    # スマートフォン録音の場合は自動的に強いノイズゲートを適用
    if g == 'smartphone':
        ng = NoiseGate(threshold_db=-42.0, ratio=5.0, attack_ms=0.8, release_ms=180.0)
        print(f"[Noise Gate] Created SMARTPHONE gate: th=-42.0dB, ratio=5.0:1, atk=0.8ms, rel=180.0ms")
        return ng
    elif mode_clean == 'strong':
        ng = NoiseGate(threshold_db=-45.0, ratio=4.0, attack_ms=1.0, release_ms=150.0)
        print(f"[Noise Gate] Created STRONG gate: th=-45.0dB, ratio=4.0:1, atk=1.0ms, rel=150.0ms")
        return ng
    else:
        ng = NoiseGate(threshold_db=-50.0, ratio=2.0, attack_ms=2.0, release_ms=120.0)
        print(f"[Noise Gate] Created WEAK gate: th=-50.0dB, ratio=2.0:1, atk=2.0ms, rel=120.0ms")
        return ng

def eq_nodes_by_gender(gender: str) -> list:
    g = (gender or 'male').lower()
    if g == 'female':
        return [
            HighpassFilter(110.0),
            PeakFilter(275.0, -1.0, 1.0),
            PeakFilter(750.0, -0.5, 1.0),
            PeakFilter(4000.0, +1.0, 1.0),
            HighShelfFilter(12000.0, +1.5),
        ]
    elif g == 'smartphone':
        # スマホ録音想定の中庸カーブ（ロー不足＋箱鳴り／サ行対策）
        return [
            HighpassFilter(140.0),                  # 低域ノイズ/机鳴り対策
            LowShelfFilter(170.0, +1.0),            # 体感ローの薄さを軽く補う
            PeakFilter(300.0,  -2.0, 1.1),          # 箱鳴り
            PeakFilter(600.0,  -1.0, 1.2),          # こもり
            PeakFilter(1000.0, -0.8, 1.0),          # 鼻づまり
            PeakFilter(3500.0, -0.5, 1.0),          # 刺さりの下地を控えめに調整
            PeakFilter(7200.0, -1.5, 2.0),          # サ行のプレ・ディエッシング
            HighShelfFilter(11000.0, +0.5),         # 空気感を少しだけ
        ]
    else:
        return [
            HighpassFilter(100.0),
            PeakFilter(300.0, -1.5, 1.0),
            PeakFilter(800.0, -1.5, 1.0),
            PeakFilter(4000.0, +1.5, 1.0),
            HighShelfFilter(12000.0, +1.0),
        ]


# === EQログ用の簡易サマライザ ===
def _eq_nodes_summary(nodes: list) -> str:
    parts = []
    for n in nodes:
        name = n.__class__.__name__
        if name == "HighpassFilter":
            parts.append(f"HPF {getattr(n, 'cutoff_frequency_hz', 0):.0f}Hz")
        elif name == "LowpassFilter":
            parts.append(f"LPF {getattr(n, 'cutoff_frequency_hz', 0):.0f}Hz")
        elif name == "LowShelfFilter":
            parts.append(
                f"LowShelf {getattr(n, 'cutoff_frequency_hz', 0):.0f}Hz "
                f"{getattr(n, 'gain_db', 0.0):+0.1f}dB Q={getattr(n, 'q', 1.0):.2f}"
            )
        elif name == "HighShelfFilter":
            parts.append(
                f"HighShelf {getattr(n, 'cutoff_frequency_hz', 0):.0f}Hz "
                f"{getattr(n, 'gain_db', 0.0):+0.1f}dB Q={getattr(n, 'q', 1.0):.2f}"
            )
        elif name == "PeakFilter":
            parts.append(
                f"Peak {getattr(n, 'center_frequency_hz', getattr(n, 'cutoff_frequency_hz', 0)):.0f}Hz "
                f"{getattr(n, 'gain_db', 0.0):+0.1f}dB Q={getattr(n, 'q', 1.0):.2f}"
            )
        else:
            parts.append(name)
    return " / ".join(parts)

def style_eq_nodes(style: str) -> list:
    """スタイル別EQノードリストを返す"""
    s = (style or 'pop').lower()
    
    if s == 'rock':
        return [
            PeakFilter(400.0,  -3.0, 1.20),   # ローミッドカット
            PeakFilter(1000.0, +3.5, 1.80),   # ミッド強調
            PeakFilter(3000.0, +5.0, 2.00),   # アタック感
            PeakFilter(8000.0, +3.0, 1.20),   # ハイの鋭さ追加
            HighShelfFilter(13000.0, +4.5),   # 空気感と最上域ブースト
        ]
    elif s == 'pop':
        # 既存設定（適切）
        return [
            PeakFilter(250.0, -1.6, 1.1),   # 緩めに
            PeakFilter(3200.0, +1.5, 1.1),  # 抜け強化
            HighShelfFilter(10000.0, +1.2), # LPFやめてHSで空気感
        ]
    elif s == 'ballad':
        # 【修正】バラードらしい温かみのある設定に変更
        return [
            PeakFilter(150.0,  +0.8, 0.8),    # 軽い低域の厚み（過度でない）
            PeakFilter(400.0,  -0.5, 1.2),    # 箱鳴り軽減
            PeakFilter(1200.0, +0.8, 1.5),    # 温かみのある中域
            PeakFilter(3000.0, +1.2, 1.3),    # 適度な明瞭度
            HighShelfFilter(8000.0, +0.8),    # 優しい高域
        ]
    elif s == 'narration':
        # 【修正】より自然で聞きやすい設定に変更
        return [
            HighpassFilter(120.0),            # 不要低域カット
            PeakFilter(300.0, -1.2, 1.0),     # 箱鳴り軽減
            PeakFilter(1800.0, +1.2, 1.4),    # 音声明瞭度（控えめ）
            PeakFilter(3500.0, +1.0, 1.3),    # 音声認識向上（控えめ）
            PeakFilter(6500.0, -0.8, 1.5),    # 刺さり軽減
            HighShelfFilter(10000.0, +0.3),   # 軽い空気感
        ]
    elif s in ('chorus', 'chorus_wide'):
        # 既存設定（適切）
        return [
            PeakFilter(120.0, +0.5, 1.0),     # 軽い低域補強
            PeakFilter(800.0, -0.3, 1.2),     # 軽いマッド感軽減
            PeakFilter(4000.0, +0.8, 1.5),    # 明瞭度
            HighShelfFilter(10000.0, +0.3),   # エアー感
        ]
    else:
        # デフォルト（不明なスタイル）
        return []

# ======================================================
# 新機能：サチュレーション処理
# ======================================================
def apply_saturation(audio: np.ndarray, sr: int, style: str) -> np.ndarray:
    """スタイル別サチュレーション処理（音量統一版）"""
    s = (style or 'pop').lower()

    # ナレーションはサチュレーションなし
    if s == 'narration':
        return audio

    # スタイル別のドライブ設定（音量補正付き）
    drive_settings = {
        'rock':   {'drive_db': 6.5, 'tone': 0.6, 'gain_compensation_db': -2.8},
        'pop':    {'drive_db': 4.5, 'tone': 0.5, 'gain_compensation_db': -1.8},
        'ballad': {'drive_db': 2.5, 'tone': 0.3, 'gain_compensation_db': -1.0},
    }

    settings = drive_settings.get(s, drive_settings['pop'])

    # 入力音量測定
    input_peak = peak_db(audio)

    # ------ 後方互換：toneが使えない環境でも動作 ------
    try:
        # tone対応環境
        saturator = Distortion(drive_db=settings['drive_db'], tone=settings['tone'])
    except TypeError:
        # tone非対応環境
        try:
            if 'app' in globals():
                app.logger.warning("Distortion.tone が未対応のため drive_db のみでフォールバックします。")
            saturator = Distortion(drive_db=settings['drive_db'])
        except Exception as e:
            if 'app' in globals():
                app.logger.warning("saturation を無効化しました: %r", e)
            return audio
    except Exception as e:
        if 'app' in globals():
            app.logger.warning("saturation の初期化に失敗しました: %r", e)
        return audio

    # サチュレーション適用
    try:
        saturated = Pedalboard([saturator])(audio.T, sr).T
        
        # 音量補正を適用
        compensation_gain = db_to_lin(settings['gain_compensation_db'])
        result = saturated * compensation_gain
        
        # 最終安全処理
        result = np.clip(result, -1.0, 1.0)
        
        # デバッグ情報
        output_peak = peak_db(result)
        volume_change = output_peak - input_peak
        print(f"[Saturation {s.upper()}] Input: {input_peak:.1f}dB, Output: {output_peak:.1f}dB, Change: {volume_change:+.1f}dB")
        
        return result
        
    except Exception as e:
        if 'app' in globals():
            app.logger.warning("saturation の適用に失敗したためバイパスします: %r", e)
        return audio

def comp_stage1() -> Compressor:
    return Compressor(threshold_db=-24.0, ratio=4.0, attack_ms=3.0, release_ms=70.0)

def _comp_debug_str(c: Compressor, name: str) -> str:
    """Compressorノードの主要パラメータをログ用に整形"""
    try:
        th = getattr(c, 'threshold_db', None)
        ra = getattr(c, 'ratio', None)
        at = getattr(c, 'attack_ms', None)
        rl = getattr(c, 'release_ms', None)
        return (f"{name} th={th:.1f}dB ratio={ra:.2f}:1 atk={at:.0f}ms rel={rl:.0f}ms")
    except Exception:
        return name

def _best_probe_slice(x: np.ndarray, sr: int,
                      win_sec: float = 0.5,
                      search_sec: Optional[float] = None,
                      skip_silence: bool = True,
                      silence_thresh_db: float = -60.0,
                      hold_ms: int = 30) -> np.ndarray:
    """
    最も鳴っている短時間区間（RMS最大の窓）を返す。
    - search_sec=None で全体探索
    - skip_silence=True で先頭無音をスキップ（RMSが閾値を超える点から探索）
    """
    if x.ndim == 1:
        x2 = x[:, None]
    else:
        x2 = x
    N = x2.shape[0]
    win = max(1, int(sr * win_sec))
    L = N if (search_sec is None) else min(N, int(sr * search_sec))
    if L <= win:
        return x[:win] if N >= win else x

    # モノ化
    mono = x2.mean(axis=1)

    # 先頭無音スキップ
    start = 0
    if skip_silence:
        # 短いホールド窓でRMSを観測し、しきい値（例:-60dBFS）を超えた最初の地点
        hold = max(1, int(sr * (hold_ms / 1000.0)))
        if hold < len(mono):
            # 累積和で高速RMS
            sq = mono * mono
            csum = np.cumsum(np.concatenate([[0.0], sq]))
            # 各位置 i の区間 [i, i+hold) の平均二乗
            length_h = len(mono) - hold
            if length_h > 0:
                energy_h = csum[hold:hold+length_h] - csum[:length_h]
                rms = np.sqrt(energy_h / hold)
                thresh = 10 ** (silence_thresh_db / 20.0)
                idx = np.argmax(rms > thresh)
                if rms[idx] > thresh:
                    start = int(idx)

    end = min(N, start + L)
    seg = mono[start:end]
    if seg.size <= win:
        return x[start:start+win] if (start + win) <= N else x[start:]

    # 2乗累積和で区間RMS（探索区間内）
    sq = seg * seg
    csum = np.cumsum(np.concatenate([[0.0], sq]))
    length = seg.size - win
    energy = csum[win:win+length] - csum[:length]
    i0 = int(np.argmax(energy))
    i0_abs = start + i0
    return x[i0_abs:i0_abs+win]

def _autotarget_comp_threshold(comp: Compressor, src: np.ndarray, sr: int,
                               target_gr_db: float = 3.0,
                               tol_db: float = 0.5,
                               max_iter: int = 5) -> Tuple[Compressor, float]:
    """
    Stage1/Stage2 のしきい値を、短い代表窓で測ったGRが target_gr_dbになるように微調整。
    収束強化：最大 max_iter=5 回、更新量にゲイン α を掛ける（th += α * (measured - target)）。
    """
    th   = float(getattr(comp, 'threshold_db', -24.0))
    ratio = float(getattr(comp, 'ratio', 4.0))
    atk   = float(getattr(comp, 'attack_ms', 3.0))
    rel   = float(getattr(comp, 'release_ms', 70.0))

    last_gr = 0.0
    for _ in range(max_iter):
        test = Compressor(threshold_db=th, ratio=ratio, attack_ms=atk, release_ms=rel)
        y = Pedalboard([test])(src.T, sr).T
        gr = gr_estimate_db(src, y)
        last_gr = gr
        delta = gr - float(target_gr_db)
        if abs(delta) <= tol_db:
            break
        # 収束を少し加速（α>1.0）
        alpha = 1.2
        th += alpha * delta
        th = float(np.clip(th, -36.0, -6.0))  # 物理的な安全ガード

    tuned = Compressor(threshold_db=th, ratio=ratio, attack_ms=atk, release_ms=rel)
    return tuned, last_gr

def _la2a_target_gr_by_style(style: str) -> Optional[float]:
    s = (style or "pop").lower()
    if s in ("narration", "narrator", "nar"):
        return None  # OFF
    if s in ("chorus", "chorus_wide"):
        return 10.0  # 強め
    if s == "rock":
        return 10.0  # 強め
    if s == "pop":
        return 7.0   # 標準
    if s == "ballad":
        return 3.0   # 控えめ
    # デフォルトは標準
    return 7.0



def comp_stage2(style: str) -> Optional[Compressor]:
    s = (style or 'pop').lower()
    if s == 'rock':
        # 目安GR ≈ -10 dB
        return Compressor(threshold_db=-18.0, ratio=2.0, attack_ms=10.0, release_ms=250.0)
    if s == 'pop':
        # 目安GR ≈ -7 dB（従来より少し強め）
        return Compressor(threshold_db=-18.5, ratio=2.0, attack_ms=10.0, release_ms=220.0)
    if s == 'ballad':
        # 目安GR ≈ -3 dB（従来より少し弱め）
        return Compressor(threshold_db=-21.0, ratio=2.0, attack_ms=12.0, release_ms=280.0)
    if s == 'chorus' or s == 'chorus_wide':
        # 横に敷く用途は現状維持
        return Compressor(threshold_db=-22.0, ratio=2.0, attack_ms=10.0, release_ms=260.0)
    return None  # narration -> OFF

def deess_lite(audio: np.ndarray, sr: int, gender: str, style: str) -> np.ndarray:
    center = 7800.0 if (gender or 'male').lower() == 'female' else 6700.0
    bw = 3000.0
    lo = max(1000.0, center - bw / 2.0)
    hi = center + bw / 2.0
    strength = {'rock': 0.6, 'pop': 0.45, 'ballad': 0.30, 'narration': 0.15}.get(
        (style or 'pop').lower(), 0.45
    )
    
    # === ディエッサー動作確認ログ ===
    print(f"[De-Esser] Starting de-ess processing:")
    print(f"[De-Esser]   Gender={gender} -> Center freq={center:.0f}Hz")
    print(f"[De-Esser]   Style={style} -> Strength={strength:.2f}")
    print(f"[De-Esser]   Band filter: {lo:.0f}Hz - {hi:.0f}Hz (BW={bw:.0f}Hz)")
    
    # 入力音声の解析
    input_peak = peak_db(audio)
    print(f"[De-Esser]   Input peak: {input_peak:.2f} dBFS")
    
    band = Pedalboard([HighpassFilter(lo), LowpassFilter(hi)])(audio.T, sr).T
    band_peak = peak_db(band)
    print(f"[De-Esser]   Band-passed peak: {band_peak:.2f} dBFS")
    
    comp = Compressor(threshold_db=-24.0, ratio=6.0, attack_ms=3.0, release_ms=60.0)
    band_comp = Pedalboard([comp])(band.T, sr).T
    band_comp_peak = peak_db(band_comp)
    
    # GR推定
    gr_estimate = max(0.0, band_peak - band_comp_peak)
    print(f"[De-Esser]   Band after compression: {band_comp_peak:.2f} dBFS")
    print(f"[De-Esser]   Estimated GR in sibilant band: {gr_estimate:.2f} dB")
    
    diff = band - band_comp
    diff_peak = peak_db(diff)
    print(f"[De-Esser]   Difference signal peak: {diff_peak:.2f} dBFS")
    
    out = np.clip(audio - strength * diff, -1.0, 1.0)
    output_peak = peak_db(out)
    
    # 総合的な効果
    total_reduction = input_peak - output_peak
    print(f"[De-Esser]   Output peak: {output_peak:.2f} dBFS")
    print(f"[De-Esser]   Total level change: {total_reduction:+.2f} dB")
    print(f"[De-Esser] De-ess processing completed")
    
    return out

# ======================================================
# 修正：ドライチェーン処理（新しいエフェクト追加）
# ======================================================
import time

from functools import wraps

class EffectProfiler:
    def __init__(self):
        self.timings = {}
        self.enabled = True  # デバッグ時のみON/OFF切り替え用
    
    def profile(self, effect_name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                elapsed_ms = (end_time - start_time) * 1000
                self.timings[effect_name] = elapsed_ms
                print(f"[Profile] {effect_name}: {elapsed_ms:.1f}ms")
                return result
            return wrapper
        return decorator
    
    def reset(self):
        self.timings = {}
    
    def get_summary(self):
        if not self.timings:
            return "No timing data available"
        
        total = sum(self.timings.values())
        summary = [f"=== PERFORMANCE PROFILE (Total: {total:.1f}ms) ==="]
        
        # 処理時間順でソート
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        for name, time_ms in sorted_timings:
            percentage = (time_ms / total) * 100 if total > 0 else 0
            summary.append(f"  {name}: {time_ms:.1f}ms ({percentage:.1f}%)")
        
        return "\n".join(summary)

# グローバルインスタンス作成
profiler = EffectProfiler()

@profiler.profile("Build_Dry_Chain_Total")
def build_dry_chain(audio: np.ndarray, sr: int, gender: str, style: str, ng_mode: str, 
                   want_ng: bool, want_eq: bool, want_comp1: bool, want_comp2: bool, 
                   want_deess: bool, want_style_eq: bool, want_saturation: bool,
                   sat_amount: float = None, comp2_offset_db: float = None) -> Tuple[np.ndarray, float, float]:
    """段階2最適化版：統合処理を試行、失敗時は段階1にフォールバック"""
    
    # プロファイラーリセット
    profiler.reset()
    
    # 統合処理を試行
    try:
        result = build_dry_chain_stage2(audio, sr, gender, style, ng_mode,
                                     want_ng, want_eq, want_comp1, want_comp2, 
                                     want_deess, want_style_eq, want_saturation, sat_amount, comp2_offset_db)
        # プロファイル結果を出力
        print(profiler.get_summary())
        return result
    except Exception as e:
        print(f"[Warning] Stage2 optimization failed, using stage1: {e}")
        result = build_dry_chain_stage1(audio, sr, gender, style, ng_mode,
                                     want_ng, want_eq, want_comp1, want_comp2, want_deess,
                                     want_style_eq, want_saturation, sat_amount, comp2_offset_db)
        # エラー時もプロファイル結果を出力
        print(profiler.get_summary())
        return result

# ======================================================
# 段階2：統合最適化関数
# ======================================================
@profiler.profile("Build_Dry_Chain_Stage2")
def build_dry_chain_stage2(audio: np.ndarray, sr: int, gender: str, style: str, ng_mode: str, 
                          want_ng: bool, want_eq: bool, want_comp1: bool, want_comp2: bool, 
                          want_deess: bool, want_style_eq: bool, want_saturation: bool,
                          sat_amount: float = None, comp2_offset_db: float = None) -> Tuple[np.ndarray, float, float]:

    """段階2：全エフェクトを1つのPedalboardチェーンに統合"""
    
    x = to_stereo(audio).astype(np.float32, copy=False)
    gr1 = gr2 = 0.0
    
    t0 = time.perf_counter()
    
    # === 統合Pedalboardチェーン構築 ===
    unified_nodes = []
    
    # === 統合Pedalboardチェーン構築 ===
    unified_nodes = []
    
    # GR計算用の参照点を記録
    before_comp1 = x if want_comp1 else None
    before_comp2 = x if want_comp2 and not want_comp1 else None
    
    # 1. Noise Gate
    if want_ng:
        start_time = time.perf_counter()
        ng_node = make_noise_gate(ng_mode, gender)
        unified_nodes.append(ng_node)
        ng_time = (time.perf_counter() - start_time) * 1000
        print(f"[Profile_Detail] NoiseGate_Setup: {ng_time:.1f}ms")
        print(f"[Unified Chain] Added Noise Gate: mode='{ng_mode}', gender='{gender}', want_ng={want_ng}")
    else:
        print(f"[Unified Chain] Noise Gate disabled: want_ng={want_ng}")

    # 2. Basic EQ
    if want_eq:
        start_time = time.perf_counter()
        _eq_basic = eq_nodes_by_gender(gender)
        unified_nodes.extend(_eq_basic)
        eq_basic_time = (time.perf_counter() - start_time) * 1000
        print(f"[Profile_Detail] EQ_Basic_Setup: {eq_basic_time:.1f}ms")
        try:
            print(f"[EQ Basic] gender={gender} :: {_eq_nodes_summary(_eq_basic)}")
        except Exception:
            pass

    # 3. Style EQ（味付け）
    if want_style_eq:
        start_time = time.perf_counter()
        _eq_style = style_eq_nodes(style)
        unified_nodes.extend(_eq_style)
        eq_style_time = (time.perf_counter() - start_time) * 1000
        print(f"[Profile_Detail] EQ_Style_Setup: {eq_style_time:.1f}ms")
        try:
            cnt = len(_eq_style)
            summary = _eq_nodes_summary(_eq_style)
            print(f"[EQ Style]  style={style}  (nodes={cnt}) :: {summary}")
            if cnt == 0:
                print(f"[Warn][EQ Style] nodes=0  style='{style}'")
        except Exception as e:
            print(f"[EQ Style] log error: {e}")

    # 4. Compressor Stage 1（平均 -3 dB を自動で狙う）
    comp1 = None
    shared_pre_result = None  # スコープ問題を防ぐための初期化
    
    if want_comp1:
        comp1 = comp_stage1()

        # --- 統一前段チェーン構築と実行（使い回し最適化） ---
        shared_pre_nodes = []
        if want_ng:
            shared_pre_nodes.append(make_noise_gate(ng_mode, gender))
        if want_eq:
            shared_pre_nodes.extend(eq_nodes_by_gender(gender))
        if want_style_eq:
            shared_pre_nodes.extend(style_eq_nodes(style))
        if want_deess:
            shared_pre_nodes.extend(safe_enhanced_deesser_nodes(gender, style))
        
        # 統一前段チェーンを一度だけ実行
        start_time = time.perf_counter()
        shared_pre_result = Pedalboard(shared_pre_nodes)(x.T, sr).T if shared_pre_nodes else x
        shared_prechain_time = (time.perf_counter() - start_time) * 1000
        print(f"[Profile_Detail] Shared_PreChain_Execution: {shared_prechain_time:.1f}ms")
        
        # Stage1自動調整（前段チェーン結果を使い回し）
        try:            
            start_time = time.perf_counter()
            probe = _best_probe_slice(shared_pre_result, sr, win_sec=0.5, search_sec=None, skip_silence=True)
            probe_time = (time.perf_counter() - start_time) * 1000
            print(f"[Profile_Detail] Comp1_Probe_Analysis: {probe_time:.1f}ms")
            
            start_time = time.perf_counter()
            comp1, gr1_probe = _autotarget_comp_threshold(comp1, probe, sr, target_gr_db=3.0)
            autotarget_time = (time.perf_counter() - start_time) * 1000
            print(f"[Profile_Detail] Comp1_AutoTarget: {autotarget_time:.1f}ms")
            
            print(f"[Comp][Auto] Stage1 tuned to target -3 dB: probe_GR={gr1_probe:.2f} dB, th={comp1.threshold_db:.1f}dB")
        except Exception as e:
            print(f"[Comp][Auto][Warn] Stage1 autotune failed (use default): {e}")

        unified_nodes.append(comp1)
        before_comp1 = x  # 既存の簡易GRログ用の参照（必要なら）

    # 5. Compressor Stage 2（LA-2A：スタイル別ターゲット）
    c2 = comp_stage2(style)
    if want_comp2 and c2:
        base_tgt = _la2a_target_gr_by_style(style)
        if base_tgt is None:
            print("[Comp][Auto] Stage2 disabled for narration")
        else:
            # shared_pre_resultが未定義の場合（want_comp1=Falseの場合）
            if shared_pre_result is None:
                shared_pre_result = x

            # 右カラム調整値を適用
            if comp2_offset_db is not None:
                tgt = abs(float(comp2_offset_db))
                print(f"[Comp][Auto] Stage2 target: user_specified={tgt}dB")
            else:
                tgt = base_tgt
                print(f"[Comp][Auto] Stage2 target: style_default={tgt}dB")

            # Stage2自動調整（前段結果 + Stage1コンプレッサーを追加）
            try:
                start_time = time.perf_counter()
                if want_comp1 and comp1 is not None:
                    # 共有前段結果にStage1コンプレッサーを追加適用
                    stage2_pre_result = Pedalboard([comp1])(shared_pre_result.T, sr).T
                else:
                    stage2_pre_result = shared_pre_result
                comp2_prechain_time = (time.perf_counter() - start_time) * 1000
                print(f"[Profile_Detail] Comp2_PreChain_Execution: {comp2_prechain_time:.1f}ms")
                
                start_time = time.perf_counter()
                probe2 = _best_probe_slice(stage2_pre_result, sr, win_sec=0.5, search_sec=None, skip_silence=True)
                probe2_time = (time.perf_counter() - start_time) * 1000
                print(f"[Profile_Detail] Comp2_Probe_Analysis: {probe2_time:.1f}ms")
                
                start_time = time.perf_counter()
                c2, gr2_probe = _autotarget_comp_threshold(c2, probe2, sr, target_gr_db=tgt)
                autotarget2_time = (time.perf_counter() - start_time) * 1000
                print(f"[Profile_Detail] Comp2_AutoTarget: {autotarget2_time:.1f}ms")
                
                print(f"[Comp][Auto] Stage2 tuned (style={style}, target=-{tgt:.1f} dB): probe_GR={gr2_probe:.2f} dB, th={c2.threshold_db:.1f}dB")
            except Exception as e:
                print(f"[Comp][Auto][Warn] Stage2 autotune failed (use default): {e}")

            # 互換: 既存の簡易GRログ用フックを維持
            before_comp2 = x
            unified_nodes.append(c2)
    
    # === 一括実行（最大の最適化ポイント） ===
    if unified_nodes:
        try:
            # 全エフェクトを1回のPedalboard呼び出しで処理
            print(f"[Unified Chain] Processing with {len(unified_nodes)} total nodes:")
            for i, node in enumerate(unified_nodes):
                node_name = node.__class__.__name__
                if hasattr(node, 'center_frequency_hz'):
                    freq = getattr(node, 'center_frequency_hz', 'N/A')
                    gain = getattr(node, 'gain_db', 'N/A')
                    print(f"[Unified Chain]   [{i+1}] {node_name} (freq={freq}Hz, gain={gain}dB)")
                else:
                    print(f"[Unified Chain]   [{i+1}] {node_name}")
            
            x_before_unified = x.copy()
            unified_input_peak = peak_db(x_before_unified)
            
            # 最重要：実際のPedalboard処理時間を計測
            start_time = time.perf_counter()
            result = Pedalboard(unified_nodes)(x.T, sr).T
            pedalboard_time = (time.perf_counter() - start_time) * 1000
            print(f"[Profile_Detail] Pedalboard_Execution: {pedalboard_time:.1f}ms")
            
            unified_output_peak = peak_db(result)
            unified_total_change = unified_input_peak - unified_output_peak
            print(f"[Unified Chain] Input peak: {unified_input_peak:.2f} dBFS")
            print(f"[Unified Chain] Output peak: {unified_output_peak:.2f} dBFS")
            print(f"[Unified Chain] Total level change: {unified_total_change:+.2f} dB")

            # ===== ディエッサー効果の周波数帯域別確認 =====
            # 注：重複回避版ではディエッサーは統合チェーンでスキップされているが、
            # コンプレッサー自動調整で適用されているため効果測定は意味がある
            if want_deess:
                try:
                    # ディエッサー対象周波数帯域での効果測定
                    g = (gender or 'male').lower()
                    center = (7800.0 if g == 'female' else 7400.0 if g == 'smartphone' else 6700.0)
                    
                    # 歯擦音帯域での効果確認
                    sibilant_band_before = Pedalboard([
                        HighpassFilter(center - 1500.0),
                        LowpassFilter(center + 1500.0)
                    ])(x_before_unified.T, sr).T
                    
                    sibilant_band_after = Pedalboard([
                        HighpassFilter(center - 1500.0),
                        LowpassFilter(center + 1500.0)
                    ])(result.T, sr).T
                    
                    sibilant_before_peak = peak_db(sibilant_band_before)
                    sibilant_after_peak = peak_db(sibilant_band_after)
                    sibilant_reduction = sibilant_before_peak - sibilant_after_peak
                    
                    print(f"[Unified Chain] De-esser frequency analysis (via comp auto-tune):")
                    print(f"[Unified Chain]   Target band: {center-1500:.0f}-{center+1500:.0f}Hz (center: {center:.0f}Hz)")
                    print(f"[Unified Chain]   Before: {sibilant_before_peak:.2f} dBFS")
                    print(f"[Unified Chain]   After:  {sibilant_after_peak:.2f} dBFS")
                    print(f"[Unified Chain]   Sibilant reduction: {sibilant_reduction:+.2f} dB")
                    print(f"[Unified Chain]   Method: Enhanced nodes via compressor auto-tune")
                    
                except Exception as e:
                    print(f"[Unified Chain] Sibilant analysis failed: {e}")
            
            unified_output_peak = peak_db(result)
            unified_total_change = unified_input_peak - unified_output_peak
            print(f"[Unified Chain] Input peak: {unified_input_peak:.2f} dBFS")
            print(f"[Unified Chain] Output peak: {unified_output_peak:.2f} dBFS")
            print(f"[Unified Chain] Total level change: {unified_total_change:+.2f} dB")

            # ===== 局所GR（最も鳴っている区間で計測：全体探索＋無音スキップ）=====

            try:
                if want_comp1 and comp1 is not None and before_comp1 is not None:
                    src1 = _best_probe_slice(before_comp1, sr, win_sec=0.5, search_sec=None, skip_silence=True)
                    y1   = Pedalboard([comp1])(src1.T, sr).T
                    gr1  = gr_estimate_db(src1, y1)
                if want_comp2 and c2 is not None and before_comp2 is not None:
                    src2 = _best_probe_slice(before_comp2, sr, win_sec=0.5, search_sec=None, skip_silence=True)
                    y2   = Pedalboard([c2])(src2.T, sr).T
                    gr2  = gr_estimate_db(src2, y2)
            except Exception as e:
                print(f"[Comp][Warn] probe failed: {e}")

            # ===== ログ出力 =====
            try:
                if want_comp1 and comp1 is not None:
                    print(f"[Comp] Stage1: GR≈{gr1:.2f} dB | " + _comp_debug_str(comp1, "1176/R-Vox like"))
                if want_comp2 and c2 is not None:
                    print(f"[Comp] Stage2: GR≈{gr2:.2f} dB | " + _comp_debug_str(c2, "LA-2A like"))
            except Exception as e:
                print(f"[Comp][Warn] log failed: {e}")

            x = result
            
        except Exception as e:
            print(f"[Warning] Unified chain failed, falling back to stage1: {e}")
            # フォールバック：段階1の処理に戻す
            return build_dry_chain_stage1(audio, sr, gender, style, ng_mode,
                                        want_ng, want_eq, want_comp1, want_comp2, want_deess,
                                        want_style_eq, want_saturation, sat_amount)
    
    t1 = time.perf_counter()
    
    # 最終的な型安全性の確保
    x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
    
    print(f"[Profiler++] Unified Chain: {(t1 - t0)*1000:.1f}ms")
    
    return x, gr1, gr2

def build_lightweight_deess_nodes(gender: str, style: str) -> list:
    """De-Esserを軽量なPedalboardノードに変換"""
    g = (gender or 'male').lower()
    center = (
        7800.0 if g == 'female'
        else 7400.0 if g == 'smartphone'
        else 6700.0
    )
    strength_map = {'rock': 0.6, 'pop': 0.45, 'ballad': 0.30, 'narration': 0.15}
    strength = strength_map.get((style or 'pop').lower(), 0.45)
    
    # 軽量版：複雑な帯域分割処理の代わりにPeakFilterで近似
    bw = 3000.0
    q_factor = center / bw  # Q値を計算
    gain_reduction = -2.0 * strength  # strengthに基づくゲイン削減
    
    # === 軽量ディエッサー設定ログ ===
    print(f"[Lightweight De-Esser] Building pedalboard nodes:")
    print(f"[Lightweight De-Esser]   Gender={gender} ({g}) -> Center={center:.0f}Hz")
    print(f"[Lightweight De-Esser]   Style={style} -> Strength={strength:.2f}")
    print(f"[Lightweight De-Esser]   Bandwidth={bw:.0f}Hz -> Q factor={q_factor:.2f}")
    print(f"[Lightweight De-Esser]   Gain reduction={gain_reduction:.1f}dB")
    
    nodes = [
        # シンプルなPeakFilterでDe-Essing効果を近似
        PeakFilter(center, gain_reduction, q_factor)
    ]
    
    print(f"[Lightweight De-Esser] Created {len(nodes)} PeakFilter node(s)")
    return nodes

def build_lightweight_deess_nodes_enhanced(gender: str, style: str) -> list:
    """強化版軽量ディエッサー：より効果的なPedalboardノード群"""
    g = (gender or 'male').lower()
    center = (
        7800.0 if g == 'female'
        else 7400.0 if g == 'smartphone'
        else 6700.0
    )
    
    # 強化された強度設定
    strength_map = {'rock': 0.8, 'pop': 0.6, 'ballad': 0.4, 'narration': 0.2}
    strength = strength_map.get((style or 'pop').lower(), 0.6)
    
    # マルチバンド近似：複数のPeakFilterで効果的な歯擦音処理
    nodes = [
        # メイン周波数での積極的な削減
        PeakFilter(center, -3.0 * strength, 2.5),
        # 隣接周波数での軽い削減（自然な効果）
        PeakFilter(center * 0.8, -1.0 * strength, 1.8),
        PeakFilter(center * 1.2, -1.5 * strength, 2.0),
    ]
    
    print(f"[Enhanced De-Esser] Building enhanced pedalboard nodes:")
    print(f"[Enhanced De-Esser]   Gender={gender} ({g}) -> Center={center:.0f}Hz")
    print(f"[Enhanced De-Esser]   Style={style} -> Enhanced strength={strength:.2f}")
    print(f"[Enhanced De-Esser]   Multi-band approach: {len(nodes)} PeakFilter nodes")
    print(f"[Enhanced De-Esser]   Main band: {center:.0f}Hz (-{3.0*strength:.1f}dB)")
    print(f"[Enhanced De-Esser]   Low band: {center*0.8:.0f}Hz (-{1.0*strength:.1f}dB)")
    print(f"[Enhanced De-Esser]   High band: {center*1.2:.0f}Hz (-{1.5*strength:.1f}dB)")
    
    return nodes

def safe_enhanced_deesser_nodes(gender: str, style: str) -> list:
    """安全なフォールバック付き強化ディエッサー"""
    try:
        return build_lightweight_deess_nodes_enhanced(gender, style)
    except Exception as e:
        print(f"[Warning] Enhanced de-esser failed, using original: {e}")
        return build_lightweight_deess_nodes(gender, style)

def build_saturation_pedalboard_node(style: str, amount: float = 1.0):
    """Saturationをpedalboardノード化（sat_amount対応版）"""
    s = (style or 'pop').lower()
    if s == 'narration' or amount <= 0.0:
        return None
    
    drive_settings = {
        'rock': 6.5,
        'pop': 4.5,
        'ballad': 2.5,
    }

    base_drive_db = drive_settings.get(s, 4.5)
    
    # amountでドライブ量をスケール（0.0-1.0の範囲）
    scaled_drive = base_drive_db * amount
    
    try:
        # tone対応環境を試行
        return Distortion(drive_db=scaled_drive, tone=0.5)
    except TypeError:
        try:
            # tone非対応環境でフォールバック
            return Distortion(drive_db=scaled_drive)
        except Exception:
            return None


def build_dry_chain_stage1(audio: np.ndarray, sr: int, gender: str, style: str, ng_mode: str, 
                          want_ng: bool, want_eq: bool, want_comp1: bool, want_comp2: bool, 
                          want_deess: bool, want_style_eq: bool, want_saturation: bool,
                          sat_amount: float = None, comp2_offset_db: float = None) -> Tuple[np.ndarray, float, float]:

    """段階1処理（フォールバック用）"""
    # 既存の段階1処理をそのまま実行
    x = to_stereo(audio).astype(np.float32, copy=False)
    gr1 = gr2 = 0.0

    t0 = time.perf_counter()

    pb_nodes = []

    if want_ng:
        ng_node = make_noise_gate(ng_mode, gender)
        pb_nodes.append(ng_node)
        print(f"[Fallback Stage1] Added Noise Gate: mode='{ng_mode}', gender='{gender}', want_ng={want_ng}")
    else:
        print(f"[Fallback Stage1] Noise Gate disabled: want_ng={want_ng}")

    if want_eq:
        pb_nodes += eq_nodes_by_gender(gender)
    if want_style_eq:
        pb_nodes += style_eq_nodes(style)

    comp1 = None
    if want_comp1:
        before_comp1 = x.copy() if len(pb_nodes) > 0 else x
        comp1 = comp_stage1()

        pre_nodes_before_comp1 = []
        if want_ng:
            pre_nodes_before_comp1.append(make_noise_gate(ng_mode, gender))
        if want_eq:
            pre_nodes_before_comp1.extend(eq_nodes_by_gender(gender))
        if want_style_eq:
            pre_nodes_before_comp1.extend(style_eq_nodes(style))
        if want_deess:
            pre_nodes_before_comp1.extend(build_lightweight_deess_nodes(gender, style))

        try:
            pre1_full = Pedalboard(pre_nodes_before_comp1)(x.T, sr).T if pre_nodes_before_comp1 else x
            probe = _best_probe_slice(pre1_full, sr, win_sec=0.5, search_sec=None, skip_silence=True)
            comp1, gr1_probe = _autotarget_comp_threshold(comp1, probe, sr, target_gr_db=3.0)
            print(f"[Comp][Auto] Stage1 tuned to target -3 dB: probe_GR≈{gr1_probe:.2f} dB, th={comp1.threshold_db:.1f}dB")
        except Exception as e:
            print(f"[Comp][Auto][Warn] Stage1 autotune failed (use default): {e}")

    pb_nodes.append(comp1)

    c2 = comp_stage2(style)
    if want_comp2 and c2:
        base_tgt = _la2a_target_gr_by_style(style)
        if base_tgt is None:
            print("[Comp][Auto] Stage2 disabled for narration")
        else:
            # 右カラム調整値を適用
            if comp2_offset_db is not None:
                # ユーザー指定値をそのまま使用（-3, -7, -10のいずれか）
                tgt = abs(float(comp2_offset_db))
                print(f"[Comp][Auto] Stage2 target: user_specified={tgt}dB")
            else:
                # デフォルト値を使用
                tgt = base_tgt
                print(f"[Comp][Auto] Stage2 target: style_default={tgt}dB")
        
            pre_nodes_before_comp2 = []
            if want_ng:
                pre_nodes_before_comp2.append(make_noise_gate(ng_mode, gender))
            if want_eq:
                pre_nodes_before_comp2.extend(eq_nodes_by_gender(gender))
            if want_style_eq:
                pre_nodes_before_comp2.extend(style_eq_nodes(style))
            if want_deess:
                pre_nodes_before_comp2.extend(safe_enhanced_deesser_nodes(gender, style))
            if want_comp1 and comp1 is not None:
                pre_nodes_before_comp2.append(comp1)

            try:
                pre2_full = Pedalboard(pre_nodes_before_comp2)(x.T, sr).T if pre_nodes_before_comp2 else x
                probe2   = _best_probe_slice(pre2_full, sr, win_sec=0.5, search_sec=None, skip_silence=True)
                c2, gr2_probe = _autotarget_comp_threshold(c2, probe2, sr, target_gr_db=tgt)
                print(f"[Comp][Auto] Stage2 tuned (style={style}, target≈-{tgt:.1f} dB): "
                      f"probe_GR≈{gr2_probe:.2f} dB, th={c2.threshold_db:.1f}dB")
            except Exception as e:
                print(f"[Comp][Auto][Warn] Stage2 autotune failed (use default): {e}")

        pb_nodes.append(c2)

    if pb_nodes:
        try:
            x_after = Pedalboard(pb_nodes)(x.T, sr).T

            # ===== 局所GR（最も鳴っている区間で計測：全体探索＋無音スキップ）=====
            try:
                if want_comp1 and comp1 is not None:
                    src1 = _best_probe_slice(before_comp1, sr, win_sec=0.5, search_sec=None, skip_silence=True)
                    y1   = Pedalboard([comp1])(src1.T, sr).T
                    gr1  = gr_estimate_db(src1, y1)
                if want_comp2 and c2 is not None:
                    src2_base = before_comp2 if 'before_comp2' in locals() else x_after
                    src2 = _best_probe_slice(src2_base, sr, win_sec=0.5, search_sec=None, skip_silence=True)
                    y2   = Pedalboard([c2])(src2.T, sr).T
                    gr2  = gr_estimate_db(src2, y2)
            except Exception as e:
                print(f"[Comp][Warn] probe failed: {e}")

            # ===== ログ出力（既存のまま）=====
            try:
                if want_comp1 and comp1 is not None:
                    print(f"[Comp] Stage1: GR≈{gr1:.2f} dB | " + _comp_debug_str(comp1, "1176/R-Vox like"))
                if want_comp2 and c2 is not None:
                    print(f"[Comp] Stage2: GR≈{gr2:.2f} dB | " + _comp_debug_str(c2, "LA-2A like"))
            except Exception as e:
                print(f"[Comp][Warn] log failed: {e}")

            x = x_after

        except Exception as e:
            print(f"[Warning] Pedalboard chain failed, using original audio: {e}")
            pass

    t1 = time.perf_counter()

    if want_deess:
        print(f"[Fallback Stage1] De-esser stage: want_deess={want_deess}, method=deess_lite")
        print(f"[Fallback Stage1] Position: After compressors, before saturation")
        try:
            x_before_deess = x.copy()
            deess_input_peak = peak_db(x_before_deess)
            print(f"[Fallback Stage1] De-esser input peak: {deess_input_peak:.2f} dBFS")
            
            x = deess_lite(x, sr, gender, style)
            
            deess_output_peak = peak_db(x)
            deess_effect = deess_input_peak - deess_output_peak
            print(f"[Fallback Stage1] De-esser output peak: {deess_output_peak:.2f} dBFS")
            print(f"[Fallback Stage1] De-esser total effect: {deess_effect:+.2f} dB")
            
        except Exception as e:
            print(f"[Warning] De-esser failed, skipping: {e}")
            pass

    t2 = time.perf_counter()

    # サチュレーション処理（sat_amount対応版）
    if want_saturation and sat_amount is not None and sat_amount > 0.0:
        try:
            x_sat = apply_saturation(x, sr, style)
            x = np.clip((1.0 - sat_amount) * x + sat_amount * x_sat, -1.0, 1.0)
            print(f"[Debug] Saturation applied in fallback: amount={sat_amount}, style={style}")
        except Exception as e:
            print(f"[Warning] Saturation failed, skipping: {e}")
            pass

    t3 = time.perf_counter()

    print(f"[Profiler+] Fallback Stage1: {(t3 - t0)*1000:.1f}ms")

    x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
    
    return x, gr1, gr2

def build_dry_chain_cached(fid: Optional[str], src_audio: np.ndarray, src_sr: int, 
                          gender: str, style: str, ng_mode: str, fxwant: dict) -> Tuple[np.ndarray, float, float]:
    """キャッシュ機能付きドライチェーン処理"""
    want_ng = fxwant.get('noise_gate', True)
    want_eq = fxwant.get('eq', True)
    want_comp1 = fxwant.get('comp1', True)
    want_comp2 = fxwant.get('comp2', True)
    want_deess = fxwant.get('deess', True)
    want_style_eq = fxwant.get('style_eq', True)      # 修正：追加
    want_saturation = fxwant.get('saturation', True)  # 修正：追加
    
    # file_idがない場合は従来通り（キャッシュなし）
    if not fid or fid not in FILES:
        return build_dry_chain(src_audio, src_sr, gender, style, ng_mode, 
                          want_ng, want_eq, want_comp1, want_comp2, want_deess,
                          want_style_eq, want_saturation, sat_amount, comp2_offset_db)  
  
    # 修正：新しいエフェクトを含むキャッシュキー
    cache_key = f"{gender}_{style}_{ng_mode}_{want_ng}_{want_eq}_{want_comp1}_{want_comp2}_{want_deess}_{want_style_eq}_{want_saturation}_{src_sr}"
    dry_cache = FILES[fid].setdefault('dry_cache', {})
    
    if cache_key not in dry_cache:
        # キャッシュにない場合は新規作成
        result = build_dry_chain(src_audio, src_sr, gender, style, ng_mode, 
                                want_ng, want_eq, want_comp1, want_comp2, want_deess,
                                want_style_eq, want_saturation)  # 修正：引数追加
        dry_cache[cache_key] = result
        
        # キャッシュサイズ制限（メモリ使用量管理）
        if len(dry_cache) > 20:
            # 古いエントリを削除（FIFO）
            oldest_key = next(iter(dry_cache))
            del dry_cache[oldest_key]
    
    # キャッシュから取得（deepcopyして返す）
    cached_audio, cached_gr1, cached_gr2 = dry_cache[cache_key]
    return cached_audio.copy(), cached_gr1, cached_gr2

def build_dry_chain_simple_cache(fid: Optional[str], src_audio: np.ndarray, src_sr: int, 
                                gender: str, style: str, ng_mode: str, fxwant: dict,
                                sat_amount: float = None, comp2_offset_db: float = None) -> Tuple[np.ndarray, float, float]:

    """最小限のドライチェーンキャッシュ（sat_amount対応版）"""
    want_ng = fxwant.get('noise_gate', True)
    want_eq = fxwant.get('eq', True)
    want_comp1 = fxwant.get('comp1', True)
    want_comp2 = fxwant.get('comp2', True)
    want_deess = fxwant.get('deess', True)
    want_style_eq = fxwant.get('style_eq', True)
    want_saturation = fxwant.get('saturation', True)
    
    # file_idがない場合は従来通り（キャッシュなし）
    if not fid or fid not in FILES:
        return build_dry_chain(src_audio, src_sr, gender, style, ng_mode, 
                              want_ng, want_eq, want_comp1, want_comp2, want_deess,
                              want_style_eq, want_saturation, sat_amount)
    
    # SR を含め、sat_amount は除外（この関数の出力に依存しないため）
    cache_key = (
        f"{gender}_{style}_{ng_mode}_"
        f"{want_ng}_{want_eq}_{want_comp1}_{want_comp2}_{want_deess}_{want_style_eq}_{want_saturation}_"
        f"{src_sr}_{comp2_offset_db}"
    )
    
    # dry_cacheが存在しない場合は作成
    if 'dry_cache' not in FILES[fid]:
        FILES[fid]['dry_cache'] = {}
    
    dry_cache = FILES[fid]['dry_cache']
    
    # キャッシュにない場合のみ新規作成
    if cache_key not in dry_cache:
        result = build_dry_chain(src_audio, src_sr, gender, style, ng_mode, 
                                want_ng, want_eq, want_comp1, want_comp2, want_deess,
                                want_style_eq, want_saturation, sat_amount, comp2_offset_db)

        dry_cache[cache_key] = result
        
        # 簡単なサイズ制限（5つまで）
        if len(dry_cache) > 5:
            # 最初のキーを削除
            first_key = next(iter(dry_cache))
            del dry_cache[first_key]
    
    # キャッシュから取得
    cached_result = dry_cache[cache_key]
    return cached_result[0].copy(), cached_result[1], cached_result[2]

def make_render_cache_key(gender: str, style: str, ng_mode: str, fxwant: dict, 
                         bpm: Optional[float], mode: str, 
                         reverb_offset: float = 0.0, delay_offset: float = 0.0,
                         comp2_offset: float = 0.0, sat_amount: Optional[float] = None) -> str:
    """レンダリング結果キャッシュ用のキーを生成"""
    bpm_str = str(bpm) if bpm is not None else "None"
    sat_str = str(sat_amount) if sat_amount is not None else "None"
    fxwant_str = json.dumps(fxwant, sort_keys=True)
    return f"{gender}_{style}_{ng_mode}_{fxwant_str}_{bpm_str}_{reverb_offset}_{delay_offset}_{comp2_offset}_{sat_str}_{mode}"

# ======================================================
# FXパラメータ & ユーティリティ
# ======================================================
def style_fx_params(style: str) -> dict:
    s = (style or 'pop').lower()
    if s == 'rock':
        return dict(delay_frac=None, delay_ms=189, fb=0.20,
                    dduck='strong', d_hpf=200, d_lpf=3000,
                    pre=80, decay=1.1,
                    r_hpf=150, r_lpf=9500, r_wet_db=-30)

    if s == 'pop':
        return dict(delay_frac=None, delay_ms=189, fb=0.26,
                    dduck='mid', d_hpf=200, d_lpf=3000,
                    pre=75, decay=1.2,
                    r_hpf=150, r_lpf=9500, r_wet_db=-28)

    if s == 'ballad':
        # 🎵 バラード専用：ドット8分ディレイ（リバーブは長めだが前に出す）
        return dict(
            delay_frac=0.5, delay_ms=300, fb=0.18, dduck='mid',
            d_hpf=150, d_lpf=9000,
            pre=80, decay=1.5,                 # ← Pre 65→80ms / Decayそのまま1.5s
            r_hpf=150, r_lpf=8000, r_wet_db=-24  # ← Reverb LPF 9000→8000 / wet基準 -24（控え目）
        )

    if s == 'chorus':
        return dict(delay_frac=None, fb=0.0, dduck='off',
                    d_hpf=150, d_lpf=9000,
                    pre=70, decay=1.4,
                    r_hpf=150, r_lpf=9000, r_wet_db=-26)

    if s == 'chorus_wide':
        return dict(delay_frac=None, fb=0.0, dduck='off',
                    d_hpf=150, d_lpf=9000,
                    pre=70, decay=1.3,
                    r_hpf=150, r_lpf=9000, r_wet_db=-28)

    # narration など
    return dict(delay_frac=None, fb=0.0, dduck='off',
                d_hpf=150, d_lpf=8000,
                pre=50, decay=0.9,
                r_hpf=150, r_lpf=8000, r_wet_db=-35)

def note_to_seconds(bpm: float, frac: float) -> float:
    """
    BPMとノート長(frac)からディレイ時間を算出する。
    - frac=1.0 → 四分音符
    - frac=0.5 → 八分音符
    - frac=2.0 → 二分音符
    - frac=0.25 → 十六分音符
    """
    if bpm <= 0:
        return 0.25  # 保険値（0.25秒）
    quarter_note = 60.0 / bpm        # 四分音符の長さ（秒）
    return quarter_note * frac

def rms_envelope(x: np.ndarray, sr: int, win_ms=10, atk_ms=8, rel_ms=160) -> np.ndarray:
    n = len(x)
    hop = max(1, int(sr * win_ms / 1000))
    kern = np.ones(hop, dtype=np.float32) / hop
    env = np.sqrt(np.convolve(x.astype(np.float32) ** 2, kern, mode='same') + 1e-12)
    atk_a = np.exp(-1.0 / (sr * atk_ms / 1000))
    rel_a = np.exp(-1.0 / (sr * rel_ms / 1000))
    y = np.zeros_like(env)
    for i in range(1, n):
        a = atk_a if env[i] > y[i-1] else rel_a
        y[i] = (1 - a) * env[i] + a * y[i-1]
    return y

def apply_duck(wet: np.ndarray, dry: np.ndarray, sr: int, strength: str) -> np.ndarray:
    if strength == 'off':
        return wet
    mono = dry.mean(axis=1)
    if strength == 'strong':
        atk, rel, depth = 8, 140, 0.75
    elif strength == 'mid':
        atk, rel, depth = 12, 180, 0.60
    else:
        atk, rel, depth = 15, 220, 0.45
    env = rms_envelope(mono, sr, atk_ms=atk, rel_ms=rel)
    env = env / (env.max() + 1e-9)
    duck = 1.0 - depth * env
    return (wet.T * duck).T

# ======================================================
# Delay / Reverb: 100% wet（duck前）をキャッシュして高速化
#   ・BPMは0.5刻みで量子化してキャッシュヒット率UP
#   ・duck（apply_duck）は毎回のドライに対して軽量に適用
# ======================================================
def _bpm_for_cache(bpm: Optional[float], style: str) -> float:
    s = (style or 'pop').lower()
    default = 120.0 if s in ('rock', 'pop') else 90.0
    use = float(bpm if bpm is not None else default)
    return round(use * 2.0) / 2.0  # 0.5 BPM 単位で量子化

def get_space_wets_cached(
    fid: Optional[str],
    dry_for_fx: np.ndarray,
    sr: int,
    style: str,
    bpm: Optional[float]
) -> tuple[np.ndarray, np.ndarray]:
    """段階3最適化版：統合処理を試行、失敗時にフォールバック"""
    return get_space_wets_cached_optimized(fid, dry_for_fx, sr, style, bpm)

# ======================================================
# 段階3：空間系エフェクト最適化関数群
# ======================================================

# --- helper: decay(秒) -> room_size(0..1) の安全マッピング ---
def _decay_to_room_size(decay: float) -> float:
    rs = 0.25 + 0.25 * float(decay)     # 速さ優先の簡易マップ
    return max(0.10, min(0.75, rs))     # 0.10〜0.75 に丸め

# --- 統合ルート（Unified） 本体 ---
# 差し替え後（引数を1つだけ追加：bpm_specified=False）
def build_space_effects_unified(
    dry_for_fx: np.ndarray, sr: int, style: str, use_bpm: float,
    delay_enabled: bool, p: dict, fx_cache: Optional[dict], cache_key: str,
    bpm_specified: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    """空間系の高速・安定版: pre-delay は前段 Delay で再現し、
       Reverb/Delay は100% wetで“個別一発”。共通HPF/LPFは1回だけ。"""
    t0 = time.perf_counter()

    # 入力をステレオ・float32 に整形
    x = to_stereo(dry_for_fx).astype(np.float32, copy=False)

    # 共通前処理（1回だけ）
    common_hpf = float(p['r_hpf'])
    common_lpf = float(min(p['r_lpf'], p.get('d_lpf', p['r_lpf'])))
    y = Pedalboard([HighpassFilter(common_hpf), LowpassFilter(common_lpf)])(x.T, sr).T

    # Reverb（100% wet）: pre-delayは前段Delayで再現
    room_size = _decay_to_room_size(float(p['decay']))
    pre_ms    = float(p.get('pre', 0.0))
    print(f"[Debug Unified Reverb] decay={p['decay']}s -> room_size={room_size:.3f}, pre_delay={pre_ms}ms")
    
    rev_chain = []
    if pre_ms > 0.0:
        rev_chain.append(Delay(delay_seconds=pre_ms/1000.0, feedback=0.0, mix=1.0))
        print(f"[Debug Unified Reverb] Added pre-delay: {pre_ms}ms")
    rev_chain.append(Reverb(room_size=room_size, damping=0.25, wet_level=1.0, dry_level=0.0, width=0.9))
    print(f"[Debug Unified Reverb] Final chain length: {len(rev_chain)} nodes")
    rev_wet = Pedalboard(rev_chain)(y.T, sr).T

    # Delay（100% wet）
    if delay_enabled:
        if bpm_specified:
            # BPM入力あり → 常に1/4音符同期
            frac = 1.0
            delay_seconds = note_to_seconds(float(use_bpm), frac)
            print(f"[Debug Unified Delay] Using BPM sync (1/4 note) at {use_bpm}BPM -> {delay_seconds:.3f}s")
        elif p.get('delay_ms') is not None:
            # BPM未入力 → プリセット固定ms
            delay_seconds = float(p['delay_ms']) / 1000.0
            print(f"[Debug Unified Delay] Using preset delay_ms: {p['delay_ms']}ms -> {delay_seconds:.3f}s")
        else:
            # 保険（fracだけ定義されている場合など）
            frac = p.get('delay_frac') or 0.25
            delay_seconds = note_to_seconds(float(use_bpm), float(frac))
            print(f"[Debug Unified Delay] Using fallback frac={frac} at {use_bpm}BPM -> {delay_seconds:.3f}s")

        print(f"[Debug Unified Delay] feedback={p['fb']}, final_delay={delay_seconds:.3f}s")
        delay_base = Pedalboard([
            Delay(delay_seconds=float(delay_seconds), feedback=float(p['fb']), mix=1.0)
        ])(y.T, sr).T
    else:
        delay_base = np.zeros_like(y)
        print(f"[Debug Unified Delay] Delay disabled, returning zeros")

    # 範囲保護
    rev_wet    = np.clip(rev_wet,    -1.0, 1.0).astype(np.float32, copy=False)
    delay_base = np.clip(delay_base, -1.0, 1.0).astype(np.float32, copy=False)

    print(f"[Profiler+++] Unified Space Effects (fast): {(time.perf_counter()-t0)*1000:.1f}ms")

    # 省メモリの簡易キャッシュ（任意）
    if fx_cache is not None:
        fx_cache[cache_key] = (rev_wet, delay_base)
        if len(fx_cache) > 12:
            fx_cache.pop(next(iter(fx_cache)))

    return rev_wet.copy(), delay_base.copy()


def get_space_wets_cached_optimized(
    fid: Optional[str],
    dry_for_fx: np.ndarray,
    sr: int,
    style: str,
    bpm: Optional[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    段階3最適化版：並列処理とキャッシュ改善で空間系エフェクトを高速化
    """
    # BPMを量子化
    use_bpm = _bpm_for_cache(bpm, style)
    p = style_fx_params(style)
    # delay_ms がある場合も有効化
    delay_enabled = (p.get('delay_ms') is not None) or (p.get('delay_frac') is not None)

    # キャッシュ確認
    fx_cache = FILES[fid]['fx_cache'] if (fid and fid in FILES) else None
    key = f"{style}_{use_bpm}_{sr}"
    if fx_cache and key in fx_cache:
        return [a.copy() for a in fx_cache[key]]

    # 統合処理を試行
    try:
        bpm_specified = (bpm is not None)
        return build_space_effects_unified(
            dry_for_fx, sr, style, use_bpm, delay_enabled, p, fx_cache, key, bpm_specified=bpm_specified
        )
    except Exception as e:
        print(f"[Warning] Unified space effects failed, using fallback: {e}")
        # フォールバック：既存処理
        return get_space_wets_cached_fallback(
            dry_for_fx, sr, style, use_bpm, delay_enabled, p, fx_cache, key,
            bpm_specified  # ←追加
        )
def get_space_wets_cached_fallback(
    dry_for_fx: np.ndarray, sr: int, style: str, use_bpm: float,
    delay_enabled: bool, p: dict, fx_cache: dict, cache_key: str,
    bpm_specified: bool  # ←追加
) -> tuple[np.ndarray, np.ndarray]:
    """フォールバック：個別処理（pre-delayは前段Delayで再現、重複排除）"""

    t0 = time.perf_counter()
    x_for_fx = to_stereo(dry_for_fx).astype(np.float32, copy=False)

    # ===== Reverb（100% wet）=====
    try:
        room_size = _decay_to_room_size(float(p['decay']))
        pre_ms    = float(p.get('pre', 0.0))

        rev_nodes = [
            HighpassFilter(float(p['r_hpf'])),
            LowpassFilter(float(p['r_lpf'])),
        ]
        # pre-delay を前段 Delay で再現（必要なときだけ）
        if pre_ms > 0.0:
            rev_nodes.append(Delay(delay_seconds=pre_ms / 1000.0,
                                   feedback=0.0, mix=1.0))
        # Reverb は 100% wet を1回だけ
        rev_nodes.append(Reverb(room_size=room_size, damping=0.25,
                                wet_level=1.0, dry_level=0.0, width=0.9))

        rev_wet = Pedalboard(rev_nodes)(x_for_fx.T, sr).T
    except Exception:
        rev_wet = np.zeros_like(x_for_fx)

    # Delay（100% wet）
    if delay_enabled:
        if bpm_specified:
            # BPM入力あり → 常に1/4音符同期
            frac = 1.0
            delay_seconds = note_to_seconds(float(use_bpm), frac)
            print(f"[Debug Unified Delay] Using BPM sync (1/4 note) at {use_bpm}BPM -> {delay_seconds:.3f}s")
        elif p.get('delay_ms') is not None:
            # BPM未入力 → プリセット固定ms
            delay_seconds = float(p['delay_ms']) / 1000.0
            print(f"[Debug Unified Delay] Using preset delay_ms: {p['delay_ms']}ms -> {delay_seconds:.3f}s")
        else:
            # 保険（fracだけ定義されている場合など）
            frac = p.get('delay_frac') or 0.25
            delay_seconds = note_to_seconds(float(use_bpm), float(frac))
            print(f"[Debug Unified Delay] Using fallback frac={frac} at {use_bpm}BPM -> {delay_seconds:.3f}s")

        print(f"[Debug Unified Delay] feedback={p['fb']}, final_delay={delay_seconds:.3f}s")
        delay_nodes = [
            HighpassFilter(float(p['d_hpf'])),
            LowpassFilter(float(p['d_lpf'])),
            Delay(delay_seconds=float(delay_seconds), feedback=float(p['fb']), mix=1.0),
        ]
        delay_base = Pedalboard(delay_nodes)(x_for_fx.T, sr).T
    else:
        delay_base = np.zeros_like(x_for_fx)
        print(f"[Debug Unified Delay] Delay disabled, returning zeros")

    # 簡易キャッシュ（任意）
    if fx_cache is not None:
        fx_cache[cache_key] = (rev_wet, delay_base)
        if len(fx_cache) > 12:
            fx_cache.pop(next(iter(fx_cache)))

    return rev_wet.copy(), delay_base.copy()

# ======================================================
# Doubler (side + center correction, 100% wet)
# ======================================================
def build_doubler_wet(dry_for_fx: np.ndarray, sr: int,
                      center_db: float, width_pct: float,
                      delay_l_ms: float, delay_r_ms: float,
                      hpf_hz: float = 150.0) -> np.ndarray:
    """
    返り値は 100% wet（x に加算）。center_db は
    「最終センター量」を dB 指定：例）-12dB => 中央は 0.25 倍へ。
    実装は『wet に (center_lin - 1.0) * mid を加える』ことで
    既存の x に対するセンター量を所望値へ調整。
    width_pct は左右ウェット成分のスケール（0〜100%）。
    """
    x = to_stereo(dry_for_fx)
    mid = to_mono_mid(x)[:, 0]  # shape: (N,)

    # ---- Center correction (negative add to reduce mid) ----
    center_lin = db_to_lin(float(center_db))
    center_corr = (center_lin - 1.0) * mid  # < 0 なら x から減算効果
    center_st = np.stack([center_corr, center_corr], axis=1)

    # ---- L/R short delays as double ----
    def mono_delay(sig: np.ndarray, ms: float) -> np.ndarray:
        if ms <= 0.0:
            return np.zeros_like(sig)
        pb = Pedalboard([Delay(delay_seconds=float(ms)/1000.0,
                               feedback=0.0, mix=1.0)])
        y = pb(sig.reshape(1, -1), sr)[0]
        # 高域中心に：低域を広げない
        y = Pedalboard([HighpassFilter(float(hpf_hz))])(y.reshape(1, -1), sr)[0]
        return y

    left  = mono_delay(mid, float(delay_l_ms))
    right = mono_delay(mid, float(delay_r_ms))
    sides = np.stack([left, right], axis=1) * (float(width_pct) / 100.0)

    wet = center_st + sides
    # 範囲保護（加算側なので軽く）
    wet = np.clip(wet, -1.0, 1.0).astype(np.float32, copy=False)
    return wet

# ======================================================
# Center Kill: 最終ミックスから「Mid（センター）」を差し引く
#   L' = L - mid, R' = R - mid  →  Sides のみが残る（モノではほぼ消える）
# ======================================================
def kill_center(x: np.ndarray) -> np.ndarray:
    x = to_stereo(x).astype(np.float32, copy=False)
    mid = to_mono_mid(x)[:, 0]  # (N,)
    x[:, 0] = x[:, 0] - mid
    x[:, 1] = x[:, 1] - mid
    # 過大振幅を保護
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


# ======================================================
# Loudness
# ======================================================

def target_lufs(style: str) -> float:
    return dict(rock=-10.0, pop=-11.0, ballad=-12.5, narration=-16.0).get((style or '').lower(), -11.0)

def limiter_only(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """プレビュー時：Limiterのみ（LUFS計測は省略）"""
    limited = Pedalboard([Limiter(threshold_db=-1.0)])(audio.T, sr).T
    tp_db = peak_db(limited)
    return limited, tp_db

# ======================================================
# プロモード用スナップショット（拡張版）
# ======================================================
def config_snapshot(gender: str, style: str, ng_mode: str, bpm: Optional[float], fx_on: Optional[dict] = None):
    s = (style or 'pop').lower()
    g = (gender or 'male').lower()
    bpm_specified = bpm is not None

    # === display_base を決める（/process と同じロジック） ===
    style_base = -16.0 if s == 'rock' else (-20.0 if s == 'pop' else -24.0)

    preset = (request.form.get('delay_preset') or '').strip().lower()
    if preset in ('weak', '控えめ'):
        display_base = -24.0
    elif preset in ('normal', '標準'):
        display_base = -20.0
    elif preset in ('strong', '強め'):
        display_base = -16.0
    else:
        display_base = style_base

    # /inspect では space_db の個別スライダを反映
    delay_off  = request.form.get('delay_offset_db',  type=float, default=None)
    reverb_off = request.form.get('reverb_offset_db', type=float, default=None)
    if delay_off is None and reverb_off is None:
        space_db   = request.form.get('space_db', type=float, default=0.0)
        delay_off  = space_db
        reverb_off = space_db
    else:
        delay_off  = 0.0 if delay_off  is None else float(delay_off)
        reverb_off = 0.0 if reverb_off is None else float(reverb_off)

    # Noise Gate（表示用）
    ng = {
        'weak':   {'threshold_db': -50.0, 'ratio': 2.0, 'attack_ms': 2.0, 'release_ms': 120.0},
        'strong': {'threshold_db': -45.0, 'ratio': 4.0, 'attack_ms': 1.0, 'release_ms': 150.0},
    }.get((ng_mode or 'weak').lower(), {})

    # 基本EQ設定（性別）
    if g == 'female':
        basic_eq = [
            {'type': 'HPF', 'freq_hz': 110.0},
            {'type': 'Peak', 'freq_hz': 275.0, 'gain_db': -1.0, 'q': 1.0},
            {'type': 'Peak', 'freq_hz': 750.0,  'gain_db': -0.5, 'q': 1.0},
            {'type': 'Peak', 'freq_hz': 4000.0, 'gain_db': +1.0, 'q': 1.0},
            {'type': 'HighShelf', 'freq_hz': 12000.0, 'gain_db': +1.5},
        ]
    elif g == 'smartphone':
        basic_eq = [
            {'type': 'HPF',      'freq_hz': 140.0},
            {'type': 'LowShelf', 'freq_hz': 170.0,  'gain_db': +1.0},
            {'type': 'Peak',     'freq_hz': 300.0,  'gain_db': -2.0, 'q': 1.1},
            {'type': 'Peak',     'freq_hz': 600.0,  'gain_db': -1.0, 'q': 1.2},
            {'type': 'Peak',     'freq_hz': 1000.0, 'gain_db': -0.8, 'q': 1.0},
            {'type': 'Peak',     'freq_hz': 3500.0, 'gain_db': -0.5, 'q': 1.0},
            {'type': 'Peak',     'freq_hz': 7200.0, 'gain_db': -1.5, 'q': 2.0},
            {'type': 'HighShelf','freq_hz': 11000.0,'gain_db': +0.5},
        ]
    else:
        basic_eq = [
            {'type': 'HPF', 'freq_hz': 100.0},
            {'type': 'Peak', 'freq_hz': 300.0,  'gain_db': -1.5, 'q': 1.0},
            {'type': 'Peak', 'freq_hz': 800.0,  'gain_db': -1.5, 'q': 1.0},
            {'type': 'Peak', 'freq_hz': 4000.0, 'gain_db': +1.5, 'q': 1.0},
            {'type': 'HighShelf', 'freq_hz': 12000.0, 'gain_db': +1.0},
        ]

    # スタイル別EQ（表示用）- 実装と一致させる
    style_eq_settings = {
        'rock': [
            {'type': 'Peak', 'freq_hz': 400.0,  'gain_db': -3.0, 'q': 1.20},
            {'type': 'Peak', 'freq_hz': 1000.0, 'gain_db': +3.5, 'q': 1.80},
            {'type': 'Peak', 'freq_hz': 3000.0, 'gain_db': +5.0, 'q': 2.00},
            {'type': 'Peak', 'freq_hz': 8000.0, 'gain_db': +3.0, 'q': 1.20},
            {'type': 'HighShelf', 'freq_hz': 13000.0, 'gain_db': +4.5},
        ],
        'pop': [
            {'type': 'Peak', 'freq_hz': 250.0, 'gain_db': -1.6, 'q': 1.1},
            {'type': 'Peak', 'freq_hz': 3200.0, 'gain_db': +1.5, 'q': 1.1},
            {'type': 'HighShelf', 'freq_hz': 10000.0, 'gain_db': +1.2},
        ],
        'ballad': [
            {'type': 'Peak', 'freq_hz': 150.0, 'gain_db': +0.8, 'q': 0.8},
            {'type': 'Peak', 'freq_hz': 400.0, 'gain_db': -0.5, 'q': 1.2},
            {'type': 'Peak', 'freq_hz': 1200.0, 'gain_db': +0.8, 'q': 1.5},
            {'type': 'Peak', 'freq_hz': 3000.0, 'gain_db': +1.2, 'q': 1.3},
            {'type': 'HighShelf', 'freq_hz': 8000.0, 'gain_db': +0.8},
        ],
        'narration': [
            {'type': 'HPF', 'freq_hz': 120.0},
            {'type': 'Peak', 'freq_hz': 300.0, 'gain_db': -1.2, 'q': 1.0},
            {'type': 'Peak', 'freq_hz': 1800.0, 'gain_db': +1.2, 'q': 1.4},
            {'type': 'Peak', 'freq_hz': 3500.0, 'gain_db': +1.0, 'q': 1.3},
            {'type': 'Peak', 'freq_hz': 6500.0, 'gain_db': -0.8, 'q': 1.5},
            {'type': 'HighShelf', 'freq_hz': 10000.0, 'gain_db': +0.3},
        ],
        'chorus': [
            {'type': 'Peak', 'freq_hz': 120.0, 'gain_db': +0.5, 'q': 1.0},
            {'type': 'Peak', 'freq_hz': 800.0, 'gain_db': -0.3, 'q': 1.2},
            {'type': 'Peak', 'freq_hz': 4000.0, 'gain_db': +0.8, 'q': 1.5},
            {'type': 'HighShelf', 'freq_hz': 10000.0, 'gain_db': +0.3},
        ],
        'chorus_wide': [
            {'type': 'Peak', 'freq_hz': 120.0, 'gain_db': +0.5, 'q': 1.0},
            {'type': 'Peak', 'freq_hz': 800.0, 'gain_db': -0.3, 'q': 1.2},
            {'type': 'Peak', 'freq_hz': 4000.0, 'gain_db': +0.8, 'q': 1.5},
            {'type': 'HighShelf', 'freq_hz': 10000.0, 'gain_db': +0.3},
        ],
    }

    # サチュレーション表示用 - 実装と一致させる
    saturation_settings = {
        'rock': {'drive_db': 6.5, 'tone': 0.6, 'character': 'aggressive'},
        'pop': {'drive_db': 4.5, 'tone': 0.5, 'character': 'balanced'},
        'ballad': {'drive_db': 2.5, 'tone': 0.3, 'character': 'warm'},
        'narration': None
    }

    sat_amount = request.form.get('sat_amount', type=float, default=None)
    sat_params = saturation_settings.get(s)
    if sat_params is not None and sat_amount is not None:
        sat_params = {**sat_params, 'amount': float(sat_amount)}

    # コンプレッサ（表示用）
    comp1 = {'model': '1176/R-Vox like', 'threshold_db': -24.0, 'ratio': 4.0, 'attack_ms': 3.0, 'release_ms': 70.0}
    comp2_map = {
        'rock':     {'model': 'LA-2A like', 'threshold_db': -18.0, 'ratio': 2.0, 'attack_ms': 10.0, 'release_ms': 250.0},
        'pop':      {'model': 'LA-2A like', 'threshold_db': -19.0, 'ratio': 2.0, 'attack_ms': 10.0, 'release_ms': 220.0},
        'ballad':   {'model': 'LA-2A like', 'threshold_db': -20.0, 'ratio': 2.0, 'attack_ms': 12.0, 'release_ms': 280.0},
        'narration': None
    }
    comp2 = comp2_map.get(s)
    t = request.form.get('comp2_target_gr_db', type=float)
    if comp2 is not None and t is not None:
        comp2 = {**comp2, 'target_gr_db': float(t)}

    # ディエッサー表示用
    center_hz = (7800.0 if g == 'female' else 7400.0 if g == 'smartphone' else 6700.0)
    deess_strength = {'rock': 0.6, 'pop': 0.45, 'ballad': 0.30, 'narration': 0.15}.get(s, 0.45)

    # エフェクト ON/OFF
    fxwant = {k: True for k in ('noise_gate','eq','comp1','comp2','deess','style_eq','saturation','delay','reverb','limiter')}
    if isinstance(fx_on, dict):
        fxwant.update({str(k): bool(v) for k, v in fx_on.items() if k in fxwant})

    # スタイル固有パラメータ
    p = style_fx_params(style)

    # 表示用 BPM（未指定ならスタイル既定値）
    use_bpm = bpm if bpm is not None else (120.0 if s in ('rock', 'pop') else 90.0)

    # ===== Delay 判定と時間算出（/process と整合）=====
    delay_enabled = (p.get('delay_ms') is not None) or (p.get('delay_frac') is not None)
    if bpm_specified:
        # BPM入力あり → 常に1/4音符同期
        frac = 1.0  # ※ 1.0を四分音符とする note_to_seconds 実装に合わせる
        delay_seconds = note_to_seconds(float(use_bpm), frac)
        print(f"[Debug Delay] Using BPM sync (1/4 note) at {use_bpm}BPM -> {delay_seconds:.3f}s")
    elif p.get('delay_ms') is not None:
        delay_seconds = float(p['delay_ms']) / 1000.0
        print(f"[Debug Delay] Using preset delay_ms: {p['delay_ms']}ms -> {delay_seconds:.3f}s")
    else:
        frac = p.get('delay_frac') or 0.25
        delay_seconds = note_to_seconds(float(use_bpm), float(frac))
        print(f"[Debug Delay] Using fallback frac={frac} at {use_bpm}BPM -> {delay_seconds:.3f}s")

    # ラウドネス目標（表示用）
    lufs_target_val = target_lufs(style)

    # === Effects 一覧を構築（表示スナップショット）===
    effects = [
        {'key': 'noise_gate', 'name': 'Noise Gate', 'enabled': fxwant['noise_gate'], 'params': ng},

        {'key': 'eq', 'name': f"EQ Basic ({'Female' if g=='female' else 'Smartphone' if g=='smartphone' else 'Male'})",
         'enabled': fxwant['eq'], 'params': basic_eq},

        {'key': 'comp1', 'name': 'Compressor Stage 1 (1176/R-Vox)', 'enabled': fxwant['comp1'], 'params': comp1},

        {'key': 'comp2', 'name': 'Compressor Stage 2 (LA-2A)',
         'enabled': fxwant['comp2'] and (comp2 is not None), 'params': comp2},

        {'key': 'deess', 'name': 'De-Esser (lite)', 'enabled': fxwant['deess'],
         'params': {'center_hz': center_hz, 'bandwidth_hz': 3000.0, 'strength': deess_strength}},

        {'key': 'style_eq', 'name': f'Style EQ ({s.title()})', 'enabled': fxwant['style_eq'],
         'params': style_eq_settings.get(s, [])},

        {'key': 'saturation', 'name': f'Saturation ({s.title()})',
         'enabled': fxwant['saturation'] and (sat_params is not None),
         'params': sat_params},

        {'key': 'delay', 'name': 'Delay (duck)',
         'enabled': fxwant['delay'] and delay_enabled,
         'params': None if not delay_enabled else {
             **({'note_fraction': float(p['delay_frac'])} if p.get('delay_frac') is not None else {}),
             'delay_time_s': delay_seconds,
             'feedback': float(p['fb']),
             'duck': p['dduck'],
             'hpf_hz': float(p['d_hpf']),
             'lpf_hz': float(p['d_lpf']),
             'mix_db': display_base + float(delay_off),
         }},

        {'key': 'reverb', 'name': 'Reverb', 'enabled': fxwant['reverb'], 'params': {
            'pre_delay_ms': float(p['pre']),
            'room_size': 0.30, 'damping': 0.25,
            'wet_db': float(p['r_wet_db']) + float(reverb_off),
            'wet_level_linear': db_to_lin(float(p['r_wet_db']) + float(reverb_off)),
            'hpf_hz': float(p['r_hpf']), 'lpf_hz': float(p['r_lpf']),
            'width': 0.9
        }},

        {'key': 'limiter', 'name': 'Safety Limiter', 'enabled': fxwant['limiter'],
         'params': {'threshold_db': -3.0, 'purpose': 'マスタリング用マージン確保'}},
    ]

    return {
        'inputs': {'gender': gender, 'style': style, 'noise_gate': ng_mode, 'bpm': bpm},
        'effects': effects
    }

# ======================================================
# #1 /upload: file_id付与 + プレビュー素材の事前生成
# ======================================================
FILES: dict[str, dict] = {}  # fid -> {'temp_path': str, 'sr': int, 'preview_mono': (audio_ds, sr_ds), 'fx_cache': {...}, 'dry_cache': {...}}

def cleanup_temp_file(fid: str):
    """Clean up temporary file for given file ID"""
    if fid in FILES and 'temp_path' in FILES[fid]:
        temp_path = FILES[fid]['temp_path']
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[Cleanup] Removed temp file: {temp_path}")
        except Exception as e:
            print(f"[Cleanup] Error removing temp file {temp_path}: {e}")
        finally:
            # Remove from FILES dict
            del FILES[fid]

def _make_file_id(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()[:16]

def _half_downsample(x: np.ndarray) -> np.ndarray:
    # 既存実装を踏襲（品質より"速さ優先"）：2:1 デシメーション
    return x[::2, :].copy()

@app.post('/upload')
def upload():
    t0 = time.perf_counter()

    if 'file' not in request.files:
        return abort(400, 'file is required')

    file = request.files['file']
    if file.filename == '':
        return abort(400, 'empty file')

    # Check file size before processing
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    print(f"[Upload] File size: {file_size / 1024 / 1024:.2f} MB")

    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return abort(413, f'File too large. Maximum size: {app.config["MAX_CONTENT_LENGTH"] / 1024 / 1024:.0f}MB')

    # Always use filename + size + timestamp for file ID to avoid memory issues
    import hashlib
    import time as time_mod
    fid = hashlib.sha1(f"{file.filename}_{file_size}_{time_mod.time()}".encode()).hexdigest()[:16]

    if fid not in FILES:
        # Save to temporary file using streaming (no memory loading)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name

        # Stream file directly to temp location without loading in memory
        try:
            file.save(temp_path)
            print(f"[Upload] Saved to temp file: {temp_path}")
        except Exception as e:
            print(f"[Upload] Error saving file: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return abort(500, f'Upload failed: {str(e)}')

        # Generate lightweight preview versions with memory limits
        try:
            # For large files, limit preview to first 30 seconds to save memory
            max_frames = None
            if file_size > 10 * 1024 * 1024:  # For files > 10MB
                # Estimate sample rate and limit to 30 seconds
                max_frames = int(44100 * 30)  # Assume 44.1kHz, 30 seconds max
                print(f"[Upload] Large file detected, limiting preview to 30 seconds")

            if max_frames is not None:
                a, sr = sf.read(temp_path, dtype='float32', always_2d=True, frames=max_frames)
            else:
                a, sr = sf.read(temp_path, dtype='float32', always_2d=True)
            a = to_stereo(a)

            # Generate lightweight preview versions only
            a_mono = to_mono_mid(a)
            a_ds_mono = _half_downsample(a_mono)
            sr_ds = sr // 2

        except Exception as e:
            print(f"[Upload] Error reading audio file: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return abort(400, f'Invalid audio file: {str(e)}')

        FILES[fid] = {
            'temp_path': temp_path,  # Store temp file path instead of full audio
            'sr': sr,
            'preview_mono': (a_ds_mono, sr_ds),  # Only keep minimal preview data
            'fx_cache': {},
            'dry_cache': {},
            'render_cache': {}
        }

    t1 = time.perf_counter()
    print(f"[Profiler] /upload: {(t1 - t0) * 1000:.1f}ms")

    return jsonify({'file_id': fid})

# ======================================================
# Routes
# ======================================================
@app.get('/')
def index():
    return render_template('index.html')

@app.route('/inspect', methods=['GET', 'POST'])
def inspect_config():
    # Handle both GET (URL params) and POST (form data)
    if request.method == 'GET':
        gender = request.args.get('gender', 'male')
        style  = request.args.get('style',  'pop')
        ng     = request.args.get('noise_gate', 'weak')
        bpm    = request.args.get('bpm', type=float, default=None)
        fx_on_raw = request.args.get('fx_on', type=str, default=None)
    else:
        gender = request.form.get('gender', 'male')
        style  = request.form.get('style',  'pop')
        ng     = request.form.get('noise_gate', 'weak')
        bpm    = request.form.get('bpm', type=float, default=None)
        fx_on_raw = request.form.get('fx_on', type=str, default=None)

    try:
        fx_map = json.loads(fx_on_raw) if fx_on_raw else None
    except Exception:
        fx_map = None
    return jsonify(config_snapshot(gender, style, ng, bpm, fx_on=fx_map))

@app.post('/process')
def process_audio():
    t0 = time.perf_counter()  # ★開始時間

    # ここに追加
    print(f"[Debug] === PROCESS START ===")

    # Add memory monitoring
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"[Memory] Initial: {initial_memory:.1f}MB")

    fid  = request.form.get('file_id')
    mode = request.form.get('mode', 'download')
    force_stereo = str(request.form.get('force_stereo', '0')).lower() in ('1','true','yes')

    print(f"[Process] fid={fid}, mode={mode}, force_stereo={force_stereo}")
    
    # さらに追加
    gender  = request.form.get('gender', 'male')
    style   = request.form.get('style',  'pop')
    reverb_offset_db = request.form.get('reverb_offset_db', type=float, default=None)
    print(f"[Debug] Received params: gender={gender}, style={style}, reverb_offset_db={reverb_offset_db}")

    fid  = request.form.get('file_id')
    mode = request.form.get('mode', 'download')
    force_stereo = str(request.form.get('force_stereo', '0')).lower() in ('1','true','yes')

    # === プレビューキャッシュヒット確認 ===
    if fid and fid in FILES:
        gender  = request.form.get('gender', 'male')
        style   = request.form.get('style',  'pop')
        ng_mode = request.form.get('noise_gate', 'weak')
        bpm     = request.form.get('bpm', type=float, default=None)
        
        # fx_on と オフセット値を先に取得
        fx_on_raw = request.form.get('fx_on', type=str, default=None)
        try:
            fxwant = json.loads(fx_on_raw) if fx_on_raw else {}
        except Exception:
            fxwant = {}
        
        reverb_offset_db = request.form.get('reverb_offset_db', type=float, default=0.0)
        delay_offset_db = request.form.get('delay_offset_db', type=float, default=0.0)
        comp2_offset_db = request.form.get('comp2_offset_db', type=float, default=0.0)  # ← 追加
        sat_amount = request.form.get('sat_amount', type=float, default=None)
    
        
        # 正しいキャッシュキーを生成
        cache_key = make_render_cache_key(gender, style, ng_mode, fxwant, bpm, mode, 
                                         reverb_offset_db or 0.0, delay_offset_db or 0.0,
                                         comp2_offset_db or 0.0, sat_amount)

        render_cache = FILES[fid].setdefault("render_cache", {})
        if mode == "preview" and cache_key in render_cache:
            cached_bytes = render_cache[cache_key]
            print(f"[Cache hit] Returning cached preview: {cache_key}")
            return send_file(io.BytesIO(cached_bytes), mimetype="audio/wav")

    if fid and fid in FILES:
        if mode == 'preview':
            src_audio, src_sr = FILES[fid]['preview_mono']
        else:
            # Load from temporary file for full quality processing
            temp_path = FILES[fid]['temp_path']
            print(f"[Memory] Before audio load: {process.memory_info().rss / 1024 / 1024:.1f}MB")

            # Check file size and use chunk processing for very large files
            file_size = os.path.getsize(temp_path)
            file_size_mb = file_size / (1024 * 1024)

            if file_size_mb > 1.5:  # For files >1.5MB, use emergency fallback
                print(f"[Emergency Fallback] Very large file detected: {file_size_mb:.1f}MB, returning original file")

                # Emergency: Just return the original file without processing to avoid SIGKILL
                resp = send_file(temp_path, mimetype='audio/wav', as_attachment=True, download_name='MixMate_out.wav')

                # Add basic headers
                resp.headers['X-EMERGENCY-FALLBACK'] = 'true'
                resp.headers['X-FILE-SIZE-MB'] = f'{file_size_mb:.1f}'

                return resp

            elif file_size_mb > 0.5:  # For files 0.5-1.5MB, use chunk processing
                print(f"[Chunk Processing] Large file detected: {file_size_mb:.1f}MB, using chunk processing")

                # Get audio info first
                with sf.SoundFile(temp_path) as f:
                    src_sr = f.samplerate
                    total_frames = len(f)
                    channels = f.channels

                # Process in 5-second chunks to limit memory usage
                chunk_size = int(src_sr * 5)  # 5 seconds per chunk
                num_chunks = (total_frames + chunk_size - 1) // chunk_size

                print(f"[Chunk Processing] Processing {num_chunks} chunks of {chunk_size} frames each")

                # Create temporary output file for chunk processing
                output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                output_temp_path = output_temp.name
                output_temp.close()

                # Process chunks and write directly to output file
                with sf.SoundFile(temp_path) as input_file:
                    with sf.SoundFile(output_temp_path, 'w', samplerate=src_sr, channels=2, format='WAV', subtype='PCM_16') as output_file:
                        for chunk_idx in range(num_chunks):
                            start_frame = chunk_idx * chunk_size
                            frames_to_read = min(chunk_size, total_frames - start_frame)

                            # Read chunk
                            input_file.seek(start_frame)
                            chunk_audio = input_file.read(frames_to_read, dtype='float32', always_2d=True)
                            chunk_audio = to_stereo(chunk_audio)

                            print(f"[Chunk {chunk_idx+1}/{num_chunks}] Processing {frames_to_read} frames, Memory: {process.memory_info().rss / 1024 / 1024:.1f}MB")

                            # Apply minimal processing to chunk (just ensure stereo and basic limiting)
                            chunk_audio = np.clip(chunk_audio, -0.95, 0.95)

                            # Write chunk to output
                            output_file.write(chunk_audio)

                            # Force memory cleanup after each chunk
                            del chunk_audio
                            import gc
                            gc.collect()

                # Return response directly from chunk-processed file
                print(f"[Chunk Processing] Completed, final memory: {process.memory_info().rss / 1024 / 1024:.1f}MB")
                resp = send_file(output_temp_path, mimetype='audio/wav', as_attachment=True, download_name='MixMate_out.wav')

                # Schedule cleanup
                import atexit
                atexit.register(lambda: os.remove(output_temp_path) if os.path.exists(output_temp_path) else None)

                return resp
            else:
                # Normal processing for smaller files
                src_audio, src_sr = sf.read(temp_path, dtype='float32', always_2d=True)
                print(f"[Memory] After audio load: {process.memory_info().rss / 1024 / 1024:.1f}MB")
                src_audio = to_stereo(src_audio)
                print(f"[Memory] After stereo conversion: {process.memory_info().rss / 1024 / 1024:.1f}MB")
    else:
        if 'file' not in request.files:
            return abort(400, 'file or file_id is required')
        data = request.files['file'].read()
        if not data:
            return abort(400, 'empty file')
        a, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
        a = to_stereo(a)
        if mode == 'preview':
            if force_stereo:
                src_audio, src_sr = _half_downsample(a), sr // 2
            else:
                a_mono = to_mono_mid(a)
                src_audio, src_sr = _half_downsample(a_mono), sr // 2
        else:
            src_audio, src_sr = (a, sr)

    gender  = request.form.get('gender', 'male')
    style   = request.form.get('style',  'pop')
    ng_mode = request.form.get('noise_gate', 'weak')
    bpm     = request.form.get('bpm', type=float, default=None)

    fx_on_raw = request.form.get('fx_on', type=str, default=None)
    try:
        fxwant = json.loads(fx_on_raw) if fx_on_raw else {}
    except Exception:
        fxwant = {}

    try:
        print(f"[Debug FX] fxwant={fxwant}")
    except Exception:
        pass
    
    sat_amount = request.form.get('sat_amount', type=float, default=None)
    print(f"[Debug Python] Received sat_amount: {sat_amount}, type: {type(sat_amount)}")
    print(f"[Debug Python] All form data keys: {list(request.form.keys())}")
    comp2_offset_db = request.form.get('comp2_offset_db', type=float, default=None)

    def want(k: str, default: bool=True) -> bool:
        return bool(fxwant.get(k, default))
    
    # ノイズゲート設定の詳細ログ
    print(f"[Debug Noise Gate] ng_mode='{ng_mode}', want_ng={want('noise_gate', True)}")
    print(f"[Debug Noise Gate] Input audio peak: {peak_db(src_audio):.2f} dBFS")
    
    # ★修正：サチュレーション情報を含むキャッシュキー用にfxwantを調整
    # サチュレーションはドライチェーン後で個別処理するが、キャッシュ効率のため情報は保持
    fxwant_for_cache = {**fxwant, 'saturation': False}  # ドライチェーンではサチュレーション無効
    
    # ★修正：sat_amountをキャッシュ考慮要素として追加
    x, gr1, gr2 = build_dry_chain_simple_cache(
        fid, src_audio, src_sr, gender, style, ng_mode, fxwant_for_cache, sat_amount, comp2_offset_db
    )

    # ノイズゲート効果の測定
    if want('noise_gate', True):
        try:
            ng_input_peak = peak_db(src_audio)
            ng_output_peak = peak_db(x)
            ng_effect = ng_input_peak - ng_output_peak
            print(f"[Debug Noise Gate] Effect measurement:")
            print(f"[Debug Noise Gate]   Input peak:  {ng_input_peak:.2f} dBFS")
            print(f"[Debug Noise Gate]   Output peak: {ng_output_peak:.2f} dBFS")
            print(f"[Debug Noise Gate]   Level change: {ng_effect:+.2f} dB")
            
            # 静音部分での効果確認（最初の0.1秒をサンプル）
            sample_len = min(int(src_sr * 0.1), len(src_audio))
            if sample_len > 0:
                sample_input = src_audio[:sample_len]
                sample_output = x[:sample_len]
                sample_input_rms = np.sqrt(np.mean(sample_input ** 2))
                sample_output_rms = np.sqrt(np.mean(sample_output ** 2))
                sample_input_db = 20 * np.log10(sample_input_rms + 1e-12)
                sample_output_db = 20 * np.log10(sample_output_rms + 1e-12)
                print(f"[Debug Noise Gate]   Sample (0.1s) input RMS:  {sample_input_db:.2f} dBFS")
                print(f"[Debug Noise Gate]   Sample (0.1s) output RMS: {sample_output_db:.2f} dBFS")
        except Exception as e:
            print(f"[Debug Noise Gate] Measurement failed: {e}")

    # サチュレーション処理のデバッグ情報
    print(f"[Debug] Saturation check: sat_amount={sat_amount}, style={style}, want_saturation={want('saturation', True)}")
    
    # 個別サチュレーション処理（統合チェーン後）
    if want('saturation', True):
        if sat_amount is not None and sat_amount > 0.0:
            print(f"[Debug] Applying user-controlled saturation: amount={sat_amount}")
            x_sat = apply_saturation(x, src_sr, style)
            x = np.clip((1.0 - sat_amount) * x + sat_amount * x_sat, -1.0, 1.0)
            print(f"[Debug] User saturation applied: clean={1.0-sat_amount:.1f}, saturated={sat_amount:.1f}")
        else:
            # sat_amountが指定されていない場合はスタイル固有の標準量を適用
            print(f"[Debug] Applying default saturation for style: {style}")
            x = apply_saturation(x, src_sr, style)
            print(f"[Debug] Default saturation applied for {style}")
    else:
        print(f"[Debug] Saturation disabled by user")

    # ダブラー設定を取得・判定
    enable_doubler_param = request.form.get('enable_doubler', '0')
    enable_doubler = str(enable_doubler_param).lower() in ('1', 'true', 'yes')

    # コーラス系スタイルで自動有効化
    if style.lower() in ('chorus', 'chorus_wide'):
        enable_doubler = True

    print(f"[Debug] Doubler setting: param={enable_doubler_param}, final={enable_doubler}, style={style}")

    # ダブラー処理
    if enable_doubler:
        try:
            # J-POP/歌ってみた向け改良版
            center_db = -12.0      # より積極的なセンター抑制
            width_pct = 95.0       # ワイドな広がり
            delay_l_ms = 15.0      # 左15ms遅延
            delay_r_ms = 9.0       # 右9ms遅延（6ms差でより明確な分離）
            
            doubler_wet = build_doubler_wet(
                x, src_sr,
                center_db=center_db,
                width_pct=width_pct,
                delay_l_ms=delay_l_ms,
                delay_r_ms=delay_r_ms,
                hpf_hz=150.0
            )
            x = np.clip(x + doubler_wet, -1.0, 1.0)
            print(f"[Debug] Doubler applied: center_db={center_db}, width={width_pct}%, L={delay_l_ms}ms, R={delay_r_ms}ms")
        except Exception as e:
            print(f"[Warning] Doubler failed, skipping: {e}")
    else:
        print(f"[Debug] Doubler skipped for style: {style}")

    x_fx = x.copy()
    
    enable_delay = want('delay', True)
    enable_reverb = want('reverb', True)

    delay_offset_db = request.form.get('delay_offset_db', type=float, default=None)
    reverb_offset_db = request.form.get('reverb_offset_db', type=float, default=None)

    if delay_offset_db is None and reverb_offset_db is None:
        space_db = request.form.get('space_db', type=float, default=0.0)
        delay_offset_db = space_db
        reverb_offset_db = space_db
    else:
        delay_offset_db = 0.0 if delay_offset_db is None else float(delay_offset_db)
        reverb_offset_db = 0.0 if reverb_offset_db is None else float(reverb_offset_db)

    p = style_fx_params(style)

    enable_doubler = want('doubler', False)

    delay_offset_db  = request.form.get('delay_offset_db',  type=float, default=None)
    reverb_offset_db = request.form.get('reverb_offset_db', type=float, default=None)

    if delay_offset_db is None and reverb_offset_db is None:
        space_db = request.form.get('space_db', type=float, default=0.0)
        delay_offset_db  = space_db
        reverb_offset_db = space_db
    else:
        delay_offset_db  = 0.0 if delay_offset_db  is None else float(delay_offset_db)
        reverb_offset_db = 0.0 if reverb_offset_db is None else float(reverb_offset_db)

    p = style_fx_params(style)
    
    # --- リバーブ・ディレイ詳細デバッグ出力 ---
    print(f"[Debug Reverb] Style={style}: pre={p.get('pre')}ms, decay={p.get('decay')}s, "
          f"HPF={p.get('r_hpf')}Hz, LPF={p.get('r_lpf')}Hz, wet_db={p.get('r_wet_db')}dB")
    print(f"[Debug Delay] Style={style}: delay_ms={p.get('delay_ms')}, delay_frac={p.get('delay_frac')}, "
          f"feedback={p.get('fb')}, duck={p.get('dduck')}, HPF={p.get('d_hpf')}Hz, LPF={p.get('d_lpf')}Hz")

    rev_wet, delay_base = get_space_wets_cached(fid, x_fx, src_sr, style, bpm)

    duck_strength = str(p.get('dduck', 'mid'))
    delay_ducked = apply_duck(delay_base, x_fx, src_sr, duck_strength) if enable_delay else np.zeros_like(x_fx)

    if enable_reverb:
        base_wet = float(p.get('r_wet_db', -14.0))
        ui_off   = float(reverb_offset_db)

        # UI ±4 dB → 内部 ±4 dB（1:1の直接適用）
        eff_off  = ui_off
        final_wet = base_wet + eff_off
        final_wet = max(-40.0, min(-10.0, final_wet))

        r_gain = db_to_lin(final_wet)
        print(f"[Debug] Reverb calc: style={style}, base={base_wet}dB, ui_off={ui_off}dB, eff_off={eff_off}dB, final={final_wet}dB, gain={r_gain:.4f}")
        x_fx = np.clip(x_fx + r_gain * rev_wet, -1.0, 1.0)

    # --- Delay（ms/frac どちらでも有効）＋プリセット対応 ---
    print(f"[Debug] Delay params: enable_delay={enable_delay}, has_ms={p.get('delay_ms') is not None}, has_frac={p.get('delay_frac') is not None}")
    print(f"[Debug] Delay_base peak(before duck) = {peak_db(delay_base):.1f} dBFS")

    print(f"[Debug] Delay_base peak(before duck) = {peak_db(delay_base):.1f} dBFS")
    if enable_delay and ((p.get('delay_frac') is not None) or (p.get('delay_ms') is not None)):
        ui_off = float(delay_offset_db)
        eff_off = 1.5 * ui_off   # UI ±4 dB → 内部 ±6 dB（明確な変化のため）
        
        # スタイル別ディレイベースレベル（右カラム「標準」の位置）
        s = (style or 'pop').lower()
        if s == 'rock':
            style_base = -16.0
        elif s == 'pop':
            style_base = -20.0
        elif s == 'ballad':
            style_base = -22.0
        elif s in ('chorus', 'chorus_wide'):
            style_base = -24.0
        elif s == 'narration':
            style_base = -30.0
        else:
            style_base = -24.0
            
        final_delay_db = style_base + eff_off
        final_delay_db = max(-40.0, min(-8.0, final_delay_db))  # 制限範囲
        
        d_gain = db_to_lin(final_delay_db)
        print(f"[Debug] Delay calc: style={style}, base={style_base}dB, ui_off={ui_off}dB, final={final_delay_db}dB, gain={d_gain:.4f}")
        
        x_fx = np.clip(x_fx + d_gain * delay_ducked, -1.0, 1.0)
        print(f"[Debug] Delay mix peak(after) = {peak_db(d_gain * delay_ducked):.1f} dBFS")

    # === 修正：確実な安全処理 ===
    if want('limiter', True):
        # 確実な-3.0 dBFS制限
        safety_threshold = db_to_lin(-3.0)  # 0.708
        pre_peak = peak_db(x_fx)
        
        # 安全な制限処理
        final = np.clip(x_fx, -safety_threshold, safety_threshold)
        tp_db = peak_db(final)
        
        print(f"[Safety Clip] Pre: {pre_peak:.2f} dBFS, Post: {tp_db:.2f} dBFS")
        
        if pre_peak > -3.0:
            print(f"[Safety Clip] ACTIVE - Clipped {pre_peak - tp_db:.2f} dB")
        else:
            print(f"[Safety Clip] TRANSPARENT - No clipping needed")
            
    else:
        final = np.clip(x_fx, -0.85, 0.85)
        tp_db = peak_db(final)
        print(f"[Hard Clipping] Peak limited to: {tp_db:.2f} dBFS")

    # Memory checkpoint at 90% completion
    memory_90 = process.memory_info().rss / 1024 / 1024
    print(f"[Memory] At 90% completion: {memory_90:.1f}MB")

    # 参考値としてのLUFS測定（強制調整はしない）
    if mode == 'preview':
        lufs_out = float('nan')  # プレビューでは省略
    else:
        try:
            meter = pyln.Meter(src_sr)
            lufs_out = float(meter.integrated_loudness(final))
        except:
            lufs_out = float('nan')

    # For large files, use temporary file to avoid memory issues
    file_size_mb = len(final) * final.itemsize / (1024 * 1024)
    use_temp_file = file_size_mb > 1.0  # Use temp file for >1MB processed audio

    if use_temp_file:
        # Use temporary file for large audio
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        output_temp_path = output_temp.name
        output_temp.close()

        print(f"[Large Audio] Using temp file for {file_size_mb:.1f}MB processed audio")
        print(f"[Memory] Before file write: {process.memory_info().rss / 1024 / 1024:.1f}MB")

        if mode == 'preview':
            sf.write(output_temp_path, final, src_sr, format='WAV', subtype='FLOAT')
        else:
            sf.write(output_temp_path, final, src_sr, format='WAV', subtype='PCM_16')

        print(f"[Memory] After file write: {process.memory_info().rss / 1024 / 1024:.1f}MB")
        buf = None  # Will use file path instead
    else:
        # Use memory buffer for smaller files
        buf = io.BytesIO()
        if mode == 'preview':
            sf.write(buf, final, src_sr, format='WAV', subtype='FLOAT')
        else:
            sf.write(buf, final, src_sr, format='WAV', subtype='PCM_16')
        buf.seek(0)
        output_temp_path = None
    # === プレビュー結果キャッシュ保存 ===
    if mode == "preview" and fid and fid in FILES:
        render_cache = FILES[fid].setdefault("render_cache", {})
        # cache_keyは既に正しく計算済み
        if use_temp_file:
            # For temp files, read the content to cache
            with open(output_temp_path, 'rb') as f:
                render_cache[cache_key] = f.read()
        else:
            render_cache[cache_key] = buf.getvalue()

    if mode == 'preview':
        if use_temp_file:
            resp = send_file(output_temp_path, mimetype='audio/wav')
        else:
            resp = send_file(io.BytesIO(render_cache[cache_key]), mimetype='audio/wav')
    else:
        # PyWebView環境の検出
        is_pywebview = request.headers.get('User-Agent', '').find('pywebview') != -1

        if is_pywebview:
            # PyWebView環境の場合はBase64エンコードしてJSONで返す
            import base64
            if use_temp_file:
                with open(output_temp_path, 'rb') as f:
                    audio_data = f.read()
            else:
                audio_data = buf.getvalue()
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')

            resp = jsonify({
                'status': 'success',
                'audio_b64': audio_b64,
                'filename': 'MixMate_out.wav',
                'pywebview': True
            })
        else:
            # 通常のブラウザ環境の場合は従来通り
            if use_temp_file:
                resp = send_file(output_temp_path, mimetype='audio/wav', as_attachment=True, download_name='MixMate_out.wav')
            else:
                resp = send_file(buf, mimetype='audio/wav', as_attachment=True, download_name='MixMate_out.wav')

    resp.headers['X-GR1-PEAK-DB'] = f'{gr1:.2f}'
    resp.headers['X-GR2-PEAK-DB'] = f'{gr2:.2f}'
    resp.headers['X-LUFS']        = 'NaN' if (mode == 'preview') else f'{lufs_out:.2f}'
    resp.headers['X-TP-DB']       = f'{tp_db:.2f}'

    if fid and fid in FILES:
        tms  = FILES[fid].get('last_wets_time_ms')
        flag = FILES[fid].get('last_wets_cache')
        if tms is not None:
            resp.headers['X-WETS-MS'] = str(tms)
        if flag:
            resp.headers['X-WETS-CACHE'] = flag

    t1 = time.perf_counter()  # ★終了時間
    print(f"[Profiler] /process: {(t1 - t0) * 1000:.1f}ms")

    # Clean up temporary files
    if mode == 'download' and fid:
        cleanup_temp_file(fid)

    # Clean up output temp file (after response is sent)
    if use_temp_file and output_temp_path and os.path.exists(output_temp_path):
        try:
            # For preview mode, keep temp file for potential reuse
            # For download mode, schedule cleanup after response
            if mode == 'download':
                import atexit
                atexit.register(lambda: os.remove(output_temp_path) if os.path.exists(output_temp_path) else None)
        except Exception as e:
            print(f"[Cleanup] Warning: Could not schedule temp file cleanup: {e}")

    return resp

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)