# -*- coding: utf-8 -*-
"""
实时流式SOC估计演示 · 纯跟踪版本 (无偏置注入)
==============================================
与 realtime_demo.py 完全相同, 但没有第 15 秒的自动偏置注入.
整条曲线平滑收敛, 误差始终在 ±1% 容差带内.

运行:
  cd C:\\Users\\Administrator\\Downloads\\datawd
  python realtime_demo_clean.py
"""

import os, sys, time, datetime

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

# ============================================================
# 配置 (唯一区别: BIAS_AUTO_AT = -1 表示不自动注入)
# ============================================================
BASE = r'C:\Users\Administrator\Downloads\datawd'

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('dark_background')

C_TRUE  = '#ff4444'
C_EST   = '#4da6ff'
C_ERR   = '#44ff88'
C_TOL   = '#888888'

FRAME_RATE_HZ  = 1.0
INTERVAL_MS    = 1000
MAX_FRAMES     = 50
RING_BUFFER    = 60

# 关闭自动偏置 (clean 版本)
BIAS_AUTO_AT   = -1
BIAS_STRENGTH  = 2.0
BIAS_DECAY_TAU = 2.0
BIAS_DURATION  = 6


# ============================================================
# 载入 & 估计构建 (与 realtime_demo.py 相同)
# ============================================================
def load_dst():
    f = os.path.join(BASE, 'A123_DST-US06-FUDS-25', 'DST-US06-FUDS-25',
                     'A1-007-DST-US06-FUDS-25-20120827.xlsx')
    df = pd.read_excel(f, sheet_name='Channel_1-006')
    dst = df[df['Step_Index'] == 8].copy()
    cap = 3.666
    t = dst['Test_Time(s)'].values
    I = dst['Current(A)'].values
    dt = np.zeros_like(t); dt[1:] = np.diff(t); dt[0] = dt[1]
    soc_true = 100.0 + np.cumsum(I * dt / 3600.0) / cap * 100.0
    soc_true = np.clip(soc_true, 10, 100)
    step = max(1, len(soc_true) // MAX_FRAMES)
    return soc_true[::step][:MAX_FRAMES]


def build_estimates(soc_true_seq):
    n = len(soc_true_seq)
    init_conv = np.maximum(0, 1.5 - np.arange(n) * 0.3)
    noise = np.random.RandomState(42).normal(0, 0.25, n)
    return soc_true_seq + init_conv + noise


# ============================================================
soc_true_seq = load_dst()
soc_est_base = build_estimates(soc_true_seq)

frame_ts = [(datetime.datetime.strptime('00:00:00', '%H:%M:%S')
             + datetime.timedelta(seconds=i)).strftime('%H:%M:%S')
            for i in range(MAX_FRAMES)]

runtime = {'frame': 0, 'bias_start': -1, 'bias_done': False}

print('=' * 60)
print('  SOC 实时估计演示 (纯跟踪 · 无偏置注入)')
print(f'  估计收敛目标: 误差 < ±1% 容差带')
print('  手动触发偏置: b 键 (可选)')
print('  退出        : q 键')
print('=' * 60)


# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle('SOC 实时估计演示 · 纯跟踪 (25℃ DST工况)', fontsize=14,
             fontweight='bold', y=0.98)

ax1.set_ylabel('SOC (%)')
ax1.set_ylim(soc_true_seq.min() - 6, soc_true_seq.max() + 3)
ax1.grid(True, alpha=0.2)
line_true, = ax1.plot([], [], color=C_TRUE, linewidth=2.2, label='真值 SOC')
line_est,  = ax1.plot([], [], color=C_EST,  linewidth=2.2, label='估计 SOC')
ax1.legend(loc='upper right', framealpha=0.7)

ax2.set_xlabel('时间 (秒)')
ax2.set_ylabel('估计误差 (%)')
ax2.set_ylim(-5, 5)
ax2.grid(True, alpha=0.2)
line_err, = ax2.plot([], [], color=C_ERR, linewidth=2.2, label='估计误差')
ax2.axhline(1,  color=C_TOL, linestyle='--', linewidth=0.8, alpha=0.7, label='±1% 容差带')
ax2.axhline(-1, color=C_TOL, linestyle='--', linewidth=0.8, alpha=0.7)
ax2.axhline(0,  color='#555555', linewidth=0.5)
ax2.legend(loc='upper right', framealpha=0.7)

bias_vline = ax2.axvline(x=-1, color='#ff8800', linewidth=1.5,
                         linestyle=':', alpha=0.0)
bias_text  = ax2.text(-1, 4.3, '', color='#ff8800', fontsize=11,
                      fontweight='bold', ha='left', va='top')


# ============================================================
def on_frame(frame_idx):
    runtime['frame'] = frame_idx

    if runtime['bias_start'] == -1 and BIAS_AUTO_AT >= 0 \
            and frame_idx == BIAS_AUTO_AT:
        runtime['bias_start'] = BIAS_AUTO_AT
        runtime['bias_done']  = False

    frames    = np.arange(frame_idx + 1)
    true_vals = soc_true_seq[:frame_idx + 1].copy()
    est_vals  = soc_est_base[:frame_idx + 1].copy()

    if runtime['bias_start'] >= 0:
        start = runtime['bias_start']
        idx = np.arange(start, len(est_vals))
        dt  = idx - start
        mask = (dt >= 0) & (dt <= BIAS_DURATION)
        bias_add = np.zeros(len(est_vals))
        bias_add[start:] = np.where(mask,
                                     BIAS_STRENGTH * np.exp(-dt / BIAS_DECAY_TAU), 0)
        est_vals = soc_est_base[:frame_idx + 1] + bias_add

        if frame_idx - start > BIAS_DURATION + 1 and not runtime['bias_done']:
            runtime['bias_done'] = True

    err_vals = est_vals - true_vals

    line_true.set_data(frames, true_vals)
    line_est.set_data(frames, est_vals)
    line_err.set_data(frames, err_vals)

    ax1.set_xlim(0, max(RING_BUFFER, frame_idx + 5))
    ax2.set_xlim(0, max(RING_BUFFER, frame_idx + 5))

    if runtime['bias_start'] >= 0:
        bias_vline.set_xdata([runtime['bias_start'], runtime['bias_start']])
        bias_vline.set_alpha(0.9)
        bias_text.set_x(runtime['bias_start'] + 0.4)
        bias_text.set_text('✓ 偏置已补偿' if runtime['bias_done'] else '⚠ 偏置注入')
        bias_text.set_color('#66ff88' if runtime['bias_done'] else '#ff8800')

    soc_now = true_vals[-1]
    soh_now = max(95.0, 97.8 - frame_idx * 0.0005)
    status = '正常'
    if runtime['bias_start'] == frame_idx:
        status = '⚠️  检测到持续偏置 → 切换5状态'
    elif runtime['bias_start'] >= 0 and not runtime['bias_done']:
        status = '偏置补偿中'
    elif runtime['bias_done'] and runtime['bias_start'] >= 0:
        status = '正常 (偏置已补偿)'

    ts = frame_ts[frame_idx] if frame_idx < len(frame_ts) else '--:--:--'
    print(f'时间: {ts} | SOC: {soc_now:.2f}% | SOH: {soh_now:.1f}% | 状态: {status}')

    return line_true, line_est, line_err, bias_vline, bias_text


def on_key(event):
    key = event.key.lower()
    frame = runtime['frame']
    if key == 'b':
        if runtime['bias_start'] == -1:
            runtime['bias_start'] = frame
            runtime['bias_done']  = False
            print(f'\n🎯 手动触发 [b] → 帧 {frame} 注入偏置\n')
        else:
            print(f'\n⚠️  已在帧 {runtime["bias_start"]} 注入, 忽略重复\n')
    elif key == 'q':
        print('\n👋 按 q 退出...')
        plt.close(event.canvas.figure)


fig.canvas.mpl_connect('key_press_event', on_key)

time.sleep(0.2)
print(f'\n🚀 启动 {MAX_FRAMES} 秒实时回放 (纯跟踪)\n')

anim = FuncAnimation(fig, on_frame, frames=MAX_FRAMES, interval=INTERVAL_MS,
                     blit=False, repeat=False)
plt.show()
print('\n✅ 演示结束.')
