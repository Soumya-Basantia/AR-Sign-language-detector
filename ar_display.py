"""
ARDisplay — Real-time AR glasses HUD simulation over an OpenCV frame.

Two render modes toggled by pressing G:
  MODE 0 — Normal UI  : clean dark overlay with structured panels
  MODE 1 — AR Glasses : minimal futuristic floating HUD elements

All drawing uses only OpenCV + numpy (no extra dependencies).
"""

import time
import math
import numpy as np
import cv2
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
C_GREEN      = (80, 230, 120)
C_GREEN_GLOW = (40, 180, 80)
C_ORANGE     = (40, 165, 255)
C_RED        = (60, 60, 255)
C_CYAN       = (230, 210, 50)
C_WHITE      = (240, 240, 240)
C_DARK       = (10, 10, 10)
C_PANEL_BG   = (18, 18, 24)
C_ACCENT     = (0, 200, 160)          # teal accent
C_DIM        = (100, 100, 100)

FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL  = cv2.FONT_HERSHEY_SIMPLEX


def _conf_color(conf: float) -> Tuple[int, int, int]:
    if conf >= 0.80:
        return C_GREEN
    if conf >= 0.60:
        return C_ORANGE
    return C_RED


def _alpha_blend(base: np.ndarray, overlay: np.ndarray,
                 alpha: float) -> np.ndarray:
    """Blend overlay onto base with given alpha (0–1)."""
    return cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)


def _draw_glow_text(img: np.ndarray, text: str, org: Tuple[int, int],
                    font_scale: float, color: Tuple[int, int, int],
                    thickness: int = 2, glow_radius: int = 3):
    """Simulate a soft glow by drawing the text multiple times slightly offset."""
    glow_color = tuple(max(0, int(c * 0.4)) for c in color)
    for dx in range(-glow_radius, glow_radius + 1, glow_radius):
        for dy in range(-glow_radius, glow_radius + 1, glow_radius):
            if dx == 0 and dy == 0:
                continue
            cv2.putText(img, text,
                        (org[0] + dx, org[1] + dy),
                        FONT, font_scale, glow_color,
                        thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, font_scale, color,
                thickness, cv2.LINE_AA)


def _rounded_rect(img: np.ndarray, x: int, y: int, w: int, h: int,
                  color: Tuple[int, int, int], radius: int = 10,
                  alpha: float = 0.7):
    """Draw a filled rounded rectangle with transparency."""
    overlay = img.copy()
    # Main fill
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
    # Corners
    for cx, cy in [(x + radius, y + radius), (x + w - radius, y + radius),
                   (x + radius, y + h - radius), (x + w - radius, y + h - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _text_size(text: str, scale: float, thickness: int = 2):
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


# ---------------------------------------------------------------------------
# Fade / animation helpers
# ---------------------------------------------------------------------------

class FadeText:
    """A text element that fades in and (optionally) out."""

    def __init__(self, fade_in: float = 0.3, hold: float = 2.0,
                 fade_out: float = 0.5):
        self._fade_in = fade_in
        self._hold = hold
        self._fade_out = fade_out
        self._start: float = 0.0
        self._text: str = ""
        self._active: bool = False

    def trigger(self, text: str):
        self._text = text
        self._start = time.time()
        self._active = True

    @property
    def alpha(self) -> float:
        if not self._active:
            return 0.0
        elapsed = time.time() - self._start
        if elapsed < self._fade_in:
            return elapsed / self._fade_in
        if elapsed < self._fade_in + self._hold:
            return 1.0
        fade_elapsed = elapsed - self._fade_in - self._hold
        if fade_elapsed < self._fade_out:
            return 1.0 - fade_elapsed / self._fade_out
        self._active = False
        return 0.0

    @property
    def text(self) -> str:
        return self._text

    @property
    def is_active(self) -> bool:
        return self._active or self.alpha > 0.0


# ---------------------------------------------------------------------------
# Main ARDisplay class
# ---------------------------------------------------------------------------

class ARDisplay:
    """
    Overlay renderer.  Call `render(frame, state_dict)` every frame.

    state_dict keys
    ---------------
    sentence        : str   — finalized / building sentence
    current_pred    : str   — current gesture label
    confidence      : float — prediction confidence 0–1
    status          : str   — "READY" / "STABILIZING" / "LOW_CONF" / "NO HANDS"
    mode            : str   — "WORD" or "LETTER"
    letter_buffer   : str   — e.g. "H E L P"
    fps             : float
    ar_mode         : bool  — True = AR glasses style
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self._w = width
        self._h = height
        self._ar_mode: bool = False

        # Animation state
        self._sentence_fade = FadeText(fade_in=0.2, hold=999, fade_out=0.3)
        self._status_fade   = FadeText(fade_in=0.15, hold=2.0, fade_out=0.4)
        self._last_sentence: str = ""
        self._last_status: str = ""

        # Scanline / pulse animation
        self._t0 = time.time()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def toggle_mode(self):
        self._ar_mode = not self._ar_mode

    @property
    def ar_mode(self) -> bool:
        return self._ar_mode

    def render(self, frame: np.ndarray, state: dict) -> np.ndarray:
        """Render HUD onto frame (in-place + returned)."""
        h, w = frame.shape[:2]
        self._w, self._h = w, h

        sentence    = state.get("sentence", "")
        pred        = state.get("current_pred", "")
        conf        = float(state.get("confidence", 0.0))
        status      = state.get("status", "")
        mode        = state.get("mode", "WORD")
        letter_buf  = state.get("letter_buffer", "")
        fps         = float(state.get("fps", 0.0))

        # Trigger animations on changes
        if sentence != self._last_sentence:
            self._sentence_fade.trigger(sentence)
            self._last_sentence = sentence
        if status != self._last_status:
            self._status_fade.trigger(status)
            self._last_status = status

        if self._ar_mode:
            return self._render_ar(frame, sentence, pred, conf,
                                   status, mode, letter_buf, fps)
        else:
            return self._render_normal(frame, state, sentence, pred, conf,
                                       status, mode, letter_buf, fps)

    # ------------------------------------------------------------------
    # Normal UI Mode
    # ------------------------------------------------------------------

    def _render_normal(self, frame, state, sentence, pred, conf,
                       status, mode, letter_buf, fps):
        out = frame.copy()
        h, w = out.shape[:2]
        now = time.time() - self._t0

        # ── Top bar ────────────────────────────────────────────────────
        _rounded_rect(out, 0, 0, w, 52, C_PANEL_BG, radius=0, alpha=0.85)
        cv2.line(out, (0, 52), (w, 52), C_ACCENT, 1)

        # Logo / title
        _draw_glow_text(out, "SIGN COMM", (14, 36), 0.85, C_ACCENT, 2, 2)

        # FPS
        fps_txt = f"FPS: {fps:.0f}"
        fw, _ = _text_size(fps_txt, 0.55)
        cv2.putText(out, fps_txt, (w - fw - 16, 34),
                    FONT_SMALL, 0.55, C_DIM, 1, cv2.LINE_AA)

        # Mode badge
        mode_txt = f"MODE: {mode}"
        mw, _ = _text_size(mode_txt, 0.6)
        mx = w // 2 - mw // 2
        _rounded_rect(out, mx - 8, 10, mw + 16, 32,
                      (30, 80, 50) if mode == "WORD" else (60, 30, 80),
                      radius=6, alpha=0.9)
        cv2.putText(out, mode_txt, (mx, 34),
                    FONT, 0.6, C_WHITE, 1, cv2.LINE_AA)

        # ── Right panel ────────────────────────────────────────────────
        panel_w = 280
        px = w - panel_w - 8
        _rounded_rect(out, px, 60, panel_w, h - 130, C_PANEL_BG,
                      radius=8, alpha=0.80)
        cv2.line(out, (px, 60), (px, h - 70), C_ACCENT, 1)

        # Current prediction
        color = _conf_color(conf)
        cv2.putText(out, "CURRENT GESTURE", (px + 10, 88),
                    FONT_SMALL, 0.45, C_DIM, 1, cv2.LINE_AA)
        _draw_glow_text(out, pred or "---", (px + 10, 128),
                        1.1, color, 2, 3)
        cv2.putText(out, f"{conf*100:.0f}% confidence",
                    (px + 10, 152), FONT_SMALL, 0.52, color, 1, cv2.LINE_AA)

        # Confidence bar
        bar_x, bar_y = px + 10, 162
        bar_w = panel_w - 20
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10),
                      (40, 40, 40), -1)
        fill = int(bar_w * conf)
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill, bar_y + 10),
                      color, -1)

        # Status
        cv2.putText(out, "STATUS", (px + 10, 196),
                    FONT_SMALL, 0.45, C_DIM, 1, cv2.LINE_AA)
        s_col = {
            "READY": C_GREEN, "STABILIZING": C_ORANGE,
            "LOW_CONF": C_RED, "NO HANDS": C_RED,
            "COOLDOWN": C_CYAN,
        }.get(status, C_WHITE)
        cv2.putText(out, status or "---", (px + 10, 222),
                    FONT, 0.65, s_col, 1, cv2.LINE_AA)

        # Pulse dot
        pulse = 0.5 + 0.5 * math.sin(now * 4)
        dot_r = int(5 + 3 * pulse)
        dot_color = s_col if status == "READY" else C_DIM
        cv2.circle(out, (px + panel_w - 20, 218), dot_r, dot_color, -1)

        # Letter buffer
        if letter_buf:
            cv2.putText(out, "LETTER BUFFER", (px + 10, 256),
                        FONT_SMALL, 0.45, C_DIM, 1, cv2.LINE_AA)
            _draw_glow_text(out, letter_buf, (px + 10, 286),
                            0.7, C_CYAN, 1, 2)

        # Context suggestions
        cv2.putText(out, "SUGGESTIONS", (px + 10, 316),
                    FONT_SMALL, 0.45, C_DIM, 1, cv2.LINE_AA)
        suggestions = state.get("suggestions", [])
        for i, s in enumerate(suggestions[:4]):
            sy = 340 + i * 24
            cv2.putText(out, f"• {s}", (px + 14, sy),
                        FONT_SMALL, 0.48, C_ACCENT, 1, cv2.LINE_AA)

        # Keyboard shortcut legend
        legend = [
            ("SPACE", "Accept word"),
            ("D",     "Delete last"),
            ("R",     "Reset"),
            ("M",     "Toggle mode"),
            ("ENTER", "Finalize"),
            ("G",     "AR mode"),
        ]
        ly_start = h - 64
        cv2.putText(out, "CONTROLS", (px + 10, ly_start - 12),
                    FONT_SMALL, 0.4, C_DIM, 1, cv2.LINE_AA)
        for i, (k, v) in enumerate(legend):
            col = i // 3
            row = i % 3
            lx = px + 10 + col * 130
            ly = ly_start + row * 18
            cv2.putText(out, f"[{k}] {v}", (lx, ly),
                        FONT_SMALL, 0.38, C_DIM, 1, cv2.LINE_AA)

        # ── Bottom sentence bar ────────────────────────────────────────
        bar_h = 68
        _rounded_rect(out, 0, h - bar_h, w - panel_w - 16, bar_h,
                      C_PANEL_BG, radius=0, alpha=0.88)
        cv2.line(out, (0, h - bar_h), (w - panel_w - 16, h - bar_h),
                 C_ACCENT, 1)

        cv2.putText(out, "SENTENCE", (14, h - bar_h + 18),
                    FONT_SMALL, 0.42, C_DIM, 1, cv2.LINE_AA)

        # Truncate sentence for display
        disp_sentence = sentence
        max_chars = (w - panel_w - 40) // 16
        if len(disp_sentence) > max_chars:
            disp_sentence = "…" + disp_sentence[-(max_chars - 1):]

        alpha = self._sentence_fade.alpha
        if alpha > 0:
            overlay = out.copy()
            _draw_glow_text(overlay, disp_sentence or "...",
                            (14, h - 18), 1.0, C_WHITE, 2, 2)
            cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        return out

    # ------------------------------------------------------------------
    # AR Glasses Mode
    # ------------------------------------------------------------------

    def _render_ar(self, frame, sentence, pred, conf,
                   status, mode, letter_buf, fps):
        """Minimal floating HUD — elements appear to float over the scene."""
        out = frame.copy()
        h, w = out.shape[:2]
        now = time.time() - self._t0

        # Subtle vignette
        self._draw_vignette(out)

        # Scanlines (very faint)
        self._draw_scanlines(out, now)

        # ── Mode pill (top-centre) ─────────────────────────────────────
        mode_txt = f"◈  {mode} MODE"
        mw, mh = _text_size(mode_txt, 0.55)
        mx = w // 2 - mw // 2
        _rounded_rect(out, mx - 12, 14, mw + 24, mh + 12,
                      (10, 10, 10), radius=14, alpha=0.65)
        cv2.putText(out, mode_txt, (mx, 14 + mh + 2),
                    FONT, 0.55, C_ACCENT, 1, cv2.LINE_AA)

        # ── Top-right: prediction bubble ──────────────────────────────
        color = _conf_color(conf)
        if pred:
            bubble_txt = f"{pred}  {conf*100:.0f}%"
            bw, bh = _text_size(bubble_txt, 0.75)
            bx = w - bw - 40
            by = 60
            _rounded_rect(out, bx - 12, by - bh - 4, bw + 24, bh + 14,
                          (10, 10, 10), radius=10, alpha=0.6)
            # Corner accent
            cv2.line(out, (bx - 12, by - bh - 4),
                     (bx - 12 + 20, by - bh - 4), color, 2)
            cv2.line(out, (bx - 12, by - bh - 4),
                     (bx - 12, by - bh + 16), color, 2)
            _draw_glow_text(out, bubble_txt, (bx, by),
                            0.75, color, 2, 3)

        # ── Status (top-left) ─────────────────────────────────────────
        s_col = {"READY": C_GREEN, "STABILIZING": C_ORANGE,
                 "LOW_CONF": C_RED, "NO HANDS": C_RED,
                 "COOLDOWN": C_CYAN}.get(status, C_DIM)
        if status:
            # Pulse ring
            pulse = 0.5 + 0.5 * math.sin(now * 5)
            ring_r = int(6 + 3 * pulse)
            cv2.circle(out, (24, 30), ring_r, s_col, 1)
            cv2.circle(out, (24, 30), 4, s_col, -1)
            cv2.putText(out, status, (36, 36),
                        FONT, 0.55, s_col, 1, cv2.LINE_AA)

        # ── FPS (top-left corner, tiny) ───────────────────────────────
        cv2.putText(out, f"{fps:.0f} fps", (10, h - 10),
                    FONT_SMALL, 0.4, C_DIM, 1, cv2.LINE_AA)

        # ── Letter buffer (center-top area) ───────────────────────────
        if letter_buf:
            lb_txt = f"[ {letter_buf} ]"
            lbw, lbh = _text_size(lb_txt, 0.7)
            lbx = w // 2 - lbw // 2
            lby = h // 2 - 60
            _rounded_rect(out, lbx - 10, lby - lbh - 4,
                          lbw + 20, lbh + 14,
                          (5, 5, 5), radius=6, alpha=0.55)
            _draw_glow_text(out, lb_txt, (lbx, lby),
                            0.7, C_CYAN, 1, 2)

        # ── Main subtitle — centre-bottom ─────────────────────────────
        disp = sentence or "..."
        sub_scale = 1.3
        sub_w, sub_h = _text_size(disp, sub_scale, 3)
        # Scale down if too wide
        while sub_w > w - 80 and sub_scale > 0.65:
            sub_scale -= 0.05
            sub_w, sub_h = _text_size(disp, sub_scale, 2)

        sub_x = w // 2 - sub_w // 2
        sub_y = h - 42

        # Floating effect — tiny sine oscillation
        float_dy = int(3 * math.sin(now * 1.2))
        sub_y += float_dy

        alpha = self._sentence_fade.alpha
        if alpha > 0.02:
            # Background pill
            _rounded_rect(out,
                          sub_x - 20, sub_y - sub_h - 10,
                          sub_w + 40, sub_h + 22,
                          (5, 5, 5), radius=14, alpha=0.70 * alpha)
            # Left/right accent lines
            line_col = tuple(int(c * alpha) for c in C_ACCENT)
            cv2.line(out,
                     (sub_x - 20, sub_y - sub_h // 2),
                     (sub_x - 20 + 30, sub_y - sub_h // 2),
                     line_col, 2)
            cv2.line(out,
                     (sub_x + sub_w + 20 - 30, sub_y - sub_h // 2),
                     (sub_x + sub_w + 20, sub_y - sub_h // 2),
                     line_col, 2)

            overlay2 = out.copy()
            _draw_glow_text(overlay2, disp, (sub_x, sub_y),
                            sub_scale, C_WHITE, 2, 4)
            cv2.addWeighted(overlay2, alpha, out, 1 - alpha, 0, out)

        # Corner brackets (AR frame decoration)
        self._draw_corner_brackets(out, w, h)

        return out

    # ------------------------------------------------------------------
    # Visual helpers
    # ------------------------------------------------------------------

    def _draw_vignette(self, img: np.ndarray):
        h, w = img.shape[:2]
        # Build gradient mask
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        mask = np.clip(dist - 0.6, 0, 0.4) / 0.4  # fades edges
        mask = (mask * 120).astype(np.uint8)
        vignette = np.zeros_like(img)
        for c in range(3):
            vignette[:, :, c] = mask
        img -= np.minimum(img, vignette)

    def _draw_scanlines(self, img: np.ndarray, t: float):
        h, w = img.shape[:2]
        offset = int(t * 60) % 4
        for y in range(offset, h, 4):
            img[y, :] = (img[y, :] * 0.88).astype(np.uint8)

    def _draw_corner_brackets(self, img: np.ndarray, w: int, h: int):
        length = 28
        thick = 2
        color = C_ACCENT
        margin = 20
        corners = [
            ((margin, margin),         (1, 1)),
            ((w - margin, margin),     (-1, 1)),
            ((margin, h - margin),     (1, -1)),
            ((w - margin, h - margin), (-1, -1)),
        ]
        for (cx, cy), (dx, dy) in corners:
            cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thick)
            cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thick)
