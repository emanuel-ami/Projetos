import cv2
import numpy as np
import mss
import keyboard
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import threading

# ==========================
# SETTINGS
# ==========================
CONFIG_FILE = "regions.json"
THRESHOLD = 0.4  # Template matching sensitivity (0.0-1.0)
TEMPLATES = {
    "up": "up.png",
    "down": "down.png",
    "left": "left.png",
    "right": "right.png"
}
KEY_MAP = {"up": "w", "down": "s", "left": "a", "right": "d"}

# ==========================
# OPTIMIZATION SETTINGS
# ==========================
MAX_WORKERS = 4                # Parallel template matching threads (reduzido para evitar overhead)
MAX_TEMPLATE_SCALE = 0.25      # Max fraction of frame a template can occupy (auto-resize if larger)
HIT_OFFSET_COMP_MS = -0.4       # Compensate timing (ms): positive = press earlier
PREDICT_WINDOW = 10             # Number of detections to keep for velocity calculation
MIN_VELOCITY_PIX_PER_SEC = 0.5   # Minimum velocity threshold
SHOW_DEBUG = True              # Show debug window (set False for max performance)
PROCESSING_ALPHA = 0.85        # Suavização do delay de processamento

# ==========================
# TEMPLATE SCALE SETTINGS
# ==========================
# Use [ and ] to decrease/increase template scale at runtime.
template_scale = 1.0
scale_step = 0.1
min_scale = 0.2
max_scale = 3.0
_last_scale_update = 0.0
_scale_debounce = 0.2  # seconds

# ==========================
# RECALIBRATION CONTROLS
# ==========================
_last_recalib_update = 0.0
_recalib_debounce = 1.0  # seconds

drawing = False
ix, iy = -1, -1
rect = None
hit_zone = None


# ==========================
# MOUSE DRAW CALLBACKS
# ==========================
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img2 = param.copy()
        cv2.rectangle(img2, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Select Arrow Region", img2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        # Ensure minimum size
        if rect[2] < 10 or rect[3] < 10:
            print("Region too small. Please draw a larger region.")
            rect = None
        cv2.destroyWindow("Select Arrow Region")


def draw_hit_zone(event, x, y, flags, param):
    global ix, iy, drawing, hit_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img2 = param.copy()
        cv2.rectangle(img2, (ix, iy), (x, y), (0, 0, 255), 2)
        cv2.imshow("Select HIT Zone", img2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        hit_zone = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        # Ensure minimum size
        if hit_zone[2] < 10 or hit_zone[3] < 10:
            print("HIT zone too small. Please draw a larger region.")
            hit_zone = None
        cv2.destroyWindow("Select HIT Zone")


# ==========================
# REGION SELECTION
# ==========================
def select_arrow_region():
    """Select only the arrow detection region."""
    global rect
    sct = mss.mss()
    monitor = sct.monitors[1]
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cv2.imshow("Select Arrow Region", frame)
    cv2.setMouseCallback("Select Arrow Region", draw_rectangle, frame)
    cv2.waitKey(0)
    
    if rect:
        print(f"Arrow region updated: {rect}")
        save_config()
        return rect
    else:
        print("Arrow region selection cancelled.")
        return None


def select_hit_zone_only():
    """Select only the HIT zone."""
    global hit_zone
    sct = mss.mss()
    monitor = sct.monitors[1]
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cv2.imshow("Select HIT Zone", frame)
    cv2.setMouseCallback("Select HIT Zone", draw_hit_zone, frame)
    cv2.waitKey(0)
    
    if hit_zone:
        print(f"HIT zone updated: {hit_zone}")
        save_config()
        return hit_zone
    else:
        print("HIT zone selection cancelled.")
        return None


def select_regions():
    global rect, hit_zone
    sct = mss.mss()
    monitor = sct.monitors[1]
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Select arrow region
    cv2.imshow("Select Arrow Region", frame)
    cv2.setMouseCallback("Select Arrow Region", draw_rectangle, frame)
    cv2.waitKey(0)

    # Select hit zone
    cv2.imshow("Select HIT Zone", frame)
    cv2.setMouseCallback("Select HIT Zone", draw_hit_zone, frame)
    cv2.waitKey(0)

    if rect and hit_zone:
        save_config()
        print("Regions saved to", CONFIG_FILE)
        return rect, hit_zone
    else:
        print("Selection incomplete.")
        exit()


def save_config():
    """Save current rect and hit_zone to config file."""
    if rect and hit_zone:
        # Validate before saving
        if rect[2] <= 0 or rect[3] <= 0:
            print(f"Error: Cannot save invalid rect (width or height <= 0): {rect}")
            return
        if hit_zone[2] <= 0 or hit_zone[3] <= 0:
            print(f"Error: Cannot save invalid hit_zone (width or height <= 0): {hit_zone}")
            return
            
        data = {"rect": rect, "hit_zone": hit_zone}
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f)
        print(f"Saved: rect={rect}, hit_zone={hit_zone}")


# ==========================
# LOAD CONFIG
# ==========================
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            loaded_rect = tuple(data["rect"])
            loaded_hit_zone = tuple(data["hit_zone"])
            
            # Validate loaded regions
            if loaded_rect[2] <= 0 or loaded_rect[3] <= 0:
                print(f"Warning: Invalid rect in config (width or height <= 0): {loaded_rect}")
                return None, None
            if loaded_hit_zone[2] <= 0 or loaded_hit_zone[3] <= 0:
                print(f"Warning: Invalid hit_zone in config (width or height <= 0): {loaded_hit_zone}")
                return None, None
                
            return loaded_rect, loaded_hit_zone
        except Exception as e:
            print(f"Error loading config: {e}")
            return None, None
    else:
        return None, None


# ==========================
# VALIDATION HELPERS
# ==========================
def validate_region(region, monitor_bounds):
    """Validate that region has positive dimensions and is within monitor bounds."""
    if not region or len(region) != 4:
        return False
    x, y, w, h = region
    # Check positive dimensions
    if w <= 0 or h <= 0:
        print(f"Error: Region has invalid dimensions (w={w}, h={h})")
        return False
    # Check within monitor bounds
    if x < 0 or y < 0:
        print(f"Error: Region has negative coordinates (x={x}, y={y})")
        return False
    return True


# ==========================
# NON-BLOCKING KEY PRESS
# ==========================
def press_key_async(key):
    """Press key in separate thread to avoid blocking main loop."""
    def _press(k):
        try:
            keyboard.press_and_release(k)
        except Exception as e:
            print(f"Keyboard exception: {e}")
    threading.Thread(target=_press, args=(key,), daemon=True).start()


# ==========================
# TEMPLATE RESIZING HELPERS
# ==========================
def _resize_template_to_fit(template, max_w, max_h, scale=1.0):
    """Scale template by 'scale' then clamp to fit within (max_w, max_h)."""
    if template is None:
        return None
    # Ensure grayscale
    if template.ndim == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h_t, w_t = template.shape[:2]
    # Initial scaled size
    new_w = max(1, int(w_t * scale))
    new_h = max(1, int(h_t * scale))
    # Clamp if larger than ROI
    if new_w > max_w or new_h > max_h:
        fit = min(max_w / max(new_w, 1), max_h / max(new_h, 1), 1.0)
        new_w = max(1, int(new_w * fit))
        new_h = max(1, int(new_h * fit))
    
    # Apply MAX_TEMPLATE_SCALE constraint
    max_allowed_w = int(max_w * MAX_TEMPLATE_SCALE)
    max_allowed_h = int(max_h * MAX_TEMPLATE_SCALE)
    if new_w > max_allowed_w or new_h > max_allowed_h:
        fit = min(max_allowed_w / new_w, max_allowed_h / new_h, 1.0)
        new_w = max(4, int(new_w * fit))
        new_h = max(4, int(new_h * fit))
    
    interp = cv2.INTER_AREA if (new_w < w_t or new_h < h_t) else cv2.INTER_LINEAR
    return cv2.resize(template, (new_w, new_h), interpolation=interp)


def _build_scaled_templates(base_templates, roi_w, roi_h, scale):
    """Return dict of templates scaled and guaranteed to fit ROI."""
    scaled = {}
    for name, tmpl in base_templates.items():
        scaled[name] = _resize_template_to_fit(tmpl, roi_w, roi_h, scale)
    return scaled


# ==========================
# MAIN
# ==========================
rect, hit_zone = load_config()

if rect is None or hit_zone is None:
    print("No saved regions found. Draw new ones.")
    rect, hit_zone = select_regions()
else:
    print("Loaded saved regions from", CONFIG_FILE)
    print(f"Arrow region: {rect}")
    print(f"HIT zone: {hit_zone}")

print("Press 'R' to reset arrow region, '/' to reset HIT zone, '['/']' to adjust template scale, ESC to exit.")

# Load base templates
base_templates = {k: cv2.imread(v, cv2.IMREAD_GRAYSCALE) for k, v in TEMPLATES.items()}
for name, tmpl in base_templates.items():
    if tmpl is None:
        print(f"Warning: template for '{name}' not found at '{TEMPLATES[name]}'.")
sct = mss.mss()

time.sleep(2)
print("Bot started... watching screen!")
print("Optimization: Parallel template matching + predictive timing enabled")

# Build initial scaled templates to fit the selected ROI
templates = _build_scaled_templates(base_templates, rect[2], rect[3], template_scale)

# Calculate relative hit zone coordinates
hx_rel = hit_zone[0] - rect[0]
hy_rel = hit_zone[1] - rect[1]
hit_w = hit_zone[2]
hit_h = hit_zone[3]

# History for velocity-based prediction
history = {d: deque(maxlen=PREDICT_WINDOW) for d in TEMPLATES.keys()}
last_loop_time = time.perf_counter()
processing_delay_est = 0.001  # Moving average of processing delay (inicializado com valor pequeno)
fps_est = 0.0  # FPS estimado
last_scores = {k: 0.0 for k in TEMPLATES.keys()}  # Últimos scores dos templates

# Thread pool for parallel template matching
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Pressed keys cache para evitar spam
pressed_keys = {}  # {direction: timestamp}
KEY_PRESS_COOLDOWN = 0.15  # segundos entre pressionar a mesma tecla

try:
    while True:
        loop_start = time.perf_counter()
        
        # Handle recalibration hotkeys with debounce
        now = time.time()
        
        if keyboard.is_pressed("r") and (now - _last_recalib_update) > _recalib_debounce:
            print("Resetting arrow region...")
            _last_recalib_update = now
            new_rect = select_arrow_region()
            if new_rect:
                rect = new_rect
                # Recalculate relative hit zone
                hx_rel = hit_zone[0] - rect[0]
                hy_rel = hit_zone[1] - rect[1]
                # Rebuild scaled templates for the new ROI
                templates = _build_scaled_templates(base_templates, rect[2], rect[3], template_scale)
                # Reset history
                history = {d: deque(maxlen=PREDICT_WINDOW) for d in TEMPLATES.keys()}
            continue
        
        if keyboard.is_pressed("/") and (now - _last_recalib_update) > _recalib_debounce:
            print("Resetting HIT zone...")
            _last_recalib_update = now
            new_hit_zone = select_hit_zone_only()
            if new_hit_zone:
                hit_zone = new_hit_zone
                # Recalculate relative coordinates
                hx_rel = hit_zone[0] - rect[0]
                hy_rel = hit_zone[1] - rect[1]
                hit_w = hit_zone[2]
                hit_h = hit_zone[3]
            continue

        # Validate rect before capture
        if not rect or rect[2] <= 0 or rect[3] <= 0:
            print("Error: Invalid arrow region. Press 'R' to recalibrate.")
            time.sleep(0.1)
            continue

        try:
            frame = np.array(
                sct.grab(
                    {
                        "top": rect[1],
                        "left": rect[0],
                        "width": rect[2],
                        "height": rect[3],
                    }
                )
            )
        except Exception as e:
            print(f"Error capturing screen: {e}")
            print(f"Current region: {rect}")
            print("Press 'R' to recalibrate the arrow region.")
            time.sleep(0.5)
            continue
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        # Runtime template scale adjustment with debounce (otimizado)
        scale_now = time.time()
        scale_changed = False
        if keyboard.is_pressed("[") and (scale_now - _last_scale_update) > _scale_debounce:
            template_scale = max(min_scale, round(template_scale - scale_step, 2))
            _last_scale_update = scale_now
            scale_changed = True
        elif keyboard.is_pressed("]") and (scale_now - _last_scale_update) > _scale_debounce:
            template_scale = min(max_scale, round(template_scale + scale_step, 2))
            _last_scale_update = scale_now
            scale_changed = True

        if scale_changed:
            templates = _build_scaled_templates(base_templates, rect[2], rect[3], template_scale)
            print(f"Template scale set to {template_scale:.2f}")
            # Limpar histórico ao mudar escala
            history = {d: deque(maxlen=PREDICT_WINDOW) for d in TEMPLATES.keys()}

        fh, fw = frame_gray.shape[:2]

        # === PARALLEL TEMPLATE MATCHING (OTIMIZADO) ===
        futures = {}
        for direction, template in templates.items():
            if template is None:
                continue
            
            th, tw = template.shape[:2]
            # Skip if template too large
            if th > fh or tw > fw:
                continue
            
            # Ensure dtype compatibility (reutilizar templates já compatíveis)
            if frame_gray.dtype != template.dtype:
                template = template.astype(frame_gray.dtype)
            
            # Submit parallel matching task
            future = executor.submit(cv2.matchTemplate, frame_gray, template, cv2.TM_CCOEFF_NORMED)
            futures[future] = direction

        # Process results as they complete (otimizado)
        hits = []  # list of (direction, x, y, match_val)
        for fut in as_completed(futures):
            direction = futures[fut]
            try:
                res = fut.result()
                # Pegar apenas o melhor match (mais eficiente)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if max_val >= THRESHOLD:
                    x, y = max_loc
                    hits.append((direction, x, y, max_val))
            except Exception as e:
                continue

        # Update processing delay estimate com suavização
        loop_end = time.perf_counter()
        loop_time = loop_end - last_loop_time
        last_loop_time = loop_end
        # Low-pass filter para suavizar delay
        processing_delay_est = PROCESSING_ALPHA * processing_delay_est + (1 - PROCESSING_ALPHA) * loop_time
        # Atualizar FPS estimado
        if loop_time > 0:
            fps_est = 0.9 * fps_est + 0.1 * (1.0 / loop_time)

        # === PREDICTIVE TIMING ===
        now_perf = time.perf_counter()
        for direction, x, y, val in hits:
            # Salvar último score
            last_scores[direction] = val
            
            # Verificar cooldown para evitar spam da mesma tecla
            if direction in pressed_keys:
                time_since_press = now_perf - pressed_keys[direction]
                if time_since_press < KEY_PRESS_COOLDOWN:
                    continue  # Skip, tecla pressionada recentemente
            
            # Record detection time and x position
            history[direction].append((now_perf, x))
            
            # Compute velocity (pixels/second) from history
            vel = None
            arr = history[direction]
            if len(arr) >= 2:
                t0, x0 = arr[0]
                tn, xn = arr[-1]
                dt = tn - t0
                if dt > 0:
                    # Assuming arrows move left (from right to left)
                    vel = (x0 - xn) / dt  # positive if moving left
            
            # Decide when to press
            should_press = False
            if vel is None or vel < MIN_VELOCITY_PIX_PER_SEC:
                # Fallback: immediate press if overlapping hit zone
                if hx_rel <= x <= hx_rel + hit_w:
                    should_press = True
                    if SHOW_DEBUG:
                        print(f"[IMMEDIATE] {direction} at x={x:.1f}")
            else:
                # Predict time until arrow reaches center of hit zone
                target_x = hx_rel + hit_w // 2
                distance = x - target_x
                time_to_hit = distance / vel  # seconds
                
                # Compensate for processing delay + user offset
                advance = processing_delay_est + (HIT_OFFSET_COMP_MS / 1000.0)
                
                # Press if time to hit <= advance
                if time_to_hit <= advance:
                    should_press = True
                    if SHOW_DEBUG:
                        print(f"[PRED] {direction} x={x:.1f} dt={time_to_hit:.3f}s adv={advance:.3f}s vel={vel:.1f}px/s")
            
            if should_press:
                press_key_async(KEY_MAP[direction])
                pressed_keys[direction] = now_perf  # Registrar timestamp do pressionamento

        # === DEBUG VISUALIZATION ===
        if SHOW_DEBUG:
            # Draw HIT zone
            cv2.rectangle(frame, (hx_rel, hy_rel), (hx_rel + hit_w, hy_rel + hit_h), (0, 0, 255), 2)
            
            # Draw detected arrows
            for direction, x, y, val in hits:
                template = templates.get(direction)
                if template is not None:
                    th, tw = template.shape[:2]
                    cv2.rectangle(frame, (x, y), (x + tw, y + th), (0, 255, 0), 1)
                    cv2.putText(frame, f"{direction} {val:.2f}", (x, max(0, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Overlay runtime info com mais métricas
            _y0 = 18
            _dy = 18
            _lines = [
                f"FPS: {fps_est:5.1f}",
                f"Delay: {processing_delay_est*1000:5.1f} ms",
                f"Tracked: {len(hits):3d}",
                f"Threshold: {THRESHOLD:.2f} | Scale: {template_scale:.2f}",
                "[R] Arrow | [/] HIT | [ ] ] scale | ESC exit",
            ]
            for _i, _line in enumerate(_lines):
                cv2.putText(frame, _line, (8, _y0 + _i * _dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Últimos scores dos templates
            _ys = _y0 + len(_lines) * _dy + 6
            for _i, (_k, _v) in enumerate(last_scores.items()):
                cv2.putText(frame, f"{_k}: {_v:.2f}", (8, _ys + _i * _dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            
            cv2.imshow("Detection", frame)

            if cv2.waitKey(1) == 27:  # ESC
                break
        else:
            # Quando debug está desligado, pequeno sleep para não consumir 100% CPU
            time.sleep(0.001)
        
        # Non-blocking ESC check
        if keyboard.is_pressed("esc"):
            print("ESC pressed -> exiting")
            break

finally:
    executor.shutdown(wait=False)
    cv2.destroyAllWindows()
