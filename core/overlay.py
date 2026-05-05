import cv2
import colorsys


def color_for_id(track_id):
    hue = (track_id * 31) % 180
    r, g, b = colorsys.hsv_to_rgb(hue / 180.0, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw(frame, tracks, trail_fn):
    out = frame.copy()
    for t in tracks:
        color = color_for_id(t.track_id)
        trail = trail_fn(t.track_id)
        for i in range(1, len(trail)):
            p1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            p2 = (int(trail[i][0]), int(trail[i][1]))
            cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
        cv2.rectangle(out, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), color, 2, cv2.LINE_AA)
        label = f"#{t.track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.6, 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        lx = int(t.x1)
        ly = max(int(t.y1) - 6, th + baseline)
        cv2.rectangle(out, (lx - 1, ly - th - baseline), (lx + tw + 1, ly + baseline), color, cv2.FILLED)
        cv2.putText(out, label, (lx, ly), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out
