import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import pygame


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    calibration_seconds: float = 4.0
    default_baseline: float = 0.28
    close_ratio: float = 0.75
    open_ratio: float = 0.85
    smoothing_frames: int = 3
    max_blink_seconds: float = 0.5
    alert_seconds: float = 1.0
    face_loss_grace_seconds: float = 0.3
    perclos_window_seconds: float = 30.0
    perclos_threshold: float = 0.25
    alarm_frequency_hz: int = 1000
    alarm_duration_seconds: float = 0.6
    sample_rate_hz: int = 44100


RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(points):
    p1, p2, p3, p4, p5, p6 = points
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    return vertical / horizontal if horizontal > 0 else 0.0


def landmarks_to_points(landmarks, indices, width, height):
    return np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in indices])


class Calibrator:
    def __init__(self, duration):
        self.duration = duration
        self.started_at = time.monotonic()
        self.samples = []

    def add(self, value):
        self.samples.append(value)

    def remaining(self):
        return max(self.duration - (time.monotonic() - self.started_at), 0.0)

    def finished(self):
        return self.remaining() <= 0.0

    def baseline(self, fallback):
        if len(self.samples) < 15:
            return fallback
        return float(np.median(self.samples))


class ClosureTracker:
    def __init__(self, config):
        self.config = config
        self.close_threshold = None
        self.open_threshold = None
        self.closed = False
        self.closure_start = None
        self.last_seen = None
        self.blink_count = 0
        self.history = deque()
        self.recent = deque(maxlen=config.smoothing_frames)

    def set_baseline(self, baseline):
        self.close_threshold = baseline * self.config.close_ratio
        self.open_threshold = baseline * self.config.open_ratio
        self.reset()

    def reset(self):
        self.closed = False
        self.closure_start = None
        self.recent.clear()
        self.history.clear()

    def update(self, ear, now):
        self.recent.append(ear)
        smoothed = float(np.median(self.recent))
        self.last_seen = now

        if self.closed:
            if smoothed >= self.open_threshold:
                self._end_closure(now)
        elif smoothed < self.close_threshold:
            self.closed = True
            self.closure_start = now

        self.history.append((now, self.closed))
        return smoothed

    def mark_missing(self, now):
        if self.last_seen is None:
            return
        if now - self.last_seen > self.config.face_loss_grace_seconds:
            self.closed = False
            self.closure_start = None
            self.recent.clear()

    def closure_duration(self, now):
        if not self.closed or self.closure_start is None:
            return 0.0
        return now - self.closure_start

    def perclos(self, now):
        window = self.config.perclos_window_seconds
        while self.history and now - self.history[0][0] > window:
            self.history.popleft()
        if not self.history or now - self.history[0][0] < 0.8 * window:
            return None
        return sum(1 for _, closed in self.history if closed) / len(self.history)

    def _end_closure(self, now):
        if now - self.closure_start <= self.config.max_blink_seconds:
            self.blink_count += 1
        self.closed = False
        self.closure_start = None


class AlarmPlayer:
    def __init__(self, config):
        pygame.mixer.init(frequency=config.sample_rate_hz, size=-16, channels=1)
        t = np.linspace(
            0,
            config.alarm_duration_seconds,
            int(config.sample_rate_hz * config.alarm_duration_seconds),
            endpoint=False,
        )
        wave = np.sin(2 * np.pi * config.alarm_frequency_hz * t) * 32767 * 0.8
        self.sound = pygame.sndarray.make_sound(wave.astype(np.int16))

    def play(self):
        if not pygame.mixer.get_busy():
            self.sound.play()

    def close(self):
        pygame.mixer.quit()


class DrowsinessMonitor:
    def __init__(self, config):
        self.config = config
        self.tracker = ClosureTracker(config)
        self.calibrator = Calibrator(config.calibration_seconds)
        self.alarm = AlarmPlayer(config)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.capture = cv2.VideoCapture(config.camera_index)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open the webcam.")

    def restart_calibration(self):
        self.calibrator = Calibrator(self.config.calibration_seconds)

    def run(self):
        try:
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                self._process(frame)
                cv2.imshow("Drowsiness Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    self.restart_calibration()
        finally:
            self._release()

    def _process(self, frame):
        height, width = frame.shape[:2]
        now = time.monotonic()
        result = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        ear = None
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            right = landmarks_to_points(landmarks, RIGHT_EYE, width, height)
            left = landmarks_to_points(landmarks, LEFT_EYE, width, height)
            ear = (eye_aspect_ratio(right) + eye_aspect_ratio(left)) / 2.0
            for eye in (right, left):
                cv2.polylines(frame, [eye.astype(np.int32)], True, (0, 255, 0), 1)
        else:
            self.tracker.mark_missing(now)

        if self.calibrator is not None and not self.calibrator.finished():
            if ear is not None:
                self.calibrator.add(ear)
            self._draw_status(
                frame,
                f"Calibrating: keep your eyes open ({self.calibrator.remaining():.0f}s)",
                (255, 200, 0),
            )
            return

        if self.calibrator is not None:
            self.tracker.set_baseline(self.calibrator.baseline(self.config.default_baseline))
            self.calibrator = None

        smoothed = self.tracker.update(ear, now) if ear is not None else None
        closure = self.tracker.closure_duration(now)
        perclos = self.tracker.perclos(now)

        microsleep = closure >= self.config.alert_seconds
        fatigue = perclos is not None and perclos >= self.config.perclos_threshold

        if microsleep or fatigue:
            self.alarm.play()
            self._draw_alert(frame, "EYES CLOSED" if microsleep else "FATIGUE DETECTED")
            status, color = "ALERT", (0, 0, 255)
        elif ear is None:
            status, color = "Face not detected", (0, 165, 255)
        elif self.tracker.closed:
            status, color = f"Eyes closed: {closure:.1f}s", (0, 200, 255)
        else:
            status, color = "Attentive", (0, 200, 0)

        self._draw_status(frame, status, color)
        self._draw_metrics(frame, smoothed, perclos)

    def _draw_status(self, frame, text, color):
        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _draw_metrics(self, frame, ear, perclos):
        ear_text = f"{ear:.2f}" if ear is not None else "--"
        perclos_text = f"{perclos * 100:.0f}%" if perclos is not None else "--"
        line = (
            f"EAR: {ear_text} | Threshold: {self.tracker.close_threshold:.2f} | "
            f"PERCLOS: {perclos_text} | Blinks: {self.tracker.blink_count}"
        )
        cv2.putText(frame, line, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def _draw_alert(self, frame, message):
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 12)
        size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)[0]
        origin = ((width - size[0]) // 2, height // 2)
        cv2.putText(frame, message, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)

    def _release(self):
        self.capture.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
        self.alarm.close()


def main():
    DrowsinessMonitor(Config()).run()


if __name__ == "__main__":
    main()
