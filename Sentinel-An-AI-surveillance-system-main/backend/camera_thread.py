"""
camera_thread.py — Threaded camera capture for low-latency frame retrieval.

Supports:
  • Webcam (integer index, e.g. 0)
  • Video file (path string, e.g. "video.mp4")
  • WiFi / IP camera (URL string, e.g. "http://192.168.1.5:8080/video")

Continuously reads frames in a background thread so the main pipeline
never blocks waiting for the camera/video I/O. Only the latest frame
is kept in memory, eliminating buffer build-up.

Usage:
    cam = CameraThread(source=0)                                    # webcam
    cam = CameraThread(source="v.mp4")                              # video file
    cam = CameraThread(source="http://192.168.1.5:8080/video")      # WiFi cam
    cam.start()
    ...
    ret, frame = cam.read()            # non-blocking, returns latest frame
    ...
    cam.stop()
"""

import threading
import time
import cv2


def _is_ip_camera(source) -> bool:
    """Check if the source is an IP/WiFi camera URL."""
    if isinstance(source, str):
        return source.startswith(("http://", "https://", "rtsp://", "rtmp://"))
    return False


class CameraThread:
    """Background-threaded camera/video capture with minimal latency."""

    def __init__(self, source=0, width=640, height=480, fps=30, fallback_to_webcam=True):
        """
        Parameters
        ----------
        source : int | str
            Webcam index, path to video file, or WiFi/IP camera URL.
        width, height : int
            Requested capture resolution (hints for webcam/IP cam).
        fps : int
            Requested capture FPS (hint for webcam).
        fallback_to_webcam : bool
            If True and the source fails to open, automatically
            fallback to webcam index 0.
        """
        self.source = source
        self.is_ip_camera = _is_ip_camera(source)
        self._fallback = fallback_to_webcam

        # Open the video source
        self.cap = self._open_source(source, width, height, fps)

        # Shared state — protected by a lock
        self._lock = threading.Lock()
        self._ret = False
        self._frame = None
        self._stopped = True
        self._thread = None

    def _open_source(self, source, width=640, height=480, fps=30):
        """
        Open a VideoCapture for the given source.
        Applies latency-reduction settings for live sources (webcam / IP cam).
        Falls back to webcam 0 on failure if fallback_to_webcam is True.
        """
        is_ip = _is_ip_camera(source)

        if is_ip:
            # [WiFi Camera] Use FFMPEG backend for better IP stream handling
            print(f"[INFO] Connecting to WiFi/IP camera: {source}")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[WARNING] Failed to open source: {source}")
            if self._fallback and source != 0:
                print("[INFO] Falling back to default webcam (index 0)...")
                cap = cv2.VideoCapture(0)
                self.source = 0
                self.is_ip_camera = False
                is_ip = False
            if not cap.isOpened():
                print("[ERROR] Fallback webcam also failed.")
                return cap

        # [OPTIMIZATION] Minimise internal OpenCV buffer to 1 frame
        # This is critical for live sources to prevent frame queue build-up
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set resolution & FPS hints (webcam / IP cam will use closest supported)
        if isinstance(self.source, int) or is_ip:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

        src_type = "WiFi/IP camera" if is_ip else ("Webcam" if isinstance(self.source, int) else "Video file")
        print(f"[INFO] {src_type} opened successfully.")
        return cap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        """Start the background capture thread."""
        if not self._stopped:
            return self  # already running
        self._stopped = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def read(self):
        """
        Return the most recent frame (non-blocking).

        Returns
        -------
        ret : bool
            True if a valid frame is available.
        frame : np.ndarray | None
            The latest BGR frame, or None.
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def stop(self):
        """Signal the capture thread to stop and release resources."""
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()

    def is_opened(self):
        """Check if the underlying VideoCapture is open."""
        return self.cap.isOpened()

    def get_fps(self):
        """Return the native FPS of the video source."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def reset(self):
        """Reset video to beginning (for looping video files)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    @property
    def stopped(self):
        return self._stopped

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _capture_loop(self):
        """Continuously grab frames until stopped."""
        consecutive_failures = 0
        MAX_FAILURES = 30  # ~1 second of failures before giving up

        while not self._stopped:
            # [OPTIMIZATION] grab() + retrieve() is faster than read()
            # For IP cameras, grab() also discards stale buffered frames
            grabbed = self.cap.grab()
            if not grabbed:
                consecutive_failures += 1
                # For IP cameras, brief network hiccups are normal
                if consecutive_failures > MAX_FAILURES:
                    with self._lock:
                        self._ret = False
                    if self.is_ip_camera:
                        print("[WARNING] WiFi camera connection lost. Retrying...")
                        consecutive_failures = 0  # keep retrying
                time.sleep(0.01)
                continue

            consecutive_failures = 0  # reset on success
            ret, frame = self.cap.retrieve()
            with self._lock:
                self._ret = ret
                self._frame = frame
