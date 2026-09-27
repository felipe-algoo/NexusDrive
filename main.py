import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import dearpygui.dearpygui as dpg
from ultralytics import YOLO

warnings.filterwarnings('ignore')

class SecurityContext:
    @staticmethod
    def validate_resource(target_path: str) -> Path:
        resolved = Path(target_path).resolve()
        if not resolved.exists() and not resolved.parent.exists():
            raise ValueError()
        return resolved

    @staticmethod
    def sanitize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        if frame is None or frame.size == 0 or len(frame.shape) != 3:
            raise ValueError()
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

class GeometryEngine:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        src_points = np.float32([
            [width * 0.45, height * 0.65],
            [width * 0.55, height * 0.65],
            [width * 0.95, height * 0.95],
            [width * 0.05, height * 0.95]
        ])
        
        dst_points = np.float32([
            [width * 0.2, 0],
            [width * 0.8, 0],
            [width * 0.8, height],
            [width * 0.2, height]
        ])
        
        self.matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inv_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    def warp_perspective(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.matrix, (self.width, self.height), flags=cv2.INTER_LINEAR)

    def unwarp_perspective(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.inv_matrix, (self.width, self.height), flags=cv2.INTER_LINEAR)

class CurvatureAnalyzer:
    def __init__(self):
        self.left_fit: Optional[np.ndarray] = None
        self.right_fit: Optional[np.ndarray] = None
        self.ym_per_pix = 30 / 720
        self.xm_per_pix = 3.7 / 700
        self.smoothing_factor = 0.8

    def _validate_points(self, points: np.ndarray) -> bool:
        return len(points) >= 10 and len(np.unique(points)) >= 5 and (np.max(points) - np.min(points)) > 50

    def analyze(self, binary_warped: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = int(binary_warped.shape[0] // nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        curve_radius = None

        if self._validate_points(lefty) and self._validate_points(righty):
            curr_left_fit = np.polyfit(lefty, leftx, 2)
            curr_right_fit = np.polyfit(righty, rightx, 2)
            
            if self.left_fit is None or self.right_fit is None:
                self.left_fit = curr_left_fit
                self.right_fit = curr_right_fit
            else:
                self.left_fit = self.smoothing_factor * self.left_fit + (1 - self.smoothing_factor) * curr_left_fit
                self.right_fit = self.smoothing_factor * self.right_fit + (1 - self.smoothing_factor) * curr_right_fit

            left_fit_cr = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
            right_fit_cr = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)
            
            y_eval = np.max(nonzeroy)
            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
            curve_radius = float((left_curverad + right_curverad) / 2)

        if self.left_fit is not None and self.right_fit is not None:
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]
            right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            
            cv2.fillPoly(out_img, np.int32([pts]), (0, 255, 0))

        return out_img, curve_radius

class SpatialEntity:
    def __init__(self, bbox: Tuple[float, float, float, float], conf: float, cls_id: int):
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.conf = conf
        self.cls_id = cls_id
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0
        self.area = (self.x2 - self.x1) * (self.y2 - self.y1)

class ContextAwareTracker:
    def __init__(self, iou_threshold: float = 0.3, max_history: int = 30):
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_history = max_history

    def _compute_iou(self, boxA: SpatialEntity, boxB: SpatialEntity) -> float:
        xA = max(boxA.x1, boxB.x1)
        yA = max(boxA.y1, boxB.y1)
        xB = min(boxA.x2, boxB.x2)
        yB = min(boxA.y2, boxB.y2)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        iou = interArea / float(boxA.area + boxB.area - interArea)
        return iou

    def update(self, detections: List[SpatialEntity]) -> List[Dict[str, Any]]:
        updated_tracks = []
        assigned_detections = set()

        for track_id, track_data in self.tracks.items():
            best_iou = 0.0
            best_det_idx = -1
            for i, det in enumerate(detections):
                if i in assigned_detections:
                    continue
                iou = self._compute_iou(track_data["last_entity"], det)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            if best_iou >= self.iou_threshold:
                det = detections[best_det_idx]
                track_data["history"].append(det)
                if len(track_data["history"]) > self.max_history:
                    track_data["history"].pop(0)
                track_data["last_entity"] = det
                track_data["missed_frames"] = 0
                track_data["confirmed"] = len(track_data["history"]) >= 3
                assigned_detections.add(best_det_idx)
                updated_tracks.append({"id": track_id, "entity": det, "confirmed": track_data["confirmed"], "history": track_data["history"]})
            else:
                track_data["missed_frames"] += 1

        for i, det in enumerate(detections):
            if i not in assigned_detections:
                self.tracks[self.next_id] = {
                    "history": [det],
                    "last_entity": det,
                    "missed_frames": 0,
                    "confirmed": False
                }
                self.next_id += 1

        keys_to_delete = [t_id for t_id, t_data in self.tracks.items() if t_data["missed_frames"] > 5]
        for k in keys_to_delete:
            del self.tracks[k]

        return updated_tracks

class CollisionPredictor:
    def __init__(self, focal_length: float = 800.0, real_width_m: float = 1.8):
        self.focal_length = focal_length
        self.real_width_m = real_width_m

    def assess_threat(self, tracks: List[Dict[str, Any]], frame_width: int) -> Tuple[bool, List[Dict[str, Any]]]:
        critical_threat = False
        analyzed_tracks = []

        for track in tracks:
            if not track["confirmed"]:
                continue
            
            entity: SpatialEntity = track["entity"]
            history: List[SpatialEntity] = track["history"]
            
            pixel_width = entity.x2 - entity.x1
            distance_m = (self.real_width_m * self.focal_length) / (pixel_width + 1e-6)
            
            ttc = 999.0
            if len(history) >= 5:
                past_entity = history[-5]
                past_distance_m = (self.real_width_m * self.focal_length) / ((past_entity.x2 - past_entity.x1) + 1e-6)
                velocity_m_per_frame = past_distance_m - distance_m
                if velocity_m_per_frame > 0.1:
                    ttc = distance_m / velocity_m_per_frame

            in_path = (entity.cx > frame_width * 0.35) and (entity.cx < frame_width * 0.65)
            threat_level = "LOW"
            if in_path:
                if ttc < 15.0:
                    threat_level = "CRITICAL"
                    critical_threat = True
                elif distance_m < 20.0:
                    threat_level = "WARNING"

            analyzed_tracks.append({
                "id": track["id"],
                "bbox": (entity.x1, entity.y1, entity.x2, entity.y2),
                "distance": distance_m,
                "ttc": ttc,
                "threat": threat_level
            })

        return critical_threat, analyzed_tracks

class NeuralInferenceWrapper:
    def __init__(self, model_identifier: str = "yolov8n.pt"):
        safe_path = SecurityContext.validate_resource(model_identifier)
        self.model = YOLO(str(safe_path), task="detect")
        self.target_classes = {0, 1, 2, 3, 5, 7}

    def infer(self, frame: np.ndarray) -> List[SpatialEntity]:
        results = self.model(frame, verbose=False)[0]
        entities = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.target_classes:
                conf = float(box.conf[0])
                if conf > 0.45:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    entities.append(SpatialEntity((x1, y1, x2, y2), conf, cls_id))
        return entities

class NexusDriveCore:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.capture = None
        self.geometry = GeometryEngine(self.width, self.height)
        self.curvature = CurvatureAnalyzer()
        self.ai_model = NeuralInferenceWrapper()
        self.tracker = ContextAwareTracker()
        self.predictor = CollisionPredictor()
        self.frame_buffer = np.zeros((self.height, self.width, 4), dtype=np.float32)

    def bind_source(self, source: Any) -> bool:
        if self.capture is not None:
            self.capture.release()
            
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture = None
            return False
            
        self.tracker = ContextAwareTracker()
        self.curvature.left_fit = None
        self.curvature.right_fit = None
        return True

    def generate_binary_mask(self, frame: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
        
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1
        
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    def pipeline(self) -> np.ndarray:
        if self.capture is None:
            return self.frame_buffer

        ret, raw_frame = self.capture.read()
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, raw_frame = self.capture.read()
            if not ret:
                return self.frame_buffer

        frame = SecurityContext.sanitize_frame(raw_frame, (self.width, self.height))
        overlay = frame.copy()

        binary = self.generate_binary_mask(frame)
        warped = self.geometry.warp_perspective(binary)
        lane_overlay, radius = self.curvature.analyze(warped)
        unwarped_lane = self.geometry.unwarp_perspective(lane_overlay)
        
        alpha = 0.4
        cv2.addWeighted(unwarped_lane, alpha, overlay, 1 - alpha, 0, overlay)

        if radius is not None:
            cv2.putText(overlay, f"Curve Radius: {radius:.2f}m", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        raw_entities = self.ai_model.infer(frame)
        tracked_entities = self.tracker.update(raw_entities)
        critical_alert, analyzed_data = self.predictor.assess_threat(tracked_entities, self.width)

        if critical_alert:
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 255), 10)
            cv2.putText(overlay, "CRITICAL COLLISION WARNING", (self.width // 4, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        for obj in analyzed_data:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            color = (0, 255, 0)
            if obj["threat"] == "WARNING":
                color = (0, 165, 255)
            elif obj["threat"] == "CRITICAL":
                color = (0, 0, 255)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            lbl = f"Dist: {obj['distance']:.1f}m"
            if obj["ttc"] < 999.0:
                lbl += f" TTC: {obj['ttc']:.1f}s"
            cv2.putText(overlay, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        rgba_frame = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        return rgba_frame

    def release(self):
        if self.capture is not None:
            self.capture.release()

class GUIManager:
    def __init__(self):
        dpg.create_context()
        self.core = NexusDriveCore()
        
        self.core.bind_source(0)

        self.texture_data = np.zeros((self.core.height, self.core.width, 4), dtype=np.float32)
        
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.core.width, self.core.height, self.texture_data, format=dpg.mvFormat_Float_rgba, tag="main_feed")

        with dpg.file_dialog(directory_selector=False, show=False, callback=self.file_picker_callback, tag="file_dialog", width=600, height=400):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255))
            dpg.add_file_extension(".avi", color=(0, 255, 0, 255))
            dpg.add_file_extension(".mkv", color=(0, 255, 0, 255))

        with dpg.window(tag="Primary Window", no_scrollbar=True, no_title_bar=True):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Video", callback=lambda: dpg.show_item("file_dialog"))
                dpg.add_button(label="Use WebCam", callback=lambda: self.core.bind_source(0))
                dpg.add_input_text(hint="Paste or Drag & Drop File Path Here...", width=-1, callback=self.text_input_callback)
            dpg.add_image("main_feed")

        dpg.create_viewport(title="NexusDrive Spatial", width=self.core.width, height=self.core.height + 60)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)

    def text_input_callback(self, sender, app_data):
        target = str(app_data).strip("\"' ")
        if target:
            try:
                safe_path = str(SecurityContext.validate_resource(target))
                self.core.bind_source(safe_path)
            except Exception:
                pass

    def file_picker_callback(self, sender, app_data):
        if 'selections' in app_data and app_data['selections']:
            selected_file = list(app_data['selections'].values())[0]
            try:
                safe_path = str(SecurityContext.validate_resource(selected_file))
                self.core.bind_source(safe_path)
            except Exception:
                pass

    def execute(self):
        while dpg.is_dearpygui_running():
            processed_matrix = self.core.pipeline()
            dpg.set_value("main_feed", processed_matrix.ravel())
            dpg.render_dearpygui_frame()
            
        self.core.release()
        dpg.destroy_context()

if __name__ == "__main__":
    app = GUIManager()
    app.execute()
