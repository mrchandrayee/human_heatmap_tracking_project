#!/usr/bin/env python3
"""
Human Detection, ByteTrack Tracking, and Density Heatmap Pipeline.

Reconstructed from the supplied screen recording. The video shows a YOLO +
ByteTrack workflow, Gaussian density accumulation, configurable polygon zones,
zone density/count classification, and visual overlays. This is a clean,
runnable implementation of that workflow, not a byte-for-byte OCR dump.
"""
from __future__ import annotations
import argparse, json, os, time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

COLORMAP_DICT = {
    "JET": cv2.COLORMAP_JET, "INFERNO": cv2.COLORMAP_INFERNO,
    "TURBO": cv2.COLORMAP_TURBO, "HOT": cv2.COLORMAP_HOT,
    "PLASMA": cv2.COLORMAP_PLASMA, "MAGMA": cv2.COLORMAP_MAGMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS, "RAINBOW": cv2.COLORMAP_RAINBOW,
}

def generate_colors(num_colors=256):
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_colors, 3), dtype=np.uint8)
    return [tuple(map(int, c)) for c in colors]

COLOR_PALETTE = generate_colors(500)

def get_track_color(track_id):
    return COLOR_PALETTE[int(track_id) % len(COLOR_PALETTE)]

def draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, corner_length_ratio=0.25):
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w-1, int(x2)), min(h-1, int(y2))
    if x2 <= x1 or y2 <= y1: return
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cl = max(10, int(min(x2-x1, y2-y1) * corner_length_ratio))
    t = thickness + 1
    for x, y, sx, sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img,(x,y),(x+sx*cl,y),color,t)
        cv2.line(img,(x,y),(x,y+sy*cl),color,t)

def create_gaussian_kernel(radius, sigma=None):
    sigma = sigma or max(1.0, radius / 2.0)
    size = 2 * int(radius) + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx*xx + yy*yy) / (2*sigma*sigma))
    kernel /= max(float(kernel.max()), 1e-8)
    return kernel.astype(np.float32)

def default_zones(width, height):
    # Three editable zones, scaled to the current video resolution.
    return [
        {"name":"Line 1","color":[0,255,0],"points":[[.05,.18],[.42,.18],[.42,.92],[.05,.92]]},
        {"name":"Line 2","color":[0,165,255],"points":[[.43,.18],[.72,.18],[.72,.92],[.43,.92]]},
        {"name":"Line 3","color":[255,0,128],"points":[[.73,.18],[.98,.18],[.98,.92],[.73,.92]]},
    ]

class HeatmapTrackerPipeline:
    def __init__(self, args):
        self.args = args
        self.model = YOLO(args.model)
        self.trails = defaultdict(lambda: deque(maxlen=args.tail_len))
        self.heat = None
        self.kernel = create_gaussian_kernel(args.radius)
        self.zones = self.load_zones(args.zones)

    def load_zones(self, path):
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("zones", data)
        return None

    def prepare_zones(self, width, height):
        if not self.zones:
            self.zones = default_zones(width, height)
        self.zone_points = []
        self.zone_masks = []
        for zone in self.zones:
            pts = []
            for x, y in zone["points"]:
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append([int(x*width), int(y*height)])
                else:
                    pts.append([int(x), int(y)])
            pts = np.asarray(pts, dtype=np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            self.zone_points.append(pts)
            self.zone_masks.append(mask.astype(bool))
        print(f"[ZONES] Initialized {len(self.zones)} polygon zones")

    def add_heat(self, x, y):
        r = self.args.radius
        h, w = self.heat.shape
        x, y = int(x), int(y)
        x1, x2 = max(0,x-r), min(w,x+r+1)
        y1, y2 = max(0,y-r), min(h,y+r+1)
        kx1, ky1 = x1-(x-r), y1-(y-r)
        kx2, ky2 = kx1+(x2-x1), ky1+(y2-y1)
        self.heat[y1:y2,x1:x2] += self.kernel[ky1:ky2,kx1:kx2]

    def analyze_zones(self, centers):
        results = []
        for i, zone in enumerate(self.zones):
            mask = self.zone_masks[i]
            values = self.heat[mask]
            density_mean = float(values.mean()) if values.size else 0.0
            density_max = float(values.max()) if values.size else 0.0
            people = sum(cv2.pointPolygonTest(
                self.zone_points[i], (float(x),float(y)), False) >= 0 for x,y in centers)
            if density_mean > 5.0: level, color = "HIGH", (0,0,255)
            elif density_mean > 1.0: level, color = "MEDIUM", (0,165,255)
            elif density_mean > 0.1: level, color = "LOW", (0,255,255)
            else: level, color = "CLEAR", (0,255,0)
            results.append({"name":zone["name"],"persons":int(people),
                "density_mean":density_mean,"density_max":density_max,"level":level,
                "color":list(color)})
        return results

    def draw_zones(self, frame, results):
        overlay = frame.copy()
        for i, zone in enumerate(self.zones):
            pts = self.zone_points[i]
            color = tuple(zone.get("color",[0,255,0]))
            cv2.fillPoly(overlay,[pts],color)
            cv2.polylines(frame,[pts],True,color,2)
            cx, cy = np.mean(pts,axis=0).astype(int)
            r = results[i]
            label = f'{r["name"]}: {r["persons"]} | {r["level"]}'
            cv2.rectangle(frame,(cx-95,cy-17),(cx+95,cy+17),(20,20,20),-1)
            cv2.putText(frame,label,(cx-88,cy+6),cv2.FONT_HERSHEY_SIMPLEX,.42,
                        (255,255,255),1,cv2.LINE_AA)
        cv2.addWeighted(overlay,0.12,frame,0.88,0,frame)

    def render_heatmap(self, frame):
        heat = self.heat
        if heat.max() > 0:
            # Robust percentile normalization prevents one hotspot washing out the map.
            vmax = max(float(np.percentile(heat, 99.5)), 1e-6)
            normalized = np.clip(heat / vmax * 255,0,255).astype(np.uint8)
        else:
            normalized = np.zeros(heat.shape,dtype=np.uint8)
        cmap = COLORMAP_DICT.get(self.args.colormap.upper(),cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(normalized,cmap)
        return cv2.addWeighted(frame,1.0,colored,self.args.alpha,0)

    def run(self):
        source = int(self.args.source) if str(self.args.source).isdigit() else self.args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.args.source}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1 else 25.0
        self.heat = np.zeros((height,width),dtype=np.float32)
        self.prepare_zones(width,height)
        out_path = Path(self.args.output)
        out_path.parent.mkdir(parents=True,exist_ok=True)
        writer = cv2.VideoWriter(str(out_path),cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps,(width,height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {out_path}")
        frame_idx, last_results, total_seen = 0, [], set()
        zone_history = []
        started = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                frame_idx += 1
                if self.args.frame_skip > 1 and frame_idx % self.args.frame_skip:
                    continue
                result = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                    classes=[0], conf=self.args.conf, iou=self.args.iou,
                    imgsz=self.args.imgsz, verbose=False)[0]
                centers = []
                if result.boxes is not None and len(result.boxes):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id
                    ids = ids.int().cpu().tolist() if ids is not None else list(range(len(boxes)))
                    for box, tid in zip(boxes,ids):
                        x1,y1,x2,y2 = map(int,box)
                        cx,cy = (x1+x2)//2, (y1+y2)//2
                        centers.append((cx,cy)); total_seen.add(int(tid))
                        self.trails[int(tid)].append((cx,cy))
                        self.add_heat(cx,cy)
                        color = get_track_color(tid)
                        draw_corner_box(frame,x1,y1,x2,y2,color)
                        cv2.putText(frame,f"Person {tid}",(x1,max(20,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,color,2,cv2.LINE_AA)
                        pts = list(self.trails[int(tid)])
                        for j in range(1,len(pts)):
                            cv2.line(frame,pts[j-1],pts[j],color,2)
                        cv2.circle(frame,(cx,cy),4,color,-1)
                if frame_idx % self.args.decay_interval == 0:
                    self.heat *= self.args.decay
                zone_results = self.analyze_zones(centers)
                self.draw_zones(frame,zone_results)
                heat_frame = self.render_heatmap(frame)
                cv2.putText(heat_frame,f"Human Heatmap & Tracking | Frame {frame_idx}",
                    (12,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(heat_frame,f"Active: {len(centers)}  Unique IDs: {len(total_seen)}",
                    (12,53),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
                writer.write(heat_frame)
                if frame_idx % self.args.report_interval == 0:
                    zone_history.append({"frame":frame_idx,"time_sec":frame_idx/fps,
                                         "zones":zone_results})
                if self.args.display:
                    cv2.imshow("Human Heatmap & Tracking",heat_frame)
                    if cv2.waitKey(1)&0xFF == ord("q"): break
        finally:
            cap.release(); writer.release()
            if self.args.display: cv2.destroyAllWindows()
        # Save a standalone heatmap image.
        heat_norm = np.clip(self.heat/max(float(self.heat.max()),1e-6)*255,0,255).astype(np.uint8)
        cmap = COLORMAP_DICT.get(self.args.colormap.upper(),cv2.COLORMAP_JET)
        cv2.imwrite(str(out_path.with_name(out_path.stem+"_heatmap.png")),
                    cv2.applyColorMap(heat_norm,cmap))
        report = {"source":str(self.args.source),"frames_processed":frame_idx,
            "unique_track_ids":len(total_seen),"elapsed_seconds":round(time.time()-started,2),
            "zones":self.zones,"zone_history":zone_history}
        out_path.with_name(out_path.stem+"_report.json").write_text(
            json.dumps(report,indent=2),encoding="utf-8")
        print(f"[DONE] Video: {out_path}")
        print(f"[DONE] Heatmap: {out_path.with_name(out_path.stem+'_heatmap.png')}")
        print(f"[DONE] Report: {out_path.with_name(out_path.stem+'_report.json')}")

def parse_args():
    p = argparse.ArgumentParser(description="YOLO + ByteTrack human heatmap and zone analytics")
    p.add_argument("--source",default="0",help="Camera index or video path")
    p.add_argument("--model",default="yolo11n.pt",help="Ultralytics YOLO weights")
    p.add_argument("--output",default="output/human_heatmap_tracking.mp4")
    p.add_argument("--zones",default="config/zones.json",help="Polygon zones JSON")
    p.add_argument("--conf",type=float,default=.35)
    p.add_argument("--iou",type=float,default=.5)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--radius",type=int,default=28)
    p.add_argument("--alpha",type=float,default=.42)
    p.add_argument("--decay",type=float,default=.995)
    p.add_argument("--decay-interval",type=int,default=1)
    p.add_argument("--tail-len",type=int,default=30)
    p.add_argument("--frame-skip",type=int,default=1)
    p.add_argument("--report-interval",type=int,default=25)
    p.add_argument("--colormap",default="JET",choices=list(COLORMAP_DICT))
    p.add_argument("--display",action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    HeatmapTrackerPipeline(parse_args()).run()
