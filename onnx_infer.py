import numpy as np
import onnxruntime as ort
import cv2
import argparse
from pathlib import Path
from pyzbar.pyzbar import decode

def apply_clahe(image):
    """光照补偿（对比度受限自适应直方图均衡化）"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def apply_super_resolution(image, scale=2.0):
    """超分辨率重建（基于插值放大 + 锐化）"""
    if image.shape[0] < 100:  # 仅对小图像放大
        h, w = image.shape[:2]
        enlarged = cv2.resize(image, (int(w * scale), int(h * scale)), 
                            interpolation=cv2.INTER_CUBIC)
        # 锐化处理
        kernel = np.array([[-1, -1, -1], 
                          [-1, 9, -1], 
                          [-1, -1, -1]])
        return cv2.filter2D(enlarged, -1, kernel)
    return image

def preprocess(image, img_size=(640, 640), enhance=True):
    """图像预处理（含增强选项）"""
    if enhance:
        image = apply_clahe(image)  # 光照补偿
    image = cv2.resize(image, img_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1)
    return np.expand_dims(image.astype(np.float32) / 255.0, 0)

def postprocess(output, conf_thresh=0.5, iou_thresh=0.45, max_det=100):
    """后处理（NMS 过滤）"""
    detections = []
    for pred in output[0]:  # 遍历 25200 个预测框
        x, y, w, h, obj_conf, class_conf = pred
        confidence = obj_conf * class_conf
        if confidence < conf_thresh:
            continue
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        detections.append([x1, y1, x2, y2, confidence, 0])  # cls_id=0（单类别）
    
    boxes = np.array([d[:4] for d in detections])
    confidences = np.array([d[4] for d in detections])
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_thresh, iou_thresh)
    return [detections[i] for i in indices] if len(indices) > 0 else []

def expand_and_crop(image, box, expand_ratio=0.3):
    """外扩框并裁剪图像（动态调整外扩比例）"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    # 动态外扩：小目标外扩更多
    qr_size = max(x2 - x1, y2 - y1)
    expand_ratio = 0.4 if qr_size < 0.1 * min(h, w) else 0.2
    dw, dh = (x2 - x1) * expand_ratio / 2, (y2 - y1) * expand_ratio / 2
    x1 = max(0, int((x1 - dw) * w))
    y1 = max(0, int((y1 - dh) * h))
    x2 = min(w, int((x2 + dw) * w))
    y2 = min(h, int((y2 + dh) * h))
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

def decode_with_zbar_enhanced(image):
    """增强版 ZBar 解码（含超分辨率 + 多策略尝试）"""
    # 超分辨率重建
    sr_image = apply_super_resolution(image)
    
    # 多策略解码尝试
    for gray in [
        cv2.cvtColor(sr_image, cv2.COLOR_BGR2GRAY),  # 原始灰度
        cv2.threshold(cv2.cvtColor(sr_image, cv2.COLOR_BGR2GRAY), 0, 255, 
                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],  # 二值化
        cv2.adaptiveThreshold(cv2.cvtColor(sr_image, cv2.COLOR_BGR2GRAY), 255, 
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                            cv2.THRESH_BINARY, 11, 2)  # 自适应阈值
    ]:
        results = decode(gray)
        if results:
            return results[0].data.decode('utf-8'), gray
    return None, gray

def run_inference(weights, source, img_size=(640, 640), conf_thresh=0.5, 
                 iou_thresh=0.45, max_det=100, enhance=True, sr_scale=2.0,
                 save_img=False, view_img=False, show_crop=False, project=None, name=None):
    # 初始化 ONNX Runtime
    sess = ort.InferenceSession(weights, providers=['CUDAExecutionProvider' if ort.get_device() == 'GPU' else 'CPUExecutionProvider'])
    output_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name

    # 输入源处理（支持摄像头/视频/图片）
    cap = cv2.VideoCapture(int(source)) if source.isnumeric() else cv2.VideoCapture(source) if isinstance(source, str) else None

    # 创建保存目录
    save_dir = Path(project) / name if (save_img and project) else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    while True:
        # 读取帧
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = cv2.imread(source)
            if frame is None:
                raise FileNotFoundError(f"Input not found: {source}")

        orig_h, orig_w = frame.shape[:2]
        input_tensor = preprocess(frame.copy(), img_size, enhance)
        output = sess.run([output_name], {input_name: input_tensor})[0]

        # 检测后处理
        detections = postprocess(output, conf_thresh, iou_thresh, max_det)
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            # 转换为绝对坐标
            x1_abs, y1_abs = int(x1 * orig_w / img_size[0]), int(y1 * orig_h / img_size[1])
            x2_abs, y2_abs = int(x2 * orig_w / img_size[0]), int(y2 * orig_h / img_size[1])
            
            # 外扩裁剪
            cropped, _ = expand_and_crop(frame, 
                [x1_abs/orig_w, y1_abs/orig_h, x2_abs/orig_w, y2_abs/orig_h])
            
            if cropped.size > 0:
                # 增强解码
                qr_data, processed_img = decode_with_zbar_enhanced(cropped)
                
                # 在主窗口绘制检测框
                cv2.rectangle(frame, (x1_abs, y1_abs), (x2_abs, y2_abs), (0, 255, 0), 2)
                cv2.putText(frame, f"QR: {qr_data}" if qr_data else "No QR", 
                          (x1_abs, y1_abs-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示裁剪窗口（调试用）
                if show_crop:
                    cv2.imshow("Cropped QR", processed_img)
                    cv2.waitKey(1)
                
                if qr_data:
                    print(f"Decoded: {qr_data} | Confidence: {conf:.2f}")

        # 显示主窗口
        if view_img:
            cv2.imshow("QR Detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break
        
        # 保存结果
        if save_img and save_dir:
            if cap:
                save_path = save_dir / f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"
            else:
                save_path = save_dir / Path(source).name
            cv2.imwrite(str(save_path), frame)

        if not cap:
            break  # 单张图片处理完后退出

    if cap:
        cap.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 + ZBar QR Code Detector with Enhancement")
    parser.add_argument('--weights', type=str, default='yolov5s.onnx', help='模型路径')
    parser.add_argument('--source', type=str, default='test.jpg', help='输入源（图片/视频/摄像头 ID）')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='推理尺寸 [h, w]')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thresh', type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--max-det', type=int, default=100, help='每帧最大检测数')
    parser.add_argument('--no-enhance', action='store_false', dest='enhance', help='禁用光照补偿')
    parser.add_argument('--sr-scale', type=float, default=2.0, help='超分辨率放大倍数')
    parser.add_argument('--save-img', action='store_true', help='保存结果')
    parser.add_argument('--view-img', action='store_true', help='实时显示检测窗口')
    parser.add_argument('--show-crop', action='store_true', help='显示裁剪的二维码区域')
    parser.add_argument('--project', type=str, default='runs/detect', help='结果保存目录')
    parser.add_argument('--name', type=str, default='exp', help='结果保存子目录名')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(
        weights=args.weights,
        source=args.source,
        img_size=tuple(args.img_size),
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        max_det=args.max_det,
        enhance=args.enhance,
        sr_scale=args.sr_scale,
        save_img=args.save_img,
        view_img=args.view_img,
        show_crop=args.show_crop,
        project=args.project,
        name=args.name
    )