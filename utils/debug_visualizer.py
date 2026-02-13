"""
Debug Visualizer Module
用于在项目运行时保存真实的中间数据和可视化图，不影响正常运行。
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示窗口
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime


class DebugVisualizer:
    """调试可视化器，保存运行时的中间数据和可视化图"""

    def __init__(self, output_dir, enabled=True):
        """
        初始化可视化器

        Args:
            output_dir: 输出目录路径
            enabled: 是否启用可视化（设为False可完全禁用，不影响运行）
        """
        self.enabled = enabled
        if not enabled:
            return

        self.output_dir = os.path.join(output_dir, "debug_vis")
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建时间戳用于区分不同运行
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _safe_execute(self, func, *args, **kwargs):
        """安全执行函数，出错时不影响主流程"""
        if not self.enabled:
            return None
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[DebugVisualizer Warning] {func.__name__} failed: {e}")
            return None

    # =========================================================================
    # 1. 保存 Qwen 的 System Prompt 和用户指令
    # =========================================================================
    def save_qwen_prompts(self, messages, input_text, base64_image_preview=None):
        """
        保存 Qwen-VL-Max 的 System Prompt 和用户指令

        Args:
            messages: 发送给 Qwen 的完整消息列表
            input_text: 用户的任务文本
            base64_image_preview: 可选，base64图像的前100字符（用于确认）
        """
        def _save():
            output_path = os.path.join(self.output_dir, "1_qwen_prompts.txt")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("QWEN-VL-MAX INPUT PROMPTS\n")
                f.write("=" * 80 + "\n\n")

                for msg in messages:
                    f.write(f"[{msg['role'].upper()}]\n")
                    f.write("-" * 40 + "\n")

                    if isinstance(msg['content'], str):
                        f.write(msg['content'] + "\n")
                    elif isinstance(msg['content'], list):
                        for item in msg['content']:
                            if item.get('type') == 'text':
                                f.write(f"Text: {item['text']}\n")
                            elif item.get('type') == 'image_url':
                                f.write(f"Image: [Base64 Image, length={len(item['image_url']['url'])}]\n")
                    f.write("\n")

                f.write("=" * 80 + "\n")
                f.write(f"User Task Input: {input_text}\n")
                f.write("=" * 80 + "\n")

            print(f"[DebugVisualizer] Saved Qwen prompts to {output_path}")
            return output_path

        return self._safe_execute(_save)

    # =========================================================================
    # 2. 保存 Qwen 的实际文本输出
    # =========================================================================
    def save_qwen_output(self, raw_output, parsed_result):
        """
        保存 Qwen 的实际文本输出（包括推理过程和最终格式结果）

        Args:
            raw_output: Qwen 的原始输出文本
            parsed_result: 解析后的结果字典 (selected_object_id, class_name)
        """
        def _save():
            output_path = os.path.join(self.output_dir, "2_qwen_output.txt")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("QWEN-VL-MAX RAW OUTPUT\n")
                f.write("=" * 80 + "\n\n")
                f.write(raw_output + "\n\n")

                f.write("=" * 80 + "\n")
                f.write("PARSED RESULT\n")
                f.write("=" * 80 + "\n")
                f.write(json.dumps(parsed_result, indent=2, ensure_ascii=False) + "\n")

            print(f"[DebugVisualizer] Saved Qwen output to {output_path}")
            return output_path

        return self._safe_execute(_save)

    # =========================================================================
    # 3. 两阶段实例分割可视化 - LangSAM 所有掩码
    # =========================================================================
    def save_langsam_all_masks(self, image_pil, masks, boxes, logits, class_name):
        """
        可视化 LangSAM 输出的所有同类实例掩码，用不同颜色半透明覆盖

        Args:
            image_pil: PIL Image 原图
            masks: LangSAM 输出的所有掩码 (list of tensors)
            boxes: 对应的边界框
            logits: 对应的置信度分数
            class_name: 查询的类别名称
        """
        def _save():
            if masks is None or len(masks) == 0:
                print("[DebugVisualizer] No masks to visualize")
                return None

            output_path = os.path.join(self.output_dir, "3_langsam_all_masks.png")

            # 转换图像
            image_np = np.array(image_pil)

            # 创建图像
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.imshow(image_np)

            # 不同颜色的掩码
            colors = [
                (1, 0, 0, 0.4),      # 红色
                (0, 0, 1, 0.4),      # 蓝色
                (0, 1, 0, 0.4),      # 绿色
                (1, 1, 0, 0.4),      # 黄色
                (1, 0, 1, 0.4),      # 洋红
                (0, 1, 1, 0.4),      # 青色
                (1, 0.5, 0, 0.4),    # 橙色
                (0.5, 0, 1, 0.4),    # 紫色
            ]

            h, w = image_np.shape[:2]

            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze()

                color = colors[i % len(colors)]

                # 创建彩色掩码覆盖
                colored_mask = np.zeros((h, w, 4), dtype=np.float32)
                colored_mask[mask_np > 0] = color
                ax.imshow(colored_mask)

                # 获取掩码中心位置用于标注
                ys, xs = np.where(mask_np > 0)
                if len(xs) > 0 and len(ys) > 0:
                    cx, cy = xs.mean(), ys.mean()
                    logit_val = logits[i].item() if hasattr(logits[i], 'item') else logits[i]
                    ax.text(cx, cy, f'Mask {i}\nlogit={logit_val:.2f}',
                           fontsize=10, ha='center', va='center',
                           color='white', fontweight='bold',
                           bbox=dict(facecolor=color[:3], alpha=0.8, edgecolor='white', pad=2))

                # 绘制边界框
                if boxes is not None and i < len(boxes):
                    box = boxes[i].cpu().numpy() if hasattr(boxes[i], 'cpu') else boxes[i]
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                         linewidth=2, edgecolor=color[:3],
                                         facecolor='none', linestyle='--')
                    ax.add_patch(rect)

            ax.set_title(f'LangSAM Stage 1: All masks for "{class_name}" ({len(masks)} instances)',
                        fontsize=14, fontweight='bold')
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[DebugVisualizer] Saved LangSAM all masks to {output_path}")
            return output_path

        return self._safe_execute(_save)

    # =========================================================================
    # 4. 坐标匹配阶段可视化
    # =========================================================================
    def save_coordinate_matching(self, image_pil, masks, goal_coor, selected_index):
        """
        可视化坐标匹配阶段：Molmo 坐标点（红色十字）和掩码匹配

        Args:
            image_pil: PIL Image 原图
            masks: LangSAM 输出的所有掩码
            goal_coor: Molmo 提供的目标坐标 (x, y)
            selected_index: 被选中的掩码索引
        """
        def _save():
            if masks is None or len(masks) == 0:
                return None

            output_path = os.path.join(self.output_dir, "4_coordinate_matching.png")

            image_np = np.array(image_pil)
            h, w = image_np.shape[:2]

            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.imshow(image_np)

            colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

            goal_x, goal_y = goal_coor

            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze()

                color = colors[i % len(colors)]
                alpha = 0.35 if i != selected_index else 0.5
                linestyle = '--' if i != selected_index else '-'
                linewidth = 1 if i != selected_index else 3

                # 绘制掩码轮廓
                colored_mask = np.zeros((h, w, 4), dtype=np.float32)
                colored_mask[mask_np > 0] = (*color, alpha)
                ax.imshow(colored_mask)

                # 检查坐标是否在掩码内
                is_inside = mask_np[goal_y, goal_x] if 0 <= goal_y < h and 0 <= goal_x < w else False

                # 标注
                ys, xs = np.where(mask_np > 0)
                if len(xs) > 0:
                    cx, cy = xs.mean(), ys.mean()
                    marker = '✓' if is_inside else '✗'
                    marker_color = '#00cc00' if is_inside else '#cc0000'
                    ax.text(cx, cy - 20, marker, fontsize=24, ha='center',
                           color=marker_color, fontweight='bold')
                    ax.text(cx, cy + 10, f'Mask {i}', fontsize=9, ha='center',
                           color='white', fontweight='bold',
                           bbox=dict(facecolor=color, alpha=0.7, pad=2))

            # 绘制 Molmo 目标坐标点（红色十字）
            cross_size = 25
            ax.plot([goal_x - cross_size, goal_x + cross_size], [goal_y, goal_y],
                   'r-', linewidth=3, zorder=10)
            ax.plot([goal_x, goal_x], [goal_y - cross_size, goal_y + cross_size],
                   'r-', linewidth=3, zorder=10)
            ax.plot(goal_x, goal_y, 'r*', markersize=15, zorder=11)

            # 坐标标注框
            ax.text(goal_x + 30, goal_y, f'Molmo goal_coor\n({goal_x}, {goal_y})',
                   fontsize=11, color='red', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', linewidth=2, pad=4),
                   zorder=12)

            ax.set_title(f'Stage 2: Coordinate Matching - Selected Mask {selected_index}\n'
                        f'mask[{goal_y}, {goal_x}] == True', fontsize=14, fontweight='bold')
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[DebugVisualizer] Saved coordinate matching to {output_path}")
            return output_path

        return self._safe_execute(_save)

    # =========================================================================
    # 5. 最终选中的单一实例掩码高亮
    # =========================================================================
    def save_final_selected_mask(self, image_pil, goal_mask, goal_bbox=None):
        """
        可视化最终选中的单一实例掩码

        Args:
            image_pil: PIL Image 原图
            goal_mask: 选中的目标掩码
            goal_bbox: 目标边界框 (可选)
        """
        def _save():
            if goal_mask is None:
                return None

            output_path = os.path.join(self.output_dir, "5_final_selected_mask.png")

            image_np = np.array(image_pil)
            h, w = image_np.shape[:2]

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # 左图：原图 + 掩码覆盖
            ax1 = axes[0]
            ax1.imshow(image_np)

            mask_np = goal_mask.cpu().numpy() if hasattr(goal_mask, 'cpu') else goal_mask
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze()

            # 绿色高亮掩码
            colored_mask = np.zeros((h, w, 4), dtype=np.float32)
            colored_mask[mask_np > 0] = (0, 1, 0, 0.5)
            ax1.imshow(colored_mask)

            # 绘制边界框
            if goal_bbox is not None:
                x1, y1, x2, y2 = goal_bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=3, edgecolor='#00ff00',
                                     facecolor='none')
                ax1.add_patch(rect)

            ax1.set_title('Final Selected Mask (RGB + Overlay)', fontsize=12, fontweight='bold')
            ax1.axis('off')

            # 右图：仅掩码（二值图）
            ax2 = axes[1]
            ax2.imshow(mask_np, cmap='gray')
            ax2.set_title(f'Binary Mask ({mask_np.shape[0]}x{mask_np.shape[1]})',
                         fontsize=12, fontweight='bold')
            ax2.axis('off')

            plt.suptitle('Stage 3: Final goal_mask -> Pass to GraspNet', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[DebugVisualizer] Saved final selected mask to {output_path}")
            return output_path

        return self._safe_execute(_save)

    # =========================================================================
    # 6. 点云裁剪时带有实例掩码覆盖的 RGB 图像
    # =========================================================================
    def save_rgb_with_mask_for_pointcloud(self, image_np, goal_mask, cropping_box):
        """
        保存点云裁剪阶段的 RGB 图像 + 掩码覆盖

        Args:
            image_np: numpy array RGB 图像
            goal_mask: 目标掩码
            cropping_box: 裁剪框 (x1, y1, x2, y2)
        """
        def _save():
            output_path = os.path.join(self.output_dir, "6_rgb_mask_for_pointcloud.png")

            h, w = image_np.shape[:2]

            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.imshow(image_np)

            # 绘制掩码
            if goal_mask is not None:
                mask_np = goal_mask.cpu().numpy() if hasattr(goal_mask, 'cpu') else goal_mask
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze()

                colored_mask = np.zeros((h, w, 4), dtype=np.float32)
                colored_mask[mask_np > 0] = (0, 1, 0, 0.4)
                ax.imshow(colored_mask)

            # 绘制裁剪框
            if cropping_box is not None:
                x1, y1, x2, y2 = cropping_box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=3, edgecolor='red',
                                     facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.text(x1, y1-10, f'Cropping Box\n({x1}, {y1}) - ({x2}, {y2})',
                       fontsize=10, color='red', fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.8, pad=2))

            ax.set_title('RGB Image + Instance Mask + Cropping Box for Point Cloud',
                        fontsize=14, fontweight='bold')
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[DebugVisualizer] Saved RGB with mask to {output_path}")
            return output_path

        return self._safe_execute(_save)

    # =========================================================================
    # 7. 场景点云可视化（目标物体高亮）
    # =========================================================================
    def save_scene_pointcloud(self, cloud_full, color_full, goal_mask, depth, camera):
        """
        可视化从深度图反投影生成的场景点云，目标物体用不同颜色高亮

        Args:
            cloud_full: 完整场景点云 (H, W, 3) 或 (N, 3)
            color_full: 点云颜色 (H, W, 3) 或 (N, 3)
            goal_mask: 目标物体掩码
            depth: 深度图
            camera: 相机参数
        """
        def _save():
            output_path = os.path.join(self.output_dir, "7_scene_pointcloud.png")

            # 处理掩码
            mask_np = goal_mask.cpu().numpy() if hasattr(goal_mask, 'cpu') else goal_mask
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze()

            # 确保点云是正确的形状
            if cloud_full.ndim == 3:
                h, w, _ = cloud_full.shape
                cloud_flat = cloud_full.reshape(-1, 3)
                color_flat = color_full.reshape(-1, 3) if color_full is not None else None
                mask_flat = mask_np.reshape(-1)
                depth_flat = depth.reshape(-1)
            else:
                cloud_flat = cloud_full
                color_flat = color_full
                mask_flat = mask_np.reshape(-1) if mask_np.ndim > 1 else mask_np
                depth_flat = depth.reshape(-1)

            # 过滤有效点（深度 > 0）
            valid_mask = depth_flat > 0
            cloud_valid = cloud_flat[valid_mask]
            mask_valid = mask_flat[valid_mask]

            if color_flat is not None:
                color_valid = color_flat[valid_mask]
            else:
                color_valid = np.ones_like(cloud_valid) * 0.5

            # 下采样以便可视化
            max_points = 50000
            if len(cloud_valid) > max_points:
                indices = np.random.choice(len(cloud_valid), max_points, replace=False)
                cloud_vis = cloud_valid[indices]
                color_vis = color_valid[indices]
                mask_vis = mask_valid[indices]
            else:
                cloud_vis = cloud_valid
                color_vis = color_valid
                mask_vis = mask_valid

            # 创建 3D 图
            fig = plt.figure(figsize=(16, 7))

            # 左图：完整场景，目标高亮
            ax1 = fig.add_subplot(121, projection='3d')

            # 非目标点（灰色）
            non_target = ~mask_vis.astype(bool)
            if np.any(non_target):
                ax1.scatter(cloud_vis[non_target, 0],
                           cloud_vis[non_target, 1],
                           cloud_vis[non_target, 2],
                           c=color_vis[non_target] * 0.5 + 0.3,
                           s=0.5, alpha=0.3)

            # 目标点（绿色高亮）
            target = mask_vis.astype(bool)
            if np.any(target):
                ax1.scatter(cloud_vis[target, 0],
                           cloud_vis[target, 1],
                           cloud_vis[target, 2],
                           c='lime', s=3, alpha=0.9)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('Scene Point Cloud\n(Target Object Highlighted in Green)',
                         fontsize=11, fontweight='bold')
            ax1.view_init(elev=25, azim=-60)

            # 右图：仅目标物体点云
            ax2 = fig.add_subplot(122, projection='3d')

            if np.any(target):
                target_cloud = cloud_vis[target]
                target_color = color_vis[target]
                ax2.scatter(target_cloud[:, 0],
                           target_cloud[:, 1],
                           target_cloud[:, 2],
                           c=target_color, s=5, alpha=0.8)

                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_zlabel('Z (m)')
                ax2.set_title(f'Target Object Point Cloud\n({np.sum(target)} points)',
                             fontsize=11, fontweight='bold')
                ax2.view_init(elev=30, azim=-45)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[DebugVisualizer] Saved scene pointcloud to {output_path}")
            return output_path

        return self._safe_execute(_save)

    # =========================================================================
    # 辅助方法：保存完整的调试摘要
    # =========================================================================
    def save_summary(self, data_dict):
        """
        保存运行摘要信息

        Args:
            data_dict: 包含所有关键信息的字典
        """
        def _save():
            output_path = os.path.join(self.output_dir, "0_debug_summary.json")

            # 过滤不可序列化的对象
            serializable_data = {}
            for k, v in data_dict.items():
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    serializable_data[k] = v
                elif isinstance(v, np.ndarray):
                    serializable_data[k] = f"<numpy.ndarray shape={v.shape}>"
                elif hasattr(v, 'shape'):
                    serializable_data[k] = f"<tensor shape={v.shape}>"
                else:
                    serializable_data[k] = str(type(v))

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

            print(f"[DebugVisualizer] Saved summary to {output_path}")
            return output_path

        return self._safe_execute(_save)
