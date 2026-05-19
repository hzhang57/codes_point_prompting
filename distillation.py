"""
知识蒸馏模块（论文 Section 4）：将 PointPrompter 教师模型生成的伪标签
用于训练轻量级学生模型（CoTracker 风格），实现实时点跟踪推理。

训练流程：
  1. 对无标注视频运行教师模型（PointPrompter），得到伪标签轨迹。
  2. 用伪标签监督学生模型（ConvGRU），损失 = L1 位置误差 + BCE 遮挡预测。
  3. 学生推理时无需扩散模型，速度大幅提升。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
#  伪标签数据集                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class PseudoSample:
    """单个视频的伪标签样本。"""
    frames: np.ndarray    # (T, H, W, 3) BGR uint8，视频帧
    tracks: np.ndarray    # (N, T, 2) float32 — N 个查询点在每帧的坐标
    visible: np.ndarray   # (N, T) bool，True 表示该帧该点可见（未遮挡）


class PseudoLabelDataset(Dataset):
    """将伪标签样本列表封装为 PyTorch Dataset，支持随机访问和批量加载。"""

    def __init__(self, samples: List[PseudoSample], T: int = 16, size: Tuple[int,int] = (256, 256)):
        self.samples = samples
        self.T = T
        self.size = size  # 统一缩放到此分辨率 (H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        import cv2
        # 将每帧缩放到统一分辨率，方便批次化处理
        frames = np.stack([
            cv2.resize(f, self.size[::-1]) for f in s.frames
        ], axis=0)  # (T, H, W, 3)
        # 归一化到 [-1, 1]，并将通道维移到第二维
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        tracks  = torch.from_numpy(s.tracks).float()   # (N, T, 2) 像素坐标
        visible = torch.from_numpy(s.visible).bool()   # (N, T) 可见性标志
        return frames_t, tracks, visible


# --------------------------------------------------------------------------- #
#  轻量级学生模型：基于 ConvGRU 的点跟踪器                                      #
# --------------------------------------------------------------------------- #

class ConvFeatureExtractor(nn.Module):
    """浅层 CNN 主干，提取每帧的特征图。

    三层卷积将输入分辨率下采样 8 倍，输出密集特征图用于后续相关计算。
    """

    def __init__(self, in_ch: int = 3, feat_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # 第 1 层：7×7 卷积，stride=2 → 分辨率减半
            nn.Conv2d(in_ch, 32, 7, stride=2, padding=3), nn.ReLU(True),
            # 第 2 层：3×3 卷积，stride=2 → 再减半
            nn.Conv2d(32, 64, 3, stride=2, padding=1),    nn.ReLU(True),
            # 第 3 层：3×3 卷积，stride=2 → 总共 8 倍下采样
            nn.Conv2d(64, feat_ch, 3, stride=2, padding=1), nn.ReLU(True),
        )

    def forward(self, x):  # (B, C, H, W) → (B, feat_ch, H/8, W/8)
        return self.net(x)


class StudentTracker(nn.Module):
    """
    轻量级循环跟踪器：提取特征后，通过局部相关 + GRU 迭代精细化轨迹。

    输入：
        frames: (B, T, 3, H, W) — 视频帧序列
        query:  (B, N, 2) — 第 0 帧中的查询点像素坐标 (x, y)

    输出：
        tracks:     (B, N, T, 2) — 预测的每帧像素坐标
        vis_logits: (B, N, T)   — 可见性 logits（sigmoid 后为可见概率）
    """

    def __init__(self, feat_ch: int = 64, hidden: int = 128, corr_r: int = 4):
        super().__init__()
        self.feat = ConvFeatureExtractor(feat_ch=feat_ch)
        self.corr_r = corr_r
        corr_dim = (2 * corr_r + 1) ** 2  # 局部窗口内的采样点数
        # GRU 输入 = 相关体积特征 + 当前预测位置 (2维)
        self.gru   = nn.GRUCell(input_size=corr_dim + 2, hidden_size=hidden)
        # 位置残差预测头：输出 Δ(x, y)
        self.head  = nn.Linear(hidden, 2)
        # 可见性预测头：输出单个 logit
        self.vis   = nn.Linear(hidden, 1)

    def _sample_feat(self, feat_map, points_norm):
        """在归一化坐标 [-1,1] 处双线性采样特征图，返回 (B, N, C)。"""
        grid = points_norm.unsqueeze(2)  # (B, N, 1, 2)
        vals = F.grid_sample(feat_map, grid, align_corners=True)  # (B, C, N, 1)
        return vals.squeeze(-1).permute(0, 2, 1)  # (B, N, C)

    def _correlation(self, feat_map, points_norm):
        """
        局部相关体积：在每个查询点周围采样 (2r+1)×(2r+1) 的邻域特征均值。
        返回 (B, N, (2r+1)^2)，编码了局部外观线索。
        """
        B, C, H, W = feat_map.shape
        r = self.corr_r
        # 构建归一化偏移量网格（相对偏移，适应特征图分辨率）
        offsets = torch.stack(
            torch.meshgrid(
                torch.linspace(-r/W, r/W, 2*r+1, device=feat_map.device),
                torch.linspace(-r/H, r/H, 2*r+1, device=feat_map.device),
                indexing='xy'
            ), dim=-1
        ).reshape(-1, 2)   # ((2r+1)^2, 2)

        N = points_norm.shape[1]
        # 将偏移量加到当前预测位置，得到采样坐标
        pts_exp = points_norm.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)  # (B,N,K,2)
        grid = pts_exp.reshape(B, N * (2*r+1)**2, 1, 2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, padding_mode="border")
        # 对通道维取均值，得到标量相关值
        sampled = sampled.reshape(B, C, N, (2*r+1)**2).mean(dim=1)  # (B, N, K)
        return sampled

    def forward(self, frames, query):
        B, T, _, H, W = frames.shape
        N = query.shape[1]

        # 批量提取所有帧的特征，减少循环开销
        frames_flat = frames.reshape(B * T, 3, H, W)
        feats_flat  = self.feat(frames_flat)
        _, C, fH, fW = feats_flat.shape
        feats = feats_flat.reshape(B, T, C, fH, fW)

        # 将查询点像素坐标归一化到 [-1, 1]（与 grid_sample 约定一致）
        scale = torch.tensor([W - 1, H - 1], device=frames.device, dtype=frames.dtype)
        pts = query / scale * 2 - 1   # (B, N, 2)

        # 初始化 GRU 隐藏状态
        h = torch.zeros(B * N, self.gru.hidden_size, device=frames.device, dtype=frames.dtype)
        all_tracks = []
        all_vis    = []

        for t in range(T):
            feat_t = feats[:, t]  # (B, C, fH, fW) 当前帧特征
            # 计算当前点位置的局部相关体积
            corr   = self._correlation(feat_t, pts)                  # (B, N, K)
            inp    = torch.cat([corr, pts], dim=-1)                  # (B, N, K+2)
            inp_   = inp.reshape(B * N, -1)
            # GRU 更新隐藏状态
            h      = self.gru(inp_, h)
            # 预测位置残差和可见性
            delta  = self.head(h).reshape(B, N, 2)
            vis_l  = self.vis(h).reshape(B, N)

            # 小步长残差更新：0.05 防止预测跳变过大
            pts = (pts + delta * 0.05).clamp(-1, 1)
            all_tracks.append(pts)
            all_vis.append(vis_l)

        tracks_norm = torch.stack(all_tracks, dim=2)   # (B, N, T, 2) 归一化坐标
        vis_logits  = torch.stack(all_vis,    dim=2)   # (B, N, T) 可见性 logits

        # 反归一化回像素坐标
        tracks_px = (tracks_norm + 1) / 2 * scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return tracks_px, vis_logits


# --------------------------------------------------------------------------- #
#  训练循环                                                                      #
# --------------------------------------------------------------------------- #

def train_student(
    student: StudentTracker,
    dataset: PseudoLabelDataset,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """用伪标签数据集训练学生模型。

    损失函数 = L1 位置损失（仅对可见帧）+ 0.1 × BCE 遮挡损失。
    使用梯度裁剪（max_norm=1.0）防止训练不稳定。
    """
    student = student.to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        total_loss = 0.0
        for frames, tracks_gt, visible in loader:
            frames    = frames.to(device)       # (B, T, 3, H, W)
            tracks_gt = tracks_gt.to(device)    # (B, N, T, 2)
            visible   = visible.to(device)      # (B, N, T)

            # 用第 0 帧的真实坐标作为初始查询点
            query = tracks_gt[:, :, 0, :]       # (B, N, 2)

            tracks_pred, vis_logits = student(frames, query)

            # 仅对可见帧计算位置损失，不惩罚遮挡帧的位置误差
            vis_f = visible.float()
            pos_loss = (F.l1_loss(tracks_pred, tracks_gt, reduction='none') * vis_f.unsqueeze(-1)).mean()
            # 二元交叉熵损失：监督可见性预测
            vis_loss = F.binary_cross_entropy_with_logits(vis_logits, vis_f)
            loss = pos_loss + 0.1 * vis_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

    return student


# --------------------------------------------------------------------------- #
#  伪标签生成辅助函数                                                            #
# --------------------------------------------------------------------------- #

def generate_pseudo_labels(
    tracker,                          # PointPrompter 实例（教师模型）
    video_paths: List[str],
    n_points_per_video: int = 10,     # 每个视频随机采样的查询点数量
    frame_limit: int = 24,            # 最多读取的帧数（防止显存溢出）
) -> List[PseudoSample]:
    """对无标注视频集合运行教师跟踪器，生成伪标签样本列表。

    查询点在视频第 0 帧的中心区域（10%~90% 范围内）随机采样，
    避免选取图像边缘区域导致标记超出画面。
    """
    import cv2
    samples = []
    for path in video_paths:
        # 逐帧读取视频，达到帧数上限后停止
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < frame_limit:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()
        if len(frames) < 2:
            continue  # 跳过过短的视频

        H, W = frames[0].shape[:2]
        # 在图像中心区域随机采样查询点（避免边缘区域）
        xs = np.random.uniform(0.1 * W, 0.9 * W, n_points_per_video)
        ys = np.random.uniform(0.1 * H, 0.9 * H, n_points_per_video)
        query_points = list(zip(xs.tolist(), ys.tolist()))

        # 教师模型批量跟踪所有查询点（每点独立运行完整流水线）
        results = tracker.track_multiple(frames, query_points)

        tracks  = np.stack([r.tracks  for r in results], axis=0)   # (N, T, 2)
        visible = np.stack([r.visible for r in results], axis=0)   # (N, T)
        samples.append(PseudoSample(
            frames  = np.stack(frames, axis=0),
            tracks  = tracks,
            visible = visible,
        ))
    return samples
