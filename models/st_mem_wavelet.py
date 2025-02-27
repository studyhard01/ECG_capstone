# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
import pywt
import numpy as np

from models.encoder.st_mem_vit import ST_MEM_ViT, TransformerBlock


__all__ = ['ST_MEM', 'st_mem_vit_small_dec256d4b', 'st_mem_vit_base_dec256d4b']


def get_1d_sincos_pos_embed(embed_dim: int,
                            grid_size: int,
                            temperature: float = 10000,
                            sep_embed: bool = False):
    """Positional embedding for 1D patches.
    """
    assert (embed_dim % 2) == 0, \
        'feature dimension must be multiple of 2 for sincos emb.'
    grid = torch.arange(grid_size, dtype=torch.float32)

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / (temperature ** omega)

    grid = grid.flatten()[:, None] * omega[None, :]
    pos_embed = torch.cat((grid.sin(), grid.cos()), dim=1)
    if sep_embed:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed, torch.zeros(1, embed_dim)],
                              dim=0)
    return pos_embed


class ST_MEM(nn.Module):
    def __init__(self,
                 seq_len: int = 2250,
                 patch_size: int = 75,
                 num_leads: int = 12,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 decoder_embed_dim: int = 256,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 4,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm,
                 norm_pix_loss: bool = False):
        super().__init__()
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'embed_dim': embed_dim,
                           'depth': depth,
                           'num_heads': num_heads,
                           'decoder_embed_dim': decoder_embed_dim,
                           'decoder_depth': decoder_depth,
                           'decoder_num_heads': decoder_num_heads,
                           'mlp_ratio': mlp_ratio,
                           'qkv_bias': qkv_bias,
                           'norm_layer': str(norm_layer),
                           'norm_pix_loss': norm_pix_loss}
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.num_leads = num_leads
        # --------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = ST_MEM_ViT(seq_len=seq_len,
                                  patch_size=patch_size,
                                  num_leads=num_leads,
                                  width=embed_dim,
                                  depth=depth,
                                  mlp_dim=mlp_ratio * embed_dim,
                                  heads=num_heads,
                                  qkv_bias=qkv_bias)
        self.to_patch_embedding = self.encoder.to_patch_embedding
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # MAE decoder specifics
        self.to_decoder_embedding = nn.Linear(embed_dim, decoder_embed_dim)

        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # self.decoder_pos_embed = nn.Parameter(
        #     torch.zeros(1, self.num_patches + 2, decoder_embed_dim),
        #     requires_grad=False
        # )

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, 7 + 2, decoder_embed_dim), ## 25% 마스킹 기준 = 7 !! (나중에 비율 다르게 변경하면 값 변경 필요)
            requires_grad=False
        )

        # self.decoder_blocks = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
        #                                                       output_dim=decoder_embed_dim,
        #                                                       hidden_dim=decoder_embed_dim * mlp_ratio,
        #                                                       heads=decoder_num_heads,
        #                                                       dim_head=64,
        #                                                       qkv_bias=qkv_bias)
        #                                      for _ in range(decoder_depth)])

        self.decoder_blocks_L = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
                                                                output_dim=decoder_embed_dim,
                                                                hidden_dim=decoder_embed_dim * mlp_ratio,
                                                                heads=decoder_num_heads,
                                                                dim_head=64,
                                                                qkv_bias=qkv_bias)
                                               for _ in range(decoder_depth)])
        self.proj_out_L = nn.Linear(decoder_embed_dim * 7, 435)

        self.decoder_blocks_H1 = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
                                                                 output_dim=decoder_embed_dim,
                                                                 hidden_dim=decoder_embed_dim * mlp_ratio,
                                                                 heads=decoder_num_heads,
                                                                 dim_head=64,
                                                                 qkv_bias=qkv_bias)
                                                for _ in range(decoder_depth)])
        self.proj_out_H1 = nn.Linear(decoder_embed_dim * 7, 435)

        self.decoder_blocks_H2 = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
                                                              output_dim=decoder_embed_dim,
                                                              hidden_dim=decoder_embed_dim * mlp_ratio,
                                                              heads=decoder_num_heads,
                                                              dim_head=64,
                                                              qkv_bias=qkv_bias)
                                             for _ in range(decoder_depth)])
        self.proj_out_H2 = nn.Linear(decoder_embed_dim * 7, 865)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_size)
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.encoder.pos_embedding.shape[-1],
                                            self.num_patches,
                                            sep_embed=True)
        self.encoder.pos_embedding.data.copy_(pos_embed.float().unsqueeze(0))
        self.encoder.pos_embedding.requires_grad = False

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    7, #self.num_patches,
                                                    sep_embed=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.sep_embedding, std=.02)
        torch.nn.init.normal_(self.mask_embedding, std=.02)
        for i in range(self.num_leads):
            torch.nn.init.normal_(self.encoder.lead_embeddings[i], std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, series):
        """
        series: (batch_size, num_leads, seq_len)
        x: (batch_size, num_leads, n, patch_size)
        """
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b c (n p) -> b c n p', p=p)
        return x

    def unpatchify(self, x):
        """
        x: (batch_size, num_leads, n, patch_size)
        series: (batch_size, num_leads, seq_len)
        """
        series = rearrange(x, 'b c n p -> b c (n p)')
        return series

  
    def sequence_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: (batch_size, num_leads, n, embed_dim)
        """
        b, num_leads, n, d = x.shape
        len_keep = int(n * (1 - mask_ratio))  ## 마스킹되지 않고 남은 패치 개수

        input_patch = x[:,:,:len_keep,:] ## 앞의 25%만 입력으로 들어갈거니깐

        mask = torch.ones([b, num_leads, n], device=x.device) ## 원래 시퀀스와 동일한 마스크 텐서 생성 (처음 모든 패치 1로 생성 => 마스킹X 부분은 0)
        mask[:, :, :len_keep] = 0 ## 마스킹                  

        return input_patch, mask
        

    def forward_encoder(self, x, mask_ratio):
        """
        x: (batch_size, num_leads, seq_len) [256, 12, 2250]
        """
        # embed patches
        x = self.to_patch_embedding(x) ## [256, 12, 30, 768]
        b, _, n, _ = x.shape

        # add positional embeddings
        x = x + self.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            # x, mask, ids_restore = self.random_masking(x, mask_ratio)
            x, mask = self.sequence_masking(x, mask_ratio)
        else:
            mask = torch.zeros([b, self.num_leads, n], device=x.device)
            ids_restore = torch.arange(n, device=x.device).unsqueeze(0).repeat(b, self.num_leads, 1)

        # apply lead indicating modules
        ## 시작과 끝 알려주는 임베딩
        # 1) SEP embedding
        sep_embedding = self.encoder.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
            ## 구분자 + 원본 시퀀스 + 구분자
            
        # 2) lead embeddings
        n_masked_with_sep = x.shape[2] ## 패치에 해당
        lead_embeddings = torch.stack([self.encoder.lead_embeddings[i] for i in range(self.num_leads)]).unsqueeze(0) ## 리드별 임베딩을 한 배열로 모음(stack) -> unsqueeze 통해서 batch 차원 추가
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n_masked_with_sep, -1) ## unsqueeze(2) 통해서 sequence 차원 추가 -> expand 통해서 배치 크기와 리드 개수, 시퀀스 길이 만큼 리드 임베딩 확장
        x = x + lead_embeddings ## 리드 임베딩 더해줌

        x = rearrange(x, 'b c n p -> b (c n) p')
        for i in range(self.encoder.depth):
            x = getattr(self.encoder, f'block{i}')(x)
        x = self.encoder.norm(x)
        ## 마스킹 되지 않은 부분만 encoder의 입력으로 들어가서 representation을 뽑음

        return x, mask
        # return x, mask, ids_restore
            ## x: 마스킹되지 않은 부분의 representation 값
            ## mask: 각 패치의 마스킹 여부
            ## ids_restore: 복원 해야할 패치 인덱스
    
    def forward_decoder(self, x):
        # 1. coeff 예측
        #  -> (관련 정보) pywt.wavedec(ecg_signal[:1725], 'db3', level=2) 기준 coeff 차원: (435,), (435), (865)
        # 2. 예측한 coeff 기준으로 pywt.waverec(coeffs2, 'db3') 실행하여 원본 복원 (예측)
        # 3. 2번 결과 반환

        ## 원래 코드
        x = self.to_decoder_embedding(x) ## Linear 층 태움. [256, 108, 256] 
            ## 108 = (7 (= 입력 패치 수) + 2 (= 앞뒤 패치)) * 12 (리드)

        ## 원래 코드
        x = rearrange(x, 'b (c n) p -> b c n p', c=self.num_leads) ## [256, 12, 9, 256] (마스킹 X)
        b, l, n_masked_with_sep, d = x.shape ## b=256, n_masked_with_sep=9, d=256

        ## 원래 코드
        ## 위치 임베딩 및 SEP 임베딩 추가
        # positional embedding and SEP embedding
        x_wo_sep = x[:,:,1:-1,:]
        x_wo_sep = x_wo_sep + self.decoder_pos_embed[:, 1:-1, :].unsqueeze(1) ## 위치 임베딩 추가 # [256, 12, 7?, 256]
        left_sep = x[:, :, :1, :] + self.decoder_pos_embed[:, :1, :].unsqueeze(1) ## 왼쪽 구분자에 위치 임베딩 추가
        right_sep = x[:, :, -1:, :] + self.decoder_pos_embed[:, -1:, :].unsqueeze(1) ## 오른쪽 구분자에 위치 임베딩 추가
        x = torch.cat([left_sep, x_wo_sep, right_sep], dim=2) ## 구분자와 시퀀스 합침 # [256, 12, 9?, 256]
        
        # lead-wise decoding
        ## 리드별 디코딩 --> coeff 예측으로 !!
        x_decoded = []
        for i in range(self.num_leads):  # 12
            x_lead = x[:, i, :, :]  # 리드별 데이터 추출 [256, 9?, 256]

            ## 저주파
            for block in self.decoder_blocks_L:  # 각 리드에 디코더 블록 적용
                x_lead = block(x_lead)  # [256, 9?, 256]
            x_lead_L = x_lead[:, 1:-1, :]  # 구분자 제거
            batch, patches, embed_dim = x_lead_L.shape
            x_lead_L = x_lead_L.view(batch, patches * embed_dim)  # 차원 변경
            x_lead_L = self.proj_out_L(x_lead_L)  # coeff 예측

            ## 고주파1
            for block in self.decoder_blocks_H1:  # 각 리드에 디코더 블록 적용
                x_lead = block(x_lead)  # [256, 9?, 256]
            x_lead_H1 = x_lead[:, 1:-1, :]  # 구분자 제거
            batch, patches, embed_dim = x_lead_H1.shape
            x_lead_H1 = x_lead_H1.view(batch, patches * embed_dim)  # 차원 변경
            x_lead_H1 = self.proj_out_H1(x_lead_H1)  # coeff 예측

            ## 고주파2
            for block in self.decoder_blocks_H2:  # 각 리드에 디코더 블록 적용
                x_lead = block(x_lead)  # [256, 9?, 256]
            x_lead_H2 = x_lead[:, 1:-1, :]  # 구분자 제거
            batch, patches, embed_dim = x_lead_H2.shape
            x_lead_H2 = x_lead_H2.view(batch, patches * embed_dim)  # 차원 변경
            x_lead_H2 = self.proj_out_H2(x_lead_H2)  # coeff 예측

            ## 예측한 coeff로 신호 복원
            recon_batches = []
            for c_batch in range(batch):
                coeffs = (
                    x_lead_L[c_batch].tolist(), #.cpu().numpy(), 
                    x_lead_H1[c_batch].tolist(), #.cpu().numpy(), 
                    x_lead_H2[c_batch].tolist(), #.cpu().numpy()
                )  # GPU -> CPU로 이동 및 NumPy 배열로 변환
                recon_signal = pywt.waverec(coeffs, 'db3')  # 신호 복원
                recon_batches.append(torch.tensor(recon_signal, device=x.device))  # 원래 device 유지

            recon_batches = torch.stack(recon_batches)
            x_decoded.append(recon_batches)

        x = torch.stack(x_decoded, dim=1)  # 리드별 디코딩 결과를 다시 합침
        return x

    # def forward_loss(self, series, pred, mask):
    def forward_loss(self, series, pred, mask_ratio):
        """
        series: (batch_size, num_leads, seq_len)
        pred: (batch_size, num_leads, n, patch_size)
        mask: (batch_size, num_leads, n), 0 is keep, 1 is remove,
        """
        ## series 데이터 패치화
        target = self.patchify(series)
        b, num_leads, n, d = target.shape
        target_part = int(n * (1 - mask_ratio))
        target = target[:,:,target_part:,:]
        
        ## 픽셀값 정규화 (옵션, default True 같음)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True) ## 패치별 평균
            var = target.var(dim=-1, keepdim=True) ## 패치별 분산
            target = (target - mean) / (var + 1.e-6)**.5 ## 정규화 수행

        ## 예측값과 실제값의 차이 제곱
        target = rearrange(target, 'b c n p -> b c (n p)', n=23)
        pred = pred[:,:,:-1]
        
        loss = (pred - target) ** 2 ## MSE Loss
        loss = loss.mean()
        
        # ## 각 패치에 대한 평균 손실 계산
        # loss = loss.mean(dim=-1)  # (batch_size, num_leads, n), mean loss per patch ## 패치 단위로 손실 정규화
        
        # ## 마스킹된 패치에 대해서만 손실 적용
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches ## 마스킹된 패치만을 대상으로 손실 계산
        return loss

    def forward(self,
                series,
                mask_ratio=0.75):
        recon_loss = 0
        pred = None
        mask = None

        # latent, mask, ids_restore = self.forward_encoder(series, mask_ratio)
        # pred = self.forward_decoder(latent, ids_restore)
        # recon_loss = self.forward_loss(series, pred, mask)
        
        latent, mask = self.forward_encoder(series, mask_ratio)
        pred = self.forward_decoder(latent)
        recon_loss = self.forward_loss(series, pred, mask_ratio)

        return {"loss": recon_loss, "pred": pred, "mask": mask}

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def st_mem_vit_small_dec256d4b(**kwargs):
    model = ST_MEM(embed_dim=384,
                   depth=12,
                   num_heads=6,
                   decoder_embed_dim=256,
                   decoder_depth=4,
                   decoder_num_heads=4,
                   mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                   **kwargs)
    return model


def st_mem_vit_base_dec256d4b(**kwargs):
    model = ST_MEM(embed_dim=768,
                   depth=12,
                   num_heads=12,
                   decoder_embed_dim=256,
                   decoder_depth=4,
                   decoder_num_heads=4,
                   mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                   **kwargs)
    return model
