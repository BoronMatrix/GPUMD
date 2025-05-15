from fastapi import FastAPI, File, UploadFile, HTTPException, Query,Response, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
import shutil
import os
import io
import pandas as pd
import uvicorn

# micro_sam
import os
from glob import glob
from typing import Optional, Union, Tuple
from PIL import Image
from typing import List
from skimage import measure
from typing import List, Dict

import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label as connected_components
from scipy.stats import scoreatpercentile

import torch

from pydantic import BaseModel
import uuid
import json

import tempfile

from ase.build import graphene_nanoribbon
from ase.io import write

# custom-docs-ui-assets
app = FastAPI(docs_url=None, redoc_url=None)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://unpkg.com/redoc@2/bundles/redoc.standalone.js",
    )


@app.get("/")
async def root():
    return {"message": "欢迎使用图片上传服务"}

# ===== 请求模型定义 =====
class ConfigRequest(BaseModel):
    # 必填项
    potential_filename: str

    # velocity 参数
    velocity_temperature: Optional[float] = 300.0
    velocity_seed: Optional[int] = None

    # dftd3 参数
    dftd3_functional: Optional[str] = None
    dftd3_potential_cutoff: Optional[float] = None
    dftd3_coordination_number_cutoff: Optional[float] = None

    # change_box 参数
    change_box_delta: Optional[float] = None
    change_box_delta_xx: Optional[float] = None
    change_box_delta_yy: Optional[float] = None
    change_box_delta_zz: Optional[float] = None
    change_box_epsilon_yz: Optional[float] = None
    change_box_epsilon_xz: Optional[float] = None
    change_box_epsilon_xy: Optional[float] = None

    deform_A_per_step: Optional[float] = None
    deform_x: Optional[int] = None
    deform_y: Optional[int] = None
    deform_z: Optional[int] = None

    ensemble_type: Optional[str] = None  # nve, nvt_ber, npt_ber, etc.
    T_1: Optional[float] = None
    T_2: Optional[float] = None
    T_coup: Optional[float] = None
    p_hydro: Optional[float] = None
    C_hydro: Optional[float] = None
    p_xx: Optional[float] = None
    p_yy: Optional[float] = None
    p_zz: Optional[float] = None
    p_yz: Optional[float] = None
    p_xz: Optional[float] = None
    p_xy: Optional[float] = None
    C_xx: Optional[float] = None
    C_yy: Optional[float] = None
    C_zz: Optional[float] = None
    C_yz: Optional[float] = None
    C_xz: Optional[float] = None
    C_xy: Optional[float] = None
    p_coup: Optional[float] = None

    fix_group_label: Optional[str] = None  # 新增字段用于fix关键字

    time_step_dt: Optional[float] = 1
    time_step_max_distance: Optional[float] = 1

    run_steps: Optional[int] = 1  # 表示要运行的步数

    compute_hac_sampling_interval: Optional[int] = 1
    compute_hac_correlation_steps: Optional[int] = 1
    compute_hac_output_interval: Optional[int] = 1

class ConfigBuilder:
    # ===== 构建 potential 行的函数 =====
    @staticmethod
    def build_potential_line(potential_filename: str) -> str:
        if not potential_filename:
            raise ValueError("Potential filename cannot be empty.")
        return f"potential   {potential_filename}"

    # ===== 构建 velocity 行的函数 =====
    @staticmethod
    def build_velocity_line(temperature: Optional[float] = 300.0, seed: Optional[int] = None) -> str:
        line = "velocity"
        if temperature is not None:
            line += f" {temperature}"
        else:
            line += " 300"
        if seed is not None:
            line += f" seed {seed}"
        return line

    # ===== 构建 dftd3 行的函数 =====
    @staticmethod
    def build_dftd3_line(
        functional: Optional[str] = None,
        potential_cutoff: Optional[float] = None,
        coordination_number_cutoff: Optional[float] = None
    ) -> str:
        if not all([functional, potential_cutoff, coordination_number_cutoff]):
            return ""
        return f"dftd3 {functional} {potential_cutoff} {coordination_number_cutoff}"

    # ===== 构建 change_box 行的函数 =====
    @staticmethod
    def build_change_box_line(
        delta: Optional[float] = None,
        delta_xx: Optional[float] = None,
        delta_yy: Optional[float] = None,
        delta_zz: Optional[float] = None,
        epsilon_yz: Optional[float] = None,
        epsilon_xz: Optional[float] = None,
        epsilon_xy: Optional[float] = None
    ) -> str:
        # Case 1: Single parameter
        if delta is not None and all(v is None for v in [delta_xx, delta_yy, delta_zz, epsilon_yz, epsilon_xz, epsilon_xy]):
            return f"change_box {delta}"

        # Case 2: Three parameters (delta_xx, delta_yy, delta_zz)
        elif all([delta_xx is not None, delta_yy is not None, delta_zz is not None]):
            return f"change_box {delta_xx} {delta_yy} {delta_zz}"

        # Case 3: Six parameters (must have triclinic box type assumed)
        elif all([delta_xx is not None, delta_yy is not None, delta_zz is not None,
                epsilon_yz is not None, epsilon_xz is not None, epsilon_xy is not None]):
            return f"change_box {delta_xx} {delta_yy} {delta_zz} {epsilon_yz} {epsilon_xz} {epsilon_xy}"

        else:
            return ""  # 不合法或不完整的参数组合则忽略
    
    @staticmethod
    def build_deform_line(
        A_per_step: Optional[float] = None,
        deform_x: Optional[int] = None,
        deform_y: Optional[int] = None,
        deform_z: Optional[int] = None
    ) -> str:
        if not all([A_per_step is not None, deform_x is not None, deform_y is not None, deform_z is not None]):
            return ""

        if deform_x not in (0, 1) or deform_y not in (0, 1) or deform_z not in (0, 1):
            raise ValueError("deform_x, deform_y, and deform_z must be either 0 or 1.")

        return f"deform {A_per_step} {deform_x} {deform_y} {deform_z}"
    
    @staticmethod
    def build_ensemble_line(config) -> str:
        if config.ensemble_type == "nve":
            return "ensemble nve"
        elif config.ensemble_type in ["nvt_ber", "nvt_nhc", "nvt_bdp", "nvt_lan", "nvt_bao"]:
            if all([config.T_1, config.T_2, config.T_coup]):
                return f"ensemble {config.ensemble_type} {config.T_1} {config.T_2} {config.T_coup}"
        elif config.ensemble_type in ["npt_ber", "npt_scr"]:
            if config.ensemble_type == "npt_ber" and all([config.T_1, config.T_2, config.T_coup, config.p_hydro, config.C_hydro, config.p_coup]):
                return f"ensemble {config.ensemble_type} {config.T_1} {config.T_2} {config.T_coup} {config.p_hydro} {config.C_hydro} {config.p_coup}"
            elif all([config.T_1, config.T_2, config.T_coup, config.p_xx, config.p_yy, config.p_zz, config.C_xx, config.C_yy, config.C_zz, config.p_coup]):
                return f"ensemble {config.ensemble_type} {config.T_1} {config.T_2} {config.T_coup} {config.p_xx} {config.p_yy} {config.p_zz} {config.C_xx} {config.C_yy} {config.C_zz} {config.p_coup}"
            elif all([config.T_1, config.T_2, config.T_coup, config.p_xx, config.p_yy, config.p_zz, config.p_yz, config.p_xz, config.p_xy, config.C_xx, config.C_yy, config.C_zz, config.C_yz, config.C_xz, config.C_xy, config.p_coup]):
                return f"ensemble {config.ensemble_type} {config.T_1} {config.T_2} {config.T_coup} {config.p_xx} {config.p_yy} {config.p_zz} {config.p_yz} {config.p_xz} {config.p_xy} {config.C_xx} {config.C_yy} {config.C_zz} {config.C_yz} {config.C_xz} {config.C_xy} {config.p_coup}"
        return ""
    
    @staticmethod
    def build_fix_line(group_label: Optional[str]) -> str:
        if group_label is not None and group_label.strip() != "":
            return f"fix {group_label}"
        else:
            return ""
    
    @staticmethod
    def build_time_step_line(dt: Optional[float], max_distance: Optional[float]) -> str:
        if dt is None:
            return ""

        if dt <= 0:
            raise ValueError("time_step_dt must be greater than 0.")

        if max_distance is not None:
            if max_distance <= 0:
                raise ValueError("time_step_max_distance must be greater than 0.")
            return f"time_step {dt} {max_distance}"
        else:
            return f"time_step {dt}"

    @staticmethod 
    def build_run_line(steps: Optional[int]) -> str:
        if steps is None:
            return ""  # 如果没有提供步数，就不写入该行
        
        if steps <= 0:
            raise ValueError("run_steps 必须大于 0")
        
        return f"run {steps}"
    
    @staticmethod 
    def build_compute_hac_line(
        sampling_interval: Optional[int],
        correlation_steps: Optional[int],
        output_interval: Optional[int]
    ) -> str:
        if None in [sampling_interval, correlation_steps, output_interval]:
            return ""  # 如果有任何一个参数缺失，就不写入该行

        if any(v <= 0 for v in [sampling_interval, correlation_steps, output_interval]):
            raise ValueError("compute_hac 参数都必须是大于 0 的正整数")

        return f"compute_hac {sampling_interval} {correlation_steps} {output_interval}"




@app.post("/generate-config")
async def generate_config_file(config: ConfigRequest):
    content = ""
    builder = ConfigBuilder()
    # 使用 ConfigBuilder 构建各部分

    content += builder.build_potential_line(config.potential_filename) + "\n"
    content += builder.build_velocity_line(
        config.velocity_temperature,
        config.velocity_seed
    ) + "\n"
    
    dftd3_line = builder.build_dftd3_line(
        config.dftd3_functional,
        config.dftd3_potential_cutoff,
        config.dftd3_coordination_number_cutoff
    )
    if dftd3_line:
        content += dftd3_line + "\n"
        
    # 写入 change_box 行
    change_box_line = builder.build_change_box_line(
        delta=config.change_box_delta,
        delta_xx=config.change_box_delta_xx,
        delta_yy=config.change_box_delta_yy,
        delta_zz=config.change_box_delta_zz,
        epsilon_yz=config.change_box_epsilon_yz,
        epsilon_xz=config.change_box_epsilon_xz,
        epsilon_xy=config.change_box_epsilon_xy
    )
    if change_box_line:
        content += change_box_line + "\n"

    deform_line = builder.build_deform_line(
            A_per_step=config.deform_A_per_step,
            deform_x=config.deform_x,
            deform_y=config.deform_y,
            deform_z=config.deform_z
        )
    if deform_line:
        content += deform_line + "\n"

    # 添加 ensemble 行
    ensemble_line = builder.build_ensemble_line(config)
    if ensemble_line:
        content += ensemble_line + "\n"

    # 添加 fix 行
    fix_line = builder.build_fix_line(config.fix_group_label)
    if fix_line:
        content += fix_line + "\n"
    
    time_step_line = builder.build_time_step_line(
        dt=config.time_step_dt,
        max_distance=config.time_step_max_distance
    )
    if time_step_line:
        content += time_step_line + "\n"

    run_line =  builder.build_run_line(config.run_steps)
    if run_line:
        content += run_line + "\n"

    compute_hac_line = builder.build_compute_hac_line(
        config.compute_hac_sampling_interval,
        config.compute_hac_correlation_steps,
        config.compute_hac_output_interval
    )
    if compute_hac_line:
        content += compute_hac_line + "\n"
        

    # 写入空行用于分隔
    content += "\n"

    # 写入临时文件并返回下载响应
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmpfile:
        tmpfile.write(content)
        tmpfilepath = tmpfile.name

    return FileResponse(
        path=tmpfilepath,
        media_type='text/plain',
        filename="config.txt",
        headers={"Content-Disposition": "attachment; filename=config.txt"}
    )


# 存放生成的文件
OUTPUT_FILE = "model.xyz"

@app.get("/generate")
def generate_gnr(
    n: int = 60,
    m: int = 36,
    gnr_type: str = 'armchair',
    sheet: bool = True,
    vacuum: float = 3.35 / 2,
    c_c: float = 1.44
):
    """
    生成石墨烯纳米带并保存为 XYZ 文件。
    
    参数:
    - n, m: 纳米带尺寸参数
    - gnr_type: armchair 或 zigzag
    - sheet: 是否为二维结构
    - vacuum: 真空层大小
    - c_c: 碳-碳键长
    """
    try:
        gnr = graphene_nanoribbon(n, m, type=gnr_type, sheet=sheet, vacuum=vacuum, C_C=c_c)
        gnr.euler_rotate(theta=90)
        l = gnr.cell.lengths()
        gnr.cell = gnr.cell.new((l[0], l[2], l[1]))
        l = l[2]
        gnr.center()
        gnr.pbc = [True, True, False]

        # 写出 xyz 文件
        write(OUTPUT_FILE, gnr)

        return {
            "message": f"已成功生成 {gnr_type} 型石墨烯纳米带",
            "atoms": len(gnr),
            "file": OUTPUT_FILE
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download")
def download_file():
    if not os.path.exists(OUTPUT_FILE):
        raise HTTPException(status_code=404, detail="文件未生成，请先访问 /generate 生成模型")
    return FileResponse(OUTPUT_FILE, media_type='chemical/x-xyz', filename=OUTPUT_FILE)




if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)