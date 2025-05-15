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


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)