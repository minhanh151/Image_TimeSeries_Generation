# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

FROM nvcr.io/nvidia/pytorch:20.12-py3


WORKDIR /app



ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install imageio-ffmpeg==0.4.3 pillow 
RUN pip install python-multipart fastapi>=0.95.0 uvicorn>=0.21.0 
RUN pip install https://github.com/podgorskiy/dnnlib/releases/download/0.0.1/dnnlib-0.0.1-py3-none-any.whl


# Unset TORCH_CUDA_ARCH_LIST and exec.  This makes pytorch run-time
# extension builds significantly faster as we only compile for the
# currently active GPU configuration.
RUN (printf '#!/bin/bash\nunset TORCH_CUDA_ARCH_LIST\nexec \"$@\"\n' >> /entry.sh) && chmod a+x /entry.sh
# ENTRYPOINT ["/entry.sh"]

# Copy code
COPY . .

# Environment variables
ENV PYTHONPATH=/app:/app/app
ENV HF_HOME=/app/huggingface_cache
ENV MODEL_TYPE=stylegan2

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]