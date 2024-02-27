## Build currently !!!BROEKN!!! due to git+https requirement
## git+https requires git which we don't want in the production image. 
## will need to do a multi-stage build
FROM ebrown/git:2.44.0 as built_git
FROM ebrown/python:3.11 as assembled
COPY --from=built_git /opt/git /opt/git
ENV PATH=/opt/git/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/git/lib:${LD_LIBRARY_PATH}
WORKDIR /app
ENV LD_LIBRARY_PATH=/opt/python/py311/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/python/py311/bin:${PATH}
RUN python3 -m virtualenv --symlinks --download /app/venv \
    && find ./ \
    		\( \
			\( -type d -a \( -name __pycache__ \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; 

RUN . /app/venv/bin/activate && \
        pip3 install --no-cache-dir --upgrade pip && \
        pip3 install --no-cache-dir --upgrade setuptools wheel && \
        pip3 install --no-cache-dir \
                torch torchvision torchaudio \
                pycocotools \
                onnxruntime \
                onnx \
                git+https://github.com/facebookresearch/segment-anything.git \
                heic2png \
                gradio \
                transformers \
                "aiohttp[speedups]" \
                blake3 \
                psutil \
                joblib \
                httpx \
                pillow \
                opencv-contrib-python-headless \
                "fastapi[all]" \
                "fastapi-cache2[redis]" \
                python-multipart \
                pydantic \
                "uvicorn[standard]"  && \
                find ./ \
                    \( \
                    \( -type d -a \( -name __pycache__ \) \) \
                    -o \
                    \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
                \) -exec rm -rf '{}' +;
WORKDIR /app
COPY ./segment_anything_webui /app/segment_anything_webui
COPY ./segment_anything_webui/images /app/segment_anything_webui/images 
RUN mkdir -p segment_anything_webui/checkpoints/models--google--owlvit-base-patch32
COPY ./checkpoints /app/segment_anything_webui/checkpoints 
COPY ./checkpoints/owlvit-base-patch32 /app/segment_anything_webui/checkpoints/models--google--owlvit-base-patch32
FROM nvidia/cuda:12.2.2-cudnn8-runtime-rockylinux8 as prod
RUN yum install dnf-plugins-core -y && \
    dnf install epel-release -y && \
    /usr/bin/crb enable -y && \
    dnf --disablerepo=cuda update -y && \
    dnf install \
                cuda-command-line-tools-12-2-12.2.2-1 \
                cuda-cudart-devel-12-2-12.2.140-1 \
                cuda-nvcc-12-2-12.2.140-1 \
                cuda-cupti-12-2-12.2.142-1 \
                cuda-nvprune-12-2-12.2.140-1 \
                cuda-nvrtc-12-2-12.2.140-1 \
                libnvinfer-plugin8-8.6.1.6-1.cuda12.0 \
                libnvinfer8-8.6.1.6-1.cuda12.0 -y &&\
                dnf clean all;
WORKDIR /tmp
COPY --from=assembled /opt/python/py311 /opt/python/py311
ENV LD_LIBRARY_PATH=/opt/python/py311/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/python/py311/bin:${PATH}
COPY --from=assembled /app/venv /app/venv/
COPY --from=assembled /app/segment_anything_webui /app/segment_anything_webui
ENV PATH /app/venv/bin:$PATH
WORKDIR /app/segment_anything_webui
ENV TERM=xterm-256color
ENV SHELL=/bin/bash
ENTRYPOINT ["python"]
CMD ["app.py"]