FROM ebrown/python:3.11 as assembled
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
                streamlit==1.32.2 \
                "aiohttp[speedups]" \
                blake3 \
                psutil \
                joblib \
                httpx \
                pillow \
                "pandas[performance, excel, computation, plot, output_formatting, html, parquet, hdf5]" \
                "polars[pandas, numpy, pyarrow, fsspec, connectorx, xlsx2csv, deltalake, timezone]" \
                opencv-contrib-python-headless \
                numpy \
                scikit-image && \ 
                find ./ \
                    \( \
                    \( -type d -a \( -name __pycache__ \) \) \
                    -o \
                    \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
                \) -exec rm -rf '{}' +;           
FROM rockylinux:8-minimal as prod
WORKDIR /tmp
COPY --from=assembled /opt/python/py311 /opt/python/py311
ENV LD_LIBRARY_PATH=/opt/python/py311/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/python/py311/bin:${PATH}
COPY --from=assembled /app/venv /app/venv/
ENV PATH /app/venv/bin:$PATH
WORKDIR /app
COPY . ./
ENV TERM=xterm-256color
ENV SHELL=/bin/bash
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit"]
CMD ["run", "src/app-gotmeals.py", "--server.address=0.0.0.0", "--server.port=8501"]
