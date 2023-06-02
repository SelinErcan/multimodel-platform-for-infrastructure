FROM python:3.8.5 as b

COPY requirements.txt .

RUN pip install --user -r requirements.txt


FROM python:3.8.5-slim

WORKDIR /app

COPY --from=b /root/.local /root/.local

COPY source/ .

ENV PATH=/root/.local:$PATH

RUN python ./model.py
CMD python ./api.py



