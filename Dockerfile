#FROM arm64v8/python:3.9.20-bullseye
#
#COPY requirements.txt app.py ./
#
#RUN pip install --upgrade pip
#RUN pip install --no-cache-dir -r ./requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
#
#EXPOSE 8050
#
#CMD [ "python3", "-m" , "flask", "run", "--host" ,"0.0.0.0", "--port", "8050"]

# 构建阶段
FROM arm64v8/python:3.9.20-bullseye AS build
WORKDIR /app
COPY . /app
#RUN /usr/local/bin/pip3 install --upgrade pip
#RUN /usr/local/bin/pip3 install --no-cache-dir -r ./requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行阶段
FROM arm64v8/python:3.9.20-bullseye
WORKDIR /app
COPY --from=build /app /app
#EXPOSE 8000
EXPOSE 8050
#CMD ["gunicorn", "-w", "4", "-b", "127.0.0.1:8050", "app:app"]
#CMD ["python", "app.py"]
CMD [ "python3", "-m" , "flask", "run", "--host" ,"127.0.0.1", "--port", "8050"]