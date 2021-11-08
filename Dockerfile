FROM kdh1477/knu_cu11_nvidia

RUN apt-get update

COPY ./corpus /workspace/corpus

WORKDIR /workspace