FROM alpine/git AS novnc
RUN git clone --depth 1 --branch v1.5.0 https://github.com/novnc/noVNC.git /novnc

FROM nginx:1.27-alpine
COPY --from=novnc /novnc/core /usr/share/nginx/html/novnc/core
