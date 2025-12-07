FROM nginx:stable-alpine

WORKDIR /usr/share/nginx/html

RUN rm -rf ./*
COPY /web/index.html ./
COPY /web/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

