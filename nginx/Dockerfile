FROM alpine/git AS novnc
RUN git clone --depth 1 --branch v1.5.0 https://github.com/novnc/noVNC.git /novnc

FROM nginx:1.27-alpine
# core/ alone is not a loadable module graph: core/rfb.js statically imports
# ./inflator.js and ./deflator.js, which import ../vendor/pako/lib/zlib/*.js —
# ten files in a sibling directory. A browser resolves the whole graph before
# the module evaluates, so omitting vendor/ made the dynamic
# import('/novnc/core/rfb.js') in web/app.js fail outright (issue #102). Both
# subtrees keep their own names under the same root, which is what lets the
# relative ../vendor/ specifiers resolve.
COPY --from=novnc /novnc/core   /usr/share/nginx/html/novnc/core
COPY --from=novnc /novnc/vendor /usr/share/nginx/html/novnc/vendor
