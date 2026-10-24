FROM ghcr.io/astral-sh/uv:0.13.5 AS uv
FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm
COPY --from=uv /uv /uvx /usr/local/bin/
