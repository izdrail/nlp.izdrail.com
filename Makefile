#!/bin/sh
.PHONY: build dev down ssh publish
build:
	docker image rm -f izdrail/nlp.izdrail.com:latest && docker build -t izdrail/nlp.izdrail.com:latest --progress=plain . 
	docker-compose -f docker-compose.yml up  --remove-orphans

dev:
	docker-compose up

down:
	docker-compose down
ssh:
	docker exec -it nlp.izdrail.com /bin/zsh
publish:
	docker push izdrail/nlp.izdrail.com:latest
