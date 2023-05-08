#!/bin/bash
source ./.env
echo $client_id
axway auth login --client-id $client_id --client-secret $client_secret

# Get API Services (can scope to an env if desired)
# axway central get apiservices -o json > service-docs/apiservices.json
axway central get apiservices -o json > service-docs/apiservices.json


# Extract and store API Service names
## TO DO -- inline the get apiservices here and iterate over the result
jq --slurp -f apiservices.jq service-docs/apiservices.json > service-docs/apiservicenames.json

# Fetch APIService definition which contains the API spec.
jq -r '.[]' service-docs/apiservicenames.json | while IFS='' read -r api; do axway central get apiservicerevision -q "metadata.references.name==$api" -o json > apiservices/$api.json | jq -r .name; echo "Fetching API $api"; done;

# Extract API spec
for file in ./apiservices/*
do
    # stuff
    echo "Extracting OAS for ${file##*/}"
    jq -r '.[0].spec.definition.value' ./apiservices/${file##*/} | base64 --decode > ./apispecs/${file##*/}
done

