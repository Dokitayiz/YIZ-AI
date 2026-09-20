#!/usr/bin/env bash
set -e
echo "1. Ensure Floot CLI is installed and authenticated."
echo "2. Set the environment variables from .env.example in Floot's dashboard."
echo "3. Attach a Postgres instance and set DATABASE_URL."
echo "4. Attach a persistent volume mounted at /app/media."
echo "5. Run the deploy command from Floot's docs, e.g.:"
echo "     floot deploy"
echo ""
echo "If Floot reads a Dockerfile, this folder is ready as-is."
echo "If Floot reads a Procfile, the Procfile is here too."
