# SPOTER Web

To test Spoter model in the web, follow these steps:
* Convert your latest Pytorch model to Onnx by running `python convert.py`. This is best done inside the Docker container. You will need to install additional dependencies for the conversions (see commented lines in requirements.txt)
* The ONNX should be generated in the `web` folder, otherwise copy it there.
* run `npx light-server -s . -p 8080` in the `web` folder. (`npx` comes with `npm`)

Enjoy!
