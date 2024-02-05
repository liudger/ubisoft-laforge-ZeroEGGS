import sys
sys.path.insert(0, '/app/ZEGGS')

import json
from pathlib import Path

from flask import Flask, request, Response
import os

from ZEGGS.generate import generate_gesture


app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def run_generate():
    if 'audio' not in request.files or 'style' not in request.form:
        return 'Missing file part', 400

    # style_encoding_type = request.form.get('style_label')
    style = request.form.get('style')
    audio_file = request.files['audio']
    # filepath = request.files['filepath']
    # temperature = request.form.get('temperature')
    # seed = request.form.get('seed')
    # use_gpu = request.form.get('use_gpu')
    results_path = request.form.get('results_path')



    # bvh_stream = generate_gesture(...)
    # subprocess.run(["python", "ZEGGS/generate.py", "-o", "options.json", "-se", style_label, "-s", style, "-a", "audio.wav", "-fp", "filepath.bvh"], check=True)
    # function returns the path to the output file

    # Define the arguments
    style_encoding_type = 'label'
    first_pose = 'first_pose_bvh_file'
    temperature = 1.0
    # open json file for options
    
    path_options = "data/outputs/v2/options.json"
    options = json.load(path_options)
    
    train_options = options["train_opt"]
    network_options = options["net_opt"]
    paths = options["paths"]
    
    base_path = Path(paths["base_path"])
    data_path = base_path / paths["path_processed_data"]

    network_path = Path(paths["models_dir"])
    output_path = Path(paths["output_dir"])

    if results_path is None:
        results_path = Path(output_path) / "results"

    # Call generate_gesture
    final_style_encoding, stream = generate_gesture(
        audio_stream=audio_file, 
        styles=style, 
        network_path=network_path, 
        data_path=data_path,
        results_path=output_path,
        style_encoding_type=style_encoding_type,
        first_pose=first_pose,
        temperature=temperature,
        # seed=seed,
        # use_gpu=use_gpu,
        )
    return Response(
        stream,
        mimetype='application/octet-stream',
        headers={
            'Content-Disposition': 'attachment; filename=gesture.bvh'
        }
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)