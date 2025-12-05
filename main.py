import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import glob

# Import validation functions
# Assuming these files are in the same directory
from modules.step1_reprojection_error import validate_reprojection_analysis
from modules.step2_bone_consistency import validate_bone_consistency_analysis

app = Flask(__name__, static_folder='templates')

# Configuration
# DATA_DIR is now dynamic based on request, but we can keep a default
DEFAULT_DATA_DIR = 'trajectory__2'

@app.route('/')
def index():
    return send_from_directory('templates', 'dashboard.html')

@app.route('/<path:path>')
def serve_static(path):
    # Check if file exists in templates folder
    if os.path.exists(os.path.join('templates', path)):
        return send_from_directory('templates', path)
    # Otherwise serve from root directory (for data files, etc.)
    return send_from_directory('.', path)

@app.route('/api/folders', methods=['GET'])
def list_folders():
    """List all subdirectories in the current directory."""
    try:
        # List directories in current path, excluding hidden/system ones
        folders = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.') and not d.startswith('__')]
        return jsonify({"folders": folders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    """List all JSON files in the specified directory."""
    folder = request.args.get('folder', DEFAULT_DATA_DIR)
    
    if not os.path.exists(folder):
        return jsonify({"error": f"Directory {folder} not found"}), 404
    
    files = glob.glob(os.path.join(folder, '*.json'))
    # Return relative paths
    files = [f.replace('\\', '/') for f in files]
    return jsonify({"files": files})

@app.route('/api/validate/step1', methods=['POST'])
def validate_step1():
    data = request.json
    try:
        json_3d = data.get('json_3d')
        json_2d_side = data.get('json_2d_side')
        json_2d_45 = data.get('json_2d_45')
        p1_data = data.get('p1')
        p2_data = data.get('p2')

        if not all([json_3d, json_2d_side, json_2d_45, p1_data, p2_data]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Convert matrices to numpy arrays
        P1 = np.array(p1_data)
        P2 = np.array(p2_data)

        # Run validation
        # The function returns the results dict, but also saves to a file.
        # We want the output file path to pass to the frontend.
        # We can let the function generate the default path, or specify one.
        # Let's specify one to be sure.
        
        base_name = os.path.splitext(os.path.basename(json_3d))[0]
        output_filename = f"{base_name}_step1_reprojection_error_results.json"
        # Use the directory of the input file for output
        output_dir = os.path.dirname(json_3d)
        output_path = os.path.join(output_dir, output_filename)
        
        validate_reprojection_analysis(
            json_3d,
            json_2d_side,
            json_2d_45,
            P1, P2,
            output_json_path=output_path
        )
        
        return jsonify({
            "status": "success", 
            "result_file": output_path.replace('\\', '/')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/validate/step2', methods=['POST'])
def validate_step2():
    data = request.json
    try:
        json_3d = data.get('json_3d')
        
        if not json_3d:
            return jsonify({"error": "Missing required parameters"}), 400

        base_name = os.path.splitext(os.path.basename(json_3d))[0]
        output_filename = f"{base_name}_step2_bone_consistency_results.json"
        # Use the directory of the input file for output
        output_dir = os.path.dirname(json_3d)
        output_path = os.path.join(output_dir, output_filename)

        validate_bone_consistency_analysis(
            json_3d,
            output_json_path=output_path,
            config_path=None
        )

        return jsonify({
            "status": "success", 
            "result_file": output_path.replace('\\', '/')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting server on http://localhost:5000")
    app.run(debug=True, port=5000)
