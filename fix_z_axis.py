import json
import os

def fix_y_axis(input_path, output_path):
    print(f"Reading from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    for frame_data in data:
        # frame_data is a dict like {"frame": 0, "nose": {"x":..., "y":..., "z":...}, ...}
        for key, value in frame_data.items():
            if isinstance(value, dict) and 'x' in value and 'y' in value and 'z' in value:
                # Flip Z axis if it's not None
                if value['z'] is not None:
                    value['z'] = -value['z']
                count += 1
                
    print(f"Processed {len(data)} frames.")
    print(f"Flipped Z axis for {count} body parts.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # Target file
    target_file = r"c:\Users\d93xj\OneDrive\Desktop\AI_Coach_Detection-tennis\data\trajectory__new\tsung__19_45(3D_trajectory_smoothed).json"
    
    # Create output filename
    dir_name = os.path.dirname(target_file)
    base_name = os.path.basename(target_file)
    name_without_ext = os.path.splitext(base_name)[0]
    new_name = f"{name_without_ext}_fixed.json"
    output_file = os.path.join(dir_name, new_name)
    
    if os.path.exists(target_file):
        fix_y_axis(target_file, output_file)
    else:
        print(f"Error: File not found: {target_file}")
