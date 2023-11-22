# IPL_labeling_tool

## Transform Tracking Result Files to Labeling Format
- First, 
    ```
    conda activate suspoints
    python kradar/tracking_to_viz_format.py <path to the folder containing all the tracking files>
    ```
    the generated file will be inside the parent folder of your input path, and is named viz_format.json.

- Second,
open main.py, modify the prediction_file_path to the file path of viz_format.json.


## Start the annotation server
- First,
    ```
    conda activate suspoints
    python main
    ```
- Second, open the browser in Incognito mode and type localhost:8002 in the url bar.