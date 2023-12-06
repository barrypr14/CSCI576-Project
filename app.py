from pydoc import visiblename
from flask import Flask, render_template, request, session
import autoEncoder, torch, time
from torchvision.models import resnet50
import numpy as np
import torch.nn as nn

app = Flask(__name__)
app.config['upload_folder'] = './data/Queries'

@app.route('/', methods=['GET'])
def homepage() :
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload() :
    if 'video' not in request.files :
        print("Nothing")
        return render_template('index.html', error='No file part')
        
    video_file = request.files['video']
    video_name = video_file.filename.split('_')[0] # type: ignore

    # Detect the implement device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Load the pretrained model
    encoder = autoEncoder.Encoder(100).to(device) # model for finding the specific frame in the target video
    model_ft = resnet50() # model for finding target video in the database
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 20)

    if device.type == "cpu":
        encoder.load_state_dict(torch.load('./model/encoder_100.pt', map_location=torch.device('cpu')))
        model_ft.load_state_dict(torch.load('./model/resnet50_poru.pt', map_location=torch.device('cpu')))
    else:
        encoder.load_state_dict(torch.load('./model/encoder_100.pt'))
    # Turn on the evaluation mode
    encoder.eval()
    model_ft = model_ft.to(device)
    model_ft.eval()

    video_path = f'./static/{video_name}.mp4'

    start = time.time()
    # Find the target video in the database
    idx = autoEncoder.videoDetermine(video_file.filename, 5, model_ft)
    end_videoTime = time.time()
    print("video classification costs", end_videoTime - start)

    encoded_frames = np.load(f'./signatures/100/video{idx+1}.npy')
    # get the frist frame and last frame in the query video
    frame_embedding, query_num_frames, fps = autoEncoder.get_first_and_last_frames(video_file.filename, encoder)
    end_getFrame = time.time()
    print("function get_first_and_last_frames costs", end_getFrame-end_videoTime)

    # predict the start frame in the video
    min_idx = autoEncoder.predict(video_name, frame_embedding, query_num_frames, encoded_frames)
    end = time.time()
    print("comparison costs", end-end_getFrame)

    duration = end - start
    start_time = min_idx / fps # type: ignore
    end_time = start_time + query_num_frames / fps

    print(min_idx)
    print(video_path)
    print(fps)
    print("frames", query_num_frames)
    print(start_time)
    print(end_time)
    return render_template('index.html', video_path = video_path, start_time = start_time, duration = round(duration,2), end_time = end_time, start_frame = min_idx )

if __name__ == '__main__' :
    app.run(debug=True)