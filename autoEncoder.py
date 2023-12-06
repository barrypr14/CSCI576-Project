import torch
from torch import nn
from torchvision.models import resnet50
import numpy as np
import cv2, math, time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Encoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        # input image size = (1, 3, 224, 224)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_1 = nn.Linear(64 * 13 * 13, 4096)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(4096, self.embedding_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.linear_1 = nn.Linear(self.embedding_size, 4096)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(4096, 64 * 13 * 13)
        self.relu_2 = nn.ReLU()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 13, 13))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 3, stride=2, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

def videoDetermine(queryVideoName,num_samples, model_ft) :
    # model_ft = resnet50()
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 20)
    # model_ft.load_state_dict(torch.load('./model/resnet50_poru.pt', map_location=torch.device('cpu')))

    # model_ft = model_ft.to(device)
    # model_ft.eval()

    frame_samples = np.zeros((num_samples, 3, 224, 224))
    cap = cv2.VideoCapture(f'./data/Queries/{queryVideoName}')
    query_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    frame_interval = math.floor(query_num_frames / num_samples)
    assert (frame_interval > 0), "number of frame samples larger than query frame numbers"

    for i in range(num_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = cap.read()
        frame = cv2.resize(frame, (224, 224))
        channel1, channel2, channel3 = cv2.split(frame)

        frame_samples[i, 0, :, :] = channel1
        frame_samples[i, 1, :, :] = channel2
        frame_samples[i, 2, :, :] = channel3

        frame_number += frame_interval


    # print("Load frames to numpy in video classification stage cost", end-start)

    input_frames = torch.from_numpy(frame_samples).type(torch.float32).to(device) / 255
    output_cat = model_ft(input_frames) # [num_samples, 20]
    _, preds = torch.max(output_cat, 1)
    preds = preds.cpu().detach().numpy()

    return max(preds)

def get_first_and_last_frames(videoName, encoder):
    frame_embedding = np.zeros((2, 100))

    cap = cv2.VideoCapture(f'./data/Queries/{videoName}')
    query_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first_frame = cap.read()
    first_frame = cv2.resize(first_frame, (224, 224))

    video_frames = np.zeros((3, 224, 224), dtype=np.uint8)
    # Split the frame into channels
    channel1, channel2, channel3 = cv2.split(first_frame)

    # Store each channel in the 3D array
    video_frames[0, :, :] = channel1
    video_frames[1, :, :] = channel2
    video_frames[2, :, :] = channel3

    test_image = torch.from_numpy(video_frames).view(-1, 3, 224, 224).to(device) / 255
    first_frame_embedding = encoder(test_image)
    frame_embedding[0] = first_frame_embedding[0].cpu().detach().numpy()

    # Set the video position to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    ret, last_frame = cap.read()
    last_frame = cv2.resize(last_frame, (224, 224))

    video_frames = np.zeros((3, 224, 224), dtype=np.uint8)
    # Split the frame into channels
    channel1, channel2, channel3 = cv2.split(last_frame)

    # Store each channel in the 3D array
    video_frames[0, :, :] = channel1
    video_frames[1, :, :] = channel2
    video_frames[2, :, :] = channel3

    test_image = torch.from_numpy(video_frames).view(-1, 3, 224, 224).to(device) / 255
    last_frame_embedding = encoder(test_image)
    frame_embedding[1] = last_frame_embedding[0].cpu().detach().numpy()

    # Release the video capture object
    cap.release()

    return frame_embedding, query_num_frames, fps

def predict(videoName, frame_embedding, query_num_frames, encoded_frames):
    # encoded_frames = np.load(f'./signatures/{videoName}.npy')
    min_dis = 10000000
    min_idx = -1
    for idx, frame in enumerate(encoded_frames):
        if(idx + query_num_frames > encoded_frames.shape[0]):
            break
        temp_dis = 0
        for a, b in zip(frame, frame_embedding[0]):
            temp_dis += abs(a - b)
        for a, b in zip(encoded_frames[idx + query_num_frames - 1], frame_embedding[1]):
            temp_dis += abs(a - b)
        if(temp_dis < min_dis):
            min_dis = temp_dis
            min_idx = idx
    return min_idx
##############
def findVideo(avg_frames, frame_embedding) :
    temp_dis = 0
    for avg, query in zip(avg_frames, frame_embedding[0]) :
        temp_dis += abs(avg - query)

    return temp_dis
